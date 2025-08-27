"""
SPDX-License-Identifier: Apache-2.0
Portions copyright (c) Hugging Face

This file contains code adapted from Hugging Face Diffusers and remains
under the Apache License, Version 2.0. See LICENSE-APACHE-2.0 for details.
"""
from typing import Callable, List, Optional, Tuple, Union
import numpy as np
import os
import csv
import sys
import yaml
from functools import partial

import torch
import torch.nn.functional as F
import torch.optim as optim

from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin, Transformer2DModel, VQModel
from diffusers.schedulers import VQDiffusionScheduler
from diffusers.utils import logging
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.schedulers.scheduling_vq_diffusion import gumbel_noised, index_to_log_onehot

from .models import LearnedClassifierFreeSamplingEmbeddings
from .optimization import (
    optimize_model_output_kl
)
from .sampling import (
    add_gumbel_noise,
    sample_gumbel_softmax_from_log_p,
    sample_cat_from_model_output,
)
from .utils import truncate, decode_image_from_sample, decode_image_from_gumbel_sample

from ..operators import (
    GaussianBlurOperator,
    InpaintingOperator,
    RandomInpaintingOperator,
    SuperResolutionOperator,
    FaceIDLoss,
    NonlinearBlurOperator,
)

from util.img_utils import Blurkernel
from util.resizer import Resizer
from util.arcface.model import IDLoss
import util.plot_utils as plot_utils

from PIL import Image
import matplotlib.pyplot as plt
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class InvProbAncVQDiffusionPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using VQ Diffusion

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vqvae ([`VQModel`]):
            Vector Quantized Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent
            representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. VQ Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        transformer ([`Transformer2DModel`]):
            Conditional transformer to denoise the encoded image latents.
        scheduler ([`VQDiffusionScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
    """


    vqvae: VQModel
    text_encoder: CLIPTextModel
    tokenizer: CLIPTokenizer
    transformer: Transformer2DModel
    learned_classifier_free_sampling_embeddings: LearnedClassifierFreeSamplingEmbeddings
    scheduler: VQDiffusionScheduler

    def __init__(
        self,
        vqvae: VQModel,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        transformer: Transformer2DModel,
        scheduler: VQDiffusionScheduler,
        learned_classifier_free_sampling_embeddings: LearnedClassifierFreeSamplingEmbeddings,
    ):
        super().__init__()

        self.register_modules(
            vqvae=vqvae,
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            learned_classifier_free_sampling_embeddings=learned_classifier_free_sampling_embeddings,
        )
        
    def _encode_prompt(self, prompt, num_images_per_prompt, do_classifier_free_guidance):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        # get prompt text embeddings
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        if text_input_ids.shape[-1] > self.tokenizer.model_max_length:
            removed_text = self.tokenizer.batch_decode(text_input_ids[:, self.tokenizer.model_max_length :])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )
            text_input_ids = text_input_ids[:, : self.tokenizer.model_max_length]
        prompt_embeds = self.text_encoder(text_input_ids.to(self.device))[0]

        # NOTE: This additional step of normalizing the text embeddings is from VQ-Diffusion.
        # While CLIP does normalize the pooled output of the text transformer when combining
        # the image and text embeddings, CLIP does not directly normalize the last hidden state.
        #
        # CLIP normalizing the pooled output.
        # https://github.com/huggingface/transformers/blob/d92e22d1f28324f513f3080e5c47c071a3916721/src/transformers/models/clip/modeling_clip.py#L1052-L1053
        prompt_embeds = prompt_embeds / prompt_embeds.norm(dim=-1, keepdim=True)

        # duplicate text embeddings for each generation per prompt
        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)

        if do_classifier_free_guidance:
            if self.learned_classifier_free_sampling_embeddings.learnable:
                negative_prompt_embeds = self.learned_classifier_free_sampling_embeddings.embeddings
                negative_prompt_embeds = negative_prompt_embeds.unsqueeze(0).repeat(batch_size, 1, 1)
            else:
                uncond_tokens = [""] * batch_size

                max_length = text_input_ids.shape[-1]
                uncond_input = self.tokenizer(
                    uncond_tokens,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                negative_prompt_embeds = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
                # See comment for normalizing text embeddings
                negative_prompt_embeds = negative_prompt_embeds / negative_prompt_embeds.norm(dim=-1, keepdim=True)

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def calc_log_p_z0_zt(self, guidance_scale, truncation_rate, do_classifier_free_guidance, prompt_embeds, sample, t):
        latent_model_input = torch.cat([sample] * 2) if do_classifier_free_guidance else sample

            # predict the un-noised image
            # model_output == `log_p_x_0`
        model_output = self.transformer(latent_model_input, encoder_hidden_states=prompt_embeds, timestep=t).sample

        if do_classifier_free_guidance:
            model_output_uncond, model_output_text = model_output.chunk(2)
            model_output = model_output_uncond + guidance_scale * (model_output_text - model_output_uncond)
            model_output -= torch.logsumexp(model_output, dim=1, keepdim=True)

        model_output = truncate(model_output, truncation_rate)

            # remove `log(0)`'s (`-inf`s)
        model_output = model_output.clamp(-70)
        return model_output

    def realloc_mask_model_output(self, model_output, model_output_bf_optimized, sample, k=1):
        """
        Compare model_output and model_output_bf_optimized, 
        
        and for tokens in the sample that are unmasked, 
        replace the top k tokens with the largest differences 
        between model_output_bf_optimized and model_output. 
        These are the tokens that need the most adjustment.

        On the other hand, for masked tokens within the sample, 
        replace the top k tokens with the ones having the highest values in model_output, 
        indicating the most certain tokens. 
        These are replaced with tokens that have the highest probabilities.
        """

        model_output = model_output.clamp(-70)
        model_output_bf_optimized = model_output_bf_optimized.clamp(-70)
        
        # count the already masked

        mask_class = model_output.shape[1]
        indices_unmasked = torch.nonzero(sample[0] != mask_class).squeeze(1)
        # n_unmasked = torch.sum(sample[0] != mask_class).item()
        _, indices_to_be_masked = torch.sort(
            torch.max(model_output_bf_optimized[:, :, indices_unmasked] - model_output[:, :, indices_unmasked], dim=1).values, 
            descending=True)

        indices_masked = torch.nonzero(sample[0] == mask_class).squeeze(1)
        values_max_model_output, indices_max_model_output = torch.max(model_output[:, :, indices_masked], dim=1)
        _, indices_to_be_unmasked = torch.sort(
            values_max_model_output,
            descending=True
        )

        k = min([indices_unmasked.numel(), indices_masked.numel(), k])

        sample[:, indices_unmasked[indices_to_be_masked.squeeze(0)[:k]]] = mask_class
        sample[:, indices_masked[indices_to_be_unmasked.squeeze(0)[:k]]] = indices_max_model_output[:, indices_to_be_unmasked.squeeze(0)[:k]]

        return sample

    def posterior_using_conf_score(self, model_output, conf_score, truncation_rate, timestep, sample_zt, generator, decouple=False):
        """
         it assumes that the posterior sampling method ("diffusion", "conf_score") remains constant throughout the inference. (There is a possibility of an error occurring due to the number of masks.)
        """

        # TODO: truncation rate
        truncation_rate = 0.95
        model_output = truncate(model_output, truncation_rate)
        model_output -= torch.logsumexp(model_output, dim=1, keepdim=True)
        sample_z0 = sample_cat_from_model_output(model_output, generator)
        
        conf_score_sample = torch.gather(conf_score, 1, sample_z0.unsqueeze(1)).squeeze(1)

        if timestep != 0:
            mask_class = self.scheduler.mask_class
            num_latent_pixels = sample_zt.shape[1]
            log_cumprod_ct = self.scheduler.log_cumprod_ct[timestep-1]
            log_cumprod_bt = self.scheduler.log_cumprod_bt[timestep-1]
            num_mask = int(num_latent_pixels * (torch.exp(log_cumprod_ct)-torch.exp(log_cumprod_bt)))
            
            num_mask_current = torch.sum(sample_zt == mask_class, dim=1)

            num_token_to_unmask = num_mask_current - num_mask

            if decouple is False:
                conf_score_sample[sample_zt != mask_class] = -float('inf')

            nb_sample = sample_zt.shape[0]

            for i in range(nb_sample):
                
                if decouple is False:
                    _, to_unmask = torch.topk(conf_score_sample[i].view(-1), k=num_token_to_unmask)
                    sample_zt[i].view(-1)[to_unmask] = sample_z0[i].view(-1)[to_unmask]
                    sample_zt_m1 = sample_zt
                else:
                    _, to_mask = torch.topk(conf_score_sample[i].view(-1), k=num_mask, largest=False)
                    sample_z0[i].view(-1)[to_mask] = mask_class
                    sample_zt_m1 = sample_z0
        else:
            sample_zt_m1 = sample_z0

        return sample_zt_m1

    def optimize_embed_and_resample(self, generator, loss_func, mask_class, sample, embeddings_shape, model_output, lr_embed, n_optim_embed, token_update_rate):
        """
            
            - After sampling z_0 from p(z_0|z_t), this func. optimizes the corresponding latent variables w.r.t ||y-f(x)||.
            - Quantize the optimized continuous latent variables and replace unmasked tokens that had more changes after the optimization with new tokens.

            Note:
                the number of replacements: [# of unmasked tokens] x token_update_rate

        """
        model_output_with_gumbel_noise = gumbel_noised(model_output, generator)
        argmax_model_output_with_gumbel = model_output_with_gumbel_noise.argmax(dim=1)
        embeddings = self.vqvae.quantize.get_codebook_entry(argmax_model_output_with_gumbel, shape=embeddings_shape)
        embeddings_bf_optim = embeddings.clone().detach()

        with torch.enable_grad():
            embeddings.requires_grad_(True)
            optimizer = optim.Adam([embeddings], lr=lr_embed, eps=1e-6)
            for i_optim in range(n_optim_embed):
                optimizer.zero_grad()

                        # image = self.vqvae.decode(embeddings.to(torch.float16), force_not_quantize=True).sample.to(torch.float32)
                image = self.vqvae.decode(embeddings.to(self.vqvae.dtype), force_not_quantize=True).sample.to(torch.float32)
                loss_reconst = loss_func(image)
                        
                if i_optim % 5 == 0:
                    print(f"reconst loss = {loss_reconst.item():.2f}, ||img||: {torch.norm(image):.2f}, ||embed|| = {torch.norm(embeddings):.2f}")

                loss_reconst.backward(retain_graph=True)
                optimizer.step()
                
        embeddings.requires_grad_(False)
        indices_optim_emb = self.vqvae.quantize(embeddings)[2][2].reshape(sample.shape)

        indices_unmask = torch.nonzero(sample[0] != mask_class).squeeze(dim=1)

        norm_emb_diff = torch.flatten((embeddings-embeddings_bf_optim), 2, 3).norm(dim=1).squeeze(dim=0)
        norm_emb_diff_unmask = norm_emb_diff[indices_unmask]
        _, indices_sorted_larger_err = torch.sort(norm_emb_diff_unmask, dim=0, descending=True)
                
        num_tokens_updated = int(np.floor(indices_unmask.shape[0] * token_update_rate))
        num_tokens_unchanged = int(torch.sum(sample[:, indices_unmask][:, indices_sorted_larger_err[:num_tokens_updated]] == indices_optim_emb[:, indices_unmask][:, indices_sorted_larger_err[:num_tokens_updated]]))
        sample[:, indices_unmask][:, indices_sorted_larger_err[:num_tokens_updated]] = indices_optim_emb[:, indices_unmask][:, indices_sorted_larger_err[:num_tokens_updated]]
                
        print(f"the number of samples updated: {num_tokens_updated - num_tokens_unchanged} / {num_tokens_updated}")

    @torch.no_grad()
    def __call__(
        self,
        image_path: str,
        prompt: Union[str, List[str]],
        num_inference_steps: int = 100,
        guidance_scale: float = 5.0,
        truncation_rate: float = 1.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        task_config=None,
        suffix_save_image=None
    ) -> Union[ImagePipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            truncation_rate (`float`, *optional*, defaults to 1.0 (equivalent to no truncation)):
                Used to "truncate" the predicted classes for x_0 such that the cumulative probability for a pixel is at
                most `truncation_rate`. The lowest probabilities that would increase the cumulative probability above
                `truncation_rate` are set to zero.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor` of shape (batch), *optional*):
                Pre-generated noisy latents to be used as inputs for image generation. Must be valid embedding indices.
                Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will
                be generated of completely masked latent pixels.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~ pipeline_utils.ImagePipelineOutput `] if `return_dict`
            is True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """
        
        default_diffusion_config = {
            "num_inference_steps": 100,
            "guidance_scale": 5.0,
            "truncation_rate": 1.0
        }

        task_config["diffusion"] = {**default_diffusion_config, **task_config["diffusion"]}

        num_inference_steps = task_config["diffusion"]["num_inference_steps"]
        guidance_scale      = task_config["diffusion"]["guidance_scale"]
        truncation_rate     = task_config["diffusion"]["truncation_rate"]
        # prompt=task_config["task"]["prompt"]
        forward_type = task_config["task"]["forward_type"]

        # ---------------------------------------------------------------------------------
        # TODO: This part should be modified so that it can be applied to various type of corruption process.
        import torchvision.transforms as transforms

        # load ground truth image
        # pil_image = Image.open("./reference_img/0/orig_2.png")
        pil_image = Image.open(image_path)
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        dir_gt = os.path.join(task_config["output_dir"], "3_gt")
        os.makedirs(dir_gt, exist_ok=True)
        pil_image.save(os.path.join(dir_gt, "gt"+suffix_save_image+".png"))

        img_gt = transform(pil_image).unsqueeze(0)
        img_gt = 2.0 * img_gt - 1.0
        img_gt = img_gt.to(self.device).to(torch.float32)

        # forward_type="gaussian_blur"
        # forward_type="inpainting"
        
        if forward_type == "inpainting":

            inpaint_mask = torch.ones_like(img_gt)
            mask_size = 128
            inpaint_mask[:, :, 128-mask_size//2:128+mask_size//2, 128-mask_size//2:128+mask_size//2] = 0.0

            self.forward_op = InpaintingOperator(mask=inpaint_mask, device=self.device)
        elif forward_type == "gaussian_blur":
            self.forward_op = GaussianBlurOperator(kernel_size=61, intensity=3.0, device=self.device, torch_dtype=torch.float32)
        elif forward_type == "super_resolution":
            in_shape = task_config["task"]["in_shape"]
            scale_factor = task_config["task"]["scale_factor"]
            self.forward_op = SuperResolutionOperator(in_shape, scale_factor, device=self.device)
        elif forward_type == "faceid":
            ref_path = task_config["task"]["ref_path"]
            self.forward_op = FaceIDLoss(ref_path, device=self.device)
        elif forward_type == "random_inpainting":
            
            random_mask_rate = task_config["task"]["random_mask_rate"]
            self.forward_op = RandomInpaintingOperator(img_gt.shape, random_mask_rate, device=self.device)
        elif forward_type == "nonlinear_blur":
            self.forward_op = NonlinearBlurOperator(opt_yml_path="./bkse/options/generate_blur/default.yml", device=self.device)

        # Loss function
        if forward_type in ["inpainting", "gaussian_blur", "super_resolution", "random_inpainting", "nonlinear_blur"]:
            # => inverse problem

            sigma_mes_noise = task_config["task"].get("sigma_mes_noise", 0.1)

            y = self.forward_op.forward(img_gt)
            y += sigma_mes_noise * torch.randn_like(y)
            self.loss_func = partial(self.forward_op.loss, y=y)
            pil_y = plot_utils.convert_to_pil(y)

            mes_dir = os.path.join(task_config["output_dir"], "2_mes")
            os.makedirs(mes_dir, exist_ok=True)
            pil_y[0].save(os.path.join(mes_dir, "mes"+suffix_save_image+".png"))
        else:
            # => guided generation 
            self.loss_func = self.forward_op.loss


        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        batch_size = batch_size * num_images_per_prompt

        do_classifier_free_guidance = guidance_scale > 1.0 or guidance_scale < 1.0

        prompt_embeds = self._encode_prompt(prompt, num_images_per_prompt, do_classifier_free_guidance)

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        # get the initial completely masked latents unless the user supplied it

        latents_shape = (batch_size, self.transformer.num_latent_pixels)
        if latents is None:
            mask_class = self.transformer.num_vector_embeds - 1
            latents = torch.full(latents_shape, mask_class).to(self.device)
        else:
            if latents.shape != latents_shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")
            if (latents < 0).any() or (latents >= self.transformer.num_vector_embeds).any():
                raise ValueError(
                    "Unexpected latents value(s). All latents be valid embedding indices i.e. in the range 0,"
                    f" {self.transformer.num_vector_embeds - 1} (inclusive)."
                )
            latents = latents.to(self.device)

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        # timesteps_tensor = self.scheduler.timesteps.to(self.device)

        # Corrected time steps
        num_train_timesteps = self.scheduler.log_at.shape[0]
        timesteps_tensor = torch.from_numpy((np.round(np.linspace(num_train_timesteps, 0, num_inference_steps+1)[1:])).astype(int))
        timesteps_tensor = timesteps_tensor.to(self.device)
        
        sample = latents

        vqvae_dict = self.vqvae.quantize.embedding.weight.to(torch.float32)
        embedding_channels = self.vqvae.config.vq_embed_dim
        embeddings_shape = (batch_size, self.transformer.height, self.transformer.width, embedding_channels)

        init_vec_kl_coef_weight = task_config["task"].get("init_vec_kl_coef_weight", 0)
        init_vec_lr_weight = task_config["task"].get("init_vec_lr_weight", 0)
        vec_temperature_gumbel_softmax = 10**(np.linspace(0, -0.2, num_inference_steps))
        vec_lr_weight                    = 10**(np.linspace(init_vec_lr_weight/2, -init_vec_lr_weight/2, num_inference_steps))
        vec_kl_coef_weight               = 10**(np.linspace(init_vec_kl_coef_weight/2, -init_vec_kl_coef_weight/2, num_inference_steps))
        # vec_lr_coef = 10**(np.linspace(0.0, 0.0, num_inference_steps))

        pil_figs = {
            "init": [],
            "af. rlc": [],
            "af. mo optim": []
        }

        ts_save_fig = []
        # optimize model output with respect to the loss
        # temperature_gumbel_softmax = vec_temperature_gumbel_softmax[i]
        
        # lr = vec_lr[i]

        vec_rlc_rate = np.linspace(0.5, 0.05, num_inference_steps)
        model_output_optimized = None

        self.list_total_loss_progress = []
        self.list_likelihood_loss_progress = []
        self.list_kl_loss_progress = []
        loss_history_lists = [self.list_total_loss_progress, self.list_likelihood_loss_progress, self.list_kl_loss_progress]

        t_start_guidance = task_config["task"].get("t_start_guidance", num_inference_steps)
        config_guidance_scale = task_config["task"].get("guidance_scale", guidance_scale)

        for i, t in enumerate(self.progress_bar(timesteps_tensor)):
            
            if t > t_start_guidance:
                guidance_scale = 0
            else:
                guidance_scale = config_guidance_scale

            flg_save_fig = (t.item() % 10 == 5) and task_config["task"]["save_intermed_figs"]
            if flg_save_fig:
                ts_save_fig.append(t.item())
                model_output = self.calc_log_p_z0_zt(guidance_scale, truncation_rate, do_classifier_free_guidance, prompt_embeds, sample, t)
                model_output_init = model_output.clone().detach()
                z0_sample_init = sample_cat_from_model_output(model_output_init, generator)
                image_init = decode_image_from_sample(self.vqvae, z0_sample_init, embeddings_shape)
                pil_figs["init"].append(plot_utils.convert_to_pil(image_init))

            # Mask reallocation algorithm 
            num_itr_optim_for_rlc = task_config["task"].get("num_itr_optim_for_rlc", 0)
            if num_itr_optim_for_rlc > 0:

                lr_optim_rlc = task_config["task"]["lr_optim_rlc"] * vec_lr_weight[i]
                coef_kl       = task_config["task"]["coef_kl"] * vec_kl_coef_weight[i]
                temperature_gumbel_softmax = task_config["task"].get("temperature_optim_model_output_kl", 1.0)

                rlc_rate = vec_rlc_rate[i]
                num_unmasked = int(torch.sum(sample[0] != mask_class))
                num_rlc_rate = int(np.floor(num_unmasked * rlc_rate))
                print(f"\n# of unmasked: num_unmaksed {num_unmasked}, # of rlc: {num_rlc_rate}")

                # num_rlc_rate = num_optim_rlc
                model_output = self.calc_log_p_z0_zt(guidance_scale, truncation_rate, do_classifier_free_guidance, prompt_embeds, sample, t)
                model_output_bf_rlc = model_output.clone().detach()
                                
                # Utilize the results optimized in the last step here.
                coef_forget = task_config["task"].get("coef_forget", 1.0)
                if model_output_optimized is not None:
                    init_model_output = coef_forget * model_output.clone().detach() + (1-coef_forget) * model_output_optimized
                    init_model_output -= torch.logsumexp(init_model_output, dim=1, keepdim=True)
                else:
                    init_model_output = None

                model_output, vqvae_dict = optimize_model_output_kl(
                    self.vqvae, 
                    generator, 
                    self.loss_func, 
                    vqvae_dict, 
                    embeddings_shape, 
                    model_output, 
                    temperature_gumbel_softmax, 
                    lr_optim_model_output_kl, 
                    num_itr_optim_model_output_kl, 
                    coef_kl,
                    init_model_output=init_model_output,
                    optimize_vqvae_dict=optimize_vqvae_dict,
                    loss_history_lists=loss_history_lists  #
                )

                model_output_optimized = model_output.clone().detach()

                sample = self.realloc_mask_model_output(model_output, model_output_bf_rlc, sample, k=num_rlc_rate)

                if flg_save_fig:
                    
                    model_output_af_rlc = self.calc_log_p_z0_zt(guidance_scale, truncation_rate, do_classifier_free_guidance, prompt_embeds, sample, t)
                    z0_sample_af_rlc = sample_cat_from_model_output(model_output_af_rlc, generator)
                    image_af_rlc = decode_image_from_sample(self.vqvae, z0_sample_af_rlc, embeddings_shape)
                    pil_figs["af. rlc"].append(plot_utils.convert_to_pil(image_af_rlc))

            model_output = self.calc_log_p_z0_zt(guidance_scale, truncation_rate, do_classifier_free_guidance, prompt_embeds, sample, t)

            # Optimize the model output and sample from p(z_{t-1}|z_t, y)
            num_itr_optim_model_output_kl = task_config["task"].get("num_itr_optim_model_output_kl", 0)
            if num_itr_optim_model_output_kl > 0:
                lr_optim_model_output_kl = task_config["task"]["lr_optim_model_output_kl"] * vec_lr_weight[i]
                coef_kl = task_config["task"]["coef_kl"] * vec_kl_coef_weight[i]
                temperature_optim_model_output_kl = task_config["task"]["temperature_optim_model_output_kl"]
                coef_forget = task_config["task"].get("coef_forget", 1.0)
                
                # coef_forget = 0.5
                if model_output_optimized is not None:
                    init_model_output = coef_forget * model_output.clone().detach() + (1-coef_forget) * model_output_optimized
                    init_model_output -= torch.logsumexp(init_model_output, dim=1, keepdim=True)
                else:
                    init_model_output = None

                # if t <= 10:
                #     optimize_vqvae_dict = True
                # else:
                #     optimize_vqvae_dict = False
                optimize_vqvae_dict = False

                model_output, vqvae_dict = optimize_model_output_kl(
                    self.vqvae, 
                    generator, 
                    self.loss_func, 
                    vqvae_dict, 
                    embeddings_shape, 
                    model_output, 
                    temperature_optim_model_output_kl, 
                    lr_optim_model_output_kl, 
                    num_itr_optim_model_output_kl, 
                    coef_kl,
                    init_model_output=init_model_output,
                    optimize_vqvae_dict=optimize_vqvae_dict,
                    loss_history_lists=loss_history_lists  #
                )
                self.vqvae.quantize.embedding.weight.data = vqvae_dict.to(torch.float16)

                model_output_optimized = model_output.clone().detach()

                if flg_save_fig:
                    z0_sample_af_mo_optim = sample_cat_from_model_output(model_output, generator)
                    image_af_mo_optim = decode_image_from_sample(self.vqvae, z0_sample_af_mo_optim, embeddings_shape)
                    pil_figs["af. mo optim"].append(plot_utils.convert_to_pil(image_af_mo_optim))

            model_output = model_output.clamp(-70)
            model_output.requires_grad_(False)

            # TODO: this part should be removed. This procedure is for experimental purpose..
            # from diffusers.schedulers.scheduling_vq_diffusion import index_to_log_onehot
            # gumbel_added_model_output = gumbel_noised(model_output, generator)
            # sample_gumbel_added_model_output = gumbel_added_model_output.argmax(dim=1)
            # log_onehot_model_output = index_to_log_onehot(sample_gumbel_added_model_output, mask_class)
            # model_output = log_onehot_model_output

            # compute the previous noisy sample x_t -> x_t-1
            # type_posterior = "diffusion"
            type_posterior = task_config["task"].get("type_posterior", "diffusion")
            if type_posterior == "diffusion":
                sample = self.scheduler.step(model_output, timestep=t, sample=sample, generator=generator).prev_sample
            elif type_posterior == "conf_score_likelihood":
                conf_score = model_output
                sample = self.posterior_using_conf_score(model_output, conf_score=conf_score, truncation_rate=truncation_rate, timestep=t, sample_zt=sample, generator=generator)
            elif type_posterior == "conf_score_random":
                conf_score = torch.rand_like(model_output)
                sample = self.posterior_using_conf_score(model_output, conf_score=conf_score, truncation_rate=truncation_rate, timestep=t, sample_zt=sample, generator=generator)
            elif type_posterior == "decouple_random":
                conf_score = torch.rand_like(model_output)
                sample = self.posterior_using_conf_score(model_output, conf_score=conf_score, truncation_rate=truncation_rate, timestep=t, sample_zt=sample, generator=generator, decouple=True)
            elif type_posterior == "decouple_likelihood":
                conf_score = model_output
                sample = self.posterior_using_conf_score(model_output, conf_score=conf_score, truncation_rate=truncation_rate, timestep=t, sample_zt=sample, generator=generator, decouple=True)

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, sample)

        embedding_channels = self.vqvae.config.vq_embed_dim
        embeddings_shape = (batch_size, self.transformer.height, self.transformer.width, embedding_channels)
        embeddings = self.vqvae.quantize.get_codebook_entry(sample, shape=embeddings_shape)
        image = self.vqvae.decode(embeddings, force_not_quantize=True).sample.to(torch.float32)

        # The loss_likelihoodd:
        loss_output = self.loss_func(image).item()

        image = image.clamp(-1, 1)
        img_gt = img_gt.clamp(-1, 1)

        fun_lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(self.device)
        score_lpips = fun_lpips(img_gt.to(self.device), image.to(self.device)) # for LPIPS, image must be in the range in [-1, 1]

        image = (image / 2 + 0.5).clamp(0, 1)
        img_gt = (img_gt/ 2 + 0.5).clamp(0, 1)
        mse = torch.mean((image - img_gt) ** 2)
        psnr = 10 * torch.log10(1 / mse)
        
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        path_csv = os.path.join(task_config["output_dir"], "metrics.csv")
        if not os.path.exists(path_csv):
            with open(path_csv, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["suffix", "Likelihood Loss", "PSNR", "LPIPS"])
        with open(path_csv, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([suffix_save_image, f"{loss_output:.2f}", f"{psnr.item():.2f}", f"{score_lpips.item():.3f}"])

        if output_type == "pil":
            image = self.numpy_to_pil(image)
            dir_res = os.path.join(task_config["output_dir"], "0_res")
            os.makedirs(dir_res, exist_ok=True)
            # TODO: if batch_size >= 2, it will not work properly
            image[0].save(os.path.join(dir_res, "res"+suffix_save_image+".png"))

        if task_config["task"]["save_intermed_figs"]:
            pil_intermed_image = plot_utils.savefig_tiled_pil(pil_figs, ts_save_fig, os.path.join(task_config["output_dir"], "intermed_images.png"))
            # image.append(pil_intermed_image)
            dir_intermed = os.path.join(task_config["output_dir"], "1_intermed")
            os.makedirs(dir_intermed, exist_ok=True)
            pil_intermed_image.save(os.path.join(dir_intermed, "intermed"+suffix_save_image+".png"))
        
        if task_config["task"].get("save_loss_history", False):
            
            if len(self.list_total_loss_progress) > 0:
                dir_loss_history = os.path.join(task_config["output_dir"], "4_loss_history")
                os.makedirs(dir_loss_history, exist_ok=True)

                list_total_loss_plot = []
                list_likelihood_loss_plot = []
                list_kl_loss_plot = []
                for i, t in enumerate(timesteps_tensor):
                    if t.item() % 10 == 5:
                        list_total_loss_plot += self.list_total_loss_progress[i]
                        list_likelihood_loss_plot += self.list_likelihood_loss_progress[i]
                        list_kl_loss_plot += self.list_kl_loss_progress[i]
                        
                plt.figure()
                plt.semilogy(list_total_loss_plot, label="Total loss")
                plt.semilogy(list_likelihood_loss_plot, label="Likelihood loss")
                plt.semilogy(list_kl_loss_plot, label="KL loss")
                plt.ylim([0.1, 200])
                plt.ylabel("Loss Value")
                plt.title("Loss Evolution")
                plt.legend()
                plt.savefig(os.path.join(dir_loss_history, "loss_history"+suffix_save_image+".png"))
                plt.close()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)