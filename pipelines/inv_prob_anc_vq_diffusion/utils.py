"""
SPDX-License-Identifier: Apache-2.0
This file contains code adapted from or interfacing with Diffusers components and remains
under the Apache License, Version 2.0. See LICENSE-APACHE-2.0 for details.
"""
import torch
import torch.nn.functional as F

def truncate(log_p_x_0: torch.FloatTensor, truncation_rate: float) -> torch.FloatTensor:
    """
    Truncates log_p_x_0 such that for each column vector, the total cumulative probability is `truncation_rate` The
    lowest probabilities that would increase the cumulative probability above `truncation_rate` are set to zero.
    """
    sorted_log_p_x_0, indices = torch.sort(log_p_x_0, 1, descending=True)
    sorted_p_x_0 = torch.exp(sorted_log_p_x_0)
    keep_mask = sorted_p_x_0.cumsum(dim=1) < truncation_rate

    # Ensure that at least the largest probability is not zeroed out
    all_true = torch.full_like(keep_mask[:, 0:1, :], True)
    keep_mask = torch.cat((all_true, keep_mask), dim=1)
    keep_mask = keep_mask[:, :-1, :]

    keep_mask = keep_mask.gather(1, indices.argsort(1))

    rv = log_p_x_0.clone()
    rv[~keep_mask] = -torch.inf  # -inf = log(0)
    return rv

def decode_image_from_gumbel_sample(vqvae, hat_z0, embeddings_shape, vqvae_dict, force_not_quantize=False):
    """
    Decode an image from a Gumbel-softmax sample.
    Input: 
        hat_z0 : [batch, pixel, class]
    """
    embeddings = torch.matmul(hat_z0, vqvae_dict)
    embeddings = embeddings.view(embeddings_shape)
    embeddings = embeddings.permute(0, 3, 1, 2).contiguous()
    img = vqvae.decode(embeddings.to(vqvae.dtype), force_not_quantize=force_not_quantize).sample.to(torch.float32)
    return img

def decode_image_from_sample(vqvae, sample, embeddings_shape):
    """
    Decode an image from a discrete latent sample.
    """
    embeddings = vqvae.quantize.get_codebook_entry(sample, shape=embeddings_shape)
    image = vqvae.decode(embeddings, force_not_quantize=True).sample
    return image