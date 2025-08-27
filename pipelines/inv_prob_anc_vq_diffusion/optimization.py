"""
SPDX-License-Identifier: Apache-2.0
This file contains code interfacing with Diffusers-derived components and remains
under the Apache License, Version 2.0. See LICENSE-APACHE-2.0 for details.
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import RAdam
from typing import Optional, Tuple, List
from .sampling import sample_gumbel_softmax_from_log_p, sample_cat_from_model_output

def optimize_model_output_kl(vqvae, generator, func_loss_likelihood, vqvae_dict, embeddings_shape, model_output, 
                            temperature_gumbel_softmax, lr, num_optim, coef_kl, init_model_output=None, 
                            optimize_vqvae_dict=False, loss_history_lists=None):
    """
    Optimize the model output with respect to the KL between the variational categorical and the condtional p(z_0|z_t, y)
    
    Args:
        vqvae: VQModel instance for decoding
        generator: Random generator
        func_loss_likelihood: Loss function for likelihood term
        vqvae_dict: Dictionary for VQ-VAE
        embeddings_shape: Shape of embeddings
        model_output: Model output to optimize
        temperature_gumbel_softmax: Temperature for Gumbel softmax
        lr: Learning rate
        num_optim: Number of optimization steps
        coef_kl: Coefficient for KL term
        init_model_output: Initial model output (optional)
        optimize_vqvae_dict: Whether to optimize vqvae_dict
        loss_history_lists: Lists to store loss history (optional)
    
    Returns:
        Tuple of optimized model output, vqvae_dict
    """
    
    total_loss_progress = []
    likelihood_loss_progress = []
    kl_loss_progress = []

    print("BEFORE OPTIMIZATION")
    print(f"Prior Logit {model_output[0].max(dim=0).values.mean():.4f}")

    with torch.enable_grad():
        model_output_prior = model_output.clone().detach()
        
        # Initialization
        if init_model_output is not None:
            model_output.copy_(init_model_output)

        model_output.requires_grad_(True)
        
        if optimize_vqvae_dict:
            # optimizer = optim.Adam([model_output, vqvae_dict], lr=lr)
            optimizer = optim.RAdam([model_output, vqvae_dict], lr=lr)
        else:
            # optimizer = optim.Adam([model_output], lr=lr)
            optimizer = optim.RAdam([model_output], lr=lr)

        for i in range(num_optim):
            optimizer.zero_grad()
            hat_z0 = sample_gumbel_softmax_from_log_p(model_output, temperature_gumbel_softmax, generator)
                
            # Likelihood loss
            embeddings = torch.matmul(hat_z0.transpose(1, 2), vqvae_dict)
            embeddings = embeddings.view(embeddings_shape)
            embeddings = embeddings.permute(0, 3, 1, 2).contiguous()

            img = vqvae.decode(embeddings.to(vqvae.dtype), force_not_quantize=True).sample.to(torch.float32)
            likelihood_loss = func_loss_likelihood(img)
            
            # Entropy loss
            entropy_q = torch.sum(F.softmax(model_output, dim=1) * model_output)
            crossentropy_q_p = torch.sum(F.softmax(model_output, dim=1) * model_output_prior)
            kl_loss = entropy_q - crossentropy_q_p

            total_loss = likelihood_loss + coef_kl * kl_loss

            total_loss_progress.append(total_loss.item())
            likelihood_loss_progress.append(likelihood_loss.item())
            kl_loss_progress.append((0.001*kl_loss).item())

            if i == 0 or i == num_optim-1:
                print(f"Total loss = {total_loss.item():.2f}, ||img||: {torch.norm(img):.2f}, ||embed|| = {torch.norm(embeddings):.2f} ||hat_z0|| = {torch.norm(hat_z0):.2f} likelihood loss = {likelihood_loss.item():.2f}, kl loss {coef_kl * kl_loss.item():.2f}")
                    
            total_loss.backward(retain_graph=True)
            optimizer.step()
            with torch.no_grad():   
                model_output -= torch.logsumexp(model_output, dim=1, keepdim=True)
        
        # record the history
        if loss_history_lists is not None:
            loss_history_lists[0].append(total_loss_progress)
            loss_history_lists[1].append(likelihood_loss_progress)
            loss_history_lists[2].append(kl_loss_progress)
    
    print("AFTER OPTIMIZATION")
    print(f"Optimized Logit {model_output[0].max(dim=0).values.mean():.4f}")

    return model_output, vqvae_dict

