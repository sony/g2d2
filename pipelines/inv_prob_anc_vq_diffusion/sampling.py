"""
SPDX-License-Identifier: Apache-2.0
This file contains code adapted from or interfacing with Diffusers components and remains
under the Apache License, Version 2.0. See LICENSE-APACHE-2.0 for details.
"""
import torch
import torch.nn.functional as F
from typing import Optional
from diffusers.schedulers.scheduling_vq_diffusion import gumbel_noised

def add_gumbel_noise(logits: torch.FloatTensor, generator: Optional[torch.Generator]) -> torch.FloatTensor:
    """
    Apply gumbel noise to `logits`, here, logits means unnormalized log_p.
    """
    uniform = torch.rand(logits.shape, device=logits.device, generator=generator)
    gumbel_noise = -torch.log(-torch.log(uniform+1e-30)+1e-30)
    noised = gumbel_noise + logits
    return noised

def sample_gumbel_softmax_from_log_p(logits: torch.FloatTensor, temperature: float, generator: Optional[torch.Generator]) -> torch.FloatTensor:
    """
    Sampling from the Gumbel-Softmax distribution 
    """
    gumbel_noised_logits = add_gumbel_noise(logits, generator=generator)
    gumbel_softmax_sample = F.softmax(gumbel_noised_logits/temperature, dim=1)
    return gumbel_softmax_sample

def sample_cat_from_model_output(model_output, generator):
    """
    Sample from the the model output (in logit) that is a categorical distribution.
    """
    model_output_with_gumbel_noise = gumbel_noised(model_output, generator)
    argmax_model_output_with_gumbel = model_output_with_gumbel_noise.argmax(dim=1)
    return argmax_model_output_with_gumbel