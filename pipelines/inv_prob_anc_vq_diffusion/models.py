"""
SPDX-License-Identifier: Apache-2.0
Portions copyright (c) Hugging Face

This file contains code adapted from Hugging Face Diffusers and remains
under the Apache License, Version 2.0. See LICENSE-APACHE-2.0 for details.
"""
from typing import Optional
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin

class LearnedClassifierFreeSamplingEmbeddings(ModelMixin, ConfigMixin):
    """
    Utility class for storing learned text embeddings for classifier free sampling
    """

    @register_to_config
    def __init__(self, learnable: bool, hidden_size: Optional[int] = None, length: Optional[int] = None):
        super().__init__()

        self.learnable = learnable

        if self.learnable:
            assert hidden_size is not None, "learnable=True requires `hidden_size` to be set"
            assert length is not None, "learnable=True requires `length` to be set"

            embeddings = torch.zeros(length, hidden_size)
        else:
            embeddings = None

        self.embeddings = torch.nn.Parameter(embeddings)