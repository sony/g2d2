import torch
import torch.nn.functional as F
from functools import partial
from util.resizer import Resizer

class SuperResolutionOperator:
    def __init__(self, in_shape, scale_factor, device, torch_dtype=torch.float32):
        self.device = device
        self.up_sample = partial(F.interpolate, scale_factor=scale_factor)
        self.down_sample = Resizer(in_shape, 1/scale_factor).to(device)

    def forward(self, data, **kwargs):
        return self.down_sample(data)

    def loss(self, data, y, **kwargs):
        return torch.norm(y - self.forward(data, **kwargs))