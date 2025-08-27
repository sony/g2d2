import numpy as np
import torch
import scipy
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt

class Blurkernel(nn.Module):
    def __init__(self, blur_type='gaussian', kernel_size=31, std=3.0, device=None, dtype=torch.float32):
        super().__init__()
        self.blur_type = blur_type
        self.kernel_size = kernel_size
        self.std = std
        self.device = device
        self.dtype = dtype
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(self.kernel_size//2),
            nn.Conv2d(3, 3, self.kernel_size, stride=1, padding=0, bias=False, groups=3)
        ).to(device=device, dtype=dtype)

        self.weights_init()

    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        if self.blur_type == "gaussian":
            n = np.zeros((self.kernel_size, self.kernel_size))
            n[self.kernel_size // 2, self.kernel_size // 2] = 1
            k = scipy.ndimage.gaussian_filter(n, sigma=self.std)
            k = torch.from_numpy(k).to(dtype=self.dtype, device=self.device)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)

    def update_weights(self, k):
        if not torch.is_tensor(k):
            k = torch.from_numpy(k).to(dtype=self.dtype, device=self.device)
        for name, f in self.named_parameters():
            f.data.copy_(k)

    def get_kernel(self):
        return self.k