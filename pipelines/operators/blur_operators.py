import torch
import torch.nn.functional as F
import yaml
from util.img_utils import Blurkernel

class GaussianBlurOperator:
    def __init__(self, kernel_size, intensity, device, torch_dtype=torch.float32):
        self.device = device
        self.kernel_size = kernel_size
        self.dtype = torch_dtype
        self.conv = Blurkernel(
            blur_type='gaussian',
            kernel_size=kernel_size,
            std=intensity,
            device=device, 
            dtype=torch_dtype
        )
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel)

    def forward(self, data, **kwargs):
        return self.conv(data)
    
    def loss(self, data, y, **kwargs):
        return torch.norm(y - self.forward(data, **kwargs))

class NonlinearBlurOperator:
    def __init__(self, opt_yml_path, device):
        self.device = device
        self.blur_model = self.prepare_nonlinear_blur_model(opt_yml_path)     
         
    def prepare_nonlinear_blur_model(self, opt_yml_path):
        '''
        Nonlinear deblur requires external codes (bkse).
        '''
        import sys
        sys.path.append("bkse")
        from bkse.models.kernel_encoding.kernel_wizard import KernelWizard

        with open(opt_yml_path, "r") as f:
            opt = yaml.safe_load(f)["KernelWizard"]
            model_path = opt["pretrained"]
        blur_model = KernelWizard(opt)
        blur_model.eval()
        blur_model.load_state_dict(torch.load(model_path)) 
        blur_model = blur_model.to(self.device)
        return blur_model
    
    def forward(self, data, **kwargs):
        import numpy as np
        # random_kernel = torch.randn(1, 512, 2, 2).to(self.device) * 1.2

        np.random.seed(0)
        kernel_np = np.random.randn(1, 512, 2, 2) * 1.2
        random_kernel = (torch.from_numpy(kernel_np)).float().to(self.device)
        data = (data + 1.0) / 2.0  #[-1, 1] -> [0, 1]
        blurred = self.blur_model.adaptKernel(data, kernel=random_kernel)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1) #[0, 1] -> [-1, 1]
        return blurred
    
    def loss(self, data, y, **kwargs):
        return torch.norm(y - self.forward(data, **kwargs))