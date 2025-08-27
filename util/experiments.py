from typing import Dict, List


def build_experiment_presets() -> Dict[str, Dict[str, List]]:
    # Paper experiment presets (RAdam)
    return {
        # Gaussian Deblur on ImageNet (best LPIPS = 0.373 at lr=15.0, lr_weight=1.0)
        'gblur_imagenet': {
            "coef_forget": [0.3],
            "coef_kl": [0.0003],
            "init_vec_kl_coef_weight": [2.0],
            "num_itr_optim_model_output_kl": [30],
            "lr_optim_model_output_kl": [15.0],
            "init_vec_lr_weight": [1.0],
            "guidance_scale": [5.0],
            "type_posterior": ["diffusion"],
        },
        # Super Resolution on ImageNet (best LPIPS = 0.360 at lr=10.0, lr_weight=2.0)
        'sr_imagenet': {
            "coef_forget": [0.3],
            "coef_kl": [0.0003],
            "init_vec_kl_coef_weight": [2.0],
            "num_itr_optim_model_output_kl": [30],
            "lr_optim_model_output_kl": [10.0],
            "init_vec_lr_weight": [2.0],
            "guidance_scale": [5.0],
            "type_posterior": ["diffusion"],
        },
        # Gaussian Deblur on FFHQ (best LPIPS = 0.292 at lr=15.0, lr_weight=1.0)
        'gblur_ffhq': {
            "coef_forget": [0.3],
            "coef_kl": [0.0003],
            "init_vec_kl_coef_weight": [2.0],
            "num_itr_optim_model_output_kl": [30],
            "lr_optim_model_output_kl": [15.0],
            "init_vec_lr_weight": [1.0],
            "guidance_scale": [1.0],
            "type_posterior": ["diffusion"],
        },
        # Super Resolution on FFHQ (best LPIPS = 0.268 at lr=10.0, lr_weight=2.0)
        'sr_ffhq': {
            "coef_forget": [0.3],
            "coef_kl": [0.0003],
            "init_vec_kl_coef_weight": [2.0],
            "num_itr_optim_model_output_kl": [30],
            "lr_optim_model_output_kl": [10.0],
            "init_vec_lr_weight": [2.0],
            "guidance_scale": [1.0],
            "type_posterior": ["diffusion"],
        },
    }


def get_preset_dataset_defaults() -> Dict[str, str]:
    return {
        'gblur_imagenet': 'ImageNet',
        'sr_imagenet': 'ImageNet',
        'gblur_ffhq': 'FFHQ',
        'sr_ffhq': 'FFHQ',
    }


def get_default_param_grid() -> Dict[str, List]:
    # Backward compatible default: keep current hardcoded SR on ImageNet if no preset provided
    return {
        "coef_forget": [0.3],
        "coef_kl": [0.0003],
        "init_vec_kl_coef_weight": [2.0],
        "num_itr_optim_model_output_kl": [30],
        "lr_optim_model_output_kl": [10.0],
        "init_vec_lr_weight": [2.0],
        "guidance_scale": [5.0],
        "type_posterior": ["diffusion"],
    }


def get_key_mapping() -> Dict[str, str]:
    return {
        "coef_forget": "fg",
        "num_itr_optim_model_output_kl": "itr_lgt",
        "num_itr_optim_for_rlc": "itr_rlc",
        "type_posterior": "type_pst",
        "init_vec_kl_coef_weight": "kl_weight",
        "init_vec_lr_weight": "lr_weight",
        "lr_optim_model_output_kl": "lr",
        "guidance_scale": "cfg",
    }
