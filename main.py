import argparse
import yaml
import shutil
import os
from datetime import datetime
from pytz import timezone
import itertools

import torch
import csv

import numpy as np

from pipelines import InvProbAncVQDiffusionPipeline
from typing import Dict, List, Tuple, Optional
from util.experiments import (
    build_experiment_presets,
    get_preset_dataset_defaults,
    get_default_param_grid,
    get_key_mapping,
)


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def save_results(output_dir, image, intermed_image=None, i=0):
    image.save(os.path.join(output_dir, f"res_{i}.png"))
    if intermed_image:
        intermed_image.save(os.path.join(output_dir, f"intermed_img_{i}.png"))


# Helper builders and resolvers (no behavior change)

# moved to util/experiments.py


def resolve_param_grid(task_config: dict,
                        experiment_presets: Dict[str, Dict[str, List]],
                        default_param_grid: Dict[str, List]) -> Tuple[Dict[str, List], Optional[str]]:
    experiment_preset = task_config.get('experiment_preset', None)
    if experiment_preset is not None:
        if experiment_preset not in experiment_presets:
            raise ValueError(f"Unknown experiment_preset: {experiment_preset}. Choose from {list(experiment_presets.keys())}")
        return experiment_presets[experiment_preset], experiment_preset
    return default_param_grid, None


def resolve_dataset(args,
                    task_config: dict,
                    preset_dataset_defaults: Dict[str, str],
                    experiment_preset: Optional[str]) -> str:
    dataset = args.dataset or task_config.get('dataset') or (preset_dataset_defaults.get(experiment_preset) if experiment_preset else None)
    if dataset is None:
        raise ValueError("Dataset must be specified via --dataset, task_config['dataset'], or implied by experiment_preset.")
    return dataset


def get_param_combinations(param_grid: Dict[str, List]) -> List[Dict]:
    keys, values = zip(*param_grid.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def create_pipeline(device: torch.device) -> InvProbAncVQDiffusionPipeline:
    pipeline = InvProbAncVQDiffusionPipeline.from_pretrained(
        "microsoft/vq-diffusion-ithq", torch_dtype=torch.float16
    )
    return pipeline.to(device)


def get_csv_path(dataset: str) -> str:
    if dataset == "ImageNet":
        return "dataset/imagenet_val_captions.csv"
    elif dataset == "FFHQ":
        return "dataset/ffhq_val_1k/image_captions_ffhq_val_1k.csv"
    raise ValueError(f"Unsupported dataset: {dataset}")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--task_config", type=str)
    parser.add_argument("--gpu", type=int, default=0)
    # Make dataset optional; can be provided via config or inferred from preset
    parser.add_argument('--dataset', type=str, choices=['ImageNet', 'FFHQ'], default=None, help='dataset for evaluation (ImageNet or FFHQ)')

    args = parser.parse_args()

    str_device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Device is set to {str_device}")
    device = torch.device(str_device)

    dt_now = datetime.now(timezone('UTC'))
    dt_str = dt_now.strftime('%Y%m%dT%H%M%SZ')

    task_config = load_yaml(args.task_config)
    fname_task_config = os.path.basename(args.task_config)
    fname_task_config_wo_ext = os.path.splitext(fname_task_config)[0]

    experiment_presets = build_experiment_presets()
    preset_dataset_defaults = get_preset_dataset_defaults()
    default_param_grid = get_default_param_grid()

    param_grid, experiment_preset = resolve_param_grid(task_config, experiment_presets, default_param_grid)
    dataset = resolve_dataset(args, task_config, preset_dataset_defaults, experiment_preset)

    key_mapping = get_key_mapping()

    combinations = get_param_combinations(param_grid)

    base_output_dir = os.path.join(task_config["output_dir"], fname_task_config_wo_ext + "_" + dt_str)
    os.makedirs(base_output_dir, exist_ok=True)

    for combination in combinations:
        config = task_config.copy()
        config["task"].update(combination)

        # combination_str = "_".join([f"{k}{v}" for k, v in combination.items()])
        combination_str = "_".join([f"{key_mapping.get(k, k)}{v}" for k, v in combination.items()])
        output_dir = os.path.join(base_output_dir, combination_str)
        os.makedirs(output_dir, exist_ok=True)

        config["output_dir"] = output_dir
        with open(os.path.join(output_dir, "config.yaml"), 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        pipeline = create_pipeline(device)

        path_csv = get_csv_path(dataset)

        with open(path_csv, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            total_gen = 0
            for i, row in enumerate(reader):

                should_process = False
                if dataset == "ImageNet":
                    should_process = True
                elif dataset == "FFHQ":
                    should_process = True

                if should_process:
                    image_path = row["image_path"]
                    input_prompt = row["caption"]
                    output = pipeline(image_path, input_prompt, task_config=config, suffix_save_image=f"_{i}")
                    total_gen += 1

if __name__ == "__main__":
    main()