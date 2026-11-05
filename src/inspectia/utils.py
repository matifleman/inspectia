"""Shared helpers: seeding, device selection, config loading and optional W&B."""

import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA) for reproducible runs.

    cuDNN is forced into deterministic mode; GPU results can still differ
    slightly between runs for some ops, but variation is minimized.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return CUDA if available, then Apple MPS (Mac GPU), otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: str | Path) -> dict:
    """Read one experiment YAML from configs/."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def init_wandb(config: dict, name: str, project: str = "inspectia", group: str | None = None,
               job_type: str = "experiment"):
    """Start a W&B run, or return None if wandb is not installed (local runs without it)."""
    try:
        import wandb
    except ImportError:
        print("wandb is not installed: running without experiment tracking.")
        return None
    return wandb.init(project=project, name=name, group=group, job_type=job_type, config=config)
