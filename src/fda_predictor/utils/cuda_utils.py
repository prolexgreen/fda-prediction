"""CUDA utilities: seeding, device resolution, AMP dtype selection."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed python, numpy, torch (CPU+CUDA) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_device(preference: str = "cuda") -> torch.device:
    if preference.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(preference)


def amp_dtype(device: torch.device, requested: str = "bfloat16") -> tuple[torch.dtype | None, bool]:
    """Return (autocast dtype, use_scaler).

    bf16 needs no GradScaler; fp16 does. Returns (None, False) on CPU,
    where the trainer runs plain fp32.
    """
    if device.type != "cuda":
        return None, False
    if requested == "bfloat16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16, False
    return torch.float16, True


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn for deterministic worker streams."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
