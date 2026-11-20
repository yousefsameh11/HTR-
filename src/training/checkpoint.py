"""Checkpoint save / load and resume-training utilities."""

from __future__ import annotations

import os
import shutil
from typing import Optional

import torch


def save_checkpoint(
    state: dict,
    checkpoint_dir: str,
    filename: str = "checkpoint.pt",
    is_best: bool = False,
    best_filename: str = "best.pt",
) -> str:
    """Save a training checkpoint.

    Parameters
    ----------
    state : dict typically containing model state, optimizer state, scheduler
        state, epoch, best metric, vocab, etc.
    checkpoint_dir : directory to write into.
    filename : name of the latest checkpoint.
    is_best : if True, also copy to best_filename.
    best_filename : name of the best checkpoint.

    Returns the path of the saved checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(checkpoint_dir, best_filename)
        shutil.copyfile(path, best_path)
    return path


def load_checkpoint(
    path: str, map_location=None
) -> dict:
    """Load a checkpoint dict from disk."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if map_location is None:
        map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.load(path, map_location=map_location, weights_only=False)


def resume_training(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    map_location=None,
) -> dict:
    """Load checkpoint into model (+ optimizer / scheduler) and return state.

    Returns the loaded state dict (so caller can read epoch / best metric).
    """
    state = load_checkpoint(checkpoint_path, map_location=map_location)
    model.load_state_dict(state["model_state_dict"], strict=False)
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in state:
        try:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        except ValueError:
            # Scheduler state may not be strictly compatible; skip gracefully.
            pass
    return state