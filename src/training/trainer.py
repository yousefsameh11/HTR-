"""Base trainer: training loop, AdamW, LR scheduler, early stopping, checkpoints."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint


@dataclass
class TrainResult:
    history: dict = field(default_factory=dict)
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    total_time: float = 0.0


class BaseTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: str = "auto",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip: Optional[float] = 5.0,
        scheduler_mode: str = "reduce_on_plateau",
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
        early_stopping_patience: int = 10,
        checkpoint_dir: str = "checkpoints",
        run_name: str = "run",
        extra_state: Optional[dict] = None,
        use_amp: Optional[bool] = None,
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.extra_state = dict(extra_state or {})
        # Mixed precision: automatic on CUDA unless explicitly disabled.
        self.use_amp = (
            (self.device.type == "cuda") if use_amp is None else bool(use_amp)
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        if scheduler_mode == "reduce_on_plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=scheduler_factor,
                patience=scheduler_patience,
            )
        elif scheduler_mode == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(1, scheduler_patience)
            )
        else:
            self.scheduler = None

        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_dir = checkpoint_dir
        self.run_name = run_name
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.history: dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
            "epoch_time": [],
        }
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.epochs_no_improve = 0

    # ------------------------------------------------------------------ steps
    def train_epoch(self, loader: DataLoader, compute_loss: Callable) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            total_loss += self._train_step(batch, compute_loss)
            n_batches += 1
        return total_loss / max(1, n_batches)

    @torch.no_grad()
    def validate(self, loader: DataLoader, compute_loss: Callable) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            total_loss += self._eval_step(batch, compute_loss)
            n_batches += 1
        return total_loss / max(1, n_batches)

    def _train_step(self, batch, compute_loss) -> float:
        self.optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(self.model, batch, self.device, True)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite training loss; check CTC alignment lengths or lower the learning rate."
            )
        self.scaler.scale(loss).backward()
        if self.grad_clip is not None:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return float(loss.detach().cpu())

    def _eval_step(self, batch, compute_loss) -> float:
        loss = compute_loss(self.model, batch, self.device, False)
        return float(loss.detach().cpu())

    # -------------------------------------------------------------------- fit
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        compute_loss: Callable,
        epochs: int,
        start_epoch: int = 0,
    ) -> TrainResult:
        print(f"\n{'=' * 60}")
        print(f"Training on device: {self.device}")
        print(f"{'=' * 60}\n")

        start_time = time.time()
        for epoch in range(start_epoch, epochs):
            epoch_start = time.time()
            train_loss = self.train_epoch(train_loader, compute_loss)
            val_loss = self.validate(val_loader, compute_loss)

            if self.scheduler is not None:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - epoch_start

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["learning_rate"].append(current_lr)
            self.history["epoch_time"].append(epoch_time)

            improved = val_loss < self.best_val_loss
            if improved:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": (
                    self.scheduler.state_dict() if self.scheduler else None
                ),
                "val_loss": val_loss,
                "train_loss": train_loss,
                "history": self.history,
                "config": self.extra_state,
            }
            save_checkpoint(
                state,
                self.checkpoint_dir,
                filename=f"{self.run_name}_latest.pt",
                is_best=improved,
                best_filename=f"{self.run_name}_best.pt",
            )

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"lr={current_lr:.2e} | time={epoch_time:.1f}s"
            )

            if self.epochs_no_improve >= self.early_stopping_patience:
                print(
                    f"\nEarly stopping at epoch {epoch + 1} "
                    f"(no improvement for {self.early_stopping_patience} epochs)."
                )
                break

        total_time = time.time() - start_time
        self._save_history(total_time)
        print(
            f"\nTraining complete in {total_time:.1f}s. "
            f"Best val_loss={self.best_val_loss:.4f} at epoch {self.best_epoch + 1}"
        )
        return TrainResult(
            history=self.history,
            best_val_loss=self.best_val_loss,
            best_epoch=self.best_epoch,
            total_time=total_time,
        )

    def _save_history(self, total_time: float | None = None) -> None:
        history = dict(self.history)
        if total_time is not None:
            history["total_time"] = total_time
        path = os.path.join(self.checkpoint_dir, f"{self.run_name}_history.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    def load_checkpoint(self, checkpoint_path: str) -> int:
        from .checkpoint import load_checkpoint as _load

        state = _load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        if "optimizer_state_dict" in state:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if self.scheduler is not None and state.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.history = state.get("history", self.history)
        # Recover the *best* epoch, not just the last one that was written.
        val_losses = self.history.get("val_loss") or []
        if val_losses:
            best = min(val_losses)
            self.best_val_loss = float(best)
            self.best_epoch = val_losses.index(best)
            self.epochs_no_improve = len(val_losses) - 1 - self.best_epoch
        else:
            self.best_val_loss = state.get("val_loss", float("inf"))
            self.best_epoch = state.get("epoch", 0)
        return int(state.get("epoch", 0)) + 1
