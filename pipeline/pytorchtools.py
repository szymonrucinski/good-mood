"""Early stopping that checkpoints the best model as a portable dict."""
from __future__ import annotations

import numpy as np
import torch


class EarlyStopping:
    """Stop training when validation loss stops improving, saving the best model.

    Saves a *checkpoint dict* ({"model_state", "classes", ...meta}) rather than
    pickling the whole nn.Module, so the artifact survives torch/torchvision
    version changes and loads with weights_only-style safety.
    """

    def __init__(
        self,
        patience: int = 7,
        verbose: bool = False,
        delta: float = 0.0,
        model_path: str = "model.pt",
        meta: dict | None = None,
        trace_func=print,
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.model_path = model_path
        self.meta = meta or {}
        self.trace_func = trace_func

    def __call__(self, val_loss: float, model: torch.nn.Module) -> None:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: torch.nn.Module) -> None:
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> "
                f"{val_loss:.6f}). Saving model ..."
            )
        ckpt = {"model_state": model.state_dict(), **self.meta}
        torch.save(ckpt, self.model_path)
        self.val_loss_min = val_loss
