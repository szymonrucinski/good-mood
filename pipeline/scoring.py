"""Evaluation metrics: accuracy, loss, confusion matrix, error analysis."""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Return (avg_loss, accuracy) over a loader, in eval mode.

    Used for BOTH validation and test so early stopping watches a real
    held-out signal (the original code validated on the train loader)."""
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss_sum += criterion(outputs, labels).item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def error_analysis(loader, model, classes: List[str], device) -> pd.DataFrame:
    """Per-example expected vs predicted, decoded to class names."""
    model.eval()
    preds, trues = [], []
    for images, labels in loader:
        outputs = model(images.to(device))
        preds.append(outputs.argmax(1).cpu().numpy())
        trues.append(labels.numpy())
    all_preds = np.concatenate(preds)
    all_trues = np.concatenate(trues)
    decode = np.array(classes)
    return pd.DataFrame(
        {"expected": decode[all_trues], "predicted": decode[all_preds]}
    )


def classification_summary(df: pd.DataFrame, classes: List[str]) -> str:
    return classification_report(
        df["expected"], df["predicted"], labels=classes, zero_division=0
    )


def confusion(df: pd.DataFrame, classes: List[str]) -> np.ndarray:
    return confusion_matrix(df["expected"], df["predicted"], labels=classes)
