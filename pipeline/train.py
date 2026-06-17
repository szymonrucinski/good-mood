"""Train the good-mood emotion classifier (reproducibly).

Run from the repo root:
    uv run python -m pipeline.train --epochs 30

Pipeline: EMO-DB wav -> MEL spectrogram PNGs -> ResNet18 transfer learning.
Everything is seeded; the best model (lowest val loss) is checkpointed.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from utils.core import (
    EMOTIONS,
    SEED,
    build_model,
    get_transforms,
    make_generator,
    seed_worker,
    set_seed,
)
from pipeline import preprocessing, scoring

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")

REPO_ROOT = Path(__file__).resolve().parents[1]


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_loaders(df, batch_size, seed):
    """Stratified 80/10/10 train/val/test split, seeded for reproducibility."""
    df = preprocessing.encode_targets(df)
    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["target"], random_state=seed
    )
    valid_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["target"], random_state=seed
    )
    log.info(
        "split: train=%d valid=%d test=%d", len(train_df), len(valid_df), len(test_df)
    )

    gen = make_generator(seed)
    train_ds = preprocessing.EmoDataset(train_df, transform=get_transforms(train=True))
    valid_ds = preprocessing.EmoDataset(valid_df, transform=get_transforms(train=False))
    test_ds = preprocessing.EmoDataset(test_df, transform=get_transforms(train=False))

    common = dict(num_workers=2, worker_init_fn=seed_worker, generator=gen)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)
    return train_loader, valid_loader, test_loader


def train(args) -> None:
    set_seed(args.seed)
    device = pick_device(args.device)
    log.info("device: %s", device)

    # 1. Data: parse labels, render spectrogram PNGs (cached).
    audio_dir = REPO_ROOT / "data/raw/wav"
    image_dir = REPO_ROOT / "data/preprocessed/images"
    df = preprocessing.decompose_emodb(str(audio_dir))
    if df.empty:
        raise SystemExit(
            f"No wav files in {audio_dir}. Run: uv run python -m pipeline.get_data"
        )
    log.info("loaded %d clips across %d classes", len(df), df["label"].nunique())
    df = preprocessing.build_spectrograms(df, str(image_dir))

    train_loader, valid_loader, test_loader = build_loaders(
        df, args.batch_size, args.seed
    )

    # 2. Model: ResNet18 transfer learning.
    model = build_model(num_classes=len(EMOTIONS), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # Local import to keep the public training surface small.
    from pipeline.pytorchtools import EarlyStopping

    out_path = REPO_ROOT / args.out
    stopper = EarlyStopping(
        patience=args.patience,
        verbose=True,
        model_path=str(out_path),
        meta={"arch": "resnet18", "classes": EMOTIONS, "img_size": 224},
    )

    # 3. Training loop.
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            running += loss.item() * labels.size(0)
        train_loss = running / len(train_loader.dataset)

        val_loss, val_acc = scoring.evaluate(model, valid_loader, criterion, device)
        scheduler.step(val_loss)
        log.info(
            "epoch %02d/%d | train_loss %.4f | val_loss %.4f | val_acc %.3f",
            epoch,
            args.epochs,
            train_loss,
            val_loss,
            val_acc,
        )
        stopper(val_loss, model)
        if stopper.early_stop:
            log.info("early stopping at epoch %d", epoch)
            break

    # 4. Reload best checkpoint and report test metrics.
    from utils.core import load_checkpoint

    best_model, _ = load_checkpoint(str(out_path), device)
    test_loss, test_acc = scoring.evaluate(best_model, test_loader, criterion, device)
    log.info("BEST MODEL | test_loss %.4f | test_acc %.3f", test_loss, test_acc)

    err = scoring.error_analysis(test_loader, best_model, EMOTIONS, device)
    log.info(
        "classification report:\n%s", scoring.classification_summary(err, EMOTIONS)
    )
    log.info("saved best model -> %s", out_path)


def main() -> None:
    p = argparse.ArgumentParser(description="Train good-mood emotion classifier")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", default="auto", help="auto|cuda|mps|cpu")
    p.add_argument("--out", default="model.pt", help="checkpoint path (repo-relative)")
    train(p.parse_args())


if __name__ == "__main__":
    main()
