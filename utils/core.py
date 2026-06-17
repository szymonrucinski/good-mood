"""Shared core for good-mood: seeds, spectrogram, transforms, model.

Both the training pipeline (``pipeline/*``) and the serving app (``main.py``)
import from here so that preprocessing, image size, normalization and the label
order are *guaranteed identical* between train and inference. Any divergence
between training and serving preprocessing silently wrecks accuracy, so it all
lives in one place.
"""
from __future__ import annotations

import os
import random
from typing import List, Tuple

import matplotlib

# Always render off-screen; never depend on a display / OS backend.
matplotlib.use("Agg")

import librosa
import librosa.display
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

# --- Canonical configuration (single source of truth) ------------------------

# EMO-DB emotions in the exact order LabelEncoder would produce (alphabetical),
# so integer class index -> emotion is fixed and reproducible everywhere.
EMOTIONS: List[str] = [
    "angry",
    "bored",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
]

# Pretty labels for the UI / API responses.
EMOTION_DISPLAY = {
    "angry": "Angry 😡",
    "bored": "Bored 🥱",
    "disgust": "Disgust 🤮",
    "fear": "Fear 🫣",
    "happy": "Happy 🤗",
    "neutral": "Neutral 😶",
    "sad": "Sad 😭",
}

IMG_SIZE = 224  # ResNet ImageNet input size
# ImageNet statistics — required because we fine-tune an ImageNet-pretrained net.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SEED = 42

# Hugging Face Hub repo the trained model is distributed from. Override with the
# MODEL_REPO env var to point at a fork / different checkpoint.
MODEL_REPO = os.environ.get("MODEL_REPO", "szymonrucinski/good-mood-emotion")
MODEL_FILENAME = "model.pt"


# --- Reproducibility ---------------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    """Seed every RNG and make cuDNN deterministic.

    Reproducibility means: same data + same seed -> same weights. We seed
    python, numpy and torch (CPU+CUDA), force deterministic cuDNN kernels and
    set the CUBLAS workspace so matmuls are deterministic too.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader worker seeding so shuffling/augmentation is reproducible."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = SEED) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# --- Spectrogram -------------------------------------------------------------

def audio_to_mel_image(audio: np.ndarray, sr: int) -> Image.Image:
    """Convert a raw audio waveform to a MEL-spectrogram RGB PIL image.

    Uses an isolated Figure + Agg canvas (NOT pyplot global state), so it is
    thread-safe under a web server and never returns a blank/closed figure.
    Replaces the old moviepy ``mplfig_to_npimage`` dependency.
    """
    stft = np.abs(librosa.stft(audio)) ** 2
    mel = librosa.feature.melspectrogram(S=stft, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig = Figure(figsize=(2.56, 2.56), dpi=100)  # -> 256x256 px
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])  # fill the canvas, no margins/axes
    ax.set_axis_off()
    librosa.display.specshow(mel_db, sr=sr, ax=ax)
    canvas.draw()

    rgba = np.asarray(canvas.buffer_rgba())
    return Image.fromarray(rgba[..., :3].copy())  # drop alpha -> RGB


# --- Transforms --------------------------------------------------------------

def get_transforms(train: bool) -> T.Compose:
    """Image transforms. Train adds light SpecAugment-style RandomErasing.

    Flips/rotations are intentionally NOT used: a spectrogram's axes (time,
    frequency) are meaningful, so flipping would corrupt the signal. RandomErasing
    masks random time/frequency bands -> the image-space analog of SpecAugment.
    """
    steps = [
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    if train:
        steps.append(T.RandomErasing(p=0.25, scale=(0.02, 0.12)))
    return T.Compose(steps)


# --- Model -------------------------------------------------------------------

def build_model(num_classes: int = len(EMOTIONS), pretrained: bool = True) -> nn.Module:
    """ResNet18 transfer-learning backbone with a fresh classification head.

    ResNet18 > AlexNet for this task: residual connections + batchnorm train
    faster and generalize better on a tiny (~500 image) dataset.
    """
    from torchvision import models

    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def ensure_model(path: str) -> str:
    """Return a local path to model.pt, downloading it from the Hugging Face Hub
    (``MODEL_REPO``) if it is not already present. The public model repo needs no
    token, so this works in clean clones and Docker builds."""
    import shutil

    if os.path.exists(path):
        return path
    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
    dest_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copyfile(cached, path)
    return path


def load_checkpoint(path: str, device: torch.device) -> Tuple[nn.Module, List[str]]:
    """Load a checkpoint dict saved by training. Rebuilds the model from
    state_dict (``weights_only=True`` safe) instead of unpickling a whole
    module, so it survives torch version changes."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    classes = ckpt.get("classes", EMOTIONS)
    model = build_model(num_classes=len(classes), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, classes
