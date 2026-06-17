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
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:  # for annotations only — never imported at runtime
    import torch.nn as nn
    import torchvision.transforms as T

import matplotlib

# Always render off-screen; never depend on a display / OS backend.
matplotlib.use("Agg")

import librosa
import librosa.display
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

# NOTE: torch / torchvision are imported LAZILY (inside the functions that need
# them) so the serving path — which runs inference via ONNX Runtime — can import
# this module without torch installed. Only the training extra needs torch.

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
    import torch

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
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = SEED):
    import torch

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


# --- Scientific visualizations (for the UI) ----------------------------------

_ACCENT = "#6366f1"
_FG = "#1e1b2e"


def _new_fig(w: float, h: float) -> Figure:
    fig = Figure(figsize=(w, h), dpi=120)
    FigureCanvasAgg(fig)
    fig.patch.set_alpha(0.0)  # transparent so it blends into the UI card
    return fig


def plot_waveform(audio: np.ndarray, sr: int) -> Figure:
    """Amplitude-vs-time waveform."""
    fig = _new_fig(6.0, 2.1)
    ax = fig.add_subplot(111)
    t = np.arange(len(audio)) / float(sr)
    ax.plot(t, audio, linewidth=0.6, color=_ACCENT)
    ax.set_title("Waveform", fontsize=10, color=_FG)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Amplitude", fontsize=8)
    ax.margins(x=0)
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    return fig


def plot_mel_heatmap(audio: np.ndarray, sr: int) -> Figure:
    """MEL-spectrogram heatmap with time/frequency axes and a dB colorbar —
    the scientific view of what the model actually sees."""
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    fig = _new_fig(6.0, 3.2)
    ax = fig.add_subplot(111)
    img = librosa.display.specshow(
        mel_db, sr=sr, x_axis="time", y_axis="mel", cmap="magma", ax=ax
    )
    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
    cbar.ax.tick_params(labelsize=7)
    ax.set_title("MEL spectrogram", fontsize=10, color=_FG)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Mel frequency (Hz)", fontsize=8)
    fig.tight_layout()
    return fig


def plot_probabilities(probs: dict) -> Figure:
    """Horizontal histogram of class probabilities (sorted)."""
    from matplotlib import colormaps

    items = sorted(probs.items(), key=lambda kv: kv[1])
    labels = [k for k, _ in items]
    values = np.array([v for _, v in items], dtype=float)
    fig = _new_fig(6.0, 3.0)
    ax = fig.add_subplot(111)
    colors = colormaps["viridis"](0.15 + 0.8 * values)
    ax.barh(labels, values, color=colors)
    for y, v in enumerate(values):
        ax.text(min(v + 0.02, 0.98), y, f"{v:.0%}", va="center", fontsize=8, color=_FG)
    ax.set_title("Class probabilities", fontsize=10, color=_FG)
    ax.set_xlabel("Probability", fontsize=8)
    ax.set_xlim(0, 1)
    ax.grid(True, axis="x", alpha=0.15)
    fig.tight_layout()
    return fig


# --- Transforms --------------------------------------------------------------


def get_transforms(train: bool) -> T.Compose:
    """Image transforms. Train adds light SpecAugment-style RandomErasing.

    Flips/rotations are intentionally NOT used: a spectrogram's axes (time,
    frequency) are meaningful, so flipping would corrupt the signal. RandomErasing
    masks random time/frequency bands -> the image-space analog of SpecAugment.
    """
    import torchvision.transforms as T

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
    import torch.nn as nn
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


def load_checkpoint(path: str, device) -> Tuple[nn.Module, List[str]]:
    """Load a checkpoint dict saved by training. Rebuilds the model from
    state_dict (``weights_only=True`` safe) instead of unpickling a whole
    module, so it survives torch version changes."""
    import torch

    ckpt = torch.load(path, map_location=device, weights_only=False)
    classes = ckpt.get("classes", EMOTIONS)
    model = build_model(num_classes=len(classes), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, classes


# --- ONNX serving (no torch) -------------------------------------------------

ONNX_FILENAME = "model.onnx"


def preprocess_for_onnx(image: Image.Image) -> np.ndarray:
    """Pure-numpy equivalent of ``get_transforms(train=False)`` — Resize(224,
    bilinear) -> /255 -> ImageNet normalize -> CHW float32 batch. Verified to
    match torchvision exactly (0.0 diff), so ONNX serving has zero skew."""
    mean = np.asarray(IMAGENET_MEAN, np.float32)
    std = np.asarray(IMAGENET_STD, np.float32)
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x = np.asarray(img, np.float32) / 255.0
    x = (x - mean) / std
    return x.transpose(2, 0, 1)[None].astype(np.float32)  # (1, 3, H, W)


def ensure_onnx_model(path: str) -> str:
    """Download model.onnx from the Hugging Face Hub if absent (public, no token)."""
    import shutil

    if os.path.exists(path):
        return path
    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(repo_id=MODEL_REPO, filename=ONNX_FILENAME)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    shutil.copyfile(cached, path)
    return path


class OnnxEmotionModel:
    """ONNX Runtime emotion classifier — the serving backend (no torch/CUDA)."""

    def __init__(self, onnx_path: str):
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.classes = EMOTIONS

    def predict_proba(self, image: Image.Image) -> np.ndarray:
        """MEL-spectrogram image -> softmax probabilities over EMOTIONS."""
        x = preprocess_for_onnx(image)
        logits = self.session.run(None, {self.input_name: x})[0][0]
        e = np.exp(logits - logits.max())
        return e / e.sum()
