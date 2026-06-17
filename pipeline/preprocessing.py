"""EMO-DB preprocessing: parse labels, render MEL spectrograms, Dataset."""

from __future__ import annotations

import os
from pathlib import Path

import librosa
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.core import EMOTIONS, audio_to_mel_image

# EMO-DB encodes the emotion in the 6th filename character (index 5), e.g.
# "03a01Wa.wav" -> 'W'. German emotion -> our English class label.
EMODB_CODE_TO_EMOTION = {
    "W": "angry",  # Wut (anger)
    "L": "bored",  # Langeweile (boredom)
    "E": "disgust",  # Ekel (disgust)
    "A": "fear",  # Angst (fear)
    "F": "happy",  # Freude (happiness)
    "T": "sad",  # Trauer (sadness)
    "N": "neutral",  # Neutral
}


def decompose_emodb(audio_dir: str) -> pd.DataFrame:
    """Scan an EMO-DB wav folder and return a DataFrame[label, source, path]."""
    rows = []
    for name in sorted(os.listdir(audio_dir)):
        if not name.lower().endswith(".wav"):
            continue
        emotion = EMODB_CODE_TO_EMOTION.get(name[5], "unknown")
        rows.append(
            {"label": emotion, "source": "EMODB", "path": os.path.join(audio_dir, name)}
        )
    df = pd.DataFrame(rows)
    df = df[df["label"] != "unknown"].reset_index(drop=True)
    return df


def build_spectrograms(df: pd.DataFrame, image_dir: str) -> pd.DataFrame:
    """Render each wav to a PNG MEL spectrogram (cached) and return a copy of
    ``df`` whose ``path`` points at the PNG. PNG (lossless) is used so the
    training images match what the serving app generates in-memory."""
    os.makedirs(image_dir, exist_ok=True)
    image_paths = []
    for wav_path in tqdm(df["path"], desc="spectrograms"):
        stem = Path(wav_path).stem
        png_path = os.path.join(image_dir, f"{stem}.png")
        if not os.path.exists(png_path):
            audio, sr = librosa.load(wav_path)
            audio_to_mel_image(audio, sr).save(png_path)
        image_paths.append(png_path)
    out = df.copy()
    out["path"] = image_paths
    return out


class EmoDataset(Dataset):
    """Spectrogram-image dataset. Expects columns ``path`` (png) and ``target``
    (int class index)."""

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")  # force 3 channels
        if self.transform:
            image = self.transform(image)
        return image, int(row["target"])


def encode_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Map string labels -> fixed integer indices (EMOTIONS order)."""
    out = df.copy()
    out["target"] = out["label"].map(lambda e: EMOTIONS.index(e))
    return out
