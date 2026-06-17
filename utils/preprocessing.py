"""Backwards-compatible helpers; real logic lives in utils.core."""

from __future__ import annotations

import io

from pydub import AudioSegment

from utils.core import audio_to_mel_image  # noqa: F401  (re-export)


def prepare_audio(raw_bytes: bytes, out_path: str = "sample.wav") -> str:
    """Decode arbitrary audio bytes to a wav file and return its path."""
    AudioSegment.from_file(io.BytesIO(raw_bytes)).export(out_path, format="wav")
    return out_path


def plot_mel(audio, rate):
    """Deprecated alias kept for notebooks. Returns a PIL image now."""
    return audio_to_mel_image(audio, rate)
