"""Serve the good-mood emotion classifier as a FastAPI + Gradio app.

    uv run uvicorn main:app --host 0.0.0.0 --port 8000

The Gradio UI is mounted at "/", a JSON health check at "/health".
Inference uses the EXACT same spectrogram + transforms as training
(utils.core), so there is no train/serve skew.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import librosa
import torch
from fastapi import FastAPI

from utils.core import (
    EMOTION_DISPLAY,
    audio_to_mel_image,
    ensure_model,
    get_transforms,
    load_checkpoint,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("good-mood")

MODEL_PATH = Path(__file__).resolve().parent / "model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORM = get_transforms(train=False)

# Load the trained model once at startup, fetching it from the Hugging Face Hub
# if it is not present locally.
MODEL = None
CLASSES: list[str] = []
try:
    ensure_model(str(MODEL_PATH))
    MODEL, CLASSES = load_checkpoint(str(MODEL_PATH), DEVICE)
    log.info("loaded model (%d classes) on %s", len(CLASSES), DEVICE)
except Exception:
    log.warning(
        "model unavailable — download from HF failed and no local model.pt. "
        "Train (uv run python -m pipeline.train) or set MODEL_REPO.",
        exc_info=True,
    )


def predict(file_path: str) -> dict:
    """Return {emotion_display: probability} for a recorded/uploaded clip."""
    if MODEL is None:
        raise gr.Error("Model not trained yet. Run: uv run python -m pipeline.train")
    if not file_path:
        raise gr.Error("Please record or upload an audio clip first.")

    try:
        audio, sr = librosa.load(file_path)
        image = audio_to_mel_image(audio, sr)
        tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(MODEL(tensor), dim=1).squeeze(0)
    except Exception as exc:  # surface a clean message instead of a 500
        log.exception("prediction failed")
        raise gr.Error(f"Could not process audio: {exc}")

    return {
        EMOTION_DISPLAY.get(cls, cls): float(probs[i])
        for i, cls in enumerate(CLASSES)
    }


demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio"),
    outputs=gr.Label(num_top_classes=3, label="Predicted emotion"),
    title="🎙️ Good Mood — Speech Emotion Recognition",
    description=(
        "Record or upload a short speech clip and the model predicts the "
        "speaker's emotion from its MEL spectrogram. Trained on EMO-DB "
        "(German emotional speech) with a fine-tuned ResNet18."
    ),
    flagging_mode="never",
    theme=gr.themes.Soft(),
)

app = FastAPI(title="good-mood")


@app.get("/health")
def health() -> dict:
    """Liveness/readiness probe."""
    return {"status": "ok", "model_loaded": MODEL is not None, "device": str(DEVICE)}


# Mount the Gradio UI at the site root.
app = gr.mount_gradio_app(app, demo, path="/")
