"""Local server: FastAPI + the Gradio UI mounted at /, health at /health.

The UI and ONNX inference live in ui.py (shared with the Hugging Face Space).

    uv run uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import gradio as gr
from fastapi import FastAPI

from ui import MODEL_LOADED, demo

app = FastAPI(title="good-mood")


@app.get("/health")
def health() -> dict:
    """Liveness/readiness probe."""
    return {"status": "ok", "model_loaded": MODEL_LOADED, "backend": "onnxruntime"}


app = gr.mount_gradio_app(app, demo, path="/")
