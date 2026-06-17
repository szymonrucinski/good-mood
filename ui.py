"""Gradio UI for good-mood — clean dashboard (dark hero + Soft body).

Inference runs on ONNX Runtime (CPU, no torch/CUDA). The MEL-spectrogram
preprocessing is shared with training via utils.core, so there is no
train/serve skew. Imported by main.py (FastAPI /health) and app.py (HF Space).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import librosa

from utils.core import (
    EMOTION_DISPLAY,
    EMOTIONS,
    OnnxEmotionModel,
    audio_to_mel_image,
    ensure_onnx_model,
    plot_mel_heatmap,
    plot_probabilities,
    plot_waveform,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("good-mood")

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.onnx"


def _example_dir() -> Path:
    for cand in (ROOT / "samples", ROOT / "data" / "raw" / "wav"):
        if cand.is_dir():
            return cand
    return ROOT


EXAMPLES_DIR = _example_dir()
_EXAMPLE_SPEC = [
    ("03a01Wa.wav", "angry"),
    ("03a01Fa.wav", "happy"),
    ("03a02Ta.wav", "sad"),
    ("03a04Ad.wav", "fear"),
]
EXAMPLES = [
    [str(EXAMPLES_DIR / f)] for f, _ in _EXAMPLE_SPEC if (EXAMPLES_DIR / f).exists()
]
DEFAULT_SAMPLE = EXAMPLES[0][0] if EXAMPLES else None

MODEL = None
MODEL_LOADED = False
try:
    ensure_onnx_model(str(MODEL_PATH))
    MODEL = OnnxEmotionModel(str(MODEL_PATH))
    MODEL_LOADED = True
    log.info("ONNX model loaded (%d classes)", len(MODEL.classes))
except Exception:
    log.warning(
        "model unavailable — ONNX download failed; set MODEL_REPO?", exc_info=True
    )


# --- Inference ---------------------------------------------------------------


def analyze(file_path: str):
    """clip -> (emoji label dict, waveform fig, MEL fig, probability fig)."""
    if MODEL is None:
        raise gr.Error("Model unavailable. Check the Space logs / MODEL_REPO.")
    if not file_path:
        raise gr.Error("Please record or upload an audio clip first.")
    try:
        audio, sr = librosa.load(file_path)
        image = audio_to_mel_image(audio, sr)  # same path as training
        probs = MODEL.predict_proba(image)
    except Exception as exc:
        log.exception("prediction failed")
        raise gr.Error(f"Could not process audio: {exc}")

    raw = {cls: float(probs[i]) for i, cls in enumerate(EMOTIONS)}
    label = {EMOTION_DISPLAY.get(cls, cls): p for cls, p in raw.items()}
    return (
        label,
        plot_waveform(audio, sr),
        plot_mel_heatmap(audio, sr),
        plot_probabilities(raw),
    )


# --- Header + stats (HTML) ---------------------------------------------------

HERO_HTML = """
<div class="gm-hero">
  <div class="gm-hero-title">🎙️ Good Mood — Acoustic Emotion Lab</div>
  <div class="gm-hero-sub">
    Record or upload a short speech clip; a fine-tuned <b>ResNet-18</b> reads the
    emotion from its <b>MEL spectrogram</b>, with the waveform, the time–frequency
    heatmap it sees, and the full probability distribution.
  </div>
  <div class="gm-badges">
    <span class="gm-badge gm-badge-accent">2020 · built for a job application</span>
    <span class="gm-badge">Berlin EMO-DB</span>
    <span class="gm-badge">7 emotions</span>
    <span class="gm-badge">ONNX · no CUDA</span>
  </div>
</div>
"""

STATS_HTML = """
<div class="gm-stats">
  <div class="gm-stat"><div class="gm-stat-num">85.2%</div><div class="gm-stat-lab">Test accuracy</div></div>
  <div class="gm-stat"><div class="gm-stat-num">0.85</div><div class="gm-stat-lab">Macro-F1</div></div>
  <div class="gm-stat"><div class="gm-stat-num">7</div><div class="gm-stat-lab">Emotion classes</div></div>
  <div class="gm-stat"><div class="gm-stat-num">ResNet-18</div><div class="gm-stat-lab">Backbone</div></div>
  <div class="gm-stat"><div class="gm-stat-num">ONNX · CPU</div><div class="gm-stat-lab">Inference</div></div>
</div>
"""

METHOD_MD = """
**How it works.** The waveform is converted to a 128-band **MEL spectrogram**
(short-time Fourier transform → decibels) and rendered to a 224×224 image — the
exact tensor the network sees. A **ResNet-18** fine-tuned on the Berlin EMO-DB
corpus classifies it into 7 emotions. The spectrogram code is shared between
training and serving (**zero train/serve skew**), and inference runs on **ONNX
Runtime** (CPU) — no PyTorch, no CUDA.

*A 2020 portfolio project, modernized: PyTorch → ONNX, packaged with `uv`, and
deployed here as a Gradio Space.*
"""

CSS = """
.gm-hero {
  background: linear-gradient(135deg, #1e1b4b 0%, #312e81 55%, #4f46e5 100%);
  color: #fff; padding: 34px 38px; border-radius: 18px; margin: 4px 0 16px;
}
.gm-hero-title { font-size: 30px; font-weight: 800; letter-spacing: -0.02em; }
.gm-hero-sub { margin-top: 10px; max-width: 760px; font-size: 15px; line-height: 1.6; color: #c7d2fe; }
.gm-hero-sub b { color: #eef2ff; }
.gm-badges { margin-top: 18px; display: flex; flex-wrap: wrap; gap: 8px; }
.gm-badge {
  background: rgba(255,255,255,0.10); border: 1px solid rgba(255,255,255,0.20);
  color: #e0e7ff; font-size: 12px; font-weight: 600; padding: 6px 12px; border-radius: 999px;
}
.gm-badge-accent { background: #facc15; border-color: #facc15; color: #422006; }

.gm-stats { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 18px; }
.gm-stat {
  flex: 1; min-width: 130px; padding: 16px 18px; border-radius: 14px;
  background: var(--block-background-fill);
  border: 1px solid var(--border-color-primary);
}
.gm-stat-num { font-size: 22px; font-weight: 800; color: #4f46e5; }
.gm-stat-lab {
  font-size: 11px; font-weight: 600; letter-spacing: 0.06em;
  text-transform: uppercase; color: var(--body-text-color-subdued); margin-top: 4px;
}
footer { display: none !important; }
"""


# --- UI ----------------------------------------------------------------------

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="indigo"),
    css=CSS,
    title="Good Mood — Acoustic Emotion Lab",
) as demo:
    gr.HTML(HERO_HTML)
    gr.HTML(STATS_HTML)

    with gr.Row(equal_height=False):
        with gr.Column(scale=4):
            audio_in = gr.Audio(
                value=DEFAULT_SAMPLE,
                sources=["microphone", "upload"],
                type="filepath",
                label="🎤 Record or upload speech",
            )
            with gr.Row():
                clear_btn = gr.ClearButton(value="Clear")
                analyze_btn = gr.Button("Analyze emotion", variant="primary", size="lg")
            verdict = gr.Label(num_top_classes=7, label="Predicted emotion")
            if EXAMPLES:
                gr.Examples(
                    examples=EXAMPLES,
                    inputs=audio_in,
                    label="Try a sample (EMO-DB)",
                    examples_per_page=4,
                )
            with gr.Accordion("How it works", open=False):
                gr.Markdown(METHOD_MD)

        with gr.Column(scale=6):
            mel_plot = gr.Plot(label="MEL spectrogram — what the model sees")
            with gr.Row():
                wave_plot = gr.Plot(label="Waveform")
                prob_plot = gr.Plot(label="Class probabilities")

    outputs = [verdict, wave_plot, mel_plot, prob_plot]
    analyze_btn.click(fn=analyze, inputs=audio_in, outputs=outputs)
    clear_btn.add([audio_in, verdict, wave_plot, mel_plot, prob_plot])
    if DEFAULT_SAMPLE:
        demo.load(fn=analyze, inputs=audio_in, outputs=outputs)


if __name__ == "__main__":
    demo.launch(ssr_mode=False)
