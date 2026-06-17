"""Gradio UI for good-mood — sleek scientific dashboard.

Inference runs on ONNX Runtime (CPU, no torch/CUDA). The MEL-spectrogram
preprocessing is shared with training via utils.core, so there is no
train/serve skew. This module is imported by:
  - main.py  -> mounts `demo` on FastAPI (local, adds /health)
  - app.py   -> demo.launch() for the Hugging Face Space
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
    """Bundled samples (on the Space) or the local dataset, whichever exists."""
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

# Load the ONNX model once at startup (downloads from the HF Hub if absent).
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
    """Run the classifier and build every dashboard output for one clip:
    (emoji label dict, waveform fig, MEL fig, probability fig)."""
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


# --- Theme -------------------------------------------------------------------

theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_lg,
    spacing_size=gr.themes.sizes.spacing_lg,
    text_size=gr.themes.sizes.text_md,
    font=(gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"),
    font_mono=(gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"),
).set(
    body_background_fill="#f6f7fb",
    block_background_fill="#ffffff",
    block_border_width="0px",
    block_shadow="0 1px 2px rgba(16,18,28,0.04), 0 10px 30px rgba(16,18,28,0.06)",
    block_radius="20px",
    block_label_background_fill="transparent",
    block_label_text_weight="600",
    panel_background_fill="#ffffff",
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
    button_primary_text_color="#ffffff",
    button_large_radius="14px",
    input_radius="14px",
)


# --- Custom CSS --------------------------------------------------------------

CSS = """
:root, .gradio-container {
    --gm-accent: #6366f1;
    --gm-accent-soft: rgba(99,102,241,0.10);
    --gm-ink: #1e2030;
    --gm-muted: #6b7280;
    --gm-line: rgba(20,22,36,0.08);
}
.gradio-container {
    background: #f6f7fb !important;
    color: var(--gm-ink) !important;
    max-width: 1160px !important;
    margin: 0 auto !important;
    padding-bottom: 56px !important;
}
.gm-header {
    margin: 26px 0 10px;
    padding: 40px 44px;
    border-radius: 26px;
    background:
        radial-gradient(120% 150% at 0% 0%, rgba(99,102,241,0.16) 0%, rgba(99,102,241,0) 55%),
        radial-gradient(120% 150% at 100% 0%, rgba(168,85,247,0.12) 0%, rgba(168,85,247,0) 55%),
        #ffffff;
    box-shadow: 0 1px 2px rgba(16,18,28,0.04), 0 18px 50px rgba(16,18,28,0.08);
}
.gm-eyebrow {
    display: inline-flex; align-items: center; gap: 8px;
    font-size: 12px; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--gm-accent);
    background: var(--gm-accent-soft);
    padding: 6px 12px; border-radius: 999px;
}
.gm-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #22c55e; box-shadow: 0 0 0 0 rgba(34,197,94,0.6);
    animation: gm-pulse 2.4s infinite;
}
@keyframes gm-pulse {
    0% { box-shadow: 0 0 0 0 rgba(34,197,94,0.55); }
    70% { box-shadow: 0 0 0 8px rgba(34,197,94,0); }
    100% { box-shadow: 0 0 0 0 rgba(34,197,94,0); }
}
.gm-title {
    margin: 18px 0 0; font-size: 40px; line-height: 1.08;
    font-weight: 700; letter-spacing: -0.022em; color: var(--gm-ink);
}
.gm-title .gm-accent-text { color: var(--gm-accent); }
.gm-sub {
    margin: 14px 0 0; max-width: 660px;
    font-size: 16px; line-height: 1.6; color: var(--gm-muted);
}
.gm-meta { margin-top: 22px; display: flex; flex-wrap: wrap; gap: 10px; }
.gm-chip {
    font-size: 12.5px; font-weight: 500; color: #43465c;
    background: #f3f4fb; border: 1px solid var(--gm-line);
    padding: 7px 13px; border-radius: 10px;
}
.gm-section {
    font-size: 12px; font-weight: 600; letter-spacing: 0.11em;
    text-transform: uppercase; color: var(--gm-muted);
    margin: 8px 2px 0;
}
.gm-card {
    padding: 20px !important; border-radius: 20px !important;
    background: #ffffff !important;
    box-shadow: 0 1px 2px rgba(16,18,28,0.04), 0 10px 30px rgba(16,18,28,0.06) !important;
}
.gm-cap { font-size: 13px; font-weight: 600; color: var(--gm-ink); margin: 2px 4px 8px; }
.gm-plot {
    background: #ffffff !important; border-radius: 16px !important;
    padding: 8px !important; border: 1px solid var(--gm-line) !important;
    box-shadow: none !important;
}
#gm-analyze {
    font-weight: 600 !important; letter-spacing: 0.01em;
    box-shadow: 0 6px 18px rgba(99,102,241,0.28) !important;
}
#gm-verdict { min-height: 240px; }
#gm-verdict .output-class, #gm-verdict .label-name { color: var(--gm-ink) !important; }
.gm-method { font-size: 14px; line-height: 1.65; color: #43465c; }
.gm-method code { background: #f3f4fb; padding: 1px 6px; border-radius: 6px; font-size: 12.5px; }
footer { display: none !important; }
.gm-foot {
    text-align: center; color: var(--gm-muted);
    font-size: 13px; padding: 26px 0 4px;
}
.gm-foot a { color: var(--gm-accent); text-decoration: none; }
@media (max-width: 760px) {
    .gm-header { padding: 30px 24px; }
    .gm-title { font-size: 30px; }
}
"""

HEADER_HTML = """
<div class="gm-header">
  <span class="gm-eyebrow"><span class="gm-dot"></span>Speech Emotion Recognition</span>
  <h1 class="gm-title">Good&nbsp;Mood <span class="gm-accent-text">·</span> Acoustic Emotion Lab</h1>
  <p class="gm-sub">
    Record or upload a short speech clip. The model renders its MEL
    spectrogram and a fine-tuned <b>ResNet-18</b> reads the emotional
    fingerprint of the voice &mdash; alongside the waveform, the time&ndash;
    frequency heatmap it actually sees, and the full probability distribution
    across all seven classes.
  </p>
  <div class="gm-meta">
    <span class="gm-chip">ResNet-18 &middot; ONNX Runtime</span>
    <span class="gm-chip">MEL spectrogram &middot; 224&times;224</span>
    <span class="gm-chip">EMO-DB &middot; German emotional speech</span>
    <span class="gm-chip">7 emotion classes</span>
  </div>
</div>
"""

METHOD_HTML = """
<div class="gm-method">
  <b>How it works.</b> The waveform is converted to a 128-band <b>MEL
  spectrogram</b> via short-time Fourier transform, scaled to decibels
  (<code>power_to_db</code>), and rendered to a 224&times;224 image &mdash; the
  exact tensor the network sees. A <b>ResNet-18</b> fine-tuned on Berlin EMO-DB
  classifies it into 7 emotions. The same spectrogram code runs at train and
  serve time, so there is <b>zero train/serve skew</b>; inference uses
  <b>ONNX Runtime</b> (CPU) &mdash; no PyTorch, no CUDA.
</div>
"""


# --- UI ----------------------------------------------------------------------

with gr.Blocks(
    theme=theme, css=CSS, title="Good Mood — Acoustic Emotion Lab", fill_width=False
) as demo:
    gr.HTML(HEADER_HTML)

    with gr.Row(equal_height=False):
        # Left: input + verdict
        with gr.Column(scale=5, min_width=330):
            gr.HTML('<div class="gm-section">01 · Input signal</div>')
            with gr.Group(elem_classes="gm-card"):
                audio_in = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Record or upload speech",
                    waveform_options=gr.WaveformOptions(
                        waveform_color="#c7cbf5",
                        waveform_progress_color="#6366f1",
                        show_controls=True,
                    ),
                )
                with gr.Row():
                    clear_btn = gr.ClearButton(value="Clear", scale=1)
                    analyze_btn = gr.Button(
                        "Analyze emotion",
                        variant="primary",
                        size="lg",
                        elem_id="gm-analyze",
                        scale=2,
                    )

            gr.HTML('<div class="gm-section">02 · Predicted emotion</div>')
            with gr.Group(elem_classes="gm-card"):
                verdict = gr.Label(
                    num_top_classes=7,
                    label="Affect probabilities",
                    show_heading=True,
                    elem_id="gm-verdict",
                )

            if EXAMPLES:
                gr.HTML('<div class="gm-section">Try a sample (EMO-DB)</div>')
                gr.Examples(
                    examples=EXAMPLES, inputs=audio_in, label="", examples_per_page=4
                )

            with gr.Accordion("Method", open=False):
                gr.HTML(METHOD_HTML)

        # Right: scientific visualization grid
        with gr.Column(scale=7, min_width=420):
            gr.HTML('<div class="gm-section">03 · Spectral analysis</div>')
            with gr.Group(elem_classes="gm-card"):
                gr.HTML(
                    '<div class="gm-cap">MEL spectrogram &middot; what the model actually sees (dB)</div>'
                )
                mel_plot = gr.Plot(
                    label="MEL spectrogram", show_label=False, elem_classes="gm-plot"
                )

            with gr.Row(equal_height=True):
                with gr.Column(min_width=200):
                    with gr.Group(elem_classes="gm-card"):
                        gr.HTML(
                            '<div class="gm-cap">Waveform &middot; amplitude vs time</div>'
                        )
                        wave_plot = gr.Plot(
                            label="Waveform", show_label=False, elem_classes="gm-plot"
                        )
                with gr.Column(min_width=200):
                    with gr.Group(elem_classes="gm-card"):
                        gr.HTML(
                            '<div class="gm-cap">Class probability distribution</div>'
                        )
                        prob_plot = gr.Plot(
                            label="Probabilities",
                            show_label=False,
                            elem_classes="gm-plot",
                        )

    gr.HTML(
        '<div class="gm-foot">Fine-tuned ResNet-18 on the Berlin EMO-DB corpus '
        "&middot; ONNX Runtime &middot; shared spectrogram pipeline for zero skew</div>"
    )

    outputs = [verdict, wave_plot, mel_plot, prob_plot]
    analyze_btn.click(fn=analyze, inputs=audio_in, outputs=outputs)
    clear_btn.add([audio_in, verdict, wave_plot, mel_plot, prob_plot])


if __name__ == "__main__":
    demo.launch()
