<p align="center">
  <img width="800" src="./documentation/good_mood.png">
</p>

# Good Mood — Speech Emotion Recognition

Classify the emotion in a speech recording from its **MEL spectrogram**.
A short clip is rendered to a spectrogram image, which a fine-tuned **ResNet18**
classifies into one of 7 emotions. Served as a **FastAPI + Gradio** web app.

<p align="center">
  <img width="400" src="./documentation/gui.png">
</p>

## Emotions

`angry 😡` · `bored 🥱` · `disgust 🤮` · `fear 🫣` · `happy 🤗` · `neutral 😶` · `sad 😭`

## Dataset

[EMO-DB](https://www.emodb.bilderbar.info/download/) — 535 German emotional-speech
clips from 10 speakers. The emotion is encoded in the 6th character of each
filename. Downloaded into `data/raw/` and rendered to MEL-spectrogram PNGs in
`data/preprocessed/images/`.

## Architecture

ResNet18 pretrained on ImageNet, with the final FC layer replaced for 7 classes
and the whole network fine-tuned. Inputs are 224×224 spectrogram images
normalized with ImageNet statistics. Training is **fully seeded** (python, numpy,
torch, CUDA, DataLoader workers) and the split uses a fixed `random_state`, so
runs are reproducible. The best model (lowest validation loss) is checkpointed
to `model.pt` as a portable dict (`model_state` + `classes` + metadata).

Train / validation / test split is stratified 80 / 10 / 10. Validation drives
early stopping + LR scheduling; the **test** set is touched only once, at the end.

> Training and serving share one preprocessing module (`utils/core.py`), so the
> spectrogram, image size, normalization and label order are guaranteed identical
> — no train/serve skew.

## Live demo

▶️ **<https://huggingface.co/spaces/szymonrucinski/good-mood>** — runs on a free
CPU Space (ONNX Runtime, no GPU).

## Stack

**Serving:** ONNX Runtime (CPU) · Gradio · FastAPI — no torch/CUDA at inference.
**Training:** PyTorch / torchvision · scikit-learn (optional `train` extra).
librosa · [uv](https://docs.astral.sh/uv/) · Docker.

## Setup (reproducible via uv)

```sh
uv sync                  # lean SERVING env (ONNX Runtime, no torch) — runs the app
uv sync --extra train    # add PyTorch/CUDA for training + ONNX export
```

`uv.lock` pins every dependency to exact versions + hashes. Inference runs on
ONNX Runtime, so the default env (and the Docker image) ship **without torch or
CUDA** — torch lives in the optional `train` extra.

## Train (needs the `train` extra)

```sh
uv sync --extra train
uv run python -m pipeline.get_data              # download EMO-DB into data/raw/
uv run python -m pipeline.train --epochs 40     # -> writes model.pt
uv run python -m pipeline.export_onnx           # model.pt -> model.onnx (+ parity check)
```

Training auto-selects CUDA → MPS → CPU. Useful flags: `--lr`, `--batch-size`,
`--patience`, `--seed`, `--device`.

## Serve (ONNX, no torch)

```sh
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

- UI:     <http://localhost:8000/>
- Health: <http://localhost:8000/health>

The UI (in `ui.py`, shared with the Space via `app.py`) is a Gradio dashboard:
waveform, MEL-spectrogram heatmap, and the 7-class probability histogram beside
the emotion verdict. Inference is ONNX Runtime on CPU.

## Trained model (Hugging Face Hub)

Both the torch checkpoint (`model.pt`) and the serving model (`model.onnx`) are
published at
**[szymonrucinski/good-mood-emotion](https://huggingface.co/szymonrucinski/good-mood-emotion)**.
The serving app downloads `model.onnx` on first startup (Docker bakes it in at
build). Override the source with the `MODEL_REPO` env var — public, no token.

## Docker

The image pulls the model from the Hub at build time (no local `model.pt`
needed):

```sh
chmod +x start_docker.sh
./start_docker.sh
```

The image installs the exact locked dependencies (`uv sync --frozen`), bakes in
the model, exposes port 8000, and has a `/health` HEALTHCHECK. Override the model
source with `docker build --build-arg MODEL_REPO=<user>/<repo>`.
