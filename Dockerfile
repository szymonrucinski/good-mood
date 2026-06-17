# Reproducible build: installs the EXACT locked dependency set (uv.lock).
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# System libs for audio decoding (librosa/soundfile/pydub).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

# Install deps first (cached unless lockfile changes). --frozen = fail if the
# lockfile is out of date, guaranteeing the image matches uv.lock exactly.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# App code.
COPY . .
RUN uv sync --frozen --no-dev

# Bake the ONNX model into the image, pulled from the Hugging Face Hub (public,
# no token). Serving is ONNX Runtime on CPU — no torch/CUDA — so the image is
# lean (~1 GB, not ~7 GB). Override with --build-arg MODEL_REPO=<user>/<repo>.
ARG MODEL_REPO=szymonrucinski/good-mood-emotion
ENV MODEL_REPO=${MODEL_REPO}
RUN uv run python -c "from utils.core import ensure_onnx_model; ensure_onnx_model('model.onnx')" \
    && test "$(stat -c%s model.onnx)" -gt 1000000

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
