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

# App code + trained model (model.pt must be present in the build context).
COPY . .
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
