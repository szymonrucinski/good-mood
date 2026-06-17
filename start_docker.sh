#!/usr/bin/env bash
# Build and run the good-mood serving container.
# Requires model.pt in the repo root (train first: uv run python -m pipeline.train).
set -euo pipefail

docker build -f Dockerfile -t good-mood:latest .
docker run --rm -p 8000:8000 good-mood:latest
# UI:     http://localhost:8000/
# Health: http://localhost:8000/health
