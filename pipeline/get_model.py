"""Fetch the trained model from the Hugging Face Hub into the repo root.

    uv run python -m pipeline.get_model

Downloads from MODEL_REPO (default szymonrucinski/good-mood-emotion). The repo is
public, so no token is required.
"""

from __future__ import annotations

import logging
from pathlib import Path

from utils.core import MODEL_REPO, ensure_model

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("get_model")

DEST = Path(__file__).resolve().parents[1] / "model.pt"


def main() -> None:
    log.info("fetching model from hf.co/%s ...", MODEL_REPO)
    path = ensure_model(str(DEST))
    log.info("model ready at %s (%d bytes)", path, Path(path).stat().st_size)


if __name__ == "__main__":
    main()
