"""Download the EMO-DB dataset into data/raw/.

    uv run python -m pipeline.get_data
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("get_data")

URL = "http://emodb.bilderbar.info/download/download.zip"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if (RAW_DIR / "wav").exists() and any((RAW_DIR / "wav").glob("*.wav")):
        log.info("dataset already present at %s", RAW_DIR / "wav")
        return
    zip_path = RAW_DIR.parent / "download.zip"
    log.info("downloading EMO-DB ...")
    subprocess.run(["curl", "-sSL", "-o", str(zip_path), URL], check=True)
    log.info("unzipping ...")
    subprocess.run(["unzip", "-oq", str(zip_path), "-d", str(RAW_DIR)], check=True)
    zip_path.unlink(missing_ok=True)
    n = len(list((RAW_DIR / "wav").glob("*.wav")))
    log.info("done: %d wav files in %s", n, RAW_DIR / "wav")
    if n == 0:
        sys.exit("no wav files extracted")


if __name__ == "__main__":
    main()
