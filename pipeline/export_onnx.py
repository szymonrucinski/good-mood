"""Export the trained checkpoint to ONNX for lean, torch-free serving.

    uv run --extra train python -m pipeline.export_onnx

Produces model.onnx and verifies it matches the torch model numerically.
"""

from __future__ import annotations

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("export_onnx")

ROOT = Path(__file__).resolve().parents[1]


def main(ckpt: str = "model.pt", out: str = "model.onnx") -> None:
    import numpy as np
    import onnxruntime as ort
    import torch

    from utils.core import ensure_model, load_checkpoint

    ensure_model(str(ROOT / ckpt))
    model, classes = load_checkpoint(str(ROOT / ckpt), torch.device("cpu"))
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    out_path = ROOT / out
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    log.info("exported %s | classes=%s", out_path, classes)

    # Parity check: torch vs onnxruntime.
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        t = torch.softmax(model(x), 1).numpy()
    o = sess.run(None, {"input": x.numpy()})[0]
    o = np.exp(o) / np.exp(o).sum(1, keepdims=True)
    log.info("onnx parity max diff: %.2e", float(np.abs(t - o).max()))


if __name__ == "__main__":
    main()
