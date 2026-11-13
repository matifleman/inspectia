"""Export the official CNN to ONNX and build the STATIC Hugging Face Space (Phase 6, D-049).

    python scripts/export_onnx.py --out space_static

Hugging Face Gradio Spaces need a paid plan; Static Spaces are free. The model therefore runs in
the browser (ONNX Runtime Web). The exported graph returns two outputs:
  logit : the CNN logit (sigmoid -> probability of "defective")
  cam   : Grad-CAM of layer4[-1] at feature-map resolution (19 x 7), un-normalized.

Why the CAM is exact: after layer4 the network does global-average-pool + Linear(512 -> 1), so
d(logit)/d(A_c) = w_c / (h*w) for every pixel. Grad-CAM = ReLU(sum_c w_c * A_c) up to a positive
constant, which disappears when the map is normalized to [0, 1]. This script checks it against
`inspectia.explain.gradcam`, the logit against the original model, and (if onnxruntime is
installed) the ONNX file against PyTorch. It never touches the test set.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inspectia.data import IMG_SIZE, get_transforms  # noqa: E402
from inspectia.demo import REVIEW_LOW  # noqa: E402
from inspectia.explain import gradcam  # noqa: E402
from inspectia.models import build_resnet18  # noqa: E402


class LogitCam(nn.Module):
    """ResNet18 forward that also returns ReLU(sum_c w_c * A_c) of the last feature map."""

    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m

    def forward(self, x):
        m = self.m
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        f = m.layer4(m.layer3(m.layer2(m.layer1(x))))              # (B, 512, h, w) = layer4 output
        logit = m.fc(f.mean(dim=(2, 3)))                           # global average pool + fc
        cam = torch.relu((f * m.fc.weight[0].view(1, -1, 1, 1)).sum(dim=1))   # (B, h, w)
        return logit.squeeze(1), cam


def norm_cam(cam: torch.Tensor, size) -> np.ndarray:
    """Upsample (h, w) -> size and min-max normalize, exactly like explain.gradcam."""
    c = F.interpolate(cam[None, None], size=size, mode="bilinear", align_corners=False)[0, 0]
    c = c - c.min()
    return (c / (c.max() + 1e-8)).detach().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="space_static")
    ap.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints"))
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "KolektorSDD2"))
    a = ap.parse_args()

    info = json.load(open(Path(a.results_dir) / "cnn_resnet18_val.json"))
    model = build_resnet18(pretrained=False)
    model.load_state_dict(torch.load(Path(a.ckpt_dir) / "cnn_resnet18.pt", map_location="cpu"))
    model.eval()
    wrap = LogitCam(model).eval()

    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # ---- 1) equivalence checks on validation images (never test) -------------------------
    val = pd.read_csv(ROOT / "splits" / "val.csv")
    rows = list(val[val.label == 1].head(3).itertuples()) + list(val[val.label == 0].head(3).itertuples())
    tf = get_transforms(False, IMG_SIZE)
    try:
        import onnxruntime as ort
    except ImportError:
        ort = None
    dummy = torch.zeros(1, 3, *IMG_SIZE)
    onnx_path = out / "model.onnx"
    kwargs = dict(input_names=["input"], output_names=["logit", "cam"], opset_version=17)
    try:
        torch.onnx.export(wrap, dummy, str(onnx_path), dynamo=False, **kwargs)   # legacy exporter
    except TypeError:
        torch.onnx.export(wrap, dummy, str(onnx_path), **kwargs)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"]) if ort else None
    if sess is None:
        print("onnxruntime not installed: skipping the ONNX-vs-PyTorch check (pip install onnxruntime)")

    worst = dict(logit=0.0, cam=0.0, onnx_logit=0.0, onnx_cam=0.0)
    for r in rows:
        x = tf(Image.open(Path(a.data_dir) / r.image).convert("RGB")).unsqueeze(0)
        cam_ref, logit_ref = gradcam(model, x, model.layer4[-1])
        with torch.no_grad():
            logit_w, cam_w = wrap(x)
        worst["logit"] = max(worst["logit"], abs(float(logit_w) - logit_ref))
        worst["cam"] = max(worst["cam"], float(np.abs(norm_cam(cam_w[0], IMG_SIZE) - cam_ref).max()))
        if sess is not None:
            lo, cm = sess.run(None, {"input": x.numpy()})
            worst["onnx_logit"] = max(worst["onnx_logit"], abs(float(lo[0]) - logit_ref))
            worst["onnx_cam"] = max(worst["onnx_cam"], float(np.abs(norm_cam(torch.from_numpy(cm[0]), IMG_SIZE) - cam_ref).max()))
    print("max abs differences over", len(rows), "val images:", {k: f"{v:.2e}" for k, v in worst.items()})
    assert worst["logit"] < 1e-3 and worst["cam"] < 1e-3, "wrapper does not reproduce model/Grad-CAM"
    if sess is not None:
        assert worst["onnx_logit"] < 1e-3 and worst["onnx_cam"] < 1e-2, "ONNX does not reproduce PyTorch"

    # ---- 2) static Space folder -----------------------------------------------------------
    shutil.copy(ROOT / "web" / "index.html", out / "index.html")
    shutil.copy(ROOT / "web" / "SPACE_README.md", out / "README.md")
    json.dump({"threshold": info["threshold"], "review_low": REVIEW_LOW, "val": info["val"],
               "size": list(IMG_SIZE), "source": "cnn_resnet18 (seed 42), val max-F2"},
              open(out / "model_info.json", "w"), indent=2)
    print(f"Static Space ready in {out}/ (model.onnx {onnx_path.stat().st_size / 1e6:.1f} MB, "
          f"threshold {info['threshold']:.4f})")


if __name__ == "__main__":
    main()
