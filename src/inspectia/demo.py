"""Phase 6: inference + Grad-CAM for the demo app. No UI code here (that lives in demo/app.py).

The demo uses the OFFICIAL CNN (`cnn_resnet18.pt`, seed 42) with the decision threshold fixed on
VALIDATION (max F2, D-024). Nothing here is tuned on the test set.

Three verdicts, because a false negative costs much more than a false positive:
    DEFECTUOSA  prob >= threshold            (official decision)
    REVISAR     REVIEW_LOW <= prob < threshold  (gray zone: send to a human)
    OK          prob < REVIEW_LOW
REVIEW_LOW = 0.30 is the lowest val-F2 threshold seen across the 3 seeds (D-041); see D-048.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from inspectia.data import IMG_SIZE, get_transforms
from inspectia.explain import gradcam
from inspectia.models import build_resnet18

REVIEW_LOW = 0.30
WEIGHTS_NAME = "cnn_resnet18.pt"


def decide(prob: float, threshold: float, review_low: float = REVIEW_LOW) -> str:
    """Map a probability to DEFECTUOSA / REVISAR / OK (see module docstring)."""
    if prob >= threshold:
        return "DEFECTUOSA"
    if prob >= min(review_low, threshold):
        return "REVISAR"
    return "OK"


def overlay_cam(img: Image.Image, cam: np.ndarray, alpha: float = 0.65) -> Image.Image:
    """Blend a [0,1] Grad-CAM heat map (H, W) over `img` resized to the CAM size (jet colormap)."""
    from matplotlib import colormaps

    h, w = cam.shape
    base = np.asarray(img.convert("RGB").resize((w, h)), dtype=np.float32) / 255.0
    heat = colormaps["jet"](cam)[..., :3].astype(np.float32)
    a = (alpha * cam)[..., None]                      # transparent where the CAM is ~0
    out = (1 - a) * base + a * heat
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))


@dataclass
class Inspection:
    prob: float            # sigmoid(logit): probability of "defective"
    threshold: float       # official val threshold
    verdict: str           # DEFECTUOSA / REVISAR / OK
    cam: np.ndarray        # (H, W) in [0, 1]
    overlay: Image.Image   # input image with the heat map on top


class Inspector:
    """ResNet18 (1 logit) + Grad-CAM on `layer4[-1]`, on CPU/CUDA."""

    def __init__(self, model: torch.nn.Module, threshold: float, device: str | None = None,
                 size: tuple[int, int] = IMG_SIZE, val_metrics: dict | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.threshold = float(threshold)
        self.size = size
        self.val_metrics = val_metrics or {}
        self.transform = get_transforms(False, size)   # same preprocessing as val/test: no augmentation

    @classmethod
    def _build(cls, weights: Path, threshold: float, val_metrics: dict | None, device: str | None):
        model = build_resnet18(pretrained=False)
        model.load_state_dict(torch.load(weights, map_location="cpu"))
        return cls(model, threshold, device, val_metrics=val_metrics)

    @classmethod
    def from_artifacts(cls, ckpt_dir, results_dir, name: str = "cnn_resnet18", device: str | None = None):
        """From the training artifacts: checkpoints/<name>.pt + results/<name>_val.json."""
        with open(Path(results_dir) / f"{name}_val.json") as f:
            info = json.load(f)
        return cls._build(Path(ckpt_dir) / f"{name}.pt", info["threshold"], info.get("val"), device)

    @classmethod
    def from_dir(cls, model_dir, device: str | None = None):
        """From a self-contained folder (used by the Hugging Face Space): weights + model_info.json."""
        model_dir = Path(model_dir)
        with open(model_dir / "model_info.json") as f:
            info = json.load(f)
        return cls._build(model_dir / WEIGHTS_NAME, info["threshold"], info.get("val"), device)

    def inspect(self, img: Image.Image) -> Inspection:
        """One image (any size, gray or RGB) -> probability, verdict and Grad-CAM overlay."""
        img = img.convert("RGB")                       # same as InspectDataset (gray -> 3 channels)
        x = self.transform(img).unsqueeze(0).to(self.device)
        cam, logit = gradcam(self.model, x, self.model.layer4[-1])
        prob = float(torch.sigmoid(torch.tensor(logit)))
        return Inspection(prob, self.threshold, decide(prob, self.threshold), cam, overlay_cam(img, cam))
