"""Grad-CAM written by hand (no extra dependency) to see WHERE the CNN looks.

Used only for analysis after the metrics are fixed; it never changes a model or a threshold.
The defect mask appears here as a ruler (does the heat fall on the defect?), never as input.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def gradcam(model: torch.nn.Module, x: torch.Tensor, layer: torch.nn.Module):
    """Grad-CAM of the single-logit ResNet18 for one image x (1, 3, H, W).

    `layer` is the convolutional block to explain (the last one, `model.layer4[-1]`).
    Returns (cam, logit): cam is (H, W) in [0, 1] upsampled to the input size.
    The hook sits on the OUTPUT TENSOR (not a module backward hook) because the in-place ReLU
    of a ResNet block breaks module backward hooks.
    """
    store: dict = {}

    def fwd(_m, _i, out):
        store["a"] = out
        out.register_hook(lambda g: store.__setitem__("g", g))

    handle = layer.register_forward_hook(fwd)
    model.eval()
    model.zero_grad()
    with torch.enable_grad():
        logit = model(x).squeeze()
        logit.backward()
    handle.remove()
    weights = store["g"].mean(dim=(2, 3), keepdim=True)            # one weight per channel
    cam = F.relu((weights * store["a"]).sum(dim=1, keepdim=True))  # weighted sum of feature maps
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy(), float(logit.detach())


def cam_vs_mask(cam: np.ndarray, mask: np.ndarray) -> tuple[float, bool]:
    """(share of the CAM energy inside the defect mask, does the CAM peak fall inside the mask?)."""
    mask = mask.astype(bool)
    inside = float(cam[mask].sum() / (cam.sum() + 1e-8))
    hit = bool(mask.flat[int(cam.argmax())])
    return inside, hit
