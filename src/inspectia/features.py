"""Feature extraction for the baselines (no network is trained here).

* `resnet_features`: frozen, ImageNet-pretrained ResNet18; the 512-d vector after global pooling.
* `hog_features`: classic Histogram of Oriented Gradients on the gray image (extra reference).

No augmentation is used (also for train): this step only extracts. Features are cached in an
.npz next to the data (Drive in Colab) together with the image names; if the names differ from
the requested DataFrame, the cache is ignored and recomputed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.feature import hog
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

from .data import IMG_SIZE, InspectDataset, get_transforms
from .utils import get_device


def _load_cache(cache: Path | None, images: list[str]):
    if cache is not None and Path(cache).exists():
        z = np.load(cache, allow_pickle=False)
        if list(z["images"]) == images:
            print(f"Features loaded from cache: {cache}")
            return z["X"], z["y"]
    return None


def _save_cache(cache: Path | None, X: np.ndarray, y: np.ndarray, images: list[str]) -> None:
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, X=X, y=y, images=np.array(images))


@torch.no_grad()
def resnet_features(df: pd.DataFrame, root: str | Path, size: tuple[int, int] = IMG_SIZE,
                    batch_size: int = 64, num_workers: int = 2, device: torch.device | None = None,
                    cache: str | Path | None = None):
    """Return (X [n, 512], y [n]) from a frozen ResNet18, in the same order as `df`."""
    images = df["image"].tolist()
    hit = _load_cache(cache, images)
    if hit is not None:
        return hit
    device = device or get_device()
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()  # keep the 512-d vector instead of the 1000 ImageNet classes
    model.eval().to(device)
    # shuffle=False is essential: features must line up with the labels of `df`
    dl = DataLoader(InspectDataset(df, root, get_transforms(False, size)), batch_size=batch_size,
                    shuffle=False, num_workers=num_workers)
    X = np.concatenate([model(x.to(device)).cpu().numpy() for x, _ in dl])
    y = df["label"].to_numpy()
    _save_cache(cache, X, y, images)
    return X, y


def hog_features(df: pd.DataFrame, root: str | Path, size: tuple[int, int] = IMG_SIZE,
                 pixels_per_cell: tuple[int, int] = (16, 16), cache: str | Path | None = None):
    """Return (X [n, d], y [n]) with HOG on the gray image resized to `size` (height, width)."""
    images = df["image"].tolist()
    hit = _load_cache(cache, images)
    if hit is not None:
        return hit
    root, feats = Path(root), []
    for rel in images:
        img = Image.open(root / rel).convert("L").resize((size[1], size[0]), Image.BILINEAR)  # PIL: (w, h)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        feats.append(hog(arr, orientations=9, pixels_per_cell=pixels_per_cell,
                         cells_per_block=(2, 2), block_norm="L2-Hys"))
    X, y = np.asarray(feats, dtype=np.float32), df["label"].to_numpy()
    _save_cache(cache, X, y, images)
    return X, y
