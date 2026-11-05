"""Models: baseline classifiers, a small CNN, and ResNet18 with staged fine-tuning.

(Phase 4 adds the convolutional autoencoder to this module.)

ResNet18 fine-tuning is staged (plan 3.2) because there are only ~197 defective training
images: unfreezing everything at once overfits.
    "head"   -> only the new final layer (the backbone acts as a fixed feature extractor)
    "layer4" -> final layer + last convolutional block (most task-specific features)
    "all"    -> the whole network (also used to train the small CNN from scratch)
The output is ONE logit: sigmoid(logit) = probability of "defective".
"""

from __future__ import annotations

import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from .data import SEED

STAGES = ("head", "layer4", "all")


# --------------------------------------------------------------------------- #
# Baseline classifiers (on precomputed features)
# --------------------------------------------------------------------------- #
DEFAULT_GRIDS = {
    "logistic": [dict(C=c) for c in (0.001, 0.01, 0.1, 1, 10)],
    "knn": [dict(k=k) for k in (1, 3, 5, 11, 21, 41)],
}


def make_baseline(kind: str, params: dict, seed: int = SEED):
    """StandardScaler + classifier. The scaler is fit inside the pipeline on train only."""
    if kind == "logistic":
        # class_weight='balanced' compensates the ~11% defect rate
        clf = LogisticRegression(C=params["C"], class_weight="balanced", max_iter=2000, random_state=seed)
    elif kind == "knn":
        # weights='distance' gives finer-grained scores than a plain majority vote
        clf = KNeighborsClassifier(n_neighbors=params["k"], weights="distance")
    else:
        raise ValueError(f"unknown baseline kind: {kind}")
    return make_pipeline(StandardScaler(), clf)


def select_baseline(kind: str, X_tr, y_tr, X_val, y_val, grid: list[dict] | None = None, seed: int = SEED):
    """Fit every grid point on train and keep the best by average precision on VALIDATION.

    Returns (model, params, val_scores, table). Average precision does not depend on the
    threshold, which is chosen afterwards with `evaluate.best_f2_threshold`.
    """
    best, rows = None, []
    for params in grid or DEFAULT_GRIDS[kind]:
        model = make_baseline(kind, params, seed).fit(X_tr, y_tr)
        scores = model.predict_proba(X_val)[:, 1]
        ap = float(average_precision_score(y_val, scores))
        rows.append({**params, "val_ap": round(ap, 4)})
        if best is None or ap > best[0]:
            best = (ap, model, params, scores)
    return best[1], best[2], best[3], pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Small CNN from scratch (Practice 7 control experiment)
# --------------------------------------------------------------------------- #
class SmallCNN(nn.Module):
    """Three conv blocks + global MAX pooling + one logit. Not expected to win: it is the
    "no pretraining" reference for the controlled experiment of Practice 7. Max pooling (not
    average) so a small defect is not diluted by the rest of the image."""

    def __init__(self):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
                                 nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(3, 16), block(16, 32), block(32, 64))
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        return self.fc(self.pool(self.features(x)).flatten(1))


# --------------------------------------------------------------------------- #
# ResNet18 with transfer learning
# --------------------------------------------------------------------------- #
def build_resnet18(pretrained: bool = True) -> nn.Module:
    """ResNet18 with the last layer replaced by Linear(512 -> 1)."""
    model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def build_model(arch: str, pretrained: bool = True) -> nn.Module:
    if arch == "resnet18":
        return build_resnet18(pretrained)
    if arch == "small_cnn":
        return SmallCNN()
    raise ValueError(f"unknown arch: {arch}")


def set_stage(model: nn.Module, stage: str) -> list[nn.Parameter]:
    """Freeze/unfreeze layers for a training stage and return the trainable parameters."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    for p in model.parameters():
        p.requires_grad = stage == "all"
    if stage != "all":
        parts = [model.fc] + ([model.layer4] if stage == "layer4" else [])
        for module in parts:
            for p in module.parameters():
                p.requires_grad = True
    return [p for p in model.parameters() if p.requires_grad]


def freeze_frozen_bn(model: nn.Module) -> None:
    """Put BatchNorm layers without trainable parameters in eval() mode.

    model.train() would otherwise keep updating their running statistics with our images and
    silently "unfreeze" part of the network. Call right after model.train().
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
            m.eval()


def count_trainable(model: nn.Module) -> tuple[int, int]:
    """(trainable parameters, total parameters)."""
    total = sum(p.numel() for p in model.parameters())
    return sum(p.numel() for p in model.parameters() if p.requires_grad), total
