"""Phase 5 helpers: score the OFFICIAL test split with models and thresholds fixed in validation.

Protocol (D-045, approved): the test split is scored ONCE, by `05_evaluacion`, with
  * the baseline, the CNN (seeds 42, 1, 2) and the autoencoder b2 (seeds 42, 1, 2);
  * each model's own threshold, the one that maximized F2 on its VALIDATION set (D-024);
  * nothing re-tuned afterwards. The test never chooses a threshold, an epoch or a model.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import InspectDataset, get_transforms
from .features import resnet_features
from .models import build_autoencoder, build_model
from .train import anomaly_scores, predict, sigmoid

SEEDS = {"s42": "", "s1": "_s1", "s2": "_s2"}   # file suffix of each seed's results/checkpoints


def _val_json(results_dir: Path, name: str) -> dict:
    with open(Path(results_dir) / f"{name}_val.json") as f:
        return json.load(f)


def build_registry(ckpt_dir: str | Path, results_dir: str | Path) -> dict[str, dict]:
    """Every model that goes to the test, with the threshold fixed in validation."""
    ckpt_dir, results_dir = Path(ckpt_dir), Path(results_dir)
    reg: dict[str, dict] = {}
    bl = joblib.load(ckpt_dir / "baseline.joblib")
    with open(results_dir / "baseline_val.json") as f:
        brow = next(r for r in json.load(f) if r["model"] == bl["name"])
    reg["baseline"] = dict(kind="baseline", family="baseline", ckpt=ckpt_dir / "baseline.joblib",
                           thr=float(bl["threshold"]),
                           val={k: brow[k] for k in ("f2", "recall", "precision", "auc_roc", "avg_precision")})
    for tag, suf in SEEDS.items():
        j = _val_json(results_dir, f"cnn_resnet18{suf}")
        reg[f"cnn_{tag}"] = dict(kind="cnn", family="cnn", ckpt=ckpt_dir / f"cnn_resnet18{suf}.pt",
                                 thr=float(j["threshold"]), val=j["val"], arch=j["config"]["model"]["arch"])
    for tag, suf in SEEDS.items():
        j = _val_json(results_dir, f"autoencoder_b2{suf}")
        reg[f"ae_{tag}"] = dict(kind="ae", family="ae", ckpt=ckpt_dir / f"autoencoder_b2{suf}.pt",
                                thr=float(j["threshold"]), val=j["val"], model_cfg=j["config"]["model"],
                                variant=j["score_variant"], smooth=j["config"]["train"]["smooth"])
    for k, v in reg.items():
        if not v["ckpt"].exists():
            raise FileNotFoundError(f"Missing checkpoint for {k}: {v['ckpt']}")
    return reg


def score_all(registry: dict, te_df, root, features_dir, size, device, batch_size: int = 32,
              num_workers: int = 2):
    """Score the test split with every registered model. Returns (y_test, {name: scores})."""
    y = te_df["label"].to_numpy().astype(int)
    dl = DataLoader(InspectDataset(te_df, root, get_transforms(False, size)), batch_size=batch_size,
                    shuffle=False, num_workers=num_workers)
    scores: dict[str, np.ndarray] = {}
    for name, v in registry.items():
        if v["kind"] == "baseline":
            bl = joblib.load(v["ckpt"])
            X, yb = resnet_features(te_df, root, size, 64, cache=Path(features_dir) / "resnet18_test.npz")
            assert (yb == y).all()
            s = bl["model"].predict_proba(X)[:, 1]
        elif v["kind"] == "cnn":
            model = build_model(v["arch"], pretrained=False)
            model.load_state_dict(torch.load(v["ckpt"], map_location="cpu"))
            yy, logits = predict(model.to(device), dl, device)
            assert (yy == y).all()
            s = sigmoid(logits)
        else:
            m = v["model_cfg"]
            model = build_autoencoder(m["base"], m["bottleneck"])
            model.load_state_dict(torch.load(v["ckpt"], map_location="cpu"))
            yy, variants, _ = anomaly_scores(model.to(device), dl, device, v["smooth"])
            assert (yy == y).all()
            s = variants[v["variant"]]
        scores[name] = np.asarray(s, dtype=float)
        print(f"scored {name:10s} ({len(s)} images)")
    return y, scores


def f2_curve(y, scores, n: int = 200):
    """(thresholds, F2) on a grid over the score range. For INFORMATION only (threshold sensitivity)."""
    y, s = np.asarray(y).astype(int), np.asarray(scores, dtype=float)
    ts = np.linspace(s.min(), s.max(), n)
    f2 = []
    for t in ts:
        p = s >= t
        tp, fp, fn = (p & (y == 1)).sum(), (p & (y == 0)).sum(), ((~p) & (y == 1)).sum()
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f2.append(5 * pr * rc / (4 * pr + rc) if 4 * pr + rc else 0.0)
    return ts, np.array(f2)
