"""Training loop (Phase 3).

Project rules respected:
  * Early stopping and epoch selection use VALIDATION only; the test set never intervenes.
  * Loss: BCEWithLogitsLoss, optionally with pos_weight = n_good / n_defective of train
    (use `pos_weight=None` when the balanced sampler is used instead, plan 3.3).
  * The mask is not used; augmentation lives in data.get_transforms (train only).
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from .evaluate import best_f2
from .models import freeze_frozen_bn, set_stage


@torch.no_grad()
def predict(model: nn.Module, loader, device, max_batches: int | None = None):
    """(y_true, logits) as NumPy arrays; no gradients, eval mode."""
    model.eval()
    ys, logits = [], []
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        logits.append(model(x.to(device)).squeeze(1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(logits)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _train_epoch(model, loader, loss_fn, optimizer, device, max_batches=None) -> float:
    model.train()
    freeze_frozen_bn(model)
    total, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x).squeeze(1), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / max(n, 1)


def fit_stage(model: nn.Module, loaders: dict, device, stage: str, epochs: int, lr: float,
              pos_weight: float | None = None, patience: int = 3, weight_decay: float = 1e-4,
              monitor: str = "ap", max_batches: int | None = None, wandb_run=None,
              tag: str = "") -> list[dict]:
    """Train one stage and leave in `model` the weights of the BEST epoch (by validation).

    monitor: "ap" (average precision) or "f2" (best F2 over thresholds) on validation.
    Returns the history: one row per epoch with train_loss, val_loss, val_ap, val_auc, val_f2.
    """
    if monitor not in ("ap", "f2"):
        raise ValueError("monitor must be 'ap' or 'f2'")
    params = set_stage(model, stage)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    pw = torch.tensor(pos_weight, device=device) if pos_weight is not None else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    best_score, best_state, bad, history = -1.0, None, 0, []

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        train_loss = _train_epoch(model, loaders["train"], loss_fn, optimizer, device, max_batches)
        y, logits = predict(model, loaders["val"], device, max_batches)
        val_loss = loss_fn(torch.tensor(logits, device=device),
                           torch.tensor(y, dtype=torch.float32, device=device)).item()
        two_classes = len(set(y.tolist())) > 1  # False only in quick smoke runs with few batches
        val_ap = float(average_precision_score(y, logits)) if two_classes else 0.0
        val_auc = float(roc_auc_score(y, logits)) if two_classes else float("nan")
        val_f2 = best_f2(y, logits)[0] if two_classes else 0.0
        row = dict(stage=stage, epoch=epoch, train_loss=train_loss, val_loss=val_loss, val_ap=val_ap,
                   val_auc=val_auc, val_f2=val_f2, seconds=time.perf_counter() - t0)
        history.append(row)
        print(f"[{stage}] ep {epoch}/{epochs} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f}"
              f" | val_AP {val_ap:.4f} | val_F2 {val_f2:.4f} | val_AUC {val_auc:.4f} | {row['seconds']:.0f}s")
        if wandb_run is not None:
            wandb_run.log({f"{tag}{stage}/{k}": v for k, v in row.items() if k != "stage"})
        score = val_ap if monitor == "ap" else val_f2
        if score > best_score:
            best_score, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[{stage}] early stopping (no improvement in {patience} epochs)")
                break

    model.load_state_dict(best_state)
    print(f"[{stage}] best val_{monitor} = {best_score:.4f}")
    return history


def save_checkpoint(model: nn.Module, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


# --------------------------------------------------------------------------- #
# Autoencoder (Phase 4)
# --------------------------------------------------------------------------- #
# Image-level anomaly score variants computed from the per-pixel reconstruction error.
# A defect is small (median 1.6 % of the image), so the plain mean dilutes it; the top-k
# variants look only at the worst pixels. Which variant to use is chosen on VALIDATION (AP).
SCORE_VARIANTS = ("mean", "top5", "top1", "max")


def error_map(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Per-pixel squared reconstruction error averaged over channels: (B, H, W)."""
    return ((model(x) - x) ** 2).mean(dim=1)


def score_variants(err: torch.Tensor, smooth: int = 9) -> dict[str, torch.Tensor]:
    """Image-level scores (B,) from error maps (B, H, W). `smooth` = odd kernel of the local
    average (suppresses isolated noisy pixels before taking the worst ones)."""
    b = err.shape[0]
    sm = F.avg_pool2d(err.unsqueeze(1), smooth, stride=1, padding=smooth // 2).reshape(b, -1)
    n = sm.shape[1]
    out = {"mean": err.reshape(b, -1).mean(dim=1), "max": sm.max(dim=1).values}
    for name, frac in (("top5", 0.05), ("top1", 0.01)):
        out[name] = sm.topk(max(1, int(frac * n)), dim=1).values.mean(dim=1)
    return out


@torch.no_grad()
def anomaly_scores(model: nn.Module, loader, device, smooth: int = 9, max_batches: int | None = None):
    """(y_true, {variant: scores}, mean reconstruction MSE) over a loader; eval mode."""
    model.eval()
    ys, acc, mse_sum, n = [], {k: [] for k in SCORE_VARIANTS}, 0.0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        # Reduce on CPU: topk over ~136k pixels is unreliable on the MPS backend (in the smoke test
        # top1 and top5 came out constant, AUC exactly 0.5), and it costs little.
        err = error_map(model, x.to(device)).cpu()
        for k, v in score_variants(err, smooth).items():
            acc[k].append(v.numpy())
        mse_sum += err.mean(dim=(1, 2)).sum().item()
        n += len(y)
        ys.append(y.numpy())
    return np.concatenate(ys), {k: np.concatenate(v) for k, v in acc.items()}, mse_sum / max(n, 1)


def fit_autoencoder(model: nn.Module, loaders: dict, device, epochs: int, lr: float,
                    weight_decay: float = 1e-5, patience: int = 6, monitor: str = "top1",
                    smooth: int = 9, max_batches: int | None = None, wandb_run=None) -> list[dict]:
    """Train on good parts only (MSE) and leave in `model` the weights of the epoch with the best
    validation AP of the `monitor` score variant. Validation labels are used ONLY to pick the
    epoch (the loss never sees them); the test set never intervenes.
    """
    if monitor not in SCORE_VARIANTS:
        raise ValueError(f"monitor must be one of {SCORE_VARIANTS}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    best_score, best_state, bad, history = -1.0, None, 0, []

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        total, n = 0.0, 0
        for i, (x, _) in enumerate(loaders["train_good"]):
            if max_batches is not None and i >= max_batches:
                break
            x = x.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), x)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x)
            n += len(x)
        y, scores, val_mse = anomaly_scores(model, loaders["val"], device, smooth, max_batches)
        two_classes = len(set(y.tolist())) > 1
        aps = {k: (float(average_precision_score(y, v)) if two_classes else 0.0) for k, v in scores.items()}
        val_auc = float(roc_auc_score(y, scores[monitor])) if two_classes else float("nan")
        val_f2 = best_f2(y, scores[monitor])[0] if two_classes else 0.0
        row = dict(stage="ae", epoch=epoch, train_loss=total / max(n, 1), val_mse=val_mse,
                   val_ap=aps[monitor], val_auc=val_auc, val_f2=val_f2,
                   **{f"ap_{k}": v for k, v in aps.items()}, seconds=time.perf_counter() - t0)
        history.append(row)
        print(f"[ae] ep {epoch}/{epochs} | train_mse {row['train_loss']:.4f} | val_mse {val_mse:.4f}"
              f" | val_AP({monitor}) {row['val_ap']:.4f} | val_AUC {val_auc:.4f} | val_F2 {val_f2:.4f}"
              f" | {row['seconds']:.0f}s")
        if wandb_run is not None:
            wandb_run.log({f"ae/{k}": v for k, v in row.items() if k != "stage"})
        if row["val_ap"] > best_score:
            best_score, best_state, bad = row["val_ap"], copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[ae] early stopping (no improvement in {patience} epochs)")
                break

    model.load_state_dict(best_state)
    print(f"[ae] best val_AP({monitor}) = {best_score:.4f}")
    return history
