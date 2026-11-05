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
