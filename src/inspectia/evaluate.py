"""Evaluation protocol (Phase 2): the single yardstick used by ALL models.

Primary metric: F2 on the defective class (recall weighs 4x precision). Also recall, precision,
AUC-ROC and average precision. Accuracy is never a headline metric (~89% of parts are good).

Threshold (plan 2.1): the one that MAXIMIZES F2 on validation (`best_f2`). Once chosen it is
frozen and applied unchanged to the test set in Phase 5. `threshold_at_recall` is a secondary
operating point (highest threshold that still reaches a target recall) reported for context.
Every model (baseline, CNN, autoencoder) is just a `score` where higher = more suspicious.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, fbeta_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)


def evaluate(y_true, scores, threshold: float) -> dict:
    """Full metrics at one threshold (prediction = score >= threshold)."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, scores)),
        "avg_precision": float(average_precision_score(y_true, scores)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def _f2_per_threshold(y_true, scores):
    """F2 at every candidate threshold, vectorized from the precision-recall curve."""
    y_true = np.asarray(y_true).astype(int)
    p, r, thr = precision_recall_curve(y_true, np.asarray(scores, dtype=float))
    p, r = p[:-1], r[:-1]  # align with `thr` (the last point is the artificial recall=0)
    denom = 4 * p + r
    f2 = np.divide(5 * p * r, denom, out=np.zeros_like(denom), where=denom > 0)
    return f2, thr


def best_f2(y_true, scores) -> tuple[float, float]:
    """(best F2, threshold). Ties go to the LOWEST threshold, i.e. more recall."""
    f2, thr = _f2_per_threshold(y_true, scores)
    i = int(np.argmax(f2))
    return float(f2[i]), float(thr[i])


def best_f2_threshold(y_val, scores_val) -> float:
    """OFFICIAL protocol: the validation threshold that maximizes F2."""
    return best_f2(y_val, scores_val)[1]


def threshold_at_recall(y_val, scores_val, min_recall: float = 0.95) -> float:
    """Secondary operating point: the HIGHEST threshold with recall >= `min_recall` on validation.

    Caveat: with poor or discrete scores (e.g. kNN) it can fall to 0 and flag everything, so
    always compare against `trivial_baselines`. If no threshold reaches the target it falls
    back to the best-F2 threshold.
    """
    y_val = np.asarray(y_val).astype(int)
    _, recall, thr = precision_recall_curve(y_val, np.asarray(scores_val, dtype=float))
    ok = np.where(recall[:-1] >= min_recall)[0]
    return float(thr[ok].max()) if len(ok) else best_f2_threshold(y_val, scores_val)


def trivial_baselines(y_true) -> dict:
    """Reference classifiers: 'always OK' (shows why accuracy misleads) and 'flag everything'
    (recall 1.0 at the lowest possible precision). A model must beat BOTH in F2."""
    y = np.asarray(y_true).astype(int)
    out = {}
    for name, pred in (("always_ok", np.zeros_like(y)), ("flag_all", np.ones_like(y))):
        out[name] = {
            "accuracy": float((pred == y).mean()),  # shown only to expose how misleading it is
            "recall": float(recall_score(y, pred, zero_division=0)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "f2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
        }
    return out


def bootstrap_ci(y_true, scores, threshold: float, n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 42) -> dict[str, tuple[float, float]]:
    """Stratified bootstrap confidence intervals for F2, recall and precision at a FIXED threshold.

    With ~49 defective parts in validation every miss moves recall by ~2 points, so a single
    number overstates certainty: report the interval too.
    """
    y = np.asarray(y_true).astype(int)
    pred = (np.asarray(scores, dtype=float) >= threshold).astype(int)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    rng = np.random.RandomState(seed)
    vals = {"f2": [], "recall": [], "precision": []}
    for _ in range(n_boot):
        p, n = rng.choice(pos, len(pos)), rng.choice(neg, len(neg))
        tp, fn, fp = pred[p].sum(), len(p) - pred[p].sum(), pred[n].sum()
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        vals["precision"].append(prec)
        vals["recall"].append(rec)
        vals["f2"].append(5 * prec * rec / (4 * prec + rec) if 4 * prec + rec else 0.0)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: (float(np.percentile(v, lo)), float(np.percentile(v, hi))) for k, v in vals.items()}


def plot_curves(results: dict[str, tuple], title: str = ""):
    """ROC and Precision-Recall curves of several models. results = {name: (y_true, scores)}."""
    import matplotlib.pyplot as plt

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, (y, s) in results.items():
        fpr, tpr, _ = roc_curve(y, s)
        p, r, _ = precision_recall_curve(y, s)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, s):.3f})")
        ax_pr.plot(r, p, label=f"{name} (AP={average_precision_score(y, s):.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax_roc.set(xlabel="False positive rate", ylabel="Recall", title="ROC")
    ax_pr.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall")
    for ax in (ax_roc, ax_pr):
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_confusion(metrics: dict, title: str = ""):
    """2x2 confusion matrix from the dict returned by `evaluate`."""
    import matplotlib.pyplot as plt

    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, int(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["pred OK", "pred defect"],
           yticklabels=["real OK", "real defect"], title=title)
    fig.tight_layout()
    return fig
