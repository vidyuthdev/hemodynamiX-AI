"""Classification metrics: AUROC, ECE, Brier, F1, sensitivity / specificity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score


@dataclass
class ThresholdMetrics:
    threshold: float
    f1: float
    sensitivity: float
    specificity: float
    accuracy: float


def auroc(p: np.ndarray, y: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return 0.5
    return float(roc_auc_score(y, p))


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 12) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(p)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if b == n_bins - 1:
            mask = (p >= lo) & (p <= hi + 1e-9)
        else:
            mask = (p >= lo) & (p < hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(p[mask]))
        acc = float(np.mean(y[mask]))
        ece += abs(conf - acc) * mask.sum() / n
    return float(ece)


def best_threshold(p: np.ndarray, y: np.ndarray) -> ThresholdMetrics:
    """Find threshold that maximizes Youden's J = sensitivity + specificity - 1."""
    if len(np.unique(y)) < 2:
        return ThresholdMetrics(0.5, 0.0, 0.0, 0.0, float(np.mean(y == 0)))
    candidates = np.unique(np.clip(p, 0.0, 1.0))
    if candidates.size > 64:
        candidates = np.quantile(p, np.linspace(0.05, 0.95, 64))
    best = ThresholdMetrics(0.5, -1, 0, 0, 0)
    for t in candidates:
        yhat = (p >= t).astype(np.int64)
        tp = int(np.sum((yhat == 1) & (y == 1)))
        fp = int(np.sum((yhat == 1) & (y == 0)))
        fn = int(np.sum((yhat == 0) & (y == 1)))
        tn = int(np.sum((yhat == 0) & (y == 0)))
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        prec = tp / max(tp + fp, 1)
        f1 = 2 * prec * sens / max(prec + sens, 1e-9)
        j = sens + spec - 1.0
        if j > best.f1:
            best = ThresholdMetrics(
                threshold=float(t),
                f1=float(f1),
                sensitivity=float(sens),
                specificity=float(spec),
                accuracy=float((tp + tn) / max(tp + fp + fn + tn, 1)),
            )
            best.f1 = j  # store J in 'f1' slot for sorting; recompute true f1 below
    # recompute true f1 at chosen threshold
    yhat = (p >= best.threshold).astype(np.int64)
    tp = int(np.sum((yhat == 1) & (y == 1)))
    fp = int(np.sum((yhat == 1) & (y == 0)))
    fn = int(np.sum((yhat == 0) & (y == 1)))
    prec = tp / max(tp + fp, 1)
    sens = tp / max(tp + fn, 1)
    f1 = 2 * prec * sens / max(prec + sens, 1e-9)
    best.f1 = float(f1)
    return best


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(brier_score_loss(y, p))


def reliability_bins(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> List[dict]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[dict] = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (p >= lo) & (p < hi) if b < n_bins - 1 else (p >= lo) & (p <= hi + 1e-9)
        if not np.any(mask):
            out.append({"pMean": float((lo + hi) / 2.0), "yMean": 0.0, "count": 0})
            continue
        out.append(
            {
                "pMean": float(np.mean(p[mask])),
                "yMean": float(np.mean(y[mask])),
                "count": int(mask.sum()),
            }
        )
    return out
