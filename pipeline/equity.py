"""Equity audit + resolution stress test."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .config import LOCATIONS
from .metrics import auroc


@dataclass
class SubgroupResult:
    subgroup: str
    n: int
    auroc: float
    positives: int


@dataclass
class ResolutionResult:
    noise_level: float
    auroc: float


def by_location(
    test_locations: List[str],
    p_test: np.ndarray,
    y_test: np.ndarray,
) -> List[SubgroupResult]:
    out: list[SubgroupResult] = []
    for loc in LOCATIONS:
        mask = np.array([l == loc for l in test_locations])
        n = int(mask.sum())
        if n < 4:
            out.append(SubgroupResult(subgroup=loc, n=n, auroc=0.5, positives=int(np.sum(y_test[mask]))))
            continue
        ss = p_test[mask]
        yy = y_test[mask]
        try:
            a = auroc(ss, yy)
        except Exception:
            a = 0.5
        out.append(SubgroupResult(subgroup=loc, n=n, auroc=float(a), positives=int(np.sum(yy))))
    return out


def resolution_stress(
    predict_fn,
    X_test: np.ndarray,
    y_test: np.ndarray,
    noise_levels=(0.0, 0.15, 0.3, 0.5, 0.75, 1.1),
    seed: int = 42,
) -> List[ResolutionResult]:
    rng = np.random.default_rng(seed)
    mu = X_test.mean(axis=0)
    sd = X_test.std(axis=0) + 1e-9
    out: list[ResolutionResult] = []
    for sig in noise_levels:
        Xn = X_test if sig == 0 else X_test + sig * sd * rng.standard_normal(X_test.shape)
        try:
            preds = predict_fn(Xn)
            a = auroc(preds, y_test)
        except Exception:
            a = 0.5
        out.append(ResolutionResult(noise_level=float(sig), auroc=float(a)))
    return out
