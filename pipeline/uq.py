"""Split-conformal prediction for binary classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConformalSummary:
    alpha: float
    q: float
    empirical_coverage: float
    interval_width_mean: float
    abstain_rate: float


def calibrate(p_cal: np.ndarray, y_cal: np.ndarray, alpha: float) -> float:
    """Compute the (1 - alpha) quantile of nonconformity scores."""
    s = np.where(y_cal == 1, 1.0 - p_cal, p_cal)
    n = len(s)
    if n == 0:
        return 0.5
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = max(1, min(n, k))
    return float(np.sort(s)[k - 1])


def evaluate(p_test: np.ndarray, y_test: np.ndarray, q: float, alpha: float) -> ConformalSummary:
    s_test = np.where(y_test == 1, 1.0 - p_test, p_test)
    covered = (s_test <= q).astype(np.float64)
    coverage = float(np.mean(covered))

    # Width of the conformal prediction set: 0, 1, or 2 labels admitted.
    # A prediction set is {y in {0,1} : nonconformity(y) <= q}.
    in_set_neg = (p_test <= q).astype(np.int64)        # would admit y=0
    in_set_pos = ((1.0 - p_test) <= q).astype(np.int64)  # would admit y=1
    width = (in_set_neg + in_set_pos).astype(np.float64)

    abstain = float(np.mean(width >= 2.0))

    return ConformalSummary(
        alpha=alpha,
        q=q,
        empirical_coverage=coverage,
        interval_width_mean=float(np.mean(width)),
        abstain_rate=abstain,
    )
