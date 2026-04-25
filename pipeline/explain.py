"""SHAP global + per-case explanations.

Uses TreeExplainer for tree models (XGBoost / RF), Permutation explainer for
others (MLP). Always returns the same shape so downstream consumers don't care.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import shap

from .config import FEATURE_KEYS


@dataclass
class FeatureImportance:
    feature: str
    delta: float


@dataclass
class CaseAttribution:
    case_id: str
    baseline: float
    prediction: float
    contributions: list[dict]


def _is_tree(est) -> bool:
    name = type(est).__name__
    return name in {"XGBClassifier", "RandomForestClassifier"}


def _shap_values(model, X_train_summary, X_query):
    """Return SHAP values (n_samples, n_features) for the positive class."""
    est = model.estimator
    predict_proba = lambda Z: model.predict_proba(Z)
    try:
        if _is_tree(est):
            explainer = shap.TreeExplainer(est)
            sv = explainer.shap_values(X_query)
        else:
            # Permutation explainer is fast, model-agnostic, and handles MLP fine
            background = shap.sample(X_train_summary, min(50, X_train_summary.shape[0]), random_state=0)
            explainer = shap.Explainer(predict_proba, background)
            sv = explainer(X_query).values
        sv = np.asarray(sv)
        # Some explainers return shape (n, k, classes) - select positive class
        if sv.ndim == 3:
            if sv.shape[-1] == 2:
                sv = sv[..., 1]
            else:
                sv = sv[..., 0]
        elif isinstance(sv, list):
            sv = np.asarray(sv[1] if len(sv) > 1 else sv[0])
        return sv
    except Exception as e:
        print(f"  ! SHAP failed ({e}); falling back to permutation importance")
        # crude fallback: per-feature one-shot permutation
        base = predict_proba(X_query)
        sv = np.zeros_like(X_query)
        rng = np.random.default_rng(0)
        for j in range(X_query.shape[1]):
            Xp = X_query.copy()
            rng.shuffle(Xp[:, j])
            sv[:, j] = base - predict_proba(Xp)
        return sv


def shap_global(model, X_train: np.ndarray, X_test: np.ndarray) -> List[FeatureImportance]:
    sv = _shap_values(model, X_train, X_test)
    importance = np.mean(np.abs(sv), axis=0)
    return [
        FeatureImportance(feature=k, delta=float(importance[i]))
        for i, k in enumerate(FEATURE_KEYS)
    ]


def shap_per_case(
    model,
    X_train: np.ndarray,
    X_cases: np.ndarray,
    case_ids: List[str],
    baseline_p: float,
) -> List[CaseAttribution]:
    sv = _shap_values(model, X_train, X_cases)
    preds = model.predict_proba(X_cases)
    out: list[CaseAttribution] = []
    for i, cid in enumerate(case_ids):
        contribs = [
            {"feature": k, "delta": float(sv[i, j])}
            for j, k in enumerate(FEATURE_KEYS)
        ]
        out.append(
            CaseAttribution(
                case_id=cid,
                baseline=float(baseline_p),
                prediction=float(preds[i]),
                contributions=contribs,
            )
        )
    return out
