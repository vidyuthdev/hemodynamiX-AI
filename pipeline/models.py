"""Model wrappers: XGBoost, Random Forest, MLP, isotonic + Platt calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainedModel:
    name: str
    estimator: object
    raw_predict: Callable
    calibrator: Optional[object] = None
    calibrator_kind: str = "none"      # "isotonic" | "platt" | "none"
    scaler: Optional[StandardScaler] = None

    def _raw(self, X: np.ndarray) -> np.ndarray:
        Z = self.scaler.transform(X) if self.scaler is not None else X
        return np.clip(self.raw_predict(Z), 1e-6, 1 - 1e-6)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self._raw(X)
        if self.calibrator is None:
            return raw
        if self.calibrator_kind == "isotonic":
            return np.clip(self.calibrator.predict(raw), 1e-6, 1 - 1e-6)
        if self.calibrator_kind == "platt":
            logits = np.log(raw / (1.0 - raw)).reshape(-1, 1)
            return self.calibrator.predict_proba(logits)[:, 1]
        return raw


def fit_xgb(X: np.ndarray, y: np.ndarray, seed: int) -> TrainedModel:
    clf = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=2,
        tree_method="hist",
    )
    clf.fit(X, y)
    return TrainedModel(name="XGBoost", estimator=clf, raw_predict=lambda Z: clf.predict_proba(Z)[:, 1])


def fit_rf(X: np.ndarray, y: np.ndarray, seed: int) -> TrainedModel:
    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        n_jobs=2,
        random_state=seed,
    )
    clf.fit(X, y)
    return TrainedModel(name="Random Forest", estimator=clf, raw_predict=lambda Z: clf.predict_proba(Z)[:, 1])


def fit_mlp(X: np.ndarray, y: np.ndarray, seed: int) -> TrainedModel:
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    clf = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        solver="adam",
        max_iter=1200,
        learning_rate_init=0.005,
        alpha=1e-3,
        random_state=seed,
    )
    clf.fit(Xs, y)
    return TrainedModel(
        name="Shallow MLP",
        estimator=clf,
        raw_predict=lambda Z: clf.predict_proba(Z)[:, 1],
        scaler=scaler,
    )


def fit_calibrator(model: TrainedModel, X_cal: np.ndarray, y_cal: np.ndarray) -> TrainedModel:
    """Choose isotonic if calibration set is large enough, otherwise Platt."""
    raw = model._raw(X_cal)
    if len(np.unique(y_cal)) < 2 or len(y_cal) < 8:
        return model
    if len(y_cal) >= 25:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-3, y_max=1 - 1e-3)
        iso.fit(raw, y_cal)
        model.calibrator = iso
        model.calibrator_kind = "isotonic"
        return model
    logits = np.log(raw / (1.0 - raw)).reshape(-1, 1)
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=400)
    lr.fit(logits, y_cal)
    model.calibrator = lr
    model.calibrator_kind = "platt"
    return model
