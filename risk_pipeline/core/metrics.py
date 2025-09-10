"""
Core metrics helpers with NaN-safe computations for regression and classification.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    log_loss,
)


def mape_safe(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = mape_safe(y_true, y_pred, eps=1e-9)
    return {"mse": mse, "mae": mae, "r2": r2, "mape": mape}


def classification_metrics(
    y_true: np.ndarray,
    proba_or_pred: np.ndarray,
    threshold: float = 0.5,
    pos_label: int = 1,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true)
    a = np.asarray(proba_or_pred)

    # Accept probabilities or hard labels
    if a.ndim == 2 and a.shape[1] == 2:
        y_prob = a[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        try:
            ll = float(log_loss(y_true, a, labels=[0, 1]))
        except Exception:
            ll = np.nan
    elif a.ndim == 1 and np.all((0.0 <= a) & (a <= 1.0)):
        y_prob = a
        y_pred = (y_prob >= threshold).astype(int)
        try:
            ll = float(log_loss(y_true, np.c_[1 - y_prob, y_prob], labels=[0, 1]))
        except Exception:
            ll = np.nan
    else:
        y_prob = None
        y_pred = a.astype(int)
        ll = np.nan

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0, pos_label=pos_label))
    pre = float(precision_score(y_true, y_pred, zero_division=0, pos_label=pos_label))
    rec = float(recall_score(y_true, y_pred, zero_division=0, pos_label=pos_label))
    return {"acc": acc, "f1": f1, "precision": pre, "recall": rec, "logloss": ll}


