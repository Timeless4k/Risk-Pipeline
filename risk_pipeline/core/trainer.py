from __future__ import annotations

import time
import random
from dataclasses import dataclass
from typing import Any, Dict, Tuple
import numpy as np

from risk_pipeline.config.global_config import GlobalConfig


@dataclass
class TrainResult:
    y_val_pred: np.ndarray
    y_val_true: np.ndarray
    fit_time_s: float
    pred_time_s: float


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, cfg: GlobalConfig) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if cfg.track_mse:
        metrics["MSE"] = float(np.mean((y_true - y_pred) ** 2))
    if cfg.track_mae:
        metrics["MAE"] = float(np.mean(np.abs(y_true - y_pred)))
    if cfg.track_spearman_ic:
        try:
            from scipy.stats import spearmanr
            ic = spearmanr(y_true, y_pred, nan_policy="omit")[0]
        except Exception:
            ic = np.nan
        metrics["IC"] = float(ic) if ic is not None else float("nan")
    return metrics


def train_once(model, X_train, y_train, X_val, y_val, cfg: GlobalConfig) -> Tuple[TrainResult, Dict[str, float]]:
    set_global_seeds(cfg.random_seed)
    start_fit = time.time()
    # Fit model
    if hasattr(model, 'fit'):
        # Pass cfg as keyword to support signatures like fit(self, X, y, **kwargs)
        model.fit(X_train, y_train, cfg=cfg)
    else:
        # legacy fallback
        model.train(X_train, y_train)
    fit_time = time.time() - start_fit

    start_pred = time.time()
    y_val_pred = model.predict(X_val)
    pred_time = time.time() - start_pred

    y_val_pred = np.asarray(y_val_pred).reshape(-1)
    y_val_true = np.asarray(y_val).reshape(-1)
    # Align lengths defensively in case a model emits fewer preds
    n = min(len(y_val_true), len(y_val_pred))
    y_val_true = y_val_true[:n]
    y_val_pred = y_val_pred[:n]

    result = TrainResult(y_val_pred=y_val_pred, y_val_true=y_val_true, fit_time_s=fit_time, pred_time_s=pred_time)
    metrics = compute_metrics(y_val_true, y_val_pred, cfg)
    return result, metrics
