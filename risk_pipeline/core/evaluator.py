from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from risk_pipeline.config.global_config import GlobalConfig
from risk_pipeline.core.splits import generate_sliding_splits
from risk_pipeline.core.trainer import train_once
from risk_pipeline.core.adapters import SeqAdapterModel, FlatAdapterModel
from risk_pipeline.models.lstm_model import LSTMModel
from risk_pipeline.models.stockmixer_model import StockMixerModel
from risk_pipeline.models.xgboost_model import XGBoostModel
from risk_pipeline.models.arima_model import ARIMAModel
from risk_pipeline.core.feature_engineer import FeatureEngineer


SEQ_MODELS = {"lstm", "stockmixer"}
FLAT_MODELS = {"xgb", "arima", "linear"}


@dataclass
class EvalRow:
    model: str
    mse: float
    mae: float
    ic: float
    fit_time_s: float
    pred_time_s: float


def evaluate_all(models: Dict[str, any], raw_df: pd.DataFrame, target: pd.Series, cfg: GlobalConfig) -> pd.DataFrame:
    fe = FeatureEngineer()

    rows: List[EvalRow] = []
    n = len(raw_df)
    for train_slc, val_slc in generate_sliding_splits(n, cfg.train_size, cfg.val_size, cfg.step):
        (X_seq_tr, X_flat_tr, y_tr), (X_seq_va, X_flat_va, y_va) = fe.create_canonical_views(raw_df, target, cfg, train_slc, val_slc)

        for model_name in cfg.models_to_run:
            if model_name not in models:
                continue
            model = models[model_name]
            if model_name in SEQ_MODELS:
                X_tr, X_va = X_seq_tr, X_seq_va
            else:
                X_tr, X_va = X_flat_tr, X_flat_va

            result, metrics = train_once(model, X_tr, y_tr, X_va, y_va, cfg)
            rows.append(EvalRow(
                model=model_name,
                mse=metrics.get("MSE", np.nan),
                mae=metrics.get("MAE", np.nan),
                ic=metrics.get("IC", np.nan),
                fit_time_s=result.fit_time_s,
                pred_time_s=result.pred_time_s,
            ))

    # Aggregate
    df = pd.DataFrame([r.__dict__ for r in rows])
    agg = df.groupby("model").agg({
        "mse": ["mean", "std"],
        "mae": ["mean", "std"],
        "ic": ["mean", "std"],
        "fit_time_s": "mean",
        "pred_time_s": "mean",
    })
    agg.columns = ["_".join(c for c in col if c) for col in agg.columns.to_flat_index()]
    agg = agg.reset_index()

    # Persist
    os.makedirs(cfg.artifacts_dir, exist_ok=True)
    out_csv = os.path.join(cfg.artifacts_dir, "results.csv")
    agg.to_csv(out_csv, index=False)

    # Markdown
    out_md = os.path.join(cfg.artifacts_dir, "RESULTS.md")
    try:
        md = agg.to_markdown(index=False)
    except Exception:
        md = agg.to_csv(index=False)
    with open(out_md, "w") as f:
        f.write("# Model Comparison Results\n\n")
        f.write(md)

    return agg


def build_models(cfg: GlobalConfig) -> Dict[str, any]:
    models: Dict[str, any] = {}
    if "lstm" in cfg.models_to_run:
        models["lstm"] = SeqAdapterModel(LSTMModel(model_type='lstm', task='regression'), "lstm")
    if "stockmixer" in cfg.models_to_run:
        models["stockmixer"] = SeqAdapterModel(StockMixerModel(model_type='stockmixer', task='regression'), "stockmixer")
    if "xgb" in cfg.models_to_run:
        models["xgb"] = FlatAdapterModel(XGBoostModel(task='regression'), "xgb")
    if "arima" in cfg.models_to_run:
        models["arima"] = FlatAdapterModel(ARIMAModel(model_type='arima', task='regression'), "arima")
    return models
