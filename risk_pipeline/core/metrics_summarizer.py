"""
Metrics summarizer helpers to aggregate per-fold results robustly.
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Any


def _v(key: str, folds: List[Dict[str, Any]], default: float = np.nan) -> np.ndarray:
    """Extract values for a specific key from fold metrics with better error handling."""
    import logging
    logger = logging.getLogger(__name__)
    
    values = []
    for i, fold in enumerate(folds):
        try:
            if key in fold:
                val = fold[key]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    values.append(float(val))
                else:
                    logger.debug(f"Fold {i+1}: {key} is None or NaN, using default {default}")
                    values.append(default)
            else:
                logger.debug(f"Fold {i+1}: {key} not found, using default {default}")
                values.append(default)
        except (ValueError, TypeError) as e:
            logger.warning(f"Fold {i+1}: Could not convert {key}={fold.get(key, 'N/A')} to float: {e}")
            values.append(default)
    
    logger.debug(f"Extracted {len(values)} values for {key}: {values}")
    return np.array(values, dtype=float)


def _nz(x: np.ndarray) -> np.ndarray:
    # Don't convert NaN to 0.0 - this masks real issues
    # Instead, filter out NaN values for proper aggregation
    arr = np.asarray(x, dtype=float)
    # Only convert inf to large numbers, keep NaN as NaN
    arr = np.nan_to_num(arr, nan=np.nan, posinf=1e6, neginf=-1e6)
    return arr


def _mean_std(xs: List[float]) -> tuple[float, float]:
    """Safe mean/std calculation with guards against empty lists."""
    import logging
    logger = logging.getLogger(__name__)
    
    if not xs or len(xs) == 0:
        logger.debug("Empty list provided to _mean_std, returning (nan, nan)")
        return (float("nan"), float("nan"))
    
    arr = np.asarray(xs, dtype=float)
    logger.debug(f"Input array: {arr}")
    
    # Filter out NaN values instead of converting to 0.0
    arr_clean = arr[~np.isnan(arr)]
    logger.debug(f"After NaN filtering: {arr_clean}")
    
    if len(arr_clean) == 0:
        logger.debug("All values were NaN, returning (nan, nan)")
        return (float("nan"), float("nan"))
    
    mean_val = float(np.mean(arr_clean))
    std_val = float(np.std(arr_clean, ddof=1))
    logger.debug(f"Calculated mean: {mean_val}, std: {std_val}")
    
    return (mean_val, std_val)


def _validate_fold_metrics(fold_metrics: List[Dict[str, Any]], task: str) -> bool:
    """Validate that fold metrics have the expected structure."""
    import logging
    logger = logging.getLogger(__name__)
    
    if not fold_metrics:
        logger.error(f"No fold metrics provided for {task}")
        return False
    
    expected_keys = {
        'regression': ['mse', 'mae', 'r2', 'mape'],
        'classification': ['accuracy', 'f1', 'precision', 'recall']
    }
    
    required_keys = expected_keys.get(task, [])
    logger.debug(f"Validating {task} metrics with required keys: {required_keys}")
    
    for i, fold in enumerate(fold_metrics):
        if not isinstance(fold, dict):
            logger.error(f"Fold {i+1} is not a dictionary: {type(fold)}")
            return False
        
        missing_keys = [key for key in required_keys if key not in fold]
        if missing_keys:
            logger.warning(f"Fold {i+1} missing keys: {missing_keys}")
            logger.debug(f"Fold {i+1} available keys: {list(fold.keys())}")
    
    logger.debug(f"Fold metrics validation completed for {len(fold_metrics)} folds")
    return True


def summarize_regression(fold_metrics: List[Dict[str, Any]], fit_times: List[float], pred_times: List[float]) -> Dict[str, Any]:
    """
    FIXED: Properly aggregate regression metrics to prevent nan values.
    Centralize fold metrics and aggregate the lists instead of computing RMSE from averaged MSE.
    """
    # Add debug logging
    import logging
    logger = logging.getLogger(__name__)
    logger.debug(f"Summarizing regression metrics for {len(fold_metrics)} folds")
    
    # Validate input
    if not _validate_fold_metrics(fold_metrics, 'regression'):
        logger.error("Invalid fold metrics for regression, returning empty results")
        return {}
    
    # Extract metrics from each fold - FIXED: extract directly from fold dictionaries
    mse_list = [fold.get("mse", np.nan) for fold in fold_metrics]
    mae_list = [fold.get("mae", np.nan) for fold in fold_metrics]
    r2_list = [fold.get("r2", np.nan) for fold in fold_metrics]
    mape_list = [fold.get("mape", np.nan) for fold in fold_metrics]
    
    # Log extracted values for debugging
    logger.debug(f"Extracted MSE values: {mse_list}")
    logger.debug(f"Extracted MAE values: {mae_list}")
    logger.debug(f"Extracted R² values: {r2_list}")
    logger.debug(f"Extracted MAPE values: {mape_list}")
    
    # Extract baseline metrics from each fold
    baseline_mse_list = [fold.get("baseline_mse", np.nan) for fold in fold_metrics]
    baseline_mae_list = [fold.get("baseline_mae", np.nan) for fold in fold_metrics]
    baseline_r2_list = [fold.get("baseline_r2", np.nan) for fold in fold_metrics]
    
    # Calculate RMSE from individual fold MSE values, not from averaged MSE
    rmse_list = [np.sqrt(max(mse, 0.0)) for mse in mse_list]
    
    # Calculate sMAPE if we have y_true and y_pred in fold metrics
    smape_list = []
    for fold in fold_metrics:
        if 'y_true' in fold and 'y_pred' in fold:
            try:
                y_true = np.asarray(fold['y_true'])
                y_pred = np.asarray(fold['y_pred'])
                eps = 1e-8
                # sMAPE = 2 * mean(|y_true - y_pred| / (|y_true| + |y_pred| + eps))
                numerator = np.abs(y_true - y_pred)
                denominator = np.abs(y_true) + np.abs(y_pred) + eps
                smape = 2.0 * np.mean(numerator / denominator)
                smape_list.append(float(smape))
            except Exception:
                smape_list.append(np.nan)
        else:
            smape_list.append(np.nan)
    
    # Aggregate metrics using safe mean/std
    mse_mean, mse_std = _mean_std(mse_list)
    mae_mean, mae_std = _mean_std(mae_list)
    r2_mean, r2_std = _mean_std(r2_list)
    rmse_mean, rmse_std = _mean_std(rmse_list)
    mape_mean, mape_std = _mean_std(mape_list)
    
    # Aggregate baseline metrics
    baseline_mse_mean, baseline_mse_std = _mean_std(baseline_mse_list)
    baseline_mae_mean, baseline_mae_std = _mean_std(baseline_mae_list)
    baseline_r2_mean, baseline_r2_std = _mean_std(baseline_r2_list)
    
    # Aggregate sMAPE
    smape_mean, smape_std = _mean_std(smape_list)
    
    # Handle timing metrics
    ft_mean, ft_std = _mean_std(fit_times)
    pt_mean, pt_std = _mean_std(pred_times)
    
    # Log final aggregated results for debugging
    logger.debug(f"Final aggregated results:")
    logger.debug(f"  MSE: {mse_mean:.6f} ± {mse_std:.6f}")
    logger.debug(f"  RMSE: {rmse_mean:.6f} ± {rmse_std:.6f}")
    logger.debug(f"  MAE: {mae_mean:.6f} ± {mae_std:.6f}")
    logger.debug(f"  R²: {r2_mean:.6f} ± {r2_std:.6f}")
    logger.debug(f"  MAPE: {mape_mean:.6f} ± {mape_std:.6f}")
    
    return {
        "MSE": mse_mean, "MSE_std": mse_std,
        "RMSE": rmse_mean, "RMSE_std": rmse_std,  # FIXED: Not sqrt(mean(MSE))
        "MAE": mae_mean, "MAE_std": mae_std,
        "R2": r2_mean, "R2_std": r2_std,
        "MAPE": mape_mean, "MAPE_std": mape_std,  # FIXED: Proper aggregation
        "sMAPE": smape_mean, "sMAPE_std": smape_std,  # NEW: Symmetric MAPE
        "Baseline MSE": baseline_mse_mean, "Baseline MSE_std": baseline_mse_std,
        "Baseline MAE": baseline_mae_mean, "Baseline MAE_std": baseline_mae_std,
        "Baseline R2": baseline_r2_mean, "Baseline R2_std": baseline_r2_std,
        "Fit Time": ft_mean, "Fit Time_std": ft_std,
        "Prediction Time": pt_mean, "Prediction Time_std": pt_std,
    }


def summarize_classification(fold_metrics: List[Dict[str, Any]], fit_times: List[float], pred_times: List[float]) -> Dict[str, Any]:
    # Add debug logging
    import logging
    logger = logging.getLogger(__name__)
    logger.debug(f"Summarizing classification metrics for {len(fold_metrics)} folds")
    
    # Validate input
    if not _validate_fold_metrics(fold_metrics, 'classification'):
        logger.error("Invalid fold metrics for classification, returning empty results")
        return {}
    
    # Extract metrics from each fold - FIXED: extract directly from fold dictionaries
    acc_list = [fold.get("accuracy", np.nan) for fold in fold_metrics]
    f1_list = [fold.get("f1", np.nan) for fold in fold_metrics]
    pre_list = [fold.get("precision", np.nan) for fold in fold_metrics]
    rec_list = [fold.get("recall", np.nan) for fold in fold_metrics]
    
    # Log extracted values for debugging
    logger.debug(f"Extracted Accuracy values: {acc_list}")
    logger.debug(f"Extracted F1 values: {f1_list}")
    logger.debug(f"Extracted Precision values: {pre_list}")
    logger.debug(f"Extracted Recall values: {rec_list}")
    
    # Extract baseline metrics
    baseline_acc_list = [fold.get("baseline_accuracy", np.nan) for fold in fold_metrics]
    baseline_f1_list = [fold.get("baseline_f1", np.nan) for fold in fold_metrics]
    
    # Extract logloss if available
    logloss_list = []
    for fold in fold_metrics:
        if 'logloss' in fold and fold['logloss'] is not None:
            logloss_list.append(float(fold['logloss']))
        else:
            logloss_list.append(np.nan)
    
    # Aggregate using safe mean/std
    acc_mean, acc_std = _mean_std(acc_list)
    f1_mean, f1_std = _mean_std(f1_list)
    pre_mean, pre_std = _mean_std(pre_list)
    rec_mean, rec_std = _mean_std(rec_list)
    baseline_acc_mean, baseline_acc_std = _mean_std(baseline_acc_list)
    baseline_f1_mean, baseline_f1_std = _mean_std(baseline_f1_list)
    logloss_mean, logloss_std = _mean_std(logloss_list)
    
    # Handle timing metrics
    ft_mean, ft_std = _mean_std(fit_times)
    pt_mean, pt_std = _mean_std(pred_times)
    
    # Log final aggregated results for debugging
    logger.debug(f"Final aggregated results:")
    logger.debug(f"  Accuracy: {acc_mean:.6f} ± {acc_std:.6f}")
    logger.debug(f"  F1: {f1_mean:.6f} ± {f1_std:.6f}")
    logger.debug(f"  Precision: {pre_mean:.6f} ± {pre_std:.6f}")
    logger.debug(f"  Recall: {rec_mean:.6f} ± {rec_std:.6f}")
    
    return {
        "Accuracy": acc_mean, "Accuracy_std": acc_std,
        "F1": f1_mean, "F1_std": f1_std,
        "Precision": pre_mean, "Precision_std": pre_std,
        "Recall": rec_mean, "Recall_std": rec_std,
        "Baseline Accuracy": baseline_acc_mean, "Baseline Accuracy_std": baseline_acc_std,
        "Baseline F1": baseline_f1_mean, "Baseline F1_std": baseline_f1_std,
        "LogLoss": logloss_mean, "LogLoss_std": logloss_std,
        "Fit Time": ft_mean, "Fit Time_std": ft_std,
        "Prediction Time": pt_mean, "Prediction Time_std": pt_std,
    }


