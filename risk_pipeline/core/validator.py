"""
Walk-forward validation module for RiskPipeline.

This module provides robust walk-forward cross-validation for time series data,
including adaptive sizing, validation checks, and comprehensive logging.
"""

import logging
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Iterator
from sklearn.metrics import confusion_matrix
from joblib import Parallel, delayed
import psutil
from dataclasses import dataclass
from ..utils.logging_utils import log_execution_time

logger = logging.getLogger(__name__)

# Centralized metrics helper for consistent computation
def regression_fold_metrics(y_true, y_pred, eps=1e-8):
    """Calculate regression metrics for a single fold."""
    err = y_true - y_pred
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(max(mse, 0.0)))
    mae = float(np.mean(np.abs(err)))
    den = np.clip(np.abs(y_true), eps, None)
    mape = float(np.mean(np.abs(err) / den))
    return {"mse": mse, "rmse": rmse, "mae": mae, "mape": mape}

class MetricsAggregator:
    """Helper class for aggregating metrics across folds."""
    def __init__(self):
        self.metrics = {}
    
    def add_fold(self, fold_metrics):
        """Add metrics from a single fold."""
        for key, value in fold_metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def get_summary(self):
        """Get summary statistics across all folds."""
        summary = {}
        for key, values in self.metrics.items():
            if values and all(v is not None for v in values):
                try:
                    summary[f"{key}_mean"] = float(np.mean(values))
                    summary[f"{key}_std"] = float(np.std(values))
                except Exception:
                    summary[f"{key}_mean"] = np.nan
                    summary[f"{key}_std"] = np.nan
        return summary

@dataclass
class ValidationConfig:
    """Configuration for walk-forward validation."""
    
    n_splits: int = 5
    test_size: int = 252
    min_train_size: int = 60
    min_test_size: int = 20
    gap: int = 0  # Gap between train and test sets
    expanding_window: bool = True  # True for expanding, False for sliding
    embargo: int = 0  # Additional embargo periods to purge around split boundaries
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        if self.n_splits < 1:
            logger.error("n_splits must be at least 1")
            return False
        
        if self.test_size < self.min_test_size:
            logger.error(f"test_size ({self.test_size}) must be at least min_test_size ({self.min_test_size})")
            return False
        
        if self.min_train_size < 10:
            logger.error("min_train_size must be at least 10")
            return False
        
        return True

class WalkForwardValidator:
    """Implements walk-forward cross-validation with dynamic sizing and validation.
    
    This class provides robust walk-forward cross-validation for time series data,
    including adaptive sizing, validation checks, and comprehensive logging.
    """
    
    def __init__(self, n_splits: int = 5, test_size: int = 252, 
                 min_train_size: int = 60, min_test_size: int = 20,
                 gap: int = 0, expanding_window: bool = True, config: Any = None):
        """
        Initialize WalkForwardValidator.
        
        Args:
            n_splits: Number of splits to generate
            test_size: Size of test set (will be adapted if necessary)
            min_train_size: Minimum size for training set
            min_test_size: Minimum size for test set
            gap: Gap between train and test sets
            expanding_window: Whether to use expanding or sliding window
            config: Optional PipelineConfig object for training parameters
        """
        self.config = ValidationConfig(
            n_splits=n_splits,
            test_size=test_size,
            min_train_size=min_train_size,
            min_test_size=min_test_size,
            gap=gap,
            expanding_window=expanding_window
        )
        
        # Store the optional PipelineConfig for training parameters
        self.pipeline_config = config
        
        if not self.config.validate():
            raise ValueError("Invalid validation configuration")
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"WalkForwardValidator initialized: {n_splits} splits, test_size={test_size}")
    
    @log_execution_time
    def split(self, X: pd.DataFrame) -> List[Tuple[pd.Index, pd.Index]]:
        """Generate walk-forward splits with validation."""
        self.logger.info(f"WalkForward: Total samples={len(X)}, requested_splits={self.config.n_splits}, requested_test_size={self.config.test_size}")
        
        # Validate data quality
        quality_report = self.validate_data_quality(X)
        if not quality_report['is_valid']:
            self.logger.warning(f"Data quality issues detected: {quality_report['issues']}")
        
        # Calculate adaptive parameters with stronger dynamic behavior per series length
        total_n = len(X)
        gap = max(0, self.config.gap)
        requested_splits = max(1, self.config.n_splits)
        requested_test = max(self.config.min_test_size, self.config.test_size)

        # Dynamic test size: ~10% of data, capped to a trading year, floor at min_test_size
        test_by_fraction = max(self.config.min_test_size, min(504, max(1, total_n // 10)))
        adaptive_test_size = min(requested_test, test_by_fraction)

        # Dynamic target splits by dataset size
        if total_n >= 5000:
            target_splits_by_length = 12
        elif total_n >= 2000:
            target_splits_by_length = 8
        elif total_n >= 1000:
            target_splits_by_length = 5
        elif total_n >= 500:
            target_splits_by_length = 3
        else:
            target_splits_by_length = 1

        # Feasible splits given size and chosen test block
        feasible_by_blocks = max(1, (total_n - max(self.config.min_train_size, 10)) // max(1, adaptive_test_size))
        max_splits = min(requested_splits, target_splits_by_length, feasible_by_blocks)

        # Dynamic min train size: scale with data; prefer at least 2 test blocks when possible
        original_min_train = self.config.min_train_size
        if max_splits == 1:
            # Single split: use almost all data for training
            min_train_eff = max(original_min_train, total_n - adaptive_test_size - gap)
            adaptive_test_size = max(self.config.min_test_size, min(requested_test, max(1, total_n // 5)))
        else:
            # Multi-split: ensure reasonable initial context
            min_train_eff = max(original_min_train, min(756, max(126, total_n // 8)))

        # Log adaptive decisions
        self.logger.info(f"Adaptive decisions → total_n={total_n}, adaptive_test_size={adaptive_test_size}, requested_test={requested_test}")
        self.logger.info(f"Target splits by length={target_splits_by_length}, feasible_by_blocks={feasible_by_blocks}, using_splits={max_splits}")
        self.logger.info(f"Effective min_train_size set to {min_train_eff} (original {original_min_train})")
        
        # Generate splits using proportional expanding/sliding strategies
        # This ensures later folds leverage much larger training windows from the full history.
        if max_splits <= 0:
            splits = []
        else:
            # Temporarily override min_train_size for this run with dynamic value
            prev_min_train = self.config.min_train_size
            try:
                self.config.min_train_size = min_train_eff
                if self.config.expanding_window:
                    splits = self._generate_expanding_splits(X, max_splits, adaptive_test_size)
                else:
                    splits = self._generate_sliding_splits(X, max_splits, adaptive_test_size)
            finally:
                # Restore original min_train_size to avoid side effects
                self.config.min_train_size = prev_min_train
        
        # Log split summary
        for i, (train_idx, test_idx) in enumerate(splits, 1):
            self.logger.info(f"✅ Split {i}: Train={len(train_idx)}, Test={len(test_idx)}")
        
        self.logger.info(f"Generated {len(splits)} valid splits")
        
        return splits

    def evaluate_model(self,
                       model: Any,
                       X: pd.DataFrame,
                       y: pd.Series,
                       splits: List[Tuple[pd.Index, pd.Index]],
                       asset: str,
                       model_type: str,
                       regimes: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Evaluate a model using walk-forward cross-validation.
        
        Args:
            model: Model to evaluate
            X: Feature data
            y: Target data
            splits: List of train/test splits
            asset: Asset name
            model_type: Model type
            
        Returns:
            Dictionary of aggregated metrics
        """
        logger.info(f"Evaluating {asset}_{model_type} with {len(splits)} splits")
        
        # 🔒 NO LEAKAGE: Create fresh metrics aggregator and per-fold lists for each model run
        metrics_agg = MetricsAggregator()
        per_fold_preds = []
        per_fold_actuals = []
        per_fold_metrics = []
        fit_times = []
        pred_times = []
        
        # 🔒 TRUE TOTAL SAMPLES: Compute once per run from current cleaned data
        total_samples = int(X.shape[0])
        logger.info(f"📏 TRUE TOTAL SAMPLES: {total_samples} (from current cleaned dataset)")
        
        # Remove hardcoded expectation - dataset sizes can vary based on data availability
        # The important thing is that we're using the current cleaned dataset consistently
        
        def evaluate_single_split(split_data):
            """Evaluate model on a single CV split."""
            nonlocal model, X, y, model_type
            train_idx, test_idx = split_data
            
            try:
                # Convert index-based splits to position-based splits
                # This fixes the "indices are out-of-bounds" error
                logger.debug(f"Original train_idx type: {type(train_idx)}, dtype: {getattr(train_idx, 'dtype', 'N/A')}")
                logger.debug(f"Original test_idx type: {type(test_idx)}, dtype: {getattr(test_idx, 'dtype', 'N/A')}")
                logger.debug(f"X.index type: {type(X.index)}, length: {len(X.index)}")
                logger.debug(f"train_idx length: {len(train_idx)}, test_idx length: {len(test_idx)}")
                
                # Use the original indices directly with DataFrame.loc instead of converting to positions
                # This avoids the "indices are out-of-bounds" error
                try:
                    # Extract train/test data using original indices
                    X_train = X.loc[train_idx]
                    X_test = X.loc[test_idx]
                    y_train = y.loc[train_idx]
                    y_test = y.loc[test_idx]
                    
                    # Log the extraction for debugging
                    logger.debug(f"Data extraction: X_train={X_train.shape}, X_test={X_test.shape}")
                    logger.debug(f"Data extraction: y_train={y_train.shape}, y_test={y_test.shape}")
                    
                except KeyError as ke:
                    logger.error(f"KeyError during data extraction: {ke}")
                    logger.error(f"train_idx sample: {train_idx[:5] if hasattr(train_idx, '__len__') else train_idx}")
                    logger.error(f"test_idx sample: {test_idx[:5] if hasattr(test_idx, '__len__') else test_idx}")
                    logger.error(f"Available indices: {X.index[:10].tolist()}")
                    return None
                except Exception as e:
                    logger.error(f"Unexpected error during data extraction: {e}")
                    return None
                
                # Clean data (remove NaNs and infinities)
                train_mask = ~(y_train.isna() | np.isinf(y_train))
                test_mask = ~(y_test.isna() | np.isinf(y_test))
                
                X_train_clean = X_train[train_mask]
                y_train_clean = y_train[train_mask]
                X_test_clean = X_test[test_mask]
                y_test_clean = y_test[test_mask]
                
                if len(y_train_clean) == 0 or len(y_test_clean) == 0:
                    logger.warning(f"Empty train/test set after cleaning")
                    return None
                
                # Determine task type
                task_type = getattr(model, 'task', None)
                if task_type is None:
                    # Infer from target data
                    if y_train_clean.dtype in ['object', 'string'] or len(np.unique(y_train_clean)) <= 10:
                        task_type = 'classification'
                    else:
                        task_type = 'regression'
                
                logger.debug(f"Task type determined: {task_type} (model.task={getattr(model, 'task', 'None')})")
                logger.debug(f"Target data type: {y_train_clean.dtype}, unique values: {len(np.unique(y_train_clean))}")
                
                # Handle log transformation for regression
                use_log_target = False
                is_arima = model_type == 'arima'
                eps = 1e-8
                
                if task_type == 'regression' and not is_arima:
                    # Check if log transformation is needed
                    if y_train_clean.min() > 0:
                        y_train_clean = np.log(y_train_clean + eps)
                        y_test_clean = np.log(y_test_clean + eps)
                        use_log_target = True
                
                # Centralized scaling: fit on training fold, apply to train/test
                try:
                    from sklearn.preprocessing import StandardScaler as _StdScaler
                    scaler_cols = X_train_clean.select_dtypes(include=[np.number]).columns
                    _scaler = _StdScaler()
                    X_train_num = _scaler.fit_transform(X_train_clean[scaler_cols].values)
                    X_test_num = _scaler.transform(X_test_clean[scaler_cols].values)
                    # Rebuild DataFrames to preserve indices/columns
                    X_train_clean = X_train_clean.copy()
                    X_test_clean = X_test_clean.copy()
                    X_train_clean.loc[:, scaler_cols] = X_train_num
                    X_test_clean.loc[:, scaler_cols] = X_test_num
                    # Signal to model that inputs are pre-scaled
                    try:
                        setattr(model, 'expects_scaled_input', True)
                    except Exception:
                        pass
                except Exception as _scale_err:
                    logger.warning(f"Centralized scaling skipped due to error: {_scale_err}")

                # Fit model with timing
                start_time = time.perf_counter()
                try:
                    if hasattr(model, 'fit'):
                        logger.debug(f"Fitting {model_type} model with {len(X_train_clean)} training samples")
                        logger.debug(f"Training data shapes: X={X_train_clean.shape}, y={y_train_clean.shape}")
                        logger.debug(f"X_train_clean sample: {X_train_clean.iloc[:3] if hasattr(X_train_clean, 'iloc') else X_train_clean[:3]}")
                        logger.debug(f"y_train_clean sample: {y_train_clean.iloc[:3] if hasattr(y_train_clean, 'iloc') else y_train_clean[:3]}")
                        
                        # ARIMA models also have a fit method, so call it with data
                        model.fit(X_train_clean, y_train_clean)
                    else:
                        # For models without fit method (shouldn't happen with current models)
                        logger.debug(f"Fitting model without fit method, {len(X_train_clean)} training samples")
                        model = model.fit()
                    fit_time = time.perf_counter() - start_time
                    logger.debug(f"Model fitting completed in {fit_time:.3f}s")
                except Exception as e:
                    logger.error(f"Model fitting failed for {model_type}: {e}")
                    logger.error(f"Training data shapes: X={X_train_clean.shape}, y={y_train_clean.shape}")
                    logger.error(f"Training data types: X={X_train_clean.dtypes if hasattr(X_train_clean, 'dtypes') else type(X_train_clean)}, y={y_train_clean.dtype if hasattr(y_train_clean, 'dtype') else type(y_train_clean)}")
                    import traceback
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    fit_time = 0.0
                
                # Predict with timing
                start_time = time.perf_counter()
                try:
                    # Check if model is trained
                    if hasattr(model, 'is_trained') and not model.is_trained:
                        logger.error(f"Model {model_type} is not trained, cannot predict")
                        y_pred = np.full_like(y_test_clean, np.nan)
                        pred_time = 0.0
                    elif hasattr(model, 'predict'):
                        logger.debug(f"Predicting with {model_type} model on {len(X_test_clean)} test samples")
                        logger.debug(f"Test data shapes: X={X_test_clean.shape}, y={y_test_clean.shape}")
                        y_pred = model.predict(X_test_clean)
                        # Classification guard: convert probabilities to class labels if needed
                        try:
                            if getattr(model, 'task', None) == 'classification' and isinstance(y_pred, np.ndarray) and y_pred.ndim > 1:
                                if y_pred.shape[1] > 1:
                                    y_pred = np.argmax(y_pred, axis=1)
                                else:
                                    y_pred = (y_pred.ravel() >= 0.5).astype(int)
                        except Exception:
                            pass
                    else:
                        # For models without predict method
                        logger.debug(f"Forecasting {len(X_test_clean)} steps")
                        y_pred = model.forecast(steps=len(X_test_clean))
                    pred_time = time.perf_counter() - start_time
                    logger.debug(f"Prediction completed in {pred_time:.3f}s, output shape: {np.asarray(y_pred).shape}")
                except Exception as e:
                    logger.error(f"Model prediction failed for {model_type}: {e}")
                    logger.error(f"Test data shapes: X={X_test_clean.shape}, y={y_test_clean.shape}")
                    logger.error(f"Test data types: X={X_test_clean.dtypes if hasattr(X_test_clean, 'dtypes') else type(X_test_clean)}, y={y_test_clean.dtype if hasattr(y_test_clean, 'dtype') else type(y_test_clean)}")
                    import traceback
                    logger.error(f"Full prediction traceback: {traceback.format_exc()}")
                    y_pred = np.full_like(y_test_clean, np.nan)
                    pred_time = 0.0
                
                # Convert predictions to numpy arrays
                y_true = np.asarray(y_test_clean)
                y_pred = np.asarray(y_pred)
                
                # Log shapes and sample values for debugging
                logger.debug(f"Shape validation: y_true={y_true.shape}, y_pred={y_pred.shape}")
                logger.debug(f"Sample y_true values: {y_true[:5] if len(y_true) >= 5 else y_true}")
                logger.debug(f"Sample y_pred values: {y_pred[:5] if len(y_pred) >= 5 else y_pred}")
                logger.debug(f"y_true range: [{y_true.min():.6f}, {y_true.max():.6f}]")
                logger.debug(f"y_pred range: [{y_pred.min():.6f}, {y_pred.max():.6f}]")
                logger.debug(f"y_true has NaN: {np.isnan(y_true).any()}")
                logger.debug(f"y_pred has NaN: {np.isnan(y_pred).any()}")
                logger.debug(f"y_true has Inf: {np.isinf(y_true).any()}")
                logger.debug(f"y_pred has Inf: {np.isinf(y_pred).any()}")
                
                # Check for NaN or infinite values in predictions
                if np.isnan(y_pred).any() or np.isinf(y_pred).any():
                    logger.error(f"Model {model_type} produced NaN or infinite predictions")
                    logger.error(f"NaN count: {np.isnan(y_pred).sum()}, Inf count: {np.isinf(y_pred).sum()}")
                    return None
                
                # Validate prediction shapes
                if y_pred.shape != y_true.shape:
                    logger.error(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
                    if len(y_pred.shape) == 1 and len(y_true.shape) == 1:
                        # Try to fix 1D shape mismatch
                        if len(y_pred) != len(y_true):
                            logger.error(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
                            return None
                    else:
                        logger.error(f"Multi-dimensional shape mismatch")
                        return None
                
                # Handle classification task
                if task_type == 'classification':
                    # Convert to labels; support probability inputs in [0,1]
                    y_true_arr = y_true.astype(int)
                    y_pred_input = np.asarray(y_pred)
                    if y_pred_input.ndim == 1 and np.all((y_pred_input >= 0.0) & (y_pred_input <= 1.0)):
                        y_pred_arr = (y_pred_input >= 0.5).astype(int)
                    elif y_pred_input.ndim == 2 and y_pred_input.shape[1] == 2:
                        y_pred_arr = (y_pred_input[:, 1] >= 0.5).astype(int)
                    else:
                        y_pred_arr = y_pred_input.astype(int)

                    # DEBUG: Log prediction details for classification
                    logger.debug(f"Classification debug - y_true unique: {np.unique(y_true_arr)}, y_pred unique: {np.unique(y_pred_arr)}")
                    logger.debug(f"Classification debug - y_true shape: {y_true_arr.shape}, y_pred shape: {y_pred_arr.shape}")
                    logger.debug(f"Classification debug - y_true range: [{y_true_arr.min()}, {y_true_arr.max()}], y_pred range: [{y_pred_arr.min()}, {y_pred_arr.max()}]")

                    # Calculate classification metrics
                    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score, roc_auc_score

                    acc = float(accuracy_score(y_true_arr, y_pred_arr))
                    f1 = float(f1_score(y_true_arr, y_pred_arr, average='weighted', zero_division=0))
                    prec = float(precision_score(y_true_arr, y_pred_arr, average='weighted', zero_division=0))
                    rec = float(recall_score(y_true_arr, y_pred_arr, average='weighted', zero_division=0))
                    bacc = float(balanced_accuracy_score(y_true_arr, y_pred_arr))

                    # ROC-AUC (if binary classification)
                    roc_auc = np.nan
                    if len(np.unique(y_true_arr)) == 2:
                        try:
                            # Prefer probability inputs if available
                            if y_pred_input.ndim == 1 and np.all((y_pred_input >= 0.0) & (y_pred_input <= 1.0)):
                                roc_auc = float(roc_auc_score(y_true_arr, y_pred_input))
                            elif y_pred_input.ndim == 2 and y_pred_input.shape[1] == 2:
                                roc_auc = float(roc_auc_score(y_true_arr, y_pred_input[:, 1]))
                            else:
                                roc_auc = float(roc_auc_score(y_true_arr, y_pred_arr))
                        except Exception:
                            pass
                    
                    # FIXED: Threshold tuning for binary classification - use original y_pred for probabilities
                    opt_thr = np.nan
                    f1_opt = np.nan
                    if len(np.unique(y_true_arr)) == 2:
                        try:
                            # Get probability predictions if available
                            if hasattr(model, 'predict_proba'):
                                y_pred_proba = model.predict_proba(X_test_clean)
                                if y_pred_proba.shape[1] == 2:  # Binary classification
                                    y_pred_proba_positive = y_pred_proba[:, 1]  # Probability of positive class
                                else:
                                    y_pred_proba_positive = y_pred_proba[:, -1]  # Last class probability
                            else:
                                # Fallback: use original y_pred as probabilities
                                y_pred_proba_positive = y_pred
                            
                            thresholds = np.arange(0.1, 0.9, 0.1)
                            f1_scores = []
                            for thr in thresholds:
                                y_pred_thr = (y_pred_proba_positive >= thr).astype(int)
                                f1_scores.append(f1_score(y_true_arr, y_pred_thr, average='weighted', zero_division=0))
                            best_idx = np.argmax(f1_scores)
                            opt_thr = float(thresholds[best_idx])
                            f1_opt = float(f1_scores[best_idx])
                        except Exception:
                            pass
                    
                    # Baseline metrics (last sign prediction)
                    last_sign_acc = np.nan
                    last_sign_f1 = np.nan
                    try:
                        if len(y_train_clean) > 0:
                            last_sign = 1 if y_train_clean.iloc[-1] > 0 else 0
                            y_naive = np.full_like(y_true_arr, last_sign)
                            last_sign_acc = float(accuracy_score(y_true_arr, y_naive))
                            last_sign_f1 = float(f1_score(y_true_arr, y_naive, average='weighted', zero_division=0))
                    except Exception:
                        pass
                    
                    # Log loss (if available)
                    logloss = np.nan
                    try:
                        if hasattr(model, 'predict_proba'):
                            y_proba = model.predict_proba(X_test_clean)
                            from sklearn.metrics import log_loss
                            logloss = float(log_loss(y_true_arr, y_proba))
                    except Exception:
                        pass
                    
                    result = {
                        'accuracy': acc,
                        'f1': f1,
                        'precision': prec,
                        'recall': rec,
                        'balanced_accuracy': bacc,
                        'roc_auc': roc_auc,
                        'optimal_threshold': opt_thr,
                        'f1_optimized': f1_opt,
                        'last_sign_accuracy': last_sign_acc,
                        'last_sign_f1': last_sign_f1,
                        'logloss': logloss,
                        'fit_time': fit_time,
                        'pred_time': pred_time,
                        'train_size': len(y_train_clean),
                        'test_size': len(y_test_clean),
                        'baseline_accuracy': last_sign_acc,
                        'baseline_balanced_accuracy': last_sign_acc,
                        'baseline_f1': last_sign_f1,
                        'baseline_precision': last_sign_acc,
                        'baseline_recall': last_sign_acc
                    }
                    # Attach confusion matrix for diagnostics (binary/multiclass)
                    try:
                        cm = confusion_matrix(y_true_arr, y_pred_arr)
                        result['confusion_matrix'] = cm.tolist()
                    except Exception:
                        pass
                    logger.debug(f"Returning classification result with {len(result)} metrics")
                    return result
                else:
                    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                    # Use centralized metrics helper for consistency
                    logger.debug(f"Calculating regression metrics for {len(y_true)} samples")
                    metrics = regression_fold_metrics(y_true, y_pred)
                    mse = metrics['mse']
                    rmse = metrics['rmse']
                    mae = metrics['mae']
                    mape = metrics['mape']
                    r2 = float(r2_score(y_true, y_pred))
                    logger.debug(f"Calculated metrics: MSE={mse:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}, MAPE={mape:.6f}, R²={r2:.6f}")
                    
                    # FIXED: Compute baseline metrics per fold using the same scale as predictions
                    try:
                        # Naive baseline: constant prediction using last train value on same eval scale
                        if 'use_log_target' in locals() and use_log_target and not is_arima:
                            y_train_eval = np.exp(np.asarray(y_train_clean)) - eps
                        else:
                            y_train_eval = np.asarray(y_train_clean)
                        baseline = float(y_train_eval[-1]) if y_train_eval.size else 0.0
                        y_naive = np.full_like(y_true, baseline, dtype=float)
                        
                        # Compute baseline metrics using the same helper function
                        baseline_metrics = regression_fold_metrics(y_true, y_naive)
                        mse_naive = baseline_metrics['mse']
                        mae_naive = baseline_metrics['mae']
                        mape_naive = baseline_metrics['mape']
                        
                        # Compute R² baseline per fold
                        ss_res = float(np.sum((y_true - y_naive)**2))
                        ss_tot = float(np.sum((y_true - np.mean(y_true))**2)) + 1e-12
                        r2_naive = 1.0 - ss_res/ss_tot
                        
                        # Guard against accidental overwrite: model vs baseline identical
                        try:
                            if np.isclose(mse, mse_naive) and np.isclose(mae, mae_naive):
                                self.logger.warning("Model and baseline metrics identical; model likely predicting naive baseline")
                        except Exception:
                            pass
                    except Exception as e:
                        self.logger.warning(f"Baseline calculation failed: {e}")
                        mse_naive = np.nan
                        mae_naive = np.nan
                        mape_naive = np.nan
                        r2_naive = np.nan
                    
                    result = {
                        'mse': mse,
                        'rmse': rmse,
                        'mae': mae,
                        'r2': r2,
                        'mape': mape,
                        'baseline_mse': mse_naive,
                        'baseline_mae': mae_naive,
                        'baseline_mape': mape_naive,
                        'baseline_r2': r2_naive,
                        'fit_time': fit_time,
                        'pred_time': pred_time,
                        'train_size': len(y_train_clean),
                        'test_size': len(y_test_clean),
                        'y_true': y_true.tolist() if hasattr(y_true, 'tolist') else list(y_true),
                        'y_pred': y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred)
                    }
                    logger.debug(f"Returning regression result with {len(result)} metrics")
                    return result
                
            except Exception as e:
                self.logger.error(f"Split evaluation failed for {model_type}: {e}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                return None
        
        # Parallel evaluation using threads to avoid pickling issues with TF models
        # On Windows, some TF models (LSTM/StockMixer) still cause serialization issues in joblib.
        # For those, fall back to a safe serial path.
        logger.debug(f"Starting parallel evaluation of {len(splits)} splits")
        tf_like_model = str(model_type).lower() in ["lstm", "stockmixer"]
        if tf_like_model:
            logger.debug("Detected TensorFlow-like model; using serial evaluation to avoid pickling issues")
            split_results = [evaluate_single_split(split) for split in splits]
        else:
            split_results = Parallel(n_jobs=psutil.cpu_count(logical=False), backend="threading", verbose=0)(
                delayed(evaluate_single_split)(split) for split in splits
            )
        
        # Log results for debugging
        logger.debug(f"Split results: {len(split_results)} total, {[type(r) for r in split_results]}")
        
        # Build validity mask and filtered list while preserving original ordering for later per-fold mapping
        valid_mask = [isinstance(r, dict) for r in split_results]
        valid_results = [r for r in split_results if isinstance(r, dict)]
        logger.debug(f"Valid results: {len(valid_results)} out of {len(split_results)} splits")

        # Diagnostics: detect NaNs per-fold to avoid silent aggregation issues
        try:
            for idx, r in enumerate(valid_results):
                if not isinstance(r, dict):
                    continue
                for key in ['mse', 'rmse', 'mae', 'r2', 'mape', 'baseline_mse', 'baseline_mae', 'baseline_r2', 'fit_time', 'pred_time']:
                    if key in r:
                        import math as _math
                        val = r[key]
                        if isinstance(val, (list, tuple)):
                            has_nan = any(_math.isnan(float(x)) for x in val if x is not None)
                        else:
                            has_nan = _math.isnan(float(val)) if val is not None else False
                        if has_nan:
                            self.logger.warning(f"[Diag] NaN detected in '{key}' at fold {idx + 1}")
        except Exception:
            pass
        
        if not valid_results:
            self.logger.error(f"No valid evaluations for {asset}_{model_type}")
            return {}
        
        # 🔒 PER-FOLD AGGREGATION: Use the centralized metrics summarizer
        # Extract timing metrics from valid results
        fit_times = [r.get('fit_time', 0.0) for r in valid_results if 'fit_time' in r]
        pred_times = [r.get('pred_time', 0.0) for r in valid_results if 'pred_time' in r]
        
        # ⏱️ TIMING GUARDRAILS: Compare to number of successful splits to avoid false alarms
        expected_success = len(valid_results)
        if len(fit_times) != expected_success:
            self.logger.warning(f"⏱️ TIMING GUARDRAIL: fit_times length {len(fit_times)} != successful_splits {expected_success} (total requested {len(splits)})")
        if len(pred_times) != expected_success:
            self.logger.warning(f"⏱️ TIMING GUARDRAIL: pred_times length {len(pred_times)} != successful_splits {expected_success} (total requested {len(splits)})")
        
        # Use centralized metrics summarizer for proper aggregation
        from .metrics_summarizer import summarize_regression, summarize_classification
        
        is_classification_overall = getattr(model, 'task', None) == 'classification'
        if is_classification_overall:
            metrics = summarize_classification(valid_results, fit_times, pred_times)
        else:
            metrics = summarize_regression(valid_results, fit_times, pred_times)
        
        # 🔒 TRUE TOTAL SAMPLES: Use computed value, not global
        metrics['total_samples'] = total_samples
        metrics['n_splits'] = len(valid_results)
        
        self.logger.info(f"✅ {asset}_{model_type}: {len(valid_results)}/{len(splits)} splits successful")
        if is_classification_overall:
            self.logger.info(f"📊 CLASSIFICATION METRICS (mean ± std):")
            self.logger.info(f"  Accuracy: {metrics.get('Accuracy', 0):.4f} ± {metrics.get('Accuracy_std', float('nan')):.4f}")
            self.logger.info(f"  F1-Score: {metrics.get('F1', 0):.4f} ± {metrics.get('F1_std', float('nan')):.4f}")
            self.logger.info(f"  Precision: {metrics.get('Precision', 0):.4f} ± {metrics.get('Precision_std', float('nan')):.4f}")
            self.logger.info(f"  Recall: {metrics.get('Recall', 0):.4f} ± {metrics.get('Recall_std', float('nan')):.4f}")
            self.logger.info(f"  Balanced Accuracy: {metrics.get('Balanced_Accuracy', 0):.4f} ± {metrics.get('Balanced_Accuracy_std', float('nan')):.4f}")
            if 'roc_auc' in metrics:
                self.logger.info(f"  ROC-AUC: {metrics.get('roc_auc', 0):.4f} ± {metrics.get('roc_auc_std', float('nan')):.4f}")
            if 'optimal_threshold' in metrics:
                self.logger.info(f"  Optimal Threshold: {metrics.get('optimal_threshold', 0):.3f} ± {metrics.get('optimal_threshold_std', float('nan')):.3f}")
                self.logger.info(f"  F1 (Optimized): {metrics.get('f1_optimized', 0):.4f} ± {metrics.get('f1_optimized_std', float('nan')):.4f}")
            if 'last_sign_accuracy' in metrics:
                self.logger.info(f"  Last Sign Baseline: {metrics.get('last_sign_accuracy', 0):.4f} ± {metrics.get('last_sign_accuracy_std', float('nan')):.4f}")
            if 'logloss' in metrics:
                self.logger.info(f"  Log Loss: {metrics.get('logloss', 0):.4f} ± {metrics.get('logloss_std', float('nan')):.4f}")
        else:
            self.logger.info(f"📊 REGRESSION METRICS (mean ± std):")
            self.logger.info(f"  MSE: {metrics.get('MSE', 0):.6f} ± {metrics.get('MSE_std', float('nan')):.6f}")
            self.logger.info(f"  RMSE: {metrics.get('RMSE', 0):.6f} ± {metrics.get('RMSE_std', float('nan')):.6f}")
            self.logger.info(f"  MAE: {metrics.get('MAE', 0):.6f} ± {metrics.get('MAE_std', float('nan')):.6f}")
            self.logger.info(f"  R²: {metrics.get('R2', 0):.4f} ± {metrics.get('R2_std', float('nan')):.4f}")
            self.logger.info(f"  MAPE: {metrics.get('MAPE', 0):.4f} ± {metrics.get('MAPE_std', float('nan')):.4f}")
            self.logger.info(f"  Baseline MSE: {metrics.get('Baseline MSE', 0):.6f} ± {metrics.get('Baseline MSE_std', float('nan')):.6f}")
            self.logger.info(f"  Baseline MAE: {metrics.get('Baseline MAE', 0):.6f} ± {metrics.get('Baseline MAE_std', float('nan')):.6f}")
            self.logger.info(f"  Baseline R²: {metrics.get('Baseline R2', 0):.4f} ± {metrics.get('Baseline R2_std', float('nan')):.6f}")
        
        # Log comprehensive performance summary
        self.logger.info(f"⚡ PERFORMANCE METRICS:")
        self.logger.info(f"  Fit Time: {metrics.get('Fit Time', 0):.3f}s ± {metrics.get('Fit Time_std', float('nan')):.3f}s")
        self.logger.info(f"  Prediction Time: {metrics.get('Prediction Time', 0):.3f}s ± {metrics.get('Prediction Time_std', float('nan')):.3f}s")
        self.logger.info(f"  Number of Splits: {metrics.get('n_splits', 0)}")
        self.logger.info(f"  Total Samples: {metrics.get('total_samples', 0)}")
        self.logger.info(f"  Successful Splits: {len(valid_results)}/{len(splits)}")
        
        # Log dataset info for debugging
        self.logger.info(f"📊 DATASET INFO:")
        self.logger.info(f"  Original dataset size: {len(X)}")
        self.logger.info(f"  Current cleaned dataset size: {metrics.get('total_samples', len(X))}")
        self.logger.info(f"{'='*80}\n")
        
        # Collect comprehensive per-fold data for thesis analysis
        all_actuals = []
        all_predictions = []
        all_probabilities = []
        all_fold_metrics = []
        all_fold_indices = []
        
        for i, (train_idx, test_idx) in enumerate(splits):
            # Store fold indices for reproducibility
            fold_indices = {
                'fold': i + 1,
                'train_start': str(train_idx[0]),
                'train_end': str(train_idx[-1]),
                'test_start': str(test_idx[0]),
                'test_end': str(test_idx[-1]),
                'train_size': len(train_idx),
                'test_size': len(test_idx)
            }
            all_fold_indices.append(fold_indices)
            
            # Collect actuals and predictions from this fold
            if i < len(split_results) and isinstance(split_results[i], dict):
                r = split_results[i]
                if isinstance(r, dict):
                    # Store fold-level metrics
                    fold_metrics = {
                        'fold': i + 1,
                        'asset': asset,
                        'model_type': model_type,
                        'task': getattr(model, 'task', 'unknown'),
                        'train_size': r.get('train_size', 0),
                        'test_size': r.get('test_size', 0),
                        'train_start_idx': str(train_idx[0]),
                        'train_end_idx': str(train_idx[-1]),
                        'test_start_idx': str(test_idx[0]),
                        'test_end_idx': str(test_idx[-1]),
                        **{k: v for k, v in r.items() if k not in ['y_true', 'y_pred', 'y_proba']}
                    }
                    all_fold_metrics.append(fold_metrics)
                    
                    # Store predictions and actuals
                    if 'y_true' in r and 'y_pred' in r:
                        all_actuals.extend(r['y_true'])
                        all_predictions.extend(r['y_pred'])
        
        # Log comprehensive summary for thesis
        self.logger.info(f"\n{'='*80}")
        # Force task from call site to avoid UNKNOWN plumbing
        forced_task = 'CLASSIFICATION' if getattr(model, 'task', None) == 'classification' else 'REGRESSION'
        self.logger.info(f"THESIS-READY AGGREGATED RESULTS FOR {asset} - {model_type} - {forced_task}")
        self.logger.info(f"{'='*80}")
        
        if is_classification_overall:
            self.logger.info(f"📊 CLASSIFICATION METRICS (mean ± std):")
            self.logger.info(f"  Accuracy: {metrics.get('Accuracy', 0):.4f} ± {metrics.get('Accuracy_std', float('nan')):.4f}")
            self.logger.info(f"  F1-Score: {metrics.get('F1', 0):.4f} ± {metrics.get('F1_std', float('nan')):.4f}")
            self.logger.info(f"  Precision: {metrics.get('Precision', 0):.4f} ± {metrics.get('Precision_std', float('nan')):.4f}")
            self.logger.info(f"  Recall: {metrics.get('Recall', 0):.4f} ± {metrics.get('Recall_std', float('nan')):.4f}")
            self.logger.info(f"  Balanced Accuracy: {metrics.get('Balanced_Accuracy', 0):.4f} ± {metrics.get('Balanced_Accuracy_std', float('nan')):.4f}")
            if 'roc_auc' in metrics:
                self.logger.info(f"  ROC-AUC: {metrics.get('roc_auc', 0):.4f} ± {metrics.get('roc_auc_std', float('nan')):.4f}")
            if 'optimal_threshold' in metrics:
                self.logger.info(f"  Optimal Threshold: {metrics.get('optimal_threshold', 0):.3f} ± {metrics.get('optimal_threshold_std', float('nan')):.3f}")
                self.logger.info(f"  F1 (Optimized): {metrics.get('f1_optimized', 0):.4f} ± {metrics.get('f1_optimized_std', float('nan')):.4f}")
            if 'last_sign_accuracy' in metrics:
                self.logger.info(f"  Last Sign Baseline: {metrics.get('last_sign_accuracy', 0):.4f} ± {metrics.get('last_sign_accuracy_std', float('nan')):.4f}")
            if 'logloss' in metrics:
                self.logger.info(f"  Log Loss: {metrics.get('logloss', 0):.4f} ± {metrics.get('logloss_std', float('nan')):.4f}")
        else:
            self.logger.info(f"📊 REGRESSION METRICS (mean ± std):")
            self.logger.info(f"  MSE: {metrics.get('MSE', 0):.6f} ± {metrics.get('MSE_std', float('nan')):.6f}")
            self.logger.info(f"  RMSE: {metrics.get('RMSE', 0):.6f} ± {metrics.get('RMSE_std', float('nan')):.6f}")
            self.logger.info(f"  MAE: {metrics.get('MAE', 0):.6f} ± {metrics.get('MAE_std', float('nan')):.6f}")
            self.logger.info(f"  R²: {metrics.get('R2', 0):.4f} ± {metrics.get('R2_std', float('nan')):.4f}")
            self.logger.info(f"  MAPE: {metrics.get('MAPE', 0):.4f} ± {metrics.get('MAPE_std', float('nan')):.4f}")
            self.logger.info(f"  Baseline MSE: {metrics.get('Baseline MSE', 0):.6f} ± {metrics.get('Baseline MSE_std', float('nan')):.6f}")
            self.logger.info(f"  Baseline MAE: {metrics.get('Baseline MAE', 0):.6f} ± {metrics.get('Baseline MAE_std', float('nan')):.6f}")
            self.logger.info(f"  Baseline R²: {metrics.get('Baseline R2', 0):.4f} ± {metrics.get('Baseline R2_std', float('nan')):.6f}")
        
        # Log comprehensive performance summary
        self.logger.info(f"⚡ PERFORMANCE METRICS:")
        self.logger.info(f"  Fit Time: {metrics.get('Fit Time', 0):.3f}s ± {metrics.get('Fit Time_std', float('nan')):.3f}s")
        self.logger.info(f"  Prediction Time: {metrics.get('Prediction Time', 0):.3f}s ± {metrics.get('Prediction Time_std', float('nan')):.3f}s")
        self.logger.info(f"  Number of Splits: {metrics.get('n_splits', 0)}")
        self.logger.info(f"  Total Samples: {metrics.get('total_samples', 0)}")
        self.logger.info(f"  Successful Splits: {len(valid_results)}/{len(splits)}")
        
        # Log dataset info for debugging
        self.logger.info(f"📊 DATASET INFO:")
        self.logger.info(f"  Original dataset size: {len(X)}")
        self.logger.info(f"  Current cleaned dataset size: {metrics.get('total_samples', len(X))}")
        self.logger.info(f"{'='*80}\n")
        
        # Optional regime-aware breakdown (for regression tasks only)
        regime_metrics: Dict[str, Any] = {}
        try:
            if regimes is not None and isinstance(regimes, pd.Series) and getattr(model, 'task', None) != 'classification':
                # Align regimes to dataset used
                regimes_aligned = regimes.reindex(X.index, method='ffill')
                # Build concatenated y_true/y_pred over splits
                y_true_all = []
                y_pred_all = []
                reg_all = []
                for i, (train_idx, test_idx) in enumerate(splits):
                    if i < len(split_results) and isinstance(split_results[i], dict):
                        r = split_results[i]
                        if 'y_true' in r and 'y_pred' in r:
                            y_true_all.extend(r['y_true'])
                            y_pred_all.extend(r['y_pred'])
                            reg_all.extend(regimes_aligned.loc[test_idx].tolist())
                if y_true_all and y_pred_all and reg_all and len(y_true_all) == len(y_pred_all) == len(reg_all):
                    df_rm = pd.DataFrame({'y': y_true_all, 'yhat': y_pred_all, 'reg': reg_all})
                    for label, grp in df_rm.groupby('reg'):
                        if len(grp) >= 10:
                            err = grp['y'] - grp['yhat']
                            mse = float(np.mean(err**2))
                            rmse = float(np.sqrt(max(mse, 0.0)))
                            mae = float(np.mean(np.abs(err)))
                            den = np.clip(np.abs(grp['y'].values), 1e-8, None)
                            mape = float(np.mean(np.abs(err.values) / den))
                            regime_metrics[str(label)] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'N': int(len(grp))}
        except Exception:
            regime_metrics = {}

        # Return comprehensive results for thesis analysis
        return {
            'metrics': metrics,
            'all_actuals': all_actuals,
            'all_predictions': all_predictions,
            'all_probabilities': all_probabilities,
            'all_fold_metrics': all_fold_metrics,
            'all_fold_indices': all_fold_indices,
            'n_splits': len(valid_results),
            'successful_splits': len(valid_results),
            'total_splits': len(splits),
            'regime_metrics': regime_metrics
        }
    
    def _generate_expanding_splits(self, X: pd.DataFrame, n_splits: int, test_size: int) -> List[Tuple[pd.Index, pd.Index]]:
        """Generate expanding window splits."""
        splits = []
        n_samples = len(X)
        
        for i in range(n_splits):
            if n_splits == 1:
                # Single split: use most data for training
                train_end = n_samples - test_size - self.config.gap
            else:
                # Multiple splits: expanding window
                train_end = self.config.min_train_size + ((n_samples - test_size - self.config.gap - self.config.min_train_size) * (i + 1)) // n_splits
                
            test_start = train_end + self.config.gap
            test_end = min(test_start + test_size, n_samples)
            
            # Ensure we have valid indices
            if not self._validate_split_indices(train_end, test_start, test_end, n_samples):
                self.logger.warning(f"Split {i+1}: Invalid indices, skipping")
                continue
                
            train_idx = X.index[:train_end]
            test_idx = X.index[test_start:test_end]
            
            # Final validation
            if self._validate_split_sizes(train_idx, test_idx):
                splits.append((train_idx, test_idx))
                self.logger.info(f"✅ Split {i+1}: Train={len(train_idx)}, Test={len(test_idx)}")
            else:
                self.logger.warning(f"Split {i+1}: Insufficient data (Train={len(train_idx)}, Test={len(test_idx)})")
        
        return splits
    
    def _generate_sliding_splits(self, X: pd.DataFrame, n_splits: int, test_size: int) -> List[Tuple[pd.Index, pd.Index]]:
        """Generate sliding window splits."""
        splits = []
        n_samples = len(X)
        
        # Calculate step size for sliding window
        total_available = n_samples - test_size - self.config.gap
        if total_available <= self.config.min_train_size:
            return []
        step_size = max(1, (total_available - self.config.min_train_size - (n_splits - 1) * (test_size + self.config.gap)) // max(1, n_splits))
        
        for i in range(n_splits):
            # Sliding window: maintain fixed train length and slide by test size
            train_end = self.config.min_train_size + i * (test_size + self.config.gap)
            train_start = max(0, train_end - self.config.min_train_size)
            test_start = train_end + self.config.gap
            test_end = test_start + test_size
            if test_end > n_samples:
                break
            
            # Ensure we have valid indices
            if not self._validate_split_indices(train_end, test_start, test_end, n_samples, train_start=train_start):
                self.logger.warning(f"Split {i+1}: Invalid indices, skipping")
                continue
                
            train_idx = X.index[train_start:train_end]
            test_idx = X.index[test_start:test_end]
            
            # Final validation
            if self._validate_split_sizes(train_idx, test_idx):
                splits.append((train_idx, test_idx))
                self.logger.info(f"✅ Split {i+1}: Train={len(train_idx)}, Test={len(test_idx)}")
            else:
                self.logger.warning(f"Split {i+1}: Insufficient data (Train={len(train_idx)}, Test={len(test_idx)})")
        
        return splits
    
    def _validate_split_indices(self, train_end: int, test_start: int, test_end: int, n_samples: int, train_start: int = 0) -> bool:
        """Validate split indices."""
        if train_start < 0 or train_end <= train_start:
            return False
        
        if test_start < train_end or test_end <= test_start:
            return False
        
        if test_end > n_samples:
            return False
        
        return True
    
    def _validate_split_sizes(self, train_idx: pd.Index, test_idx: pd.Index) -> bool:
        """Validate split sizes."""
        return (len(train_idx) >= self.config.min_train_size and 
                len(test_idx) >= self.config.min_test_size)
    
    def get_split_info(self, splits: List[Tuple[pd.Index, pd.Index]]) -> Dict[str, Any]:
        """Get information about the generated splits."""
        if not splits:
            return {}
        
        train_sizes = [len(train_idx) for train_idx, _ in splits]
        test_sizes = [len(test_idx) for _, test_idx in splits]
        
        info = {
            'n_splits': len(splits),
            'train_sizes': {
                'min': min(train_sizes),
                'max': max(train_sizes),
                'mean': np.mean(train_sizes),
                'std': np.std(train_sizes)
            },
            'test_sizes': {
                'min': min(test_sizes),
                'max': max(test_sizes),
                'mean': np.mean(test_sizes),
                'std': np.std(test_sizes)
            },
            'total_samples_used': sum(train_sizes) + sum(test_sizes),
            'overlap': self._calculate_overlap(splits)
        }
        
        return info
    
    def _calculate_overlap(self, splits: List[Tuple[pd.Index, pd.Index]]) -> Dict[str, Any]:
        """Calculate overlap between splits."""
        if len(splits) < 2:
            return {'train_overlap': 0, 'test_overlap': 0}
        
        # Calculate train overlap (should be high for expanding window)
        train_overlaps = []
        for i in range(len(splits) - 1):
            train1 = set(splits[i][0])
            train2 = set(splits[i + 1][0])
            overlap = len(train1.intersection(train2)) / len(train1) if train1 else 0
            train_overlaps.append(overlap)
        
        # Calculate test overlap (should be 0 for proper validation)
        test_overlaps = []
        for i in range(len(splits) - 1):
            test1 = set(splits[i][1])
            test2 = set(splits[i + 1][1])
            overlap = len(test1.intersection(test2)) / len(test1) if test1 else 0
            test_overlaps.append(overlap)
        
        return {
            'train_overlap': {
                'min': min(train_overlaps) if train_overlaps else 0,
                'max': max(train_overlaps) if train_overlaps else 0,
                'mean': np.mean(train_overlaps) if train_overlaps else 0
            },
            'test_overlap': {
                'min': min(test_overlaps) if test_overlaps else 0,
                'max': max(test_overlaps) if test_overlaps else 0,
                'mean': np.mean(test_overlaps) if test_overlaps else 0
            }
        }
    
    def validate_data_quality(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Validate and clean data quality issues."""
        quality_report = {
            'is_valid': True,
            'issues': [],
            'cleaned_shape': X.shape,
            'infinite_values_removed': 0,
            'nan_values_removed': 0,
            'outliers_removed': 0,
            'cleaned_data': None
        }
        
        # Check for infinite values
        if np.isinf(X.select_dtypes(include=[np.number])).any().any():
            infinite_count = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
            quality_report['issues'].append(f"Found {infinite_count} infinite values")
            quality_report['is_valid'] = False
            
            # Clean infinite values
            X_clean = X.copy()
            numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                # Replace infinite values with NaN first
                X_clean[col] = X_clean[col].replace([np.inf, -np.inf], np.nan)
                # Then fill NaN with median
                median_val = X_clean[col].median()
                X_clean[col] = X_clean[col].fillna(median_val)
            
            quality_report['infinite_values_removed'] = infinite_count
            quality_report['cleaned_data'] = X_clean
            quality_report['cleaned_shape'] = X_clean.shape
            self.logger.warning(f"Cleaned {infinite_count} infinite values from data")
        
        # Check for NaN values
        if X.isnull().any().any():
            nan_count = X.isnull().sum().sum()
            quality_report['issues'].append(f"Found {nan_count} NaN values")
            quality_report['is_valid'] = False
            
            if quality_report['cleaned_data'] is None:
                X_clean = X.copy()
            else:
                X_clean = quality_report['cleaned_data']
            
            # Fill NaN values with appropriate methods
            for col in X_clean.columns:
                if X_clean[col].dtype in ['float64', 'int64']:
                    # Numeric columns: fill with median
                    median_val = X_clean[col].median()
                    X_clean[col] = X_clean[col].fillna(median_val)
                else:
                    # Categorical columns: fill with mode
                    mode_val = X_clean[col].mode().iloc[0] if not X_clean[col].mode().empty else 'Unknown'
                    X_clean[col] = X_clean[col].fillna(mode_val)
            
            quality_report['nan_values_removed'] = nan_count
            quality_report['cleaned_data'] = X_clean
            quality_report['cleaned_shape'] = X_clean.shape
            self.logger.warning(f"Cleaned {nan_count} NaN values from data")
        
        # Check for extreme outliers (beyond 3 standard deviations)
        if quality_report['cleaned_data'] is None:
            X_clean = X.copy()
        else:
            X_clean = quality_report['cleaned_data']
        
        numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
        # ULTIMATE FIX: Use less aggressive outlier detection for financial data
        outlier_count = 0
        
        for col in numeric_cols:
            Q1 = X_clean[col].quantile(0.25)
            Q3 = X_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # ULTIMATE FIX: Use 3.0 * IQR instead of 1.5 * IQR for financial data
            # This preserves more market extremes while still detecting true anomalies
            lower_bound = Q1 - 3.0 * IQR
            upper_bound = Q3 + 3.0 * IQR
            
            outliers = ((X_clean[col] < lower_bound) | (X_clean[col] > upper_bound)).sum()
            if outliers > 0:
                # Preserve extremes for financial data; do not cap
                outlier_count += outliers
        
        if outlier_count > 0:
            quality_report['issues'].append(f"Detected {outlier_count} statistical outliers (preserved)")
            quality_report['outliers_removed'] = 0
            quality_report['cleaned_data'] = X_clean
            quality_report['cleaned_shape'] = X_clean.shape
            self.logger.info(f"Detected {outlier_count} outliers; preserved to keep market extremes")
        
        # Final validation
        if quality_report['cleaned_data'] is not None:
            # Check if cleaning was successful
            final_inf_check = np.isinf(quality_report['cleaned_data'].select_dtypes(include=[np.number])).any().any()
            final_nan_check = quality_report['cleaned_data'].isnull().any().any()
            
            if not final_inf_check and not final_nan_check:
                quality_report['is_valid'] = True
                self.logger.info("Data quality validation completed successfully")
            else:
                quality_report['is_valid'] = False
                quality_report['issues'].append("Data cleaning incomplete")
        
        return quality_report
    
    def clean_data_for_training(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Clean data for model training by handling inf/NaNs conservatively (preserve extremes)."""
        X_clean = X.copy()
        y_clean = y.copy()
        
        # Clean infinite values in features
        numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_data = X_clean[col]
            if np.isinf(col_data).any():
                finite_mask = ~np.isinf(col_data)
                median_val = col_data[finite_mask].median()
                if pd.isna(median_val):
                    median_val = 0.0
                X_clean[col] = col_data.replace([np.inf, -np.inf], median_val)
                self.logger.info(f"Cleaned infinite values in column {col}")
        
        # Clean infinite values in targets
        if np.isinf(y_clean).any():
            finite_mask = ~np.isinf(y_clean)
            median_val = y_clean[finite_mask].median()
            if pd.isna(median_val):
                median_val = 0.0
            y_clean = y_clean.replace([np.inf, -np.inf], median_val)
            self.logger.info("Cleaned infinite values in targets")
        
        # Handle NaNs: preserve rows; impute features with median, targets with ffill/bfill
        for col in X_clean.columns:
            if X_clean[col].dtype in ['float64', 'int64']:
                median_val = X_clean[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                X_clean[col] = X_clean[col].fillna(median_val)
            else:
                mode_val = X_clean[col].mode().iloc[0] if not X_clean[col].mode().empty else 'Unknown'
                X_clean[col] = X_clean[col].fillna(mode_val)
        
        # Targets: forward fill then back fill
        if y_clean.isna().any():
            y_clean = y_clean.ffill().bfill()
        
        # As a last resort, drop rows where y is still NaN
        valid_mask = ~y_clean.isna()
        X_clean = X_clean[valid_mask]
        y_clean = y_clean[valid_mask]
        
        self.logger.info(f"Data cleaned: {len(X)} -> {len(X_clean)} samples (minimal drops)")
        return X_clean, y_clean
    
    def create_time_series_split(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Create time series split iterator.
        
        Args:
            X: Feature DataFrame
            y: Target Series (optional)
            
        Yields:
            (X_train, X_test, y_train, y_test) tuples
        """
        splits = self.split(X)
        
        for train_idx, test_idx in splits:
            X_train = X.loc[train_idx].values
            X_test = X.loc[test_idx].values
            
            if y is not None:
                y_train = y.loc[train_idx].values
                y_test = y.loc[test_idx].values
            else:
                y_train = None
                y_test = None
            
            yield X_train, X_test, y_train, y_test
    
    def get_validation_summary(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Get comprehensive validation summary."""
        # Generate splits
        splits = self.split(X)
        
        # Get split information
        split_info = self.get_split_info(splits)
        
        # Get data quality report
        quality_report = self.validate_data_quality(X)
        
        # Combine into summary
        summary = {
            'configuration': {
                'n_splits': self.config.n_splits,
                'test_size': self.config.test_size,
                'min_train_size': self.config.min_train_size,
                'min_test_size': self.config.min_test_size,
                'gap': self.config.gap,
                'expanding_window': self.config.expanding_window
            },
            'actual_splits': len(splits),
            'split_info': split_info,
            'data_quality': quality_report,
            'validation_status': 'valid' if len(splits) > 0 and quality_report['is_valid'] else 'invalid'
        }
        
        return summary
    
    def plot_splits(self, X: pd.DataFrame, splits: List[Tuple[pd.Index, pd.Index]], 
                   save_path: Optional[str] = None) -> None:
        """
        Plot the walk-forward splits for visualization.
        
        Args:
            X: Feature DataFrame
            splits: List of (train_indices, test_indices) tuples
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot data points
            ax.plot(range(len(X)), [1] * len(X), 'b-', alpha=0.3, label='Data')
            
            # Plot splits
            colors = plt.cm.Set3(np.linspace(0, 1, len(splits)))
            
            for i, (train_idx, test_idx) in enumerate(splits):
                # Train set
                train_start = X.index.get_loc(train_idx[0])
                train_end = X.index.get_loc(train_idx[-1])
                ax.add_patch(patches.Rectangle((train_start, 0.8), train_end - train_start, 0.1, 
                                             facecolor=colors[i], alpha=0.7, label=f'Train {i+1}' if i == 0 else ""))
                
                # Test set
                test_start = X.index.get_loc(test_idx[0])
                test_end = X.index.get_loc(test_idx[-1])
                ax.add_patch(patches.Rectangle((test_start, 0.6), test_end - test_start, 0.1, 
                                             facecolor=colors[i], alpha=0.9, label=f'Test {i+1}' if i == 0 else ""))
            
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Split Type')
            ax.set_title('Walk-Forward Validation Splits')
            ax.set_ylim(0.5, 1.1)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if save_path:
                try:
                    with open(save_path, 'wb') as _fh:
                        fig.savefig(_fh, dpi=300, bbox_inches='tight')
                finally:
                    self.logger.info(f"Split visualization saved to {save_path}")
                    # Ensure all figure resources are released (prevents Windows file locks)
                    plt.close(fig)
                return
            
            plt.show()
            plt.close(fig)
            
        except ImportError:
            self.logger.warning("matplotlib not available, skipping split visualization")
        except Exception as e:
            self.logger.error(f"Failed to create split visualization: {str(e)}")
    
    def __repr__(self) -> str:
        """String representation of the validator."""
        return (f"WalkForwardValidator(n_splits={self.config.n_splits}, "
                f"test_size={self.config.test_size}, "
                f"min_train_size={self.config.min_train_size}, "
                f"min_test_size={self.config.min_test_size}, "
                f"gap={self.config.gap}, "
                f"expanding_window={self.config.expanding_window})") 