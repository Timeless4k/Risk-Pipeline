"""
GARCH model implementation for RiskPipeline.
"""

import logging
from typing import Dict, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from .base_model import BaseModel


class GARCHModel(BaseModel):
    """GARCH model for volatility forecasting and derived classification."""

    def __init__(self, p: int = 1, q: int = 1, task: str = 'regression', threshold: Optional[float] = None, **kwargs):
        """
        Initialize GARCH model.

        Args:
            p: GARCH lag order
            q: ARCH lag order
            **kwargs: Additional parameters
        """
        super().__init__(name="GARCH", **kwargs)
        self.p = p
        self.q = q
        self.task = task or 'regression'
        self.fitted_model = None
        self.returns_data = None
        self.last_returns = None
        self.classification_threshold = threshold  # used if task == 'classification'

        # Exogenous feature support (mean model ARX)
        # Default to False to avoid exogenous forecasting shape issues unless explicitly enabled
        self.use_exog_mean = bool(kwargs.get('use_exog_mean', False))
        self.max_exog_features = int(kwargs.get('max_exog_features', 64))
        self.exog_scaler: Optional[StandardScaler] = StandardScaler()
        self.exog_feature_names: Optional[list] = None

        self.logger.info(f"GARCH({p}, {q}) model initialized")

    def build_model(self, input_shape: Tuple[int, ...]) -> 'GARCHModel':
        """Build GARCH model (no explicit building needed)."""
        self.input_shape = input_shape
        self.model = "GARCH_READY"
        self.logger.info(f"GARCH model ready")
        return self

    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray], **kwargs) -> 'GARCHModel':
        """Fit GARCH model. For classification, we still fit volatility model."""
        self.train(X, y, **kwargs)
        return self

    def train(self, X: Union[pd.DataFrame, np.ndarray], 
              y: Union[pd.Series, np.ndarray], **kwargs) -> Dict[str, Any]:
        """
        Train GARCH model.

        For GARCH, we need returns. If y looks like volatility, synthesize
        returns matching that volatility pattern as a proxy.
        """
        self.logger.info(f"Training GARCH model with {len(y)} observations")

        try:
            # Convert inputs
            if isinstance(y, pd.Series):
                y_arr = y.values
            else:
                y_arr = np.asarray(y).ravel()

            # Determine if y appears to be volatility
            if np.all(np.isfinite(y_arr)) and np.all(y_arr >= 0) and np.nanmean(y_arr) < 1:
                rng = np.random.default_rng(42)
                synthetic_returns = rng.normal(0.0, y_arr)
                returns_data = synthetic_returns
                self.logger.info("Using synthetic returns derived from volatility targets")
            else:
                returns_data = y_arr
                self.logger.info("Using provided data as returns")

            self.returns_data = returns_data
            self.last_returns = returns_data[-10:]

            # Scale to percentage for stability
            returns_scaled = np.asarray(returns_data, dtype=float) * 100.0
            # Prepare exogenous features for mean model (ARX)
            exog_train = None
            if X is not None and self.use_exog_mean:
                try:
                    if isinstance(X, np.ndarray):
                        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
                    else:
                        X_df = X.copy()
                    # Align lengths defensively
                    n = min(len(X_df), len(returns_scaled))
                    if n <= 10:
                        self.logger.warning("Insufficient aligned samples for exogenous features; skipping exog")
                        exog_train = None
                    else:
                        X_df = X_df.iloc[-n:]
                        ret_vec = returns_scaled[-n:]
                        # Basic feature cap to avoid ill-posed ARX when features are huge
                        if X_df.shape[1] > self.max_exog_features:
                            X_df = X_df.iloc[:, :self.max_exog_features]
                        self.exog_feature_names = X_df.columns.tolist()
                        # Fit scaler on training exog unless centralized scaling is enabled
                        if bool(getattr(self, 'expects_scaled_input', False)):
                            exog_train = X_df.values
                        else:
                            exog_train = self.exog_scaler.fit_transform(X_df.values)
                        returns_scaled = ret_vec  # ensure same horizon
                except Exception as e:
                    self.logger.warning(f"Failed to prepare exogenous features for GARCH mean model: {e}")
                    exog_train = None

            if exog_train is not None:
                try:
                    model = arch_model(returns_scaled, x=exog_train, mean='ARX', lags=0, vol='Garch', p=self.p, q=self.q)
                except Exception as e:
                    self.logger.warning(f"Falling back to no exogenous mean model due to: {e}")
                    model = arch_model(returns_scaled, vol='Garch', p=self.p, q=self.q)
            else:
                model = arch_model(returns_scaled, vol='Garch', p=self.p, q=self.q)

            self.fitted_model = model.fit(disp='off')
            self.is_trained = True

            aic = getattr(self.fitted_model, 'aic', None)
            bic = getattr(self.fitted_model, 'bic', None)

            self.logger.info("GARCH model fitted successfully")
            if aic is not None:
                self.logger.info(f"AIC: {aic:.2f}")
            if bic is not None:
                self.logger.info(f"BIC: {bic:.2f}")

            return {
                'status': 'success',
                'model_order': (self.p, self.q),
                'aic': aic,
                'bic': bic,
            }

        except Exception as e:
            self.logger.error(f"GARCH training failed: {e}")
            raise

    def predict(self, X: Union[pd.DataFrame, np.ndarray],
                steps: Optional[int] = None) -> np.ndarray:
        """
        Make volatility predictions with GARCH model.

        Args:
            X: Input features (unused; only horizon inferred)
            steps: Optional number of steps to forecast

        Returns:
            Array of predictions. For regression: volatility forecasts (decimal units).
            For classification: probability of positive class (bull regime) by thresholding volatility.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        if steps is not None:
            horizon = max(1, int(steps))
        else:
            try:
                horizon = len(X) if X is not None else 1
            except Exception:
                horizon = 1

        try:
            # Prepare exogenous for forecasting if model was trained with exog
            exog_forecast = None
            if self.exog_feature_names is not None and X is not None and self.use_exog_mean:
                try:
                    if isinstance(X, np.ndarray):
                        X_df = pd.DataFrame(X, columns=self.exog_feature_names + [f'extra_{i}' for i in range(max(0, X.shape[1] - len(self.exog_feature_names)))] )
                        X_df = X_df[self.exog_feature_names]
                    else:
                        # Select only trained exog columns; missing columns will raise
                        X_df = X[self.exog_feature_names]
                    # Ensure we only take the amount needed
                    X_df = X_df.iloc[:horizon]
                    if bool(getattr(self, 'expects_scaled_input', False)):
                        exog_forecast = X_df.values
                    else:
                        exog_forecast = self.exog_scaler.transform(X_df.values)
                except Exception as e:
                    self.logger.warning(f"Failed to prepare exogenous features for forecast; proceeding without exog: {e}")
                    exog_forecast = None

            if exog_forecast is not None:
                try:
                    forecast = self.fitted_model.forecast(horizon=horizon, x=exog_forecast)
                except Exception as e:
                    self.logger.warning(f"Forecast exog failed, using no-exog forecast: {e}")
                    forecast = self.fitted_model.forecast(horizon=horizon)
            else:
                forecast = self.fitted_model.forecast(horizon=horizon)
            volatility_forecast = np.sqrt(forecast.variance.values[-1, :])
            volatility_forecast = volatility_forecast / 100.0

            if (self.task or 'regression') == 'classification':
                # Lower volatility often associated with positive/neutral regime; map to class probability.
                vol = np.asarray(volatility_forecast, dtype=float)
                # Determine threshold: use provided or median of training vol if available
                thr = self.classification_threshold
                if thr is None:
                    try:
                        hist_vol = np.abs(np.asarray(self.returns_data, dtype=float))
                        hist_vol = np.where(np.isfinite(hist_vol), hist_vol, np.nan)
                        thr = np.nanmedian(np.abs(hist_vol))
                    except Exception:
                        thr = float(np.nanmedian(vol)) if np.isfinite(np.nanmedian(vol)) else float(np.nanmean(vol))
                # Convert vol to probability: higher prob when vol below threshold
                # Use a smooth logistic transform around the threshold
                eps = 1e-9
                scale = max(eps, float(np.nanstd(vol)) or 1e-3)
                logits = -(vol - thr) / (scale + eps)
                probs = 1.0 / (1.0 + np.exp(-logits))
                return probs

            return volatility_forecast
        except Exception as e:
            self.logger.error(f"GARCH prediction failed: {e}")
            if self.last_returns is not None:
                last_vol = float(np.std(self.last_returns))
                if (self.task or 'regression') == 'classification':
                    # map to neutral probability
                    return np.full(horizon, 0.5)
                return np.full(horizon, last_vol)
            return np.zeros(horizon)

    def evaluate(self, X: Union[pd.DataFrame, np.ndarray],
                 y: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """Evaluate the GARCH model with regression or classification metrics."""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        predictions = self.predict(X)
        y_true = y.values if isinstance(y, pd.Series) else np.asarray(y)

        min_len = min(len(predictions), len(y_true))
        predictions = predictions[:min_len]
        y_true = y_true[:min_len]

        if (self.task or 'regression') == 'classification':
            # Convert probabilities to labels for metrics
            try:
                y_prob = np.asarray(predictions, dtype=float)
                y_pred = (y_prob >= 0.5).astype(int)
                y_true_int = y_true.astype(int)
                from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
                return {
                    'accuracy': float(accuracy_score(y_true_int, y_pred)),
                    'f1': float(f1_score(y_true_int, y_pred, average='weighted', zero_division=0)),
                    'precision': float(precision_score(y_true_int, y_pred, average='weighted', zero_division=0)),
                    'recall': float(recall_score(y_true_int, y_pred, average='weighted', zero_division=0)),
                    'logloss': np.nan,
                }
            except Exception as _:
                return {
                    'accuracy': 0.0,
                    'f1': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'logloss': np.nan,
                }
        else:
            mse = mean_squared_error(y_true, predictions)
            mae = mean_absolute_error(y_true, predictions)
            r2 = r2_score(y_true, predictions)
            return {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'MSE': mse,
                'MAE': mae,
                'R2': r2,
            }

    def get_model_summary(self) -> str:
        """Get model summary."""
        if not self.is_trained:
            return "Model not trained"
        try:
            return str(self.fitted_model.summary())
        except Exception as e:
            self.logger.error(f"Failed to get model summary: {e}")
            return "Summary not available"


