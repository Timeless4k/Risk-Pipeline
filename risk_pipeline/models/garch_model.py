"""
GARCH model implementation for RiskPipeline.
"""

import logging
from typing import Dict, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .base_model import BaseModel


class GARCHModel(BaseModel):
    """GARCH model for volatility forecasting."""

    def __init__(self, p: int = 1, q: int = 1, **kwargs):
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
        self.task = 'regression'
        self.fitted_model = None
        self.returns_data = None
        self.last_returns = None

        self.logger.info(f"GARCH({p}, {q}) model initialized")

    def build_model(self, input_shape: Tuple[int, ...]) -> 'GARCHModel':
        """Build GARCH model (no explicit building needed)."""
        self.input_shape = input_shape
        self.model = "GARCH_READY"
        self.logger.info(f"GARCH model ready")
        return self

    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray], **kwargs) -> 'GARCHModel':
        """Fit GARCH model (alias for train)."""
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
            Array of volatility forecasts (in decimal units)
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
            forecast = self.fitted_model.forecast(horizon=horizon)
            volatility_forecast = np.sqrt(forecast.variance.values[-1, :])
            volatility_forecast = volatility_forecast / 100.0
            return volatility_forecast
        except Exception as e:
            self.logger.error(f"GARCH prediction failed: {e}")
            if self.last_returns is not None:
                last_vol = float(np.std(self.last_returns))
                return np.full(horizon, last_vol)
            return np.zeros(horizon)

    def evaluate(self, X: Union[pd.DataFrame, np.ndarray],
                 y: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """Evaluate the GARCH model with regression metrics."""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        predictions = self.predict(X)
        y_true = y.values if isinstance(y, pd.Series) else np.asarray(y)

        min_len = min(len(predictions), len(y_true))
        predictions = predictions[:min_len]
        y_true = y_true[:min_len]

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


