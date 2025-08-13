"""
ARIMA model implementation for RiskPipeline.
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

from .base_model import BaseModel


class ARIMAModel(BaseModel):
    """ARIMA model for time series forecasting."""
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1), **kwargs):
        """
        Initialize ARIMA model.
        
        Args:
            order: ARIMA order (p, d, q)
            **kwargs: Additional parameters
        """
        super().__init__(name="ARIMA", **kwargs)
        self.order = order
        self.task = 'regression'
        self.fitted_model = None
        self.is_stationary = None
        self.diagnostics = {}
        
        self.logger.info(f"ARIMA model initialized with order {order}")
    
    # For unit tests compatibility
    def build_model(self, input_shape: Tuple[int, ...]):
        """No-op builder for ARIMA to satisfy tests."""
        return self
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], 
              y: Union[pd.Series, np.ndarray], **kwargs) -> Dict[str, Any]:
        """
        Train ARIMA model.
        
        Args:
            X: Training features (not used for ARIMA)
            y: Training targets (time series)
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training metrics and diagnostics
        """
        # Validate input (X is not used for ARIMA; allow None)
        if X is None:
            X = np.zeros((len(y), 1))
        _, y = self._validate_input(X, y)
        
        if len(y) < 50:
            raise ValueError("ARIMA requires at least 50 observations")
        
        self.logger.info(f"Training ARIMA model with {len(y)} observations")
        
        try:
            # Check stationarity
            self._check_stationarity(y)
            
            # Fit ARIMA model
            self.fitted_model = ARIMA(y, order=self.order)
            fitted_result = self.fitted_model.fit()
            
            # Store fitted model
            self.model = fitted_result
            self.is_trained = True
            
            # Calculate training metrics
            y_pred = fitted_result.fittedvalues
            metrics = self._calculate_regression_metrics(y, y_pred)
            
            # Run diagnostics
            self._run_diagnostics(y, fitted_result)
            
            self.logger.info(f"ARIMA training completed. AIC: {fitted_result.aic:.2f}")
            
            return {
                'metrics': metrics,
                'aic': fitted_result.aic,
                'bic': fitted_result.bic,
                'diagnostics': self.diagnostics,
                'is_stationary': self.is_stationary
            }
            
        except Exception as e:
            self.logger.error(f"ARIMA training failed: {e}")
            raise
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray], 
                steps: Optional[int] = None) -> np.ndarray:
        """
        Make predictions with ARIMA model.
        
        Args:
            X: Input features (not used for ARIMA)
            steps: Number of steps to forecast (default: length of X)
            
        Returns:
            Predictions array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if steps is None:
            steps = len(X) if hasattr(X, '__len__') else 1
        
        try:
            # Make forecast
            forecast = self.model.forecast(steps=steps)
            return np.array(forecast)
            
        except Exception as e:
            self.logger.error(f"ARIMA prediction failed: {e}")
            # Return naive forecast as fallback
            if hasattr(self.model, 'fittedvalues') and len(self.model.fittedvalues) > 0:
                last_value = self.model.fittedvalues.iloc[-1]
                return np.full(steps, last_value)
            else:
                return np.zeros(steps)
    
    def evaluate(self, X: Union[pd.DataFrame, np.ndarray], 
                y: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate ARIMA model.
        
        Args:
            X: Test features (not used for ARIMA)
            y: Test targets
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Validate input
        _, y = self._validate_input(X, y)
        
        try:
            # Make predictions
            y_pred = self.predict(X, steps=len(y))
            
            # Calculate metrics
            metrics = self._calculate_regression_metrics(y, y_pred)
            
            self.logger.info(f"ARIMA evaluation completed: RMSE={metrics['RMSE']:.4f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"ARIMA evaluation failed: {e}")
            return {
                'RMSE': float('inf'),
                'MAE': float('inf'),
                'R2': -float('inf')
            }
    
    def _check_stationarity(self, y: np.ndarray) -> None:
        """Check if the time series is stationary."""
        try:
            # Augmented Dickey-Fuller test
            adf_result = adfuller(y)
            self.is_stationary = adf_result[1] < 0.05
            
            self.logger.info(f"Stationarity test: p-value={adf_result[1]:.4f}, "
                           f"stationary={self.is_stationary}")
            
        except Exception as e:
            self.logger.warning(f"Stationarity test failed: {e}")
            self.is_stationary = None
    
    def _run_diagnostics(self, y: np.ndarray, fitted_result) -> None:
        """Run model diagnostics."""
        try:
            # Ljung-Box test for residuals
            residuals = fitted_result.resid
            lb_result = acorr_ljungbox(residuals, lags=10, return_df=True)
            
            self.diagnostics = {
                'ljung_box_pvalue': lb_result['lb_pvalue'].iloc[-1],
                'residuals_std': residuals.std(),
                'residuals_mean': residuals.mean(),
                'aic': fitted_result.aic,
                'bic': fitted_result.bic
            }
            
            self.logger.info(f"Diagnostics: Ljung-Box p-value={self.diagnostics['ljung_box_pvalue']:.4f}")
            
        except Exception as e:
            self.logger.warning(f"Diagnostics failed: {e}")
            self.diagnostics = {}
    
    def plot_diagnostics(self, save_path: Optional[str] = None) -> None:
        """Plot model diagnostics."""
        if not self.is_trained:
            raise ValueError("Model must be trained before plotting diagnostics")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Residuals plot
            residuals = self.model.resid
            axes[0, 0].plot(residuals)
            axes[0, 0].set_title('Residuals')
            axes[0, 0].set_xlabel('Time')
            axes[0, 0].set_ylabel('Residuals')
            
            # Residuals histogram
            axes[0, 1].hist(residuals, bins=30, alpha=0.7)
            axes[0, 1].set_title('Residuals Distribution')
            axes[0, 1].set_xlabel('Residuals')
            axes[0, 1].set_ylabel('Frequency')
            
            # ACF plot
            plot_acf(residuals, ax=axes[1, 0], lags=20)
            axes[1, 0].set_title('Autocorrelation Function')
            
            # PACF plot
            plot_pacf(residuals, ax=axes[1, 1], lags=20)
            axes[1, 1].set_title('Partial Autocorrelation Function')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Diagnostics plot saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Failed to plot diagnostics: {e}")
    
    def get_model_summary(self) -> str:
        """Get model summary."""
        if not self.is_trained:
            return "Model not trained"
        
        try:
            return str(self.model.summary())
        except Exception as e:
            self.logger.error(f"Failed to get model summary: {e}")
            return "Summary not available"
    
    def auto_arima(self, y: np.ndarray, max_p: int = 3, max_d: int = 2, max_q: int = 3) -> Tuple[int, int, int]:
        """
        Automatically select ARIMA order using AIC.
        
        Args:
            y: Time series data
            max_p: Maximum AR order
            max_d: Maximum differencing order
            max_q: Maximum MA order
            
        Returns:
            Best ARIMA order (p, d, q)
        """
        self.logger.info("Running auto ARIMA to find best order")
        
        best_aic = float('inf')
        best_order = (1, 1, 1)
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(y, order=(p, d, q))
                        fitted = model.fit()
                        
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                            
                    except Exception:
                        continue
        
        self.logger.info(f"Best ARIMA order: {best_order} (AIC: {best_aic:.2f})")
        return best_order 