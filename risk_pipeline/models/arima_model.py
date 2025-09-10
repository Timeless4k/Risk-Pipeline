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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
        self.seasonal_order = kwargs.get('seasonal_order', (0, 0, 0, 0))  # Default no seasonality
        self.seasonal = kwargs.get('seasonal', False)
        self.auto_order = kwargs.get('auto_order', False)
        self.max_p = int(kwargs.get('max_p', 5))
        self.max_d = int(kwargs.get('max_d', 2))
        self.max_q = int(kwargs.get('max_q', 5))
        self.task = 'regression'
        self.fitted_model = None
        self.is_stationary = None
        self.diagnostics = {}
        
        self.logger.info(f"ARIMA model initialized with order {order}")
    
    def build_model(self, input_shape: Tuple[int, ...]) -> 'ARIMAModel':
        """Build the ARIMA model architecture (ARIMA models don't need building)."""
        self.input_shape = input_shape
        # FIXED: Set a placeholder model attribute to indicate the model is "built"
        # ARIMA models don't need actual building, but we need to set this for compatibility
        self.model = "ARIMA_READY"  # Placeholder to indicate readiness
        self.logger.info(f"ARIMA model ready with input shape: {input_shape}")
        return self
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], **kwargs) -> 'ARIMAModel':
        """Fit ARIMA model with parallel processing optimization."""
        self.logger.info(f"Fitting ARIMA model for {self.name}")
        
        # 🚀 24-CORE OPTIMIZATION: Use all cores for ARIMA fitting
        import psutil
        cpu_count = psutil.cpu_count(logical=False)  # Physical cores (24 for your i9-14900HX)
        self.logger.info(f"🚀 Using {cpu_count} cores for ARIMA model fitting!")
        
        # Set parallel processing for statsmodels
        import os
        os.environ['OMP_NUM_THREADS'] = str(cpu_count)
        os.environ['MKL_NUM_THREADS'] = str(cpu_count)
        os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_count)
        
        try:
            # Ignore X entirely. ARIMA uses only the univariate target series y.
            # Normalize y to a 1D numpy array
            if isinstance(y, pd.Series):
                y_arr = y.values
            elif isinstance(y, pd.DataFrame):
                # Use the first column if a DataFrame is accidentally provided
                y_arr = y.iloc[:, 0].values
            else:
                y_arr = np.asarray(y)
            
            if y_arr.ndim > 1:
                y_arr = y_arr.ravel()
            
            if y_arr.size == 0:
                raise ValueError("ARIMA requires a non-empty target series")
            
            # Optionally run simple stationarity check
            try:
                self._check_stationarity(y_arr)
            except Exception:
                pass

            # Auto-order selection if enabled
            if self.auto_order:
                try:
                    best_order = self.auto_arima(y_arr, max_p=self.max_p, max_d=self.max_d, max_q=self.max_q)
                    self.logger.info(f"Using auto-selected order: {best_order}")
                    self.order = best_order
                except Exception as auto_err:
                    self.logger.warning(f"auto_arima failed, falling back to provided order {self.order}: {auto_err}")

            # Fit ARIMA model with chosen parameters
            seasonal_order = self.seasonal_order if self.seasonal else (0, 0, 0, 0)
            self.model = ARIMA(y_arr, order=self.order, seasonal_order=seasonal_order)
            self.fitted_model = self.model.fit()
            self.is_trained = True  # Set training flag
            
            self.logger.info(f"✅ ARIMA model fitted successfully with {cpu_count}-core optimization")
            self.logger.info(f"📊 Model: ARIMA{self.order} x {self.seasonal_order}")
            
            return self
            
        except Exception as e:
            self.logger.error(f"ARIMA model fitting failed: {e}")
            raise
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], **kwargs) -> Dict[str, Any]:
        """Train the ARIMA model (alias for fit)."""
        self.fit(X, y, **kwargs)
        return {
            'status': 'success',
            'model_order': self.order,
            'aic': getattr(self.fitted_model, 'aic', None),
            'bic': getattr(self.fitted_model, 'bic', None)
        }
    
    def evaluate(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """Evaluate the ARIMA model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        predictions = self.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, predictions)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)
        
        # Provide both lowercase and uppercase keys for compatibility with tests
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'MSE': mse,
            'MAE': mae,
            'R2': r2,
        }
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray], 
                steps: Optional[int] = None) -> np.ndarray:
        """
        Make predictions with ARIMA model.
        
        Args:
            X: Input features (ignored for ARIMA; used only to infer steps if provided)
            steps: Number of steps to forecast (default: len(X) if available, else 1)
            
        Returns:
            Predictions array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # FIXED: Check if fitted model is actually a fitted ARIMA model
        if not hasattr(self.fitted_model, 'forecast'):
            raise ValueError("ARIMA model not properly fitted")
        
        # Determine forecast horizon
        if steps is not None:
            horizon = int(max(1, steps))
        else:
            try:
                horizon = len(X)  # prefer provided X length
                if horizon is None or horizon <= 0:
                    horizon = 1
            except Exception:
                horizon = 1
        
        try:
            # Make forecast
            if hasattr(self, 'fitted_model') and self.fitted_model is not None and hasattr(self.fitted_model, 'forecast'):
                forecast = self.fitted_model.forecast(steps=horizon)
            else:
                forecast = self.model.forecast(steps=horizon)
            return np.asarray(forecast)
            
        except Exception as e:
            self.logger.error(f"ARIMA prediction failed: {e}")
            # Return naive forecast as fallback
            try:
                if hasattr(self.fitted_model, 'fittedvalues') and len(self.fitted_model.fittedvalues) > 0:
                    last_value = np.asarray(self.fitted_model.fittedvalues)[-1]
                    return np.full(horizon, last_value)
            except Exception:
                pass
            return np.zeros(horizon)
    
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