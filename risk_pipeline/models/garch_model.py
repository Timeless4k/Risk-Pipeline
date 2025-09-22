"""
Enhanced GARCH model implementation for RiskPipeline.
Based on paper: "A Hybrid Model Integrating LSTM with Multiple GARCH-Type Models for Volatility and VaR Forecast"

Key improvements:
- Realized volatility calculation using high-frequency data approximation
- Multiple GARCH variants (GARCH, EGARCH, TGARCH)
- Enhanced exogenous feature support
- Improved volatility forecasting and VaR estimation
"""

import logging
from typing import Dict, Tuple, Optional, Union, Any, List
import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy import stats

from .base_model import BaseModel


class GARCHModel(BaseModel):
    """
    Enhanced GARCH model for volatility forecasting and derived classification.
    
    Based on the paper methodology with improvements for:
    - Realized volatility calculation
    - Multiple GARCH variants (GARCH, EGARCH, TGARCH)
    - Enhanced exogenous feature support
    - VaR estimation capabilities
    """

    def __init__(self, p: int = 1, q: int = 1, task: str = 'regression', 
                 threshold: Optional[float] = None, garch_type: str = 'GARCH', **kwargs):
        """
        Initialize enhanced GARCH model.

        Args:
            p: GARCH lag order
            q: ARCH lag order
            task: 'regression' or 'classification'
            threshold: Classification threshold for volatility
            garch_type: Type of GARCH model ('GARCH', 'EGARCH', 'TGARCH')
            **kwargs: Additional parameters
        """
        super().__init__(name=f"{garch_type}({p},{q})", **kwargs)
        self.p = p
        self.q = q
        self.task = task or 'regression'
        self.garch_type = garch_type.upper()
        self.fitted_model = None
        self.returns_data = None
        self.last_returns = None
        self.classification_threshold = threshold
        self.realized_volatility = None

        # Enhanced exogenous feature support
        self.use_exog_mean = bool(kwargs.get('use_exog_mean', False))
        self.max_exog_features = int(kwargs.get('max_exog_features', 64))
        self.exog_scaler: Optional[StandardScaler] = StandardScaler()
        self.exog_feature_names: Optional[List[str]] = None

        # VaR estimation parameters
        self.var_confidence_levels = [0.90, 0.95, 0.99]
        self.var_estimates = {}

        # Realized volatility parameters
        self.use_realized_vol = bool(kwargs.get('use_realized_vol', True))
        self.rv_window = int(kwargs.get('rv_window', 5))  # 5-day realized volatility

        self.logger.info(f"{self.garch_type}({p}, {q}) model initialized with task={task}")

    def build_model(self, input_shape: Tuple[int, ...]) -> 'GARCHModel':
        """Build GARCH model (no explicit building needed)."""
        self.input_shape = input_shape
        self.model = f"{self.garch_type}_READY"
        self.logger.info(f"{self.garch_type} model ready")
        return self

    def calculate_realized_volatility(self, prices: pd.Series, window: int = None) -> pd.Series:
        """
        Calculate realized volatility as per paper methodology.
        
        RV_t = sqrt(sum(r_t,i^2) for i=1 to M)
        where r_t,i = 100 * ln(P_t,i / P_t,i-1)
        
        Args:
            prices: Price series (can be daily or intraday)
            window: Rolling window for RV calculation
            
        Returns:
            Realized volatility series
        """
        if window is None:
            window = self.rv_window
            
        # Calculate log returns
        log_returns = 100 * np.log(prices / prices.shift(1))
        
        # For daily data, approximate high-frequency RV using squared returns
        # This is a simplified approximation of the paper's methodology
        if len(log_returns) < window:
            self.logger.warning(f"Insufficient data for {window}-day RV calculation")
            return pd.Series(index=prices.index, dtype=float)
        
        # Calculate rolling realized volatility
        rv = log_returns.rolling(window=window, min_periods=window//2).std()
        
        # Annualize (assuming daily data)
        rv_annualized = rv * np.sqrt(252)
        
        self.logger.info(f"Calculated {window}-day realized volatility")
        return rv_annualized

    def calculate_var(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk using parametric method.
        
        VaR = μ + t_α * σ
        where μ is mean return, t_α is quantile, σ is volatility
        
        Args:
            returns: Return series
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            VaR estimate
        """
        if len(returns) == 0:
            return 0.0
            
        mean_return = np.mean(returns)
        vol = np.std(returns)
        
        # Use t-distribution for better tail behavior
        alpha = 1 - confidence_level
        t_alpha = stats.t.ppf(alpha, df=len(returns)-1)
        
        var = mean_return + t_alpha * vol
        return var

    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray], **kwargs) -> 'GARCHModel':
        """Fit GARCH model. For classification, we still fit volatility model."""
        self.train(X, y, **kwargs)
        return self

    def train(self, X: Union[pd.DataFrame, np.ndarray], 
              y: Union[pd.Series, np.ndarray], **kwargs) -> Dict[str, Any]:
        """
        Train enhanced GARCH model with realized volatility support.

        Enhanced training process:
        1. Calculate realized volatility if using price data
        2. Use appropriate GARCH variant based on configuration
        3. Support for exogenous features in mean model
        4. VaR estimation capabilities
        """
        self.logger.info(f"Training {self.garch_type} model with {len(y)} observations")

        try:
            # Convert inputs
            if isinstance(y, pd.Series):
                y_arr = y.values
            else:
                y_arr = np.asarray(y).ravel()

            # Enhanced data preparation
            returns_data = None
            realized_vol = None
            
            # Check if we have price data to calculate realized volatility
            if X is not None and hasattr(X, 'columns'):
                # Look for price columns
                price_cols = ['Close', 'Adj Close', 'Price']
                price_col = None
                for col in price_cols:
                    if col in X.columns:
                        price_col = col
                        break
                
                if price_col and self.use_realized_vol:
                    # Calculate realized volatility from price data
                    prices = X[price_col].dropna()
                    if len(prices) > self.rv_window:
                        realized_vol = self.calculate_realized_volatility(prices, self.rv_window)
                        self.realized_volatility = realized_vol
                        self.logger.info(f"Calculated realized volatility using {price_col}")
            
            # Determine target data type and prepare returns
            if np.all(np.isfinite(y_arr)) and np.all(y_arr >= 0) and np.nanmean(y_arr) < 1:
                # y appears to be volatility - use as target for realized volatility
                if realized_vol is not None:
                    returns_data = np.log(realized_vol.dropna().values + 1e-8)  # Log transform
                    self.logger.info("Using log-transformed realized volatility as returns")
                else:
                    # Fallback to synthetic returns
                    rng = np.random.default_rng(42)
                    synthetic_returns = rng.normal(0.0, y_arr)
                    returns_data = synthetic_returns
                    self.logger.info("Using synthetic returns derived from volatility targets")
            else:
                # y appears to be returns
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

            # Create appropriate GARCH model based on type
            if exog_train is not None:
                try:
                    if self.garch_type == 'GARCH':
                        model = arch_model(returns_scaled, x=exog_train, mean='ARX', lags=0, vol='Garch', p=self.p, q=self.q)
                    elif self.garch_type == 'EGARCH':
                        model = arch_model(returns_scaled, x=exog_train, mean='ARX', lags=0, vol='EGARCH', p=self.p, q=self.q)
                    elif self.garch_type == 'TGARCH':
                        model = arch_model(returns_scaled, x=exog_train, mean='ARX', lags=0, vol='GARCH', p=self.p, q=self.q, o=1)  # Threshold GARCH
                    else:
                        model = arch_model(returns_scaled, x=exog_train, mean='ARX', lags=0, vol='Garch', p=self.p, q=self.q)
                except Exception as e:
                    self.logger.warning(f"Falling back to no exogenous mean model due to: {e}")
                    if self.garch_type == 'GARCH':
                        model = arch_model(returns_scaled, vol='Garch', p=self.p, q=self.q)
                    elif self.garch_type == 'EGARCH':
                        model = arch_model(returns_scaled, vol='EGARCH', p=self.p, q=self.q)
                    elif self.garch_type == 'TGARCH':
                        model = arch_model(returns_scaled, vol='GARCH', p=self.p, q=self.q, o=1)
                    else:
                        model = arch_model(returns_scaled, vol='Garch', p=self.p, q=self.q)
            else:
                if self.garch_type == 'GARCH':
                    model = arch_model(returns_scaled, vol='Garch', p=self.p, q=self.q)
                elif self.garch_type == 'EGARCH':
                    model = arch_model(returns_scaled, vol='EGARCH', p=self.p, q=self.q)
                elif self.garch_type == 'TGARCH':
                    model = arch_model(returns_scaled, vol='GARCH', p=self.p, q=self.q, o=1)
                else:
                    model = arch_model(returns_scaled, vol='Garch', p=self.p, q=self.q)

            # Fit the model with enhanced error handling
            try:
                self.fitted_model = model.fit(disp='off', show_warning=False)
            except Exception as e:
                self.logger.warning(f"Model fitting failed with {self.garch_type}, trying GARCH(1,1): {e}")
                # Fallback to simple GARCH(1,1)
                fallback_model = arch_model(returns_scaled, vol='Garch', p=1, q=1)
                self.fitted_model = fallback_model.fit(disp='off', show_warning=False)
            self.is_trained = True

            aic = getattr(self.fitted_model, 'aic', None)
            bic = getattr(self.fitted_model, 'bic', None)

            self.logger.info(f"{self.garch_type} model fitted successfully")
            if aic is not None:
                self.logger.info(f"AIC: {aic:.2f}")
            if bic is not None:
                self.logger.info(f"BIC: {bic:.2f}")

            # Calculate VaR estimates for different confidence levels
            if self.returns_data is not None:
                for conf_level in self.var_confidence_levels:
                    var_est = self.calculate_var(self.returns_data, conf_level)
                    self.var_estimates[conf_level] = var_est
                    self.logger.info(f"VaR {conf_level*100:.0f}%: {var_est:.4f}")

            return {
                'status': 'success',
                'model_type': self.garch_type,
                'model_order': (self.p, self.q),
                'aic': aic,
                'bic': bic,
                'var_estimates': self.var_estimates,
            }

        except Exception as e:
            self.logger.error(f"GARCH training failed: {e}")
            raise

    def predict(self, X: Union[pd.DataFrame, np.ndarray],
                steps: Optional[int] = None, return_var: bool = False) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Make enhanced volatility predictions with GARCH model.

        Args:
            X: Input features (unused; only horizon inferred)
            steps: Optional number of steps to forecast
            return_var: If True, return VaR estimates along with volatility

        Returns:
            Array of predictions. For regression: volatility forecasts (decimal units).
            For classification: probability of positive class (bull regime) by thresholding volatility.
            If return_var=True, returns dict with 'volatility' and 'var' keys.
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

            # Calculate VaR estimates if requested
            var_forecasts = {}
            if return_var and self.returns_data is not None:
                for conf_level in self.var_confidence_levels:
                    # Use forecasted volatility for VaR calculation
                    mean_return = np.mean(self.returns_data) if len(self.returns_data) > 0 else 0.0
                    vol_forecast = volatility_forecast[0] if len(volatility_forecast) > 0 else 0.01
                    
                    # Calculate VaR using forecasted volatility
                    alpha = 1 - conf_level
                    t_alpha = stats.t.ppf(alpha, df=len(self.returns_data)-1) if len(self.returns_data) > 1 else -1.645
                    var_forecast = mean_return + t_alpha * vol_forecast
                    var_forecasts[conf_level] = var_forecast

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
                
                if return_var:
                    return {'volatility': probs, 'var': var_forecasts}
                return probs

            if return_var:
                return {'volatility': volatility_forecast, 'var': var_forecasts}
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

    def get_var_estimates(self, confidence_levels: List[float] = None) -> Dict[float, float]:
        """
        Get VaR estimates for specified confidence levels.
        
        Args:
            confidence_levels: List of confidence levels (e.g., [0.90, 0.95, 0.99])
            
        Returns:
            Dictionary mapping confidence levels to VaR estimates
        """
        if not self.is_trained or self.returns_data is None:
            return {}
        
        if confidence_levels is None:
            confidence_levels = self.var_confidence_levels
        
        var_estimates = {}
        for conf_level in confidence_levels:
            var_est = self.calculate_var(self.returns_data, conf_level)
            var_estimates[conf_level] = var_est
        
        return var_estimates

    def get_model_summary(self) -> str:
        """Get enhanced model summary with VaR information."""
        if not self.is_trained:
            return "Model not trained"
        
        try:
            summary = str(self.fitted_model.summary())
            
            # Add VaR information
            if self.var_estimates:
                summary += "\n\nVaR Estimates:\n"
                for conf_level, var_est in self.var_estimates.items():
                    summary += f"VaR {conf_level*100:.0f}%: {var_est:.4f}\n"
            
            return summary
        except Exception as e:
            self.logger.error(f"Failed to get model summary: {e}")
            return f"Summary not available. Model type: {self.garch_type}"

    def get_leverage_effect(self) -> Optional[float]:
        """
        Get leverage effect parameter for EGARCH models.
        
        Returns:
            Leverage effect parameter if available, None otherwise
        """
        if not self.is_trained or self.garch_type != 'EGARCH':
            return None
        
        try:
            # Extract leverage effect from EGARCH model
            params = self.fitted_model.params
            if 'alpha[1]' in params and 'gamma[1]' in params:
                alpha = params['alpha[1]']
                gamma = params['gamma[1]']
                return gamma  # Leverage effect parameter
        except Exception as e:
            self.logger.warning(f"Could not extract leverage effect: {e}")
        
        return None


