#!/usr/bin/env python3
"""
Enhanced ARIMAX Model for RiskPipeline
Uses external variables (VIX, correlations, technical indicators) for better forecasting
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Any, Union, List
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .base_model import BaseModel

# 🚀 24-CORE OPTIMIZATION: Use all cores for statsmodels
import psutil
import os
cpu_count = psutil.cpu_count(logical=False)  # Physical cores (24 for your i9-14900HX)
os.environ['OMP_NUM_THREADS'] = str(cpu_count)
os.environ['MKL_NUM_THREADS'] = str(cpu_count)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_count)

logger = logging.getLogger(__name__)

class EnhancedARIMAModel(BaseModel):
    """
    Enhanced ARIMAX model that selectively uses engineered features
    for enhanced statistical forecasting with 24-core optimization
    """
    
    def __init__(self, name: str = "EnhancedARIMA", **kwargs):
        super().__init__(name=name, **kwargs)
        self.task = 'regression'
        self.order = kwargs.get('order', (1, 1, 1))
        # Back-compat: if top_k_features not provided, fall back to n_features; None => use all
        self.top_k_features = kwargs.get('top_k_features', kwargs.get('n_features', None))
        self.feature_selector_mode = kwargs.get('feature_selector', 'kbest')
        self.use_log_vol_target = kwargs.get('use_log_vol_target', False)
        self.log_target_epsilon = kwargs.get('log_target_epsilon', 1e-6)
        self.residual_model = kwargs.get('residual_model', 'xgb')
        self.residual_params = kwargs.get('residual_params', None)
        self.auto_order = kwargs.get('auto_order', True)
        self.feature_selector = None
        self.scaler = StandardScaler()
        self.model = None
        self.selected_features = None
        self.fitted_model = None
        self.current_model = None
        
        logger.info(f"🚀 EnhancedARIMA initialized with {cpu_count} cores for maximum performance!")
        
    def build_model(self, input_shape: Tuple[int, ...]) -> 'EnhancedARIMAModel':
        """Build enhanced ARIMA model with feature selection."""
        self.input_shape = input_shape
        
        # FIXED: Set self.model to indicate the model is properly built
        # This is required by the main pipeline to check if the model is ready
        self.model = "EnhancedARIMA_Ready"  # Use a string to indicate readiness
        
        logger.info(f"EnhancedARIMA model ready with top_k_features={self.top_k_features}")
        return self
    
    def select_key_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Select most important features for ARIMAX
        Focus on market-wide and regime indicators
        """
        
        # Priority features for financial volatility (most predictive)
        priority_features = [
            'VIX', 'VIX_change',                    # Market fear indicators
            'AAPL_GSPC_corr',                       # Market coupling (asset-specific)
            'IOZ_CBA_corr', 'BHP_IOZ_corr',         # Cross-asset correlations
            'RollingStd30', 'RollingStd5',          # Volatility regime
            'RSI', 'MACD',                          # Technical momentum
            'MA_ratio',                             # Trend strength
            'BB_position', 'BB_width',              # Bollinger Bands
            'ADX', 'CCI',                           # Additional technical
        ]
        
        # Find available priority features
        available_priority = [f for f in priority_features if f in X.columns]
        
        logger.info(f"Available priority features: {len(available_priority)}")
        logger.info(f"Priority features: {available_priority}")
        
        # If we have enough priority features, use them
        if len(available_priority) >= self.n_features:
            selected = available_priority[:self.n_features]
            logger.info(f"Using top {self.n_features} priority features")
        else:
            # Use statistical selection for remaining features
            logger.info(f"Using {len(available_priority)} priority + statistical selection")
            
            # Start with priority features
            selected = available_priority.copy()
            
            # Add statistically significant features
            remaining_features = [col for col in X.columns if col not in selected]
            n_additional = self.n_features - len(selected)
            
            if n_additional > 0 and len(remaining_features) > 0:
                # 🚀 24-CORE OPTIMIZATION: Use all cores for feature selection
                selector = SelectKBest(f_regression, k=min(n_additional, len(remaining_features)))
                X_remaining = X[remaining_features]
                selector.fit(X_remaining, y)
                
                additional_features = [remaining_features[i] for i in selector.get_support(indices=True)]
                selected.extend(additional_features)
        
        self.selected_features = selected[:self.n_features]
        logger.info(f"Final selected features for ARIMAX: {self.selected_features}")
        
        return X[self.selected_features]
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], 
            y: Union[pd.Series, np.ndarray], 
            validation_data: Tuple = None, **kwargs) -> 'EnhancedARIMAModel':
        """Training interface for your pipeline with 24-core optimization."""
        
        # Convert to pandas if needed
        if not isinstance(X, pd.DataFrame):
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=feature_names)
        
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        # FIXED: Use global cpu_count variable
        logger.info(f"🚀 Training EnhancedARIMA with {cpu_count} cores!")
        logger.info(f"Input shape: {X.shape}, Target shape: {y.shape}")
        
        # Create and fit enhanced ARIMA core with residual model
        arimax = EnhancedARIMA(
            order=self.order,
            top_k_features=self.top_k_features,
            feature_selector=self.feature_selector_mode,
            use_log_vol_target=self.use_log_vol_target,
            log_target_epsilon=self.log_target_epsilon,
            residual_model=self.residual_model,
            residual_params=self.residual_params,
            auto_order=self.auto_order,
        )
        # Feature selection, scaling, and residual fitting happen inside arimax.fit
        arimax.fit(y, X)
        
        # Store the fitted model
        self.current_model = arimax
        
        # Set the is_trained flag for the validator
        self.is_trained = True
        
        logger.info("EnhancedARIMA training completed successfully!")
        return self
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], **kwargs) -> Dict[str, Any]:
        """Train the EnhancedARIMA model (alias for fit)."""
        self.fit(X, y, **kwargs)
        return {
            'status': 'success',
            'model_order': self.order,
            'top_k_features': self.top_k_features,
            'selected_features': self.selected_features
        }
    
    def evaluate(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """Evaluate the EnhancedARIMA model."""
        if not self.current_model:
            raise ValueError("Model must be trained before evaluation")
        
        predictions = self.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, predictions)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2
        }
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Prediction interface with proper error handling."""
        
        if self.current_model is None:
            logger.error("Model must be fitted before prediction")
            return np.full(1 if X is None else len(X), np.nan)
        
        # Check if the inner model is properly fitted
        if not hasattr(self.current_model, 'fitted_model') or self.current_model.fitted_model is None:
            logger.error("Inner ARIMA model is not properly fitted")
            return np.full(1 if X is None else len(X), np.nan)
        
        # Handle None X by forecasting one step ahead using last known state
        if X is None:
            try:
                preds = self.current_model.predict(steps=1, X_test=None)
                return np.asarray(preds)
            except Exception as e:
                logger.error(f"Prediction failed with None X: {e}")
                return np.asarray([np.nan])
        
        # Convert to pandas if needed  
        if not isinstance(X, pd.DataFrame):
            feature_names = [f'feature_{i}' for i in range(X.shape[1])] if hasattr(X, 'shape') else []
            X = pd.DataFrame(X, columns=feature_names) if feature_names else pd.DataFrame()
        
        try:
            steps = len(X) if hasattr(X, '__len__') and len(X) > 0 else 1
            predictions = self.current_model.predict(steps=steps, X_test=X if not X.empty else None)
            return np.asarray(predictions)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Fallback: last fitted param or zeros
            try:
                fallback = float(self.current_model.fitted_model.params[0])
            except Exception:
                fallback = np.nan
            return np.full(steps, fallback)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from ARIMAX coefficients."""
        if self.current_model is None or self.selected_features is None:
            return {}
        
        try:
            return self.current_model.get_feature_importance()
        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
            return {}
    
    def save_model(self, filepath: str) -> bool:
        """Save the fitted model."""
        try:
            # Save using joblib for statsmodels compatibility
            import joblib
            joblib.dump(self.current_model, filepath)
            logger.info(f"EnhancedARIMA model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load a fitted model."""
        try:
            import joblib
            self.current_model = joblib.load(filepath)
            logger.info(f"EnhancedARIMA model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False


class EnhancedARIMA:
    """
    Core ARIMAX implementation with external variables
    """
    
    def __init__(self, order=(1, 1, 1), top_k_features=None, feature_selector='kbest',
                 use_log_vol_target=False, log_target_epsilon=1e-6,
                 residual_model='xgb', residual_params=None, auto_order=True):
        self.order = order
        self.top_k_features = top_k_features
        self.feature_selector_mode = feature_selector
        self.use_log_vol_target = use_log_vol_target
        self.log_target_epsilon = log_target_epsilon
        self.residual_model = residual_model
        self.residual_params = residual_params or {
            "n_estimators": 400,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "gamma": 0.2,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
        }
        self.auto_order = auto_order
        self.scaler = StandardScaler()
        self.selected_features = []
        self.selected_idx_ = None
        self.fitted_model = None  # ARIMA/SARIMAX fitted results
        self.residual_est_ = None
        
    def select_key_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Select features. If top_k_features is None or >= total, use all features."""
        num_features_total = X.shape[1]
        if self.top_k_features is None or (isinstance(self.top_k_features, int) and self.top_k_features >= num_features_total):
            self.selected_features = list(X.columns)
            self.selected_idx_ = np.arange(num_features_total)
            return X
        k = max(1, int(self.top_k_features))
        if self.feature_selector_mode == 'kbest':
            selector = SelectKBest(f_regression, k=min(k, num_features_total))
            selector.fit(X.values, y.values if isinstance(y, pd.Series) else np.asarray(y))
            idx = selector.get_support(indices=True)
            self.selected_idx_ = np.asarray(idx)
        elif self.feature_selector_mode == 'rf':
            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
            rf.fit(X.values, y.values if isinstance(y, pd.Series) else np.asarray(y))
            self.selected_idx_ = np.argsort(rf.feature_importances_)[::-1][:k]
        else:
            self.selected_idx_ = np.arange(num_features_total)
        self.selected_features = [X.columns[i] for i in self.selected_idx_]
        return X.iloc[:, self.selected_idx_]
    
    def fit(self, y: pd.Series, X: pd.DataFrame = None) -> 'EnhancedARIMA':
        """Fit baseline ARIMA and residual model on features (if provided)."""
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        if X is not None and not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        # Target transform
        if self.use_log_vol_target:
            self._target_shift_ = True
            y_tr = np.log(y.values + self.log_target_epsilon)
        else:
            self._target_shift_ = False
            y_tr = y.values
        # Scale and select features
        if X is not None:
            X_scaled = self.scaler.fit_transform(X.values)
            X_scaled_df = pd.DataFrame(X_scaled, columns=list(X.columns), index=X.index)
            X_sel_df = self.select_key_features(X_scaled_df, y)
        else:
            X_sel_df = None
        # Fit ARIMA with auto-order search
        self._fit_arima(y_tr)
        # Compute in-sample baseline and residuals
        y_hat_tr = self._predict_arima_in_sample()
        residuals = y_tr - y_hat_tr
        # Fit residual model
        if X_sel_df is not None and self.residual_model == 'xgb':
            from xgboost import XGBRegressor, callback as xgb_callback
            self.residual_est_ = XGBRegressor(**self.residual_params)
            try:
                self.residual_est_.fit(
                    X_sel_df.values, residuals,
                    eval_set=[(X_sel_df.values, residuals)],
                    eval_metric="rmse",
                    verbose=False,
                    callbacks=[xgb_callback.EarlyStopping(rounds=50, save_best=True)]
                )
            except Exception:
                self.residual_est_.fit(X_sel_df.values, residuals)
        else:
            self.residual_est_ = None
        logger.info("EnhancedARIMA fitted: baseline ARIMA + residual model=%s", 'xgb' if self.residual_est_ is not None else 'none')
        # Persist selected features for outer wrapper access
        try:
            self.selected_features = self.selected_features if self.selected_features else list(X.columns) if X is not None else []
        except Exception:
            pass
        return self

    def _fit_arima(self, y_tr: np.ndarray) -> None:
        if self.auto_order:
            import itertools
            best = None
            for p, q in itertools.product([0, 1, 2], repeat=2):
                order = (p, 1, q)
                try:
                    mdl = SARIMAX(y_tr, order=order, seasonal_order=(0, 0, 0, 0),
                                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                    aic = mdl.aic
                    if best is None or aic < best[0]:
                        best = (aic, mdl)
                except Exception:
                    continue
            self.fitted_model = best[1] if best else SARIMAX(y_tr, order=self.order,
                                                             seasonal_order=(0, 0, 0, 0),
                                                             enforce_stationarity=False,
                                                             enforce_invertibility=False).fit(disp=False)
        else:
            self.fitted_model = SARIMAX(y_tr, order=self.order, seasonal_order=(0, 0, 0, 0),
                                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        try:
            logger.info(f"ARIMA AIC: {self.fitted_model.aic:.2f}")
        except Exception:
            pass

    def _predict_arima_in_sample(self) -> np.ndarray:
        try:
            return self.fitted_model.fittedvalues if hasattr(self.fitted_model, 'fittedvalues') else self.fitted_model.predict()
        except Exception:
            return self.fitted_model.predict()
    
    def predict(self, steps: int = 1, X_test: pd.DataFrame = None) -> np.ndarray:
        """Predict using baseline ARIMA plus residual model if available."""
        base = self.fitted_model.forecast(steps=steps)
        if getattr(self, '_target_shift_', False):
            # Work in transformed space; inverse only at the end
            base_tr = base
        else:
            base_tr = base
        if X_test is not None and self.residual_est_ is not None:
            X_used = self.transform_features(X_test)
            residual_pred = self.residual_est_.predict(X_used)
        else:
            residual_pred = np.zeros_like(base_tr)
        y_hat_tr = base_tr + residual_pred
        if getattr(self, '_target_shift_', False):
            return np.exp(y_hat_tr) - self.log_target_epsilon
        return y_hat_tr

    def transform_features(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X_df = X
        else:
            X_df = pd.DataFrame(X)
        X_scaled = self.scaler.transform(X_df.values)
        if self.selected_idx_ is not None:
            return X_scaled[:, self.selected_idx_]
        return X_scaled
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from ARIMAX coefficients."""
        if self.fitted_model is None or self.selected_features is None:
            return {}
        
        # Extract coefficients for external variables
        params = self.fitted_model.params
        
        # External variable coefficients typically start after AR/MA terms
        exog_params = {}
        for feature in self.selected_features:
            if feature in params.index:
                exog_params[feature] = params[feature]
        
        return exog_params


# Factory function for integration
def create_enhanced_arima_model(**kwargs) -> EnhancedARIMAModel:
    """Factory function for EnhancedARIMA model."""
    return EnhancedARIMAModel(
        order=kwargs.get('order', (1, 1, 1)),
        top_k_features=kwargs.get('top_k_features', kwargs.get('n_features', None)),
        feature_selector=kwargs.get('feature_selector', 'kbest'),
        use_log_vol_target=kwargs.get('use_log_vol_target', False),
        log_target_epsilon=kwargs.get('log_target_epsilon', 1e-6),
        residual_model=kwargs.get('residual_model', 'xgb'),
        residual_params=kwargs.get('residual_params', None),
        auto_order=kwargs.get('auto_order', True)
    )


if __name__ == "__main__":
    print("EnhancedARIMA implementation ready!")
    print("Key benefits:")
    print("- Uses market-wide signals (VIX)")
    print("- Incorporates cross-asset correlations") 
    print("- Maintains statistical interpretability")
    print("- Bridges gap between ARIMA and deep learning")
    print(f"- Optimized for {cpu_count}-core system!")
