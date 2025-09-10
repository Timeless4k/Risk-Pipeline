"""
Modern ARIMAX model implementation optimized for financial time series forecasting.
Combines classical econometric methods with modern ML techniques.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import warnings

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss, acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats

from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import psutil
import os

CPU_COUNT = psutil.cpu_count(logical=False)
os.environ['OMP_NUM_THREADS'] = str(CPU_COUNT)
os.environ['MKL_NUM_THREADS'] = str(CPU_COUNT)
os.environ['OPENBLAS_NUM_THREADS'] = str(CPU_COUNT)

from .base_model import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class ARIMADiagnostics:
    aic: float
    bic: float
    hqic: float
    ljung_box_pvalue: float
    jarque_bera_pvalue: float
    heteroscedasticity_pvalue: float
    residual_autocorr_1: float
    residual_autocorr_5: float
    durbin_watson: float
    is_stationary: bool
    seasonal_strength: float
    trend_strength: float


class FinancialTimeSeriesAnalyzer:
    @staticmethod
    def test_stationarity(series: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:
        results = {}
        try:
            adf_stat, adf_p, _, _, adf_crit, _ = adfuller(series.dropna())
            results['adf'] = {
                'statistic': adf_stat,
                'pvalue': adf_p,
                'critical_values': adf_crit,
                'is_stationary': adf_p < alpha
            }
        except Exception as e:
            logger.warning(f"ADF test failed: {e}")
            results['adf'] = {'is_stationary': False}
        try:
            kpss_stat, kpss_p, _, kpss_crit = kpss(series.dropna(), regression='c')
            results['kpss'] = {
                'statistic': kpss_stat,
                'pvalue': kpss_p,
                'critical_values': kpss_crit,
                'is_stationary': kpss_p > alpha
            }
        except Exception as e:
            logger.warning(f"KPSS test failed: {e}")
            results['kpss'] = {'is_stationary': True}
        adf_stationary = results['adf']['is_stationary']
        kpss_stationary = results['kpss']['is_stationary']
        if adf_stationary and kpss_stationary:
            conclusion = "stationary"
        elif not adf_stationary and not kpss_stationary:
            conclusion = "non_stationary"
        else:
            conclusion = "inconclusive"
        results['conclusion'] = conclusion
        results['is_stationary'] = conclusion == "stationary"
        return results

    @staticmethod
    def detect_optimal_differencing(series: pd.Series, max_d: int = 2) -> Tuple[int, pd.Series]:
        if series.dropna().empty:
            return 0, series
        original_series = series.dropna()
        current_series = original_series.copy()
        for d in range(max_d + 1):
            stationarity = FinancialTimeSeriesAnalyzer.test_stationarity(current_series)
            if stationarity['is_stationary']:
                return d, current_series if d > 0 else original_series
            if d < max_d:
                current_series = current_series.diff().dropna()
                if len(current_series) < 50:
                    logger.warning(f"Series too short after {d+1} differences")
                    return d, current_series
        logger.warning(f"Series not stationary after {max_d} differences")
        return max_d, current_series

    @staticmethod
    def analyze_seasonality(series: pd.Series, freq: int = 252) -> Dict[str, float]:
        if len(series) < 2 * freq:
            return {'seasonal_strength': 0.0, 'trend_strength': 0.0}
        try:
            decomposition = seasonal_decompose(
                series.dropna(),
                model='additive',
                period=min(freq, len(series) // 2)
            )
            seasonal_var = np.var(decomposition.seasonal.dropna())
            residual_var = np.var(decomposition.resid.dropna())
            total_var = np.var(series.dropna())
            seasonal_strength = seasonal_var / (seasonal_var + residual_var) if (seasonal_var + residual_var) > 0 else 0
            trend_strength = 1 - (residual_var / total_var) if total_var > 0 else 0
            return {
                'seasonal_strength': float(seasonal_strength),
                'trend_strength': float(trend_strength)
            }
        except Exception as e:
            logger.warning(f"Seasonality analysis failed: {e}")
            return {'seasonal_strength': 0.0, 'trend_strength': 0.0}


class AdvancedOrderSelection:
    def __init__(self, max_p: int = 5, max_d: int = 2, max_q: int = 5,
                 seasonal_max_P: int = 2, seasonal_max_D: int = 1, seasonal_max_Q: int = 2,
                 m: int = 0, information_criterion: str = 'aic',
                 selection_metric: str = 'aic', candidate_orders: Optional[List[Tuple[int, int, int]]] = None,
                 val_fraction: float = 0.2, selection_maxiter: int = 100):
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.seasonal_max_P = seasonal_max_P
        self.seasonal_max_D = seasonal_max_D
        self.seasonal_max_Q = seasonal_max_Q
        self.m = m
        self.ic = information_criterion.lower()
        self.selection_metric = selection_metric.lower()
        self.candidate_orders = candidate_orders
        self.val_fraction = max(0.05, min(0.5, float(val_fraction)))
        self.selection_maxiter = max(50, int(selection_maxiter))

    def select_order(self, endog: pd.Series, exog: pd.DataFrame = None,
                     use_seasonal: bool = False) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
        logger.info(f"Starting advanced order selection with {CPU_COUNT} cores (metric={self.selection_metric})")
        best_score = np.inf
        best_order = (1, 1, 1)
        best_seasonal_order = (0, 0, 0, 0)
        optimal_d, _ = FinancialTimeSeriesAnalyzer.detect_optimal_differencing(endog, self.max_d)

        # Build candidate orders
        if self.candidate_orders:
            candidate_orders = [(p, optimal_d, q) for (p, _, q) in self.candidate_orders]
        else:
            # Focused small set first, bounded by max_p/max_q
            base_candidates = [(0, optimal_d, 1), (1, optimal_d, 0), (1, optimal_d, 1),
                               (2, optimal_d, 1), (1, optimal_d, 2), (2, optimal_d, 2), (3, optimal_d, 1)]
            candidate_orders = [(p, d, q) for (p, d, q) in base_candidates if p <= self.max_p and q <= self.max_q]
            # Fallback to small grid if too few
            if len(candidate_orders) < 3:
                candidate_orders = []
                for p in range(min(3, self.max_p) + 1):
                    for q in range(min(3, self.max_q) + 1):
                        candidate_orders.append((p, optimal_d, q))

        if use_seasonal and self.m > 1:
            seasonal_orders = [(P, D, Q, self.m) for P in range(min(1, self.seasonal_max_P) + 1)
                               for D in range(min(1, self.seasonal_max_D) + 1)
                               for Q in range(min(1, self.seasonal_max_Q) + 1)]
        else:
            seasonal_orders = [(0, 0, 0, 0)]

        # Validation split for RMSE selection
        n = len(endog)
        n_val = max(50, int(self.val_fraction * n)) if self.selection_metric == 'rmse' else 0
        if self.selection_metric == 'rmse' and n_val < n // 2:
            y_train = endog.iloc[:-n_val]
            y_val = endog.iloc[-n_val:]
            if exog is not None:
                X_train = exog.iloc[:-n_val]
                X_val = exog.iloc[-n_val:]
            else:
                X_train = None
                X_val = None
        else:
            y_train = endog
            y_val = None
            X_train = exog
            X_val = None

        for order in candidate_orders:
            for seasonal_order in seasonal_orders:
                try:
                    if X_train is not None:
                        model = SARIMAX(y_train, exog=X_train, order=order, seasonal_order=seasonal_order,
                                        enforce_stationarity=False, enforce_invertibility=False, trend=None)
                    else:
                        model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order,
                                        enforce_stationarity=False, enforce_invertibility=False)
                    fitted = model.fit(disp=False, maxiter=self.selection_maxiter, method='lbfgs')

                    if self.selection_metric == 'rmse' and y_val is not None and len(y_val) > 0:
                        if X_val is not None:
                            fc = fitted.get_forecast(steps=len(y_val), exog=X_val)
                        else:
                            fc = fitted.get_forecast(steps=len(y_val))
                        preds = fc.predicted_mean.values
                        rmse = float(np.sqrt(np.mean((y_val.values - preds) ** 2)))
                        score = rmse
                    else:
                        if self.ic == 'aic':
                            score = float(fitted.aic)
                        elif self.ic == 'bic':
                            score = float(fitted.bic)
                        elif self.ic == 'hqic':
                            score = float(getattr(fitted, 'hqic', fitted.aic))
                        else:
                            score = float(fitted.aic)

                    if score < best_score:
                        best_score = score
                        best_order = order
                        best_seasonal_order = seasonal_order
                except Exception:
                    continue

        metric_name = 'RMSE' if self.selection_metric == 'rmse' else self.ic.upper()
        try:
            logger.info(f"Order selection complete: ARIMA{best_order} x {best_seasonal_order} ({metric_name}={best_score:.2f})")
        except Exception:
            logger.info(f"Order selection complete: ARIMA{best_order} x {best_seasonal_order}")
        return best_order, best_seasonal_order


class SmartFeatureSelector:
    def __init__(self, max_features: int = 10, selection_method: str = 'mutual_info'):
        self.max_features = max_features
        self.selection_method = selection_method
        self.selected_features_ = None
        self.feature_scores_ = None

    def fit_select(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        if X.empty or len(X.columns) == 0:
            self.selected_features_ = []
            return pd.DataFrame()
        n_features_to_select = min(self.max_features, len(X.columns))
        logger.info(f"Selecting {n_features_to_select} features from {len(X.columns)} using {self.selection_method}")
        X_clean = X.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.fillna(method='ffill').fillna(method='bfill')
        y_clean = y.copy()
        valid_mask = ~(X_clean.isna().any(axis=1) | y_clean.isna())
        X_clean = X_clean[valid_mask]
        y_clean = y_clean[valid_mask]
        if len(X_clean) < 50:
            logger.warning("Insufficient clean data for feature selection")
            self.selected_features_ = list(X.columns)[:n_features_to_select]
            return X[self.selected_features_]
        try:
            if self.selection_method == 'mutual_info':
                selector = SelectKBest(mutual_info_regression, k=n_features_to_select)
            elif self.selection_method == 'f_test':
                selector = SelectKBest(f_regression, k=n_features_to_select)
            else:
                correlations = abs(X_clean.corrwith(y_clean))
                self.selected_features_ = correlations.nlargest(n_features_to_select).index.tolist()
                self.feature_scores_ = correlations[self.selected_features_].to_dict()
                return X[self.selected_features_]
            selector.fit(X_clean, y_clean)
            self.selected_features_ = X_clean.columns[selector.get_support()].tolist()
            try:
                scores = selector.scores_
                self.feature_scores_ = dict(zip(self.selected_features_, np.asarray(scores)[selector.get_support()].tolist()))
            except Exception:
                self.feature_scores_ = None
            return X[self.selected_features_]
        except Exception as e:
            logger.warning(f"Feature selection failed: {e}, using all features")
            self.selected_features_ = list(X.columns)[:n_features_to_select]
            return X[self.selected_features_]


class ARIMAModel(BaseModel):
    """Modern ARIMAX model with advanced financial time series capabilities."""

    def __init__(self, task: str = 'regression', **kwargs):
        super().__init__(name="ARIMA", task=task, **kwargs)
        self.max_p = kwargs.get('max_p', 5)
        self.max_d = kwargs.get('max_d', 2)
        self.max_q = kwargs.get('max_q', 5)
        self.information_criterion = kwargs.get('information_criterion', 'aic')
        self.selection_metric = kwargs.get('selection_metric', 'rmse')
        self.selection_val_fraction = kwargs.get('selection_val_fraction', 0.2)
        self.selection_maxiter = kwargs.get('selection_maxiter', 100)
        self.candidate_orders = kwargs.get('candidate_orders', None)
        self.max_features = kwargs.get('max_features', 10)
        self.feature_selection_method = kwargs.get('feature_selection_method', 'mutual_info')
        self.use_exog = kwargs.get('use_exog', True)
        self.auto_order_selection = kwargs.get('auto_order_selection', True)
        self.seasonal_period = kwargs.get('seasonal_period', 0)
        self.use_trend = kwargs.get('use_trend', True)
        self.order_ = None
        self.seasonal_order_ = None
        self.fitted_model_ = None
        self.feature_selector_ = None
        self.scaler_ = None
        self.diagnostics_ = None
        self.selected_features_ = None
        self.endog_train_ = None
        self.exog_train_ = None
        logger.info(f"Modern ARIMA initialized for {CPU_COUNT}-core optimization")

    def build_model(self, input_shape: Tuple[int, ...]) -> 'ARIMAModel':
        self.input_shape = input_shape
        if self.use_exog:
            self.feature_selector_ = SmartFeatureSelector(
                max_features=self.max_features,
                selection_method=self.feature_selection_method
            )
        self.scaler_ = RobustScaler()
        self.model = "ModernARIMA_READY"
        return self

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], **kwargs) -> 'ARIMAModel':
        logger.info("Starting Modern ARIMA training")
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
        min_len = min(len(X), len(y))
        X = X.iloc[:min_len].copy()
        y = y.iloc[:min_len].copy()
        self.endog_train_ = y.copy()
        exog_processed = None
        if self.use_exog and not X.empty:
            if self.feature_selector_ is None:
                self.build_model(X.shape)
            X_selected = self.feature_selector_.fit_select(X, y)
            self.selected_features_ = self.feature_selector_.selected_features_
            if not X_selected.empty:
                # Centralized scaling compatibility: use pre-scaled inputs if flagged
                if bool(getattr(self, 'expects_scaled_input', False)):
                    exog_processed = X_selected.copy()
                else:
                    X_scaled = pd.DataFrame(
                        self.scaler_.fit_transform(X_selected),
                        columns=X_selected.columns,
                        index=X_selected.index
                    )
                    exog_processed = X_scaled
                self.exog_train_ = exog_processed.copy()
            else:
                self.use_exog = False
        if self.auto_order_selection:
            order_selector = AdvancedOrderSelection(
                max_p=self.max_p,
                max_d=self.max_d,
                max_q=self.max_q,
                seasonal_max_P=2,
                seasonal_max_D=1,
                seasonal_max_Q=2,
                m=self.seasonal_period,
                information_criterion=self.information_criterion,
                selection_metric=self.selection_metric,
                candidate_orders=self.candidate_orders,
                val_fraction=self.selection_val_fraction,
                selection_maxiter=self.selection_maxiter
            )
            self.order_, self.seasonal_order_ = order_selector.select_order(
                endog=y,
                exog=exog_processed,
                use_seasonal=(self.seasonal_period > 1)
            )
        else:
            self.order_ = kwargs.get('order', (1, 1, 1))
            self.seasonal_order_ = kwargs.get('seasonal_order', (0, 0, 0, 0))
        logger.info(f"Fitting final ARIMAX{self.order_} x {self.seasonal_order_}")
        try:
            if exog_processed is not None:
                model = SARIMAX(
                    endog=y,
                    exog=exog_processed,
                    order=self.order_,
                    seasonal_order=self.seasonal_order_,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    trend='c' if self.use_trend else None
                )
            else:
                model = SARIMAX(
                    endog=y,
                    order=self.order_,
                    seasonal_order=self.seasonal_order_,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    trend='c' if self.use_trend else None
                )
            self.fitted_model_ = model.fit(disp=False, maxiter=500, method='lbfgs')
            self._compute_diagnostics()
            self.is_trained = True
            return self
        except Exception as e:
            logger.error(f"Modern ARIMA training failed: {e}")
            raise

    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], **kwargs) -> Dict[str, Any]:
        self.fit(X, y, **kwargs)
        result = {
            'status': 'success',
            'order': self.order_,
            'seasonal_order': self.seasonal_order_,
            'selected_features': self.selected_features_,
            'n_exog_features': len(self.selected_features_) if self.selected_features_ else 0
        }
        if self.diagnostics_:
            result.update({
                'aic': self.diagnostics_.aic,
                'bic': self.diagnostics_.bic,
                'ljung_box_pvalue': self.diagnostics_.ljung_box_pvalue
            })
        return result

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        steps = 1
        exog_forecast = None
        if self.use_exog and X is not None:
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
            if self.selected_features_:
                steps = len(X)
                X_selected = X[self.selected_features_]
                # Centralized scaling compatibility: pass-through if pre-scaled
                if bool(getattr(self, 'expects_scaled_input', False)):
                    exog_forecast = X_selected.copy()
                else:
                    exog_forecast = pd.DataFrame(
                        self.scaler_.transform(X_selected),
                        columns=X_selected.columns,
                        index=X_selected.index
                    )
        else:
            try:
                steps = len(X)
            except Exception:
                steps = 1
        try:
            if exog_forecast is not None and steps > 0:
                forecast_result = self.fitted_model_.get_forecast(steps=min(steps, len(exog_forecast)), exog=exog_forecast.iloc[:steps])
            else:
                forecast_result = self.fitted_model_.get_forecast(steps=steps)
            return forecast_result.predicted_mean.values
        except Exception as e:
            logger.error(f"ARIMA prediction failed: {e}")
            if hasattr(self.fitted_model_, 'fittedvalues') and len(self.fitted_model_.fittedvalues) > 0:
                last_value = self.fitted_model_.fittedvalues.iloc[-1]
                return np.full(steps, last_value)
            return np.zeros(steps)

    def evaluate(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
        predictions = self.predict(X)
        min_len = min(len(y), len(predictions))
        y_eval = y.iloc[:min_len]
        pred_eval = predictions[:min_len]
        mse = mean_squared_error(y_eval, pred_eval)
        mae = mean_absolute_error(y_eval, pred_eval)
        r2 = r2_score(y_eval, pred_eval)
        try:
            y_direction = np.sign(y_eval.diff().dropna())
            pred_direction = np.sign(pd.Series(pred_eval).diff().dropna())
            directional_accuracy = float(np.mean(y_direction.values == pred_direction.values))
        except Exception:
            directional_accuracy = float('nan')
        try:
            mape = float(np.mean(np.abs((y_eval - pred_eval) / np.maximum(np.abs(y_eval), 1e-8))) * 100)
        except Exception:
            mape = float('nan')
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'rmse': np.sqrt(mse)
        }

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if not self.is_trained or not self.selected_features_:
            return None
        try:
            params = self.fitted_model_.params
            feature_importance: Dict[str, float] = {}
            for i, feature in enumerate(self.selected_features_):
                param_name = f'x{i+1}'
                if hasattr(params, 'index') and param_name in params.index:
                    feature_importance[feature] = abs(params[param_name])
                elif hasattr(params, 'index') and feature in params.index:
                    feature_importance[feature] = abs(params[feature])
            if hasattr(self.feature_selector_, 'feature_scores_') and self.feature_selector_.feature_scores_:
                for feature in self.selected_features_:
                    if feature in self.feature_selector_.feature_scores_:
                        coef_importance = feature_importance.get(feature, 0)
                        selection_score = self.feature_selector_.feature_scores_[feature]
                        feature_importance[feature] = 0.7 * coef_importance + 0.3 * abs(selection_score)
            if feature_importance:
                return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        except Exception as e:
            logger.warning(f"Failed to compute feature importance: {e}")
        return None

    def forecast_with_uncertainty(self, X: pd.DataFrame = None, steps: int = 5, confidence_level: float = 0.95) -> Dict[str, np.ndarray]:
        if not self.is_trained:
            raise ValueError("Model must be trained before forecasting")
        if X is not None:
            preds, lower, upper = self._predict_with_conf_int(X, steps)
        else:
            preds, lower, upper = self._predict_with_conf_int(None, steps)
        from scipy.stats import norm
        forecast_std = (upper - lower) / (2 * norm.ppf((1 + confidence_level) / 2))
        return {
            'forecast': preds,
            'lower_bound': lower,
            'upper_bound': upper,
            'forecast_std': forecast_std,
            'confidence_level': confidence_level
        }

    def _predict_with_conf_int(self, X: Union[pd.DataFrame, np.ndarray], steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.use_exog and X is not None:
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
            X_selected = X[self.selected_features_] if self.selected_features_ else pd.DataFrame()
            exog_forecast = pd.DataFrame(self.scaler_.transform(X_selected), columns=X_selected.columns, index=X_selected.index) if not X_selected.empty else None
        else:
            exog_forecast = None
        if exog_forecast is not None:
            forecast_result = self.fitted_model_.get_forecast(steps=min(steps, len(exog_forecast)), exog=exog_forecast.iloc[:steps])
        else:
            forecast_result = self.fitted_model_.get_forecast(steps=steps)
        predictions = forecast_result.predicted_mean.values
        conf_int = forecast_result.conf_int()
        lower = conf_int.iloc[:, 0].values
        upper = conf_int.iloc[:, 1].values
        return predictions, lower, upper

    def _compute_diagnostics(self) -> None:
        if not self.fitted_model_:
            return
        try:
            residuals = self.fitted_model_.resid
            aic = self.fitted_model_.aic
            bic = self.fitted_model_.bic
            try:
                lb_result = acorr_ljungbox(residuals, lags=min(10, max(1, len(residuals)//5)), return_df=True)
                ljung_box_p = float(lb_result['lb_pvalue'].iloc[-1])
            except Exception:
                ljung_box_p = float('nan')
            try:
                jb_stat, jb_p = stats.jarque_bera(residuals.dropna())
            except Exception:
                jb_p = float('nan')
            try:
                het_p = float('nan')
                if hasattr(self.fitted_model_.model, 'exog') and self.fitted_model_.model.exog is not None:
                    from statsmodels.stats.diagnostic import het_breuschpagan
                    bp_stat, bp_p, _, _ = het_breuschpagan(residuals, self.fitted_model_.model.exog)
                    het_p = float(bp_p)
            except Exception:
                het_p = float('nan')
            try:
                acf_values = acf(residuals.dropna(), nlags=min(5, len(residuals)-2), fft=True)
                autocorr_1 = float(acf_values[1]) if len(acf_values) > 1 else float('nan')
                autocorr_5 = float(acf_values[5]) if len(acf_values) > 5 else float('nan')
            except Exception:
                autocorr_1 = float('nan')
                autocorr_5 = float('nan')
            try:
                from statsmodels.stats.stattools import durbin_watson
                dw_stat = float(durbin_watson(residuals.dropna()))
            except Exception:
                dw_stat = float('nan')
            try:
                stationarity = FinancialTimeSeriesAnalyzer.test_stationarity(residuals)
                is_stationary = bool(stationarity['is_stationary'])
            except Exception:
                is_stationary = False
            try:
                seasonality = FinancialTimeSeriesAnalyzer.analyze_seasonality(self.endog_train_)
                seasonal_strength = float(seasonality['seasonal_strength'])
                trend_strength = float(seasonality['trend_strength'])
            except Exception:
                seasonal_strength = 0.0
                trend_strength = 0.0
            self.diagnostics_ = ARIMADiagnostics(
                aic=aic,
                bic=bic,
                hqic=getattr(self.fitted_model_, 'hqic', np.nan),
                ljung_box_pvalue=ljung_box_p,
                jarque_bera_pvalue=jb_p,
                heteroscedasticity_pvalue=het_p,
                residual_autocorr_1=autocorr_1,
                residual_autocorr_5=autocorr_5,
                durbin_watson=dw_stat,
                is_stationary=is_stationary,
                seasonal_strength=seasonal_strength,
                trend_strength=trend_strength
            )
        except Exception as e:
            logger.warning(f"Diagnostic computation failed: {e}")
            self.diagnostics_ = None

    def get_model_summary(self) -> str:
        if not self.is_trained:
            return "Model not trained"
        lines: List[str] = [
            "Modern ARIMAX Model Summary",
            "========================================",
            f"Order: ARIMA{self.order_} x {self.seasonal_order_}",
            f"Exogenous features: {len(self.selected_features_) if self.selected_features_ else 0}",
        ]
        if self.selected_features_:
            lines.append(f"Selected features: {', '.join(self.selected_features_[:5])}")
            if len(self.selected_features_) > 5:
                lines.append(f"... and {len(self.selected_features_) - 5} more")
        if self.diagnostics_:
            lines.extend([
                "",
                "Model Diagnostics:",
                f"AIC: {self.diagnostics_.aic:.2f}",
                f"BIC: {self.diagnostics_.bic:.2f}",
                f"Ljung-Box p-value: {self.diagnostics_.ljung_box_pvalue:.4f}",
                f"Residuals stationary: {self.diagnostics_.is_stationary}",
            ])
        try:
            lines.extend(["", "Statistical Summary:", str(self.fitted_model_.summary())])
        except Exception:
            pass
        return "\n".join(lines)
