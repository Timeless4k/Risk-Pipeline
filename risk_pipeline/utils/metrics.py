"""
Metrics calculation utilities for RiskPipeline.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import logging

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """Utility class for calculating various performance metrics."""
    
    @staticmethod
    def calculate_regression_metrics(y_true: Union[np.ndarray, pd.Series], 
                                   y_pred: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        """
        Calculate regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        try:
            # Convert to numpy arrays if needed
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            
            # Remove any NaN values
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]
            
            if len(y_true_clean) == 0:
                logger.warning("No valid data for metric calculation")
                return {}
            
            metrics = {
                'mse': mean_squared_error(y_true_clean, y_pred_clean),
                'rmse': np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)),
                'mae': mean_absolute_error(y_true_clean, y_pred_clean),
                'r2': r2_score(y_true_clean, y_pred_clean),
                'mape': np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating regression metrics: {str(e)}")
            return {}
    
    @staticmethod
    def calculate_classification_metrics(y_true: Union[np.ndarray, pd.Series], 
                                       y_pred: Union[np.ndarray, pd.Series],
                                       average: str = 'weighted') -> Dict[str, float]:
        """
        Calculate classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging method for multi-class metrics
            
        Returns:
            Dictionary of metrics
        """
        try:
            # Convert to numpy arrays if needed
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            
            # Remove any NaN values
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]
            
            if len(y_true_clean) == 0:
                logger.warning("No valid data for metric calculation")
                return {}
            
            metrics = {
                'accuracy': accuracy_score(y_true_clean, y_pred_clean),
                'precision': precision_score(y_true_clean, y_pred_clean, average=average, zero_division=0),
                'recall': recall_score(y_true_clean, y_pred_clean, average=average, zero_division=0),
                'f1': f1_score(y_true_clean, y_pred_clean, average=average, zero_division=0)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating classification metrics: {str(e)}")
            return {}
    
    @staticmethod
    def calculate_time_series_metrics(y_true: Union[np.ndarray, pd.Series], 
                                    y_pred: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        """
        Calculate time series specific metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        try:
            # Convert to numpy arrays if needed
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            
            # Remove any NaN values
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]
            
            if len(y_true_clean) == 0:
                logger.warning("No valid data for metric calculation")
                return {}
            
            # Calculate directional accuracy
            direction_true = np.diff(y_true_clean) > 0
            direction_pred = np.diff(y_pred_clean) > 0
            directional_accuracy = np.mean(direction_true == direction_pred)
            
            # Calculate Theil's U statistic
            numerator = np.sqrt(np.mean((y_true_clean - y_pred_clean) ** 2))
            denominator = np.sqrt(np.mean(y_true_clean ** 2)) + np.sqrt(np.mean(y_pred_clean ** 2))
            theil_u = numerator / denominator if denominator > 0 else 0
            
            metrics = {
                'directional_accuracy': directional_accuracy,
                'theil_u': theil_u,
                'mse': mean_squared_error(y_true_clean, y_pred_clean),
                'rmse': np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)),
                'mae': mean_absolute_error(y_true_clean, y_pred_clean)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating time series metrics: {str(e)}")
            return {}
    
    @staticmethod
    def calculate_volatility_metrics(y_true: Union[np.ndarray, pd.Series], 
                                   y_pred: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        """
        Calculate volatility forecasting specific metrics.
        
        Args:
            y_true: True volatility values
            y_pred: Predicted volatility values
            
        Returns:
            Dictionary of metrics
        """
        try:
            # Convert to numpy arrays if needed
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            
            # Remove any NaN values
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]
            
            if len(y_true_clean) == 0:
                logger.warning("No valid data for metric calculation")
                return {}
            
            # Calculate Mincer-Zarnowitz regression metrics
            # Regress actual on predicted: actual = a + b*predicted + error
            if len(y_pred_clean) > 1:
                # Add constant term for regression
                X = np.column_stack([np.ones(len(y_pred_clean)), y_pred_clean])
                try:
                    beta = np.linalg.lstsq(X, y_true_clean, rcond=None)[0]
                    a, b = beta[0], beta[1]
                    
                    # Calculate R-squared of MZ regression
                    y_pred_mz = a + b * y_pred_clean
                    r2_mz = r2_score(y_true_clean, y_pred_mz)
                    
                    # Test if a=0 and b=1 (unbiased forecast)
                    unbiased_test = np.abs(a) < 0.01 and np.abs(b - 1) < 0.01
                except:
                    r2_mz = 0
                    unbiased_test = False
            else:
                r2_mz = 0
                unbiased_test = False
            
            metrics = {
                'mse': mean_squared_error(y_true_clean, y_pred_clean),
                'rmse': np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)),
                'mae': mean_absolute_error(y_true_clean, y_pred_clean),
                'r2': r2_score(y_true_clean, y_pred_clean),
                'r2_mz': r2_mz,
                'unbiased_forecast': unbiased_test
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating volatility metrics: {str(e)}")
            return {}
    
    @staticmethod
    def aggregate_metrics(metrics_list: List[Dict[str, float]], 
                         method: str = 'mean') -> Dict[str, float]:
        """
        Aggregate metrics across multiple folds or periods.
        
        Args:
            metrics_list: List of metric dictionaries
            method: Aggregation method ('mean', 'median', 'std')
            
        Returns:
            Aggregated metrics
        """
        if not metrics_list:
            return {}
        
        # Get all unique metric names
        all_metrics = set()
        for metrics in metrics_list:
            all_metrics.update(metrics.keys())
        
        aggregated = {}
        
        for metric in all_metrics:
            values = [metrics.get(metric, np.nan) for metrics in metrics_list]
            values = [v for v in values if not np.isnan(v)]
            
            if values:
                if method == 'mean':
                    aggregated[metric] = np.mean(values)
                elif method == 'median':
                    aggregated[metric] = np.median(values)
                elif method == 'std':
                    aggregated[metric] = np.std(values)
                elif method == 'min':
                    aggregated[metric] = np.min(values)
                elif method == 'max':
                    aggregated[metric] = np.max(values)
        
        return aggregated
    
    @staticmethod
    def format_metrics(metrics: Dict[str, float], 
                      precision: int = 4) -> Dict[str, str]:
        """
        Format metrics for display.
        
        Args:
            metrics: Dictionary of metrics
            precision: Number of decimal places
            
        Returns:
            Formatted metrics
        """
        formatted = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                formatted[key] = f"{value:.{precision}f}"
            else:
                formatted[key] = str(value)
        
        return formatted 