"""
Walk-forward validation module for RiskPipeline.

This module provides robust walk-forward cross-validation for time series data,
including adaptive sizing, validation checks, and comprehensive logging.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any, Iterator
from dataclasses import dataclass
from pathlib import Path
import warnings

from ..utils.logging_utils import log_execution_time

logger = logging.getLogger(__name__)

@dataclass
class ValidationConfig:
    """Configuration for walk-forward validation."""
    
    n_splits: int = 5
    test_size: int = 252
    min_train_size: int = 60
    min_test_size: int = 20
    gap: int = 0  # Gap between train and test sets
    expanding_window: bool = True  # True for expanding, False for sliding
    
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
    """Implements walk-forward cross-validation with dynamic sizing and validation."""
    
    def __init__(self, n_splits: int = 5, test_size: int = 252, 
                 min_train_size: int = 60, min_test_size: int = 20,
                 gap: int = 0, expanding_window: bool = True):
        """
        Initialize WalkForwardValidator.
        
        Args:
            n_splits: Number of splits to generate
            test_size: Size of test set (will be adapted if necessary)
            min_train_size: Minimum size for training set
            min_test_size: Minimum size for test set
            gap: Gap between train and test sets
            expanding_window: Whether to use expanding or sliding window
        """
        self.config = ValidationConfig(
            n_splits=n_splits,
            test_size=test_size,
            min_train_size=min_train_size,
            min_test_size=min_test_size,
            gap=gap,
            expanding_window=expanding_window
        )
        
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
        
        # Calculate adaptive parameters
        adaptive_test_size = min(self.config.test_size, len(X) // 4)
        max_splits = min(self.config.n_splits, (len(X) - adaptive_test_size) // adaptive_test_size)
        
        self.logger.info(f"Adaptive test_size: {adaptive_test_size} (requested: {self.config.test_size})")
        self.logger.info(f"Maximum possible splits: {max_splits} (requested: {self.config.n_splits})")
        
        # Generate splits
        splits = self._generate_expanding_splits(X, max_splits, adaptive_test_size)
        
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
                       model_type: str) -> Dict[str, Any]:
        """Train/evaluate a model across splits and return aggregated metrics.

        Expects model implements train, predict, evaluate compatible with BaseModel.
        """
        try:
            all_predictions: List[float] = []
            all_actuals: List[float] = []

            # Ensure inputs are aligned DataFrame/Series
            X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
            y_ser = y if isinstance(y, pd.Series) else pd.Series(y, index=X_df.index[:len(y)])

            for train_idx, test_idx in splits:
                X_train, X_test = X_df.loc[train_idx], X_df.loc[test_idx]
                y_train, y_test = y_ser.loc[train_idx], y_ser.loc[test_idx]

                # Train
                model.train(X_train, y_train)

                # Predict
                y_pred = model.predict(X_test)

                # Accumulate
                all_predictions.extend(list(np.asarray(y_pred).ravel()))
                all_actuals.extend(list(np.asarray(y_test).ravel()))

            # Evaluate overall using the last trained model on a subset of data
            # Use the last test set for final evaluation
            if splits:
                last_train_idx, last_test_idx = splits[-1]
                X_final_test = X_df.loc[last_test_idx]
                y_final_test = y_ser.loc[last_test_idx]
                
                try:
                    metrics = model.evaluate(X_final_test, y_final_test)
                except Exception as eval_error:
                    logger.warning(f"Model evaluation failed, using fallback metrics: {eval_error}")
                    # Fallback: calculate basic metrics manually
                    if len(all_predictions) > 0 and len(all_actuals) > 0:
                        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                        metrics = {
                            'MSE': mean_squared_error(all_actuals, all_predictions),
                            'MAE': mean_absolute_error(all_actuals, all_predictions),
                            'R2': r2_score(all_actuals, all_predictions)
                        }
                    else:
                        metrics = {'error': 'No valid predictions for evaluation'}
            else:
                metrics = {'error': 'No valid splits for evaluation'}

            return {
                'metrics': metrics,
                'predictions': all_predictions,
                'actuals': all_actuals,
                'config': {
                    'n_splits': self.config.n_splits,
                    'test_size': self.config.test_size,
                }
            }
        except Exception as e:
            logger.error(f"Model evaluation failed for {asset}_{model_type}: {e}")
            return {'error': str(e), 'metrics': {}, 'predictions': [], 'actuals': []}
    
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
    
    def validate_data_quality(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Validate data quality for walk-forward validation."""
        quality_report = {
            'n_samples': len(X),
            'n_features': len(X.columns),
            'missing_values': X.isna().sum().sum(),
            'duplicate_indices': X.index.duplicated().sum(),
            'index_type': str(type(X.index)),
            'is_sorted': X.index.is_monotonic_increasing,
            'date_range': {
                'start': X.index.min() if isinstance(X.index, pd.DatetimeIndex) else None,
                'end': X.index.max() if isinstance(X.index, pd.DatetimeIndex) else None
            }
        }
        
        if y is not None:
            quality_report.update({
                'target_missing': y.isna().sum(),
                'target_unique': y.nunique(),
                'target_distribution': y.value_counts().to_dict() if y.dtype == 'object' else None
            })
        
        # Check for potential issues
        issues = []
        if quality_report['missing_values'] > 0:
            issues.append(f"Found {quality_report['missing_values']} missing values")
        
        if quality_report['duplicate_indices'] > 0:
            issues.append(f"Found {quality_report['duplicate_indices']} duplicate indices")
        
        if not quality_report['is_sorted']:
            issues.append("Data index is not sorted")
        
        if not isinstance(X.index, pd.DatetimeIndex):
            issues.append("Data index is not datetime")
        
        quality_report['issues'] = issues
        quality_report['is_valid'] = len(issues) == 0
        
        return quality_report
    
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
        quality_report = self.validate_data_quality(X, y)
        
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
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Split visualization saved to {save_path}")
            
            plt.show()
            
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