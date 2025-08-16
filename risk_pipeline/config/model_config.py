"""
Model configuration for RiskPipeline.

This module provides optimized configurations for different model types
specifically tuned for financial time series data.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class FinancialModelConfig:
    """Configuration for financial models with optimized parameters."""
    
    # General training parameters
    validation_split: float = 0.2
    random_state: int = 42
    
    # Early stopping and regularization
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    min_lr: float = 1e-7
    
    # Data preprocessing
    sequence_length: int = 20  # Optimized for financial data
    feature_window: int = 60   # 60 days for feature calculation
    target_horizon: int = 5    # 5-day ahead prediction
    
    # Walk-forward validation
    min_train_size: int = 252  # 1 year of trading days
    test_size: int = 63        # 3 months
    gap: int = 5               # 5-day gap to prevent data leakage
    expanding_window: bool = True

@dataclass
class LSTMConfig(FinancialModelConfig):
    """LSTM-specific configuration for financial time series."""
    
    # Architecture
    units: list = field(default_factory=lambda: [64, 32])  # Reduced complexity
    dropout: float = 0.3  # Increased dropout for regularization
    recurrent_dropout: float = 0.2
    
    # Training
    batch_size: int = 32  # Larger batch size for stability
    epochs: int = 150
    learning_rate: float = 0.001
    
    # Financial-specific
    sequence_length: int = 30  # 30 days of history
    return_sequences: bool = False  # Single output for regression
    
    # Regularization
    l2_reg: float = 1e-4
    gradient_clip: float = 1.0

@dataclass
class StockMixerConfig(FinancialModelConfig):
    """StockMixer-specific configuration for financial data."""
    
    # Architecture
    temporal_units: int = 32  # Reduced from 64
    indicator_units: int = 32
    cross_stock_units: int = 32
    fusion_units: int = 64    # Reduced from 128
    
    # Training
    batch_size: int = 64      # Larger batch for tabular data
    epochs: int = 100
    learning_rate: float = 0.001
    
    # Regularization
    dropout: float = 0.4      # Higher dropout for tabular data
    l2_reg: float = 1e-3
    batch_norm_momentum: float = 0.9

@dataclass
class XGBoostConfig(FinancialModelConfig):
    """XGBoost-specific configuration for financial data."""
    
    # Tree parameters
    n_estimators: int = 200
    max_depth: int = 4        # Reduced to prevent overfitting
    learning_rate: float = 0.05  # Lower learning rate
    
    # Regularization
    reg_alpha: float = 0.2    # L1 regularization
    reg_lambda: float = 1.5   # L2 regularization
    subsample: float = 0.8    # Row sampling
    colsample_bytree: float = 0.8  # Column sampling
    min_child_weight: int = 5
    gamma: float = 0.2
    
    # Financial-specific
    tree_method: str = 'hist'  # CPU-optimized method
    max_bin: int = 256        # Reduced for memory efficiency
    
    # Cross-validation
    cv_folds: int = 5
    use_time_series_cv: bool = True

@dataclass
class ARIMAConfig(FinancialModelConfig):
    """ARIMA-specific configuration for financial time series."""
    
    # Model order
    max_p: int = 5
    max_d: int = 2
    max_q: int = 5
    
    # Seasonal parameters
    seasonal: bool = True
    m: int = 5  # 5-day seasonality for weekly patterns
    
    # Information criteria
    ic: str = 'aic'  # Akaike Information Criterion
    trend: str = 'c'  # Constant trend
    
    # Financial-specific
    enforce_stationarity: bool = True
    enforce_invertibility: bool = True

@dataclass
class FeatureConfig(FinancialModelConfig):
    """Feature engineering configuration for financial data."""
    
    # Technical indicators
    use_technical_indicators: bool = True
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    
    # Volatility features
    use_volatility_features: bool = True
    volatility_window: int = 20
    garch_order: tuple = (1, 1)
    
    # Time features
    use_time_features: bool = True
    use_cyclical_features: bool = True
    
    # Cross-asset features
    use_cross_asset_features: bool = True
    correlation_window: int = 60
    
    # Feature selection
    feature_selection_method: str = 'mutual_info'
    max_features: Optional[int] = None
    feature_correlation_threshold: float = 0.95

@dataclass
class ValidationConfig(FinancialModelConfig):
    """Walk-forward validation configuration."""
    
    # Split configuration - Increased for better statistical significance
    n_splits: int = 8          # Reduced from 10 to ensure larger test sets
    min_train_size: int = 504  # 2 years (increased from 1 year)
    test_size: int = 126       # 6 months (increased from 3 months)
    gap: int = 10              # 10-day gap (increased to prevent overlap)
    
    # Validation strategy
    expanding_window: bool = True
    use_rolling_window: bool = False
    rolling_window_size: int = 252
    
    # Performance metrics
    primary_metric: str = 'r2'
    secondary_metrics: list = field(default_factory=lambda: ['rmse', 'mae', 'mape'])
    
    # Statistical significance
    use_statistical_tests: bool = True
    confidence_level: float = 0.95

def get_optimized_config(model_type: str, task: str = 'regression') -> Dict[str, Any]:
    """
    Get optimized configuration for a specific model type and task.
    
    Args:
        model_type: Type of model ('lstm', 'stockmixer', 'xgboost', 'arima')
        task: Task type ('regression', 'classification')
        
    Returns:
        Configuration dictionary
    """
    base_config = FinancialModelConfig()
    
    if model_type == 'lstm':
        config = LSTMConfig()
    elif model_type == 'stockmixer':
        config = StockMixerConfig()
    elif model_type == 'xgboost':
        config = XGBoostConfig()
    elif model_type == 'arima':
        config = ARIMAConfig()
    else:
        config = base_config
    
    # Task-specific adjustments
    if task == 'classification':
        config.primary_metric = 'accuracy'
        config.secondary_metrics = ['f1', 'precision', 'recall']
        
        # Adjust model-specific parameters for classification
        if model_type == 'lstm':
            config.return_sequences = False
            config.units = [32, 16]  # Smaller for classification
        elif model_type == 'xgboost':
            config.eval_metric = 'mlogloss'
            config.max_depth = 3  # Even more conservative
    
    return config.__dict__

def get_feature_config() -> Dict[str, Any]:
    """Get optimized feature engineering configuration."""
    config = FeatureConfig()
    return config.__dict__

def get_validation_config() -> Dict[str, Any]:
    """Get optimized validation configuration."""
    config = ValidationConfig()
    return config.__dict__

# Pre-configured model configurations for different use cases
CONSERVATIVE_CONFIG = {
    'lstm': {
        'units': [32, 16],
        'dropout': 0.4,
        'batch_size': 16,
        'epochs': 100,
        'learning_rate': 0.0005
    },
    'xgboost': {
        'max_depth': 3,
        'learning_rate': 0.03,
        'n_estimators': 150,
        'reg_alpha': 0.3,
        'reg_lambda': 2.0
    }
}

AGGRESSIVE_CONFIG = {
    'lstm': {
        'units': [128, 64, 32],
        'dropout': 0.2,
        'batch_size': 64,
        'epochs': 200,
        'learning_rate': 0.002
    },
    'xgboost': {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 300,
        'reg_alpha': 0.05,
        'reg_lambda': 0.5
    }
}

BALANCED_CONFIG = {
    'lstm': {
        'units': [64, 32],
        'dropout': 0.3,
        'batch_size': 32,
        'epochs': 150,
        'learning_rate': 0.001
    },
    'xgboost': {
        'max_depth': 4,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'reg_alpha': 0.2,
        'reg_lambda': 1.5
    }
}
