"""
Configuration management for RiskPipeline with dependency injection support.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    start_date: str = '2000-01-01'
    end_date: str = '2025-01-05'
    us_assets: List[str] = field(default_factory=lambda: ['AAPL', 'MSFT', '^GSPC'])
    au_assets: List[str] = field(default_factory=lambda: ['IOZ.AX', 'CBA.AX', 'BHP.AX'])
    cache_dir: str = 'data_cache'
    
    @property
    def all_assets(self) -> List[str]:
        """Get all configured assets."""
        return self.us_assets + self.au_assets


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    volatility_window: int = 5
    ma_short: int = 10
    ma_long: int = 50
    correlation_window: int = 30
    sequence_length: int = 7


@dataclass
class ModelConfig:
    """Configuration for model parameters."""
    lstm_units: List[int] = field(default_factory=lambda: [50, 30])
    lstm_dropout: float = 0.2
    stockmixer_temporal_units: int = 64
    stockmixer_indicator_units: int = 64
    stockmixer_cross_stock_units: int = 64
    stockmixer_fusion_units: int = 128
    xgboost_n_estimators: int = 100
    xgboost_max_depth: int = 5


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    walk_forward_splits: int = 5
    test_size: int = 63
    batch_size: int = 64
    epochs: int = 100
    early_stopping_patience: int = 20
    reduce_lr_patience: int = 10
    random_state: int = 42


@dataclass
class OutputConfig:
    """Configuration for output directories."""
    results_dir: str = 'results'
    plots_dir: str = 'visualizations'
    shap_dir: str = 'shap_plots'
    models_dir: str = 'models'
    log_dir: str = 'logs'


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: int = logging.INFO
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format: str = '%Y-%m-%d %H:%M:%S'


@dataclass
class SHAPConfig:
    """Configuration for SHAP analysis."""
    background_samples: int = 100
    max_display: int = 20
    plot_type: str = 'bar'  # 'bar', 'waterfall', 'beeswarm', 'heatmap'
    save_plots: bool = True


class PipelineConfig:
    """
    Main configuration class for RiskPipeline with dependency injection support.
    
    This class manages all configuration settings and provides a centralized
    way to access configuration throughout the pipeline.
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration from dictionary or load from file.
        
        Args:
            config_dict: Configuration dictionary. If None, loads from default file.
        """
        if config_dict:
            self._load_from_dict(config_dict)
        else:
            self._load_default_config()
    
    def _load_default_config(self):
        """Load configuration from default config file."""
        default_config_path = Path('configs/pipeline_config.json')
        
        if default_config_path.exists():
            with open(default_config_path, 'r') as f:
                config_dict = json.load(f)
            self._load_from_dict(config_dict)
        else:
            # Use hardcoded defaults
            self._load_from_dict({})
    
    def _load_from_dict(self, config_dict: Dict[str, Any]):
        """Load configuration from dictionary."""
        # Data configuration
        data_config = config_dict.get('data', {})
        self.data = DataConfig(
            start_date=data_config.get('start_date', '2000-01-01'),
            end_date=data_config.get('end_date', '2025-01-05'),
            us_assets=data_config.get('us_assets', ['AAPL', 'MSFT', '^GSPC']),
            au_assets=data_config.get('au_assets', ['IOZ.AX', 'CBA.AX', 'BHP.AX']),
            cache_dir=data_config.get('cache_dir', 'data_cache')
        )
        
        # Feature configuration
        feature_config = config_dict.get('features', {})
        self.features = FeatureConfig(
            volatility_window=feature_config.get('volatility_window', 5),
            ma_short=feature_config.get('ma_short', 10),
            ma_long=feature_config.get('ma_long', 50),
            correlation_window=feature_config.get('correlation_window', 30),
            sequence_length=feature_config.get('sequence_length', 7)
        )
        
        # Model configuration
        model_config = config_dict.get('models', {})
        self.models = ModelConfig(
            lstm_units=model_config.get('lstm_units', [50, 30]),
            lstm_dropout=model_config.get('lstm_dropout', 0.2),
            stockmixer_temporal_units=model_config.get('stockmixer_temporal_units', 64),
            stockmixer_indicator_units=model_config.get('stockmixer_indicator_units', 64),
            stockmixer_cross_stock_units=model_config.get('stockmixer_cross_stock_units', 64),
            stockmixer_fusion_units=model_config.get('stockmixer_fusion_units', 128),
            xgboost_n_estimators=model_config.get('xgboost_n_estimators', 100),
            xgboost_max_depth=model_config.get('xgboost_max_depth', 5)
        )
        
        # Training configuration
        training_config = config_dict.get('training', {})
        self.training = TrainingConfig(
            walk_forward_splits=training_config.get('walk_forward_splits', 5),
            test_size=training_config.get('test_size', 63),
            batch_size=training_config.get('batch_size', 64),
            epochs=training_config.get('epochs', 100),
            early_stopping_patience=training_config.get('early_stopping_patience', 20),
            reduce_lr_patience=training_config.get('reduce_lr_patience', 10),
            random_state=training_config.get('random_state', 42)
        )
        
        # Output configuration
        output_config = config_dict.get('output', {})
        self.output = OutputConfig(
            results_dir=output_config.get('results_dir', 'results'),
            plots_dir=output_config.get('plots_dir', 'visualizations'),
            shap_dir=output_config.get('shap_dir', 'shap_plots'),
            models_dir=output_config.get('models_dir', 'models'),
            log_dir=output_config.get('log_dir', 'logs')
        )
        
        # Logging configuration
        logging_config = config_dict.get('logging', {})
        self.logging = LoggingConfig(
            level=getattr(logging, logging_config.get('level', 'INFO')),
            format=logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            date_format=logging_config.get('date_format', '%Y-%m-%d %H:%M:%S')
        )
        
        # SHAP configuration
        shap_config = config_dict.get('shap', {})
        self.shap = SHAPConfig(
            background_samples=shap_config.get('background_samples', 100),
            max_display=shap_config.get('max_display', 20),
            plot_type=shap_config.get('plot_type', 'bar'),
            save_plots=shap_config.get('save_plots', True)
        )
    
    @classmethod
    def from_file(cls, config_path: str) -> 'PipelineConfig':
        """
        Create configuration from file.
        
        Args:
            config_path: Path to configuration file.
            
        Returns:
            PipelineConfig instance
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(config_dict=config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'data': {
                'start_date': self.data.start_date,
                'end_date': self.data.end_date,
                'us_assets': self.data.us_assets,
                'au_assets': self.data.au_assets,
                'cache_dir': self.data.cache_dir
            },
            'features': {
                'volatility_window': self.features.volatility_window,
                'ma_short': self.features.ma_short,
                'ma_long': self.features.ma_long,
                'correlation_window': self.features.correlation_window,
                'sequence_length': self.features.sequence_length
            },
            'models': {
                'lstm_units': self.models.lstm_units,
                'lstm_dropout': self.models.lstm_dropout,
                'stockmixer_temporal_units': self.models.stockmixer_temporal_units,
                'stockmixer_indicator_units': self.models.stockmixer_indicator_units,
                'stockmixer_cross_stock_units': self.models.stockmixer_cross_stock_units,
                'stockmixer_fusion_units': self.models.stockmixer_fusion_units,
                'xgboost_n_estimators': self.models.xgboost_n_estimators,
                'xgboost_max_depth': self.models.xgboost_max_depth
            },
            'training': {
                'walk_forward_splits': self.training.walk_forward_splits,
                'test_size': self.training.test_size,
                'batch_size': self.training.batch_size,
                'epochs': self.training.epochs,
                'early_stopping_patience': self.training.early_stopping_patience,
                'reduce_lr_patience': self.training.reduce_lr_patience,
                'random_state': self.training.random_state
            },
            'output': {
                'results_dir': self.output.results_dir,
                'plots_dir': self.output.plots_dir,
                'shap_dir': self.output.shap_dir,
                'models_dir': self.output.models_dir,
                'log_dir': self.output.log_dir
            },
            'logging': {
                'level': logging.getLevelName(self.logging.level),
                'format': self.logging.format,
                'date_format': self.logging.date_format
            },
            'shap': {
                'background_samples': self.shap.background_samples,
                'max_display': self.shap.max_display,
                'plot_type': self.shap.plot_type,
                'save_plots': self.shap.save_plots
            }
        }
    
    def save(self, config_path: str):
        """Save configuration to file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
    
    def update(self, updates: Dict[str, Any]):
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        for section, values in updates.items():
            if hasattr(self, section):
                section_obj = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
    
    def validate(self) -> bool:
        """
        Validate configuration settings.
        
        Returns:
            True if configuration is valid
        """
        try:
            # Validate data configuration
            assert self.data.start_date < self.data.end_date, "Start date must be before end date"
            assert len(self.data.all_assets) > 0, "At least one asset must be configured"
            
            # Validate feature configuration
            assert self.features.volatility_window > 0, "Volatility window must be positive"
            assert self.features.ma_short < self.features.ma_long, "Short MA must be less than long MA"
            
            # Validate training configuration
            assert self.training.walk_forward_splits > 0, "Walk forward splits must be positive"
            assert self.training.test_size > 0, "Test size must be positive"
            
            return True
            
        except AssertionError as e:
            logging.error(f"Configuration validation failed: {e}")
            return False
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """
        Get configuration for specific model type.
        
        Args:
            model_type: Type of model ('lstm', 'stockmixer', 'xgboost', 'arima')
            
        Returns:
            Model-specific configuration dictionary
        """
        if model_type == 'lstm':
            return {
                'units': self.models.lstm_units,
                'dropout': self.models.lstm_dropout,
                'batch_size': self.training.batch_size,
                'epochs': self.training.epochs,
                'early_stopping_patience': self.training.early_stopping_patience,
                'reduce_lr_patience': self.training.reduce_lr_patience
            }
        elif model_type == 'stockmixer':
            return {
                'temporal_units': self.models.stockmixer_temporal_units,
                'indicator_units': self.models.stockmixer_indicator_units,
                'cross_stock_units': self.models.stockmixer_cross_stock_units,
                'fusion_units': self.models.stockmixer_fusion_units,
                'batch_size': self.training.batch_size,
                'epochs': self.training.epochs,
                'early_stopping_patience': self.training.early_stopping_patience,
                'reduce_lr_patience': self.training.reduce_lr_patience
            }
        elif model_type == 'xgboost':
            return {
                'n_estimators': self.models.xgboost_n_estimators,
                'max_depth': self.models.xgboost_max_depth,
                'random_state': self.training.random_state
            }
        elif model_type == 'arima':
            return {
                'order': (1, 1, 1)  # Default ARIMA order
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")


# Global configuration instance for dependency injection
_global_config: Optional[PipelineConfig] = None


def get_config() -> PipelineConfig:
    """Get global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = PipelineConfig()
    return _global_config


def set_config(config: PipelineConfig):
    """Set global configuration instance."""
    global _global_config
    _global_config = config 