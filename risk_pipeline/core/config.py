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
    start_date: str = '1990-01-01'
    end_date: str = '2025-10-08'
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
    volatility_window: int = 20  # Increased from 10 to 20 for more stable volatility
    ma_short: int = 50  # Increased from 20 to 50 for better trend detection
    ma_long: int = 200  # Increased from 100 to 200 for longer-term trends
    correlation_window: int = 120  # Increased from 60 to 120 for more stable correlations
    sequence_length: int = 30  # Increased from 15 to 30 for better temporal patterns
    # New advanced features for high-performance systems
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    atr_period: int = 14
    stochastic_k: int = 14
    stochastic_d: int = 3


@dataclass
class ModelConfig:
    """Configuration for model parameters."""
    # Enhanced LSTM for high-performance systems
    lstm_units: List[int] = field(default_factory=lambda: [128, 96, 64, 32])  # Deeper architecture
    lstm_dropout: float = 0.3  # Increased dropout for regularization
    lstm_recurrent_dropout: float = 0.2  # New parameter for recurrent dropout
    lstm_bidirectional: bool = True  # Enable bidirectional LSTM
    lstm_attention: bool = True  # Enable attention mechanism
    
    # Enhanced StockMixer for high-performance systems
    stockmixer_temporal_units: int = 256  # Increased from 64 to 256
    stockmixer_indicator_units: int = 256  # Increased from 64 to 256
    stockmixer_cross_stock_units: int = 256  # Increased from 64 to 256
    stockmixer_fusion_units: int = 512  # Increased from 128 to 512
    stockmixer_num_layers: int = 6  # New parameter for deeper architecture
    stockmixer_attention_heads: int = 8  # New parameter for multi-head attention
    stockmixer_dropout: float = 0.3  # New parameter for dropout
    
    # Enhanced XGBoost for high-performance systems
    xgboost_n_estimators: int = 500  # Increased from 100 to 500
    xgboost_max_depth: int = 8  # Increased from 5 to 8
    xgboost_learning_rate: float = 0.05  # New parameter for learning rate
    xgboost_subsample: float = 0.8  # New parameter for subsampling
    xgboost_colsample_bytree: float = 0.8  # New parameter for column sampling
    xgboost_reg_alpha: float = 0.1  # New parameter for L1 regularization
    xgboost_reg_lambda: float = 1.0  # New parameter for L2 regularization
    
    # New advanced models for high-performance systems
    transformer_heads: int = 8
    transformer_layers: int = 6
    transformer_d_model: int = 256
    transformer_dff: int = 1024
    transformer_dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    walk_forward_splits: int = 12  # Increased from 8 to 12 for extended data
    test_size: int = 252  # Increased from 126 to 252 (1 year instead of 6 months)
    batch_size: int = 128  # Increased from 64 to 128 for better GPU utilization
    epochs: int = 200  # Increased from 100 to 200 as requested
    early_stopping_patience: int = 30  # Increased from 20 to 30
    reduce_lr_patience: int = 15  # Increased from 10 to 15
    random_state: int = 42
    
    # Target transformation options
    use_log_vol_target: bool = True
    log_target_epsilon: float = 1e-6
    
    # New advanced training parameters for high-performance systems
    validation_split: float = 0.2
    class_weight_balance: bool = True  # Enable class weight balancing
    learning_rate_schedule: str = 'cosine'  # 'constant', 'step', 'cosine', 'exponential'
    warmup_epochs: int = 10  # New parameter for learning rate warmup
    gradient_clip_norm: float = 1.0  # New parameter for gradient clipping
    mixed_precision: bool = True  # Enable mixed precision training
    data_augmentation: bool = True  # Enable data augmentation
    
    # DYNAMIC CPU UTILIZATION for 24-core i9-14900HX system
    # Use 23 cores for maximum performance, leave 1 core for system stability
    num_workers: int = 23  # Use 23 cores for data loading (leave 1 for system)
    parallel_backend: str = 'multiprocessing'  # 'multiprocessing', 'threading', 'joblib'
    joblib_n_jobs: int = 23  # Use 23 cores for scikit-learn operations
    ray_num_cpus: int = 23  # Use 23 cores for Ray operations
    dask_n_workers: int = 23  # Use 23 workers for Dask operations
    
    # NEW: Dynamic core detection and utilization
    auto_detect_cores: bool = True  # Automatically detect available cores
    max_core_usage: float = 0.95  # Use up to 95% of available cores
    adaptive_batch_sizing: bool = True  # Adjust batch size based on core count

    # Feature scaling/stabilization toggles
    use_robust_scaler: bool = False  # If True, use RobustScaler instead of StandardScaler
    feature_clip_q: float | None = None  # e.g., 0.01 to clip to [q, 1-q] per feature on train


@dataclass
class HyperparameterTuningConfig:
    """Configuration for advanced hyperparameter tuning."""
    enable_tuning: bool = True
    tuning_backend: str = 'optuna'  # 'optuna', 'hyperopt', 'ray[tune]', 'skopt'
    
    # Optuna configuration
    optuna_n_trials: int = 200  # Increased number of trials
    optuna_timeout: int = 3600  # 1 hour timeout
    optuna_pruner: str = 'median'  # 'median', 'percentile', 'hyperband'
    
    # Ray Tune configuration
    ray_tune_num_samples: int = 200
    ray_tune_time_budget_s: int = 3600
    ray_tune_scheduler: str = 'asha'  # 'asha', 'hyperband', 'median_stopping'
    
    # Search spaces for different models
    lstm_search_space: Dict[str, Any] = field(default_factory=lambda: {
        'units': [64, 128, 256, 512],
        'layers': [2, 3, 4, 5],
        'dropout': [0.1, 0.2, 0.3, 0.4],
        'learning_rate': [0.001, 0.01, 0.1],
        'batch_size': [32, 64, 128, 256]
    })
    
    stockmixer_search_space: Dict[str, Any] = field(default_factory=lambda: {
        'temporal_units': [128, 256, 512],
        'indicator_units': [128, 256, 512],
        'cross_stock_units': [128, 256, 512],
        'fusion_units': [256, 512, 1024],
        'num_layers': [4, 6, 8],
        'attention_heads': [4, 8, 16]
    })
    
    xgboost_search_space: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': [200, 500, 1000],
        'max_depth': [6, 8, 10, 12],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'reg_alpha': [0, 0.1, 1.0],
        'reg_lambda': [0.1, 1.0, 10.0]
    })


@dataclass
class EnsembleConfig:
    """Configuration for ensemble methods."""
    enable_ensemble: bool = True
    ensemble_methods: List[str] = field(default_factory=lambda: ['voting', 'stacking', 'blending'])
    ensemble_weights: List[float] = field(default_factory=lambda: [0.3, 0.4, 0.3])
    
    # Voting ensemble
    voting_estimators: List[str] = field(default_factory=lambda: ['lstm', 'stockmixer', 'xgboost', 'transformer'])
    voting_method: str = 'soft'  # 'hard', 'soft'
    
    # Stacking ensemble
    stacking_cv_folds: int = 5
    stacking_meta_learner: str = 'xgboost'  # 'xgboost', 'lstm', 'linear'
    
    # Blending ensemble
    blending_holdout_size: float = 0.2
    blending_meta_learner: str = 'xgboost'


@dataclass
class AdvancedFeaturesConfig:
    """Configuration for advanced features and techniques."""
    # Feature selection
    enable_feature_selection: bool = True
    feature_selection_method: str = 'mutual_info'  # 'mutual_info', 'chi2', 'f_classif', 'recursive'
    feature_selection_k: int = 50  # Number of features to select
    
    # Feature importance
    enable_feature_importance: bool = True
    importance_methods: List[str] = field(default_factory=lambda: ['shap', 'permutation', 'tree'])
    
    # Cross-validation
    cv_method: str = 'time_series_split'  # 'time_series_split', 'walk_forward', 'purged_group'
    cv_folds: int = 5
    
    # Model interpretability
    enable_interpretability: bool = True
    interpretability_methods: List[str] = field(default_factory=lambda: ['shap', 'lime', 'permutation'])
    
    # Risk metrics
    enable_risk_metrics: bool = True
    risk_metrics: List[str] = field(default_factory=lambda: ['var', 'cvar', 'sharpe', 'sortino', 'calmar'])
    confidence_level: float = 0.95


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
            start_date=data_config.get('start_date', '1990-01-01'),
            end_date=data_config.get('end_date', '2025-10-08'),
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
            walk_forward_splits=training_config.get('walk_forward_splits', 8),
            test_size=training_config.get('test_size', 252),
            batch_size=training_config.get('batch_size', 64),
            epochs=training_config.get('epochs', 100),
            early_stopping_patience=training_config.get('early_stopping_patience', 20),
            reduce_lr_patience=training_config.get('reduce_lr_patience', 10),
            random_state=training_config.get('random_state', 42)
        )
        
        # NOTE: Optimization calls moved after SHAP config initialization to avoid
        # referencing self.shap before it exists
        
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
        
        # DYNAMIC CPU OPTIMIZATION: Auto-detect and configure cores
        # ALWAYS run this to ensure maximum performance
        self._optimize_for_cpu()
        # DYNAMIC MEMORY OPTIMIZATION: Tune batch sizes/threads and SHAP background
        self._optimize_for_memory()
        
        # Hyperparameter tuning configuration
        tuning_config = config_dict.get('hyperparameter_tuning', {})
        self.hyperparameter_tuning = HyperparameterTuningConfig(
            enable_tuning=tuning_config.get('enable_tuning', True),
            tuning_backend=tuning_config.get('tuning_backend', 'optuna'),
            optuna_n_trials=tuning_config.get('optuna_n_trials', 200),
            optuna_timeout=tuning_config.get('optuna_timeout', 3600),
            optuna_pruner=tuning_config.get('optuna_pruner', 'median'),
            ray_tune_num_samples=tuning_config.get('ray_tune_num_samples', 200),
            ray_tune_time_budget_s=tuning_config.get('ray_tune_time_budget_s', 3600),
            ray_tune_scheduler=tuning_config.get('ray_tune_scheduler', 'asha'),
            lstm_search_space=tuning_config.get('lstm_search_space', {
                'units': [64, 128, 256, 512],
                'layers': [2, 3, 4, 5],
                'dropout': [0.1, 0.2, 0.3, 0.4],
                'learning_rate': [0.001, 0.01, 0.1],
                'batch_size': [32, 64, 128, 256]
            }),
            stockmixer_search_space=tuning_config.get('stockmixer_search_space', {
                'temporal_units': [128, 256, 512],
                'indicator_units': [128, 256, 512],
                'cross_stock_units': [128, 256, 512],
                'fusion_units': [256, 512, 1024],
                'num_layers': [4, 6, 8],
                'attention_heads': [4, 8, 16]
            }),
            xgboost_search_space=tuning_config.get('xgboost_search_space', {
                'n_estimators': [200, 500, 1000],
                'max_depth': [6, 8, 10, 12],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'reg_alpha': [0, 0.1, 1.0],
                'reg_lambda': [0.1, 1.0, 10.0]
            })
        )
        
        # Ensemble configuration
        ensemble_config = config_dict.get('ensemble', {})
        self.ensemble = EnsembleConfig(
            enable_ensemble=ensemble_config.get('enable_ensemble', True),
            ensemble_methods=ensemble_config.get('ensemble_methods', ['voting', 'stacking', 'blending']),
            ensemble_weights=ensemble_config.get('ensemble_weights', [0.3, 0.4, 0.3]),
            voting_estimators=ensemble_config.get('voting_estimators', ['lstm', 'stockmixer', 'xgboost', 'transformer']),
            voting_method=ensemble_config.get('voting_method', 'soft'),
            stacking_cv_folds=ensemble_config.get('stacking_cv_folds', 5),
            stacking_meta_learner=ensemble_config.get('stacking_meta_learner', 'xgboost'),
            blending_holdout_size=ensemble_config.get('blending_holdout_size', 0.2),
            blending_meta_learner=ensemble_config.get('blending_meta_learner', 'xgboost')
        )
        
        # Advanced features configuration
        advanced_config = config_dict.get('advanced_features', {})
        self.advanced_features = AdvancedFeaturesConfig(
            enable_feature_selection=advanced_config.get('enable_feature_selection', True),
            feature_selection_method=advanced_config.get('feature_selection_method', 'mutual_info'),
            feature_selection_k=advanced_config.get('feature_selection_k', 50),
            enable_feature_importance=advanced_config.get('enable_feature_importance', True),
            importance_methods=advanced_config.get('importance_methods', ['shap', 'permutation', 'tree']),
            cv_method=advanced_config.get('cv_method', 'time_series_split'),
            cv_folds=advanced_config.get('cv_folds', 5),
            enable_interpretability=advanced_config.get('enable_interpretability', True),
            interpretability_methods=advanced_config.get('interpretability_methods', ['shap', 'lime', 'permutation']),
            enable_risk_metrics=advanced_config.get('enable_risk_metrics', True),
            risk_metrics=advanced_config.get('risk_metrics', ['var', 'cvar', 'sharpe', 'sortino', 'calmar']),
            confidence_level=advanced_config.get('confidence_level', 0.95)
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
            },
            'hyperparameter_tuning': {
                'enable_tuning': self.hyperparameter_tuning.enable_tuning,
                'tuning_backend': self.hyperparameter_tuning.tuning_backend,
                'optuna_n_trials': self.hyperparameter_tuning.optuna_n_trials,
                'optuna_timeout': self.hyperparameter_tuning.optuna_timeout,
                'optuna_pruner': self.hyperparameter_tuning.optuna_pruner,
                'ray_tune_num_samples': self.hyperparameter_tuning.ray_tune_num_samples,
                'ray_tune_time_budget_s': self.hyperparameter_tuning.ray_tune_time_budget_s,
                'ray_tune_scheduler': self.hyperparameter_tuning.ray_tune_scheduler,
                'lstm_search_space': self.hyperparameter_tuning.lstm_search_space,
                'stockmixer_search_space': self.hyperparameter_tuning.stockmixer_search_space,
                'xgboost_search_space': self.hyperparameter_tuning.xgboost_search_space
            },
            'ensemble': {
                'enable_ensemble': self.ensemble.enable_ensemble,
                'ensemble_methods': self.ensemble.ensemble_methods,
                'ensemble_weights': self.ensemble.ensemble_weights,
                'voting_estimators': self.ensemble.voting_estimators,
                'voting_method': self.ensemble.voting_method,
                'stacking_cv_folds': self.ensemble.stacking_cv_folds,
                'stacking_meta_learner': self.ensemble.stacking_meta_learner,
                'blending_holdout_size': self.ensemble.blending_holdout_size,
                'blending_meta_learner': self.ensemble.blending_meta_learner
            },
            'advanced_features': {
                'enable_feature_selection': self.advanced_features.enable_feature_selection,
                'feature_selection_method': self.advanced_features.feature_selection_method,
                'feature_selection_k': self.advanced_features.feature_selection_k,
                'enable_feature_importance': self.advanced_features.enable_feature_importance,
                'importance_methods': self.advanced_features.importance_methods,
                'cv_method': self.advanced_features.cv_method,
                'cv_folds': self.advanced_features.cv_folds,
                'enable_interpretability': self.advanced_features.enable_interpretability,
                'interpretability_methods': self.advanced_features.interpretability_methods,
                'enable_risk_metrics': self.advanced_features.enable_risk_metrics,
                'risk_metrics': self.advanced_features.risk_metrics,
                'confidence_level': self.advanced_features.confidence_level
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
            model_type: Type of model ('lstm', 'stockmixer', 'xgboost', 'arima', 'transformer')
            
        Returns:
            Model-specific configuration dictionary
        """
        if model_type == 'lstm':
            return {
                'units': self.models.lstm_units,
                'dropout': self.models.lstm_dropout,
                'recurrent_dropout': self.models.lstm_recurrent_dropout,
                'bidirectional': self.models.lstm_bidirectional,
                'attention': self.models.lstm_attention,
                'batch_size': self.training.batch_size,
                'epochs': self.training.epochs,
                'early_stopping_patience': self.training.early_stopping_patience,
                'reduce_lr_patience': self.training.reduce_lr_patience,
                'validation_split': self.training.validation_split,
                'class_weight_balance': self.training.class_weight_balance,
                'learning_rate_schedule': self.training.learning_rate_schedule,
                'warmup_epochs': self.training.warmup_epochs,
                'gradient_clip_norm': self.training.gradient_clip_norm,
                'mixed_precision': self.training.mixed_precision,
                'data_augmentation': self.training.data_augmentation
            }
        elif model_type == 'stockmixer':
            return {
                'temporal_units': self.models.stockmixer_temporal_units,
                'indicator_units': self.models.stockmixer_indicator_units,
                'cross_stock_units': self.models.stockmixer_cross_stock_units,
                'fusion_units': self.models.stockmixer_fusion_units,
                'num_layers': self.models.stockmixer_num_layers,
                'attention_heads': self.models.stockmixer_attention_heads,
                'dropout': self.models.stockmixer_dropout,
                'batch_size': self.training.batch_size,
                'epochs': self.training.epochs,
                'early_stopping_patience': self.training.early_stopping_patience,
                'reduce_lr_patience': self.training.reduce_lr_patience,
                'validation_split': self.training.validation_split,
                'class_weight_balance': self.training.class_weight_balance,
                'learning_rate_schedule': self.training.learning_rate_schedule,
                'warmup_epochs': self.training.warmup_epochs,
                'gradient_clip_norm': self.training.gradient_clip_norm,
                'mixed_precision': self.training.mixed_precision,
                'data_augmentation': self.training.data_augmentation
            }
        elif model_type == 'xgboost':
            return {
                'n_estimators': self.models.xgboost_n_estimators,
                'max_depth': self.models.xgboost_max_depth,
                'learning_rate': self.models.xgboost_learning_rate,
                'subsample': self.models.xgboost_subsample,
                'colsample_bytree': self.models.xgboost_colsample_bytree,
                'reg_alpha': self.models.xgboost_reg_alpha,
                'reg_lambda': self.models.xgboost_reg_lambda,
                'random_state': self.training.random_state,
                'n_jobs': self.training.joblib_n_jobs  # Use parallel processing
            }
        elif model_type == 'transformer':
            return {
                'heads': self.models.transformer_heads,
                'layers': self.models.transformer_layers,
                'd_model': self.models.transformer_d_model,
                'dff': self.models.transformer_dff,
                'dropout': self.models.transformer_dropout,
                'batch_size': self.training.batch_size,
                'epochs': self.training.epochs,
                'early_stopping_patience': self.training.early_stopping_patience,
                'reduce_lr_patience': self.training.reduce_lr_patience,
                'validation_split': self.training.validation_split,
                'class_weight_balance': self.training.class_weight_balance,
                'learning_rate_schedule': self.training.learning_rate_schedule,
                'warmup_epochs': self.training.warmup_epochs,
                'gradient_clip_norm': self.training.gradient_clip_norm,
                'mixed_precision': self.training.mixed_precision,
                'data_augmentation': self.training.data_augmentation
            }
        elif model_type == 'arima':
            return {
                # Sensible ARIMA defaults for equities
                'order': (1, 1, 1),
                'seasonal_order': (0, 0, 0, 0),  # Disable seasonality by default
                'seasonal': False,
                # Auto-order search (configurable)
                'auto_order': True,
                'max_p': 5,
                'max_d': 2,
                'max_q': 5,
                'n_jobs': self.training.joblib_n_jobs  # Use parallel processing
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def get_parallel_config(self) -> Dict[str, Any]:
        """
        Get parallel processing configuration for high-performance systems.
        
        Returns:
            Dictionary with parallel processing settings
        """
        return {
            'num_workers': self.training.num_workers,
            'parallel_backend': self.training.parallel_backend,
            'joblib_n_jobs': self.training.joblib_n_jobs,
            'ray_num_cpus': self.training.ray_num_cpus,
            'dask_n_workers': self.training.dask_n_workers
        }
    
    def get_hyperparameter_tuning_config(self) -> Dict[str, Any]:
        """
        Get hyperparameter tuning configuration.
        
        Returns:
            Dictionary with hyperparameter tuning settings
        """
        return {
            'enable_tuning': self.hyperparameter_tuning.enable_tuning,
            'tuning_backend': self.hyperparameter_tuning.tuning_backend,
            'optuna_n_trials': self.hyperparameter_tuning.optuna_n_trials,
            'optuna_timeout': self.hyperparameter_tuning.optuna_timeout,
            'optuna_pruner': self.hyperparameter_tuning.optuna_pruner,
            'ray_tune_num_samples': self.hyperparameter_tuning.ray_tune_num_samples,
            'ray_tune_time_budget_s': self.hyperparameter_tuning.ray_tune_time_budget_s,
            'ray_tune_scheduler': self.hyperparameter_tuning.ray_tune_scheduler,
            'search_spaces': {
                'lstm': self.hyperparameter_tuning.lstm_search_space,
                'stockmixer': self.hyperparameter_tuning.stockmixer_search_space,
                'xgboost': self.hyperparameter_tuning.xgboost_search_space
            }
        }
    
    def get_ensemble_config(self) -> Dict[str, Any]:
        """
        Get ensemble configuration.
        
        Returns:
            Dictionary with ensemble settings
        """
        return {
            'enable_ensemble': self.ensemble.enable_ensemble,
            'ensemble_methods': self.ensemble.ensemble_methods,
            'ensemble_weights': self.ensemble.ensemble_weights,
            'voting_estimators': self.ensemble.voting_estimators,
            'voting_method': self.ensemble.voting_method,
            'stacking_cv_folds': self.ensemble.stacking_cv_folds,
            'stacking_meta_learner': self.ensemble.stacking_meta_learner,
            'blending_holdout_size': self.ensemble.blending_holdout_size,
            'blending_meta_learner': self.ensemble.blending_meta_learner
        }
    
    def get_advanced_features_config(self) -> Dict[str, Any]:
        """
        Get advanced features configuration.
        
        Returns:
            Dictionary with advanced features settings
        """
        return {
            'enable_feature_selection': self.advanced_features.enable_feature_selection,
            'feature_selection_method': self.advanced_features.feature_selection_method,
            'feature_selection_k': self.advanced_features.feature_selection_k,
            'enable_feature_importance': self.advanced_features.enable_feature_importance,
            'importance_methods': self.advanced_features.importance_methods,
            'cv_method': self.advanced_features.cv_method,
            'cv_folds': self.advanced_features.cv_folds,
            'enable_interpretability': self.advanced_features.enable_interpretability,
            'interpretability_methods': self.advanced_features.interpretability_methods,
            'enable_risk_metrics': self.advanced_features.enable_risk_metrics,
            'risk_metrics': self.advanced_features.risk_metrics,
            'confidence_level': self.advanced_features.confidence_level
        }

    def _optimize_for_cpu(self):
        """Dynamically optimize CPU usage based on available cores."""
        import psutil
        import os
        
        # Get actual CPU info - use PHYSICAL cores for maximum performance
        cpu_count = psutil.cpu_count(logical=False)  # Physical cores (24 for your i9-14900HX)
        cpu_count_logical = psutil.cpu_count(logical=True)  # Logical cores (32 for your i9-14900HX)
        
        # Calculate optimal core usage (use 95% of physical cores for maximum performance)
        optimal_workers = max(1, int(cpu_count * 0.95))  # Use 23 out of 24 cores
        
        # Update training config with detected values
        self.training.num_workers = optimal_workers
        self.training.joblib_n_jobs = optimal_workers
        self.training.ray_num_cpus = optimal_workers
        self.training.dask_n_workers = optimal_workers
        
        # Set environment variables for external libraries
        os.environ['OMP_NUM_THREADS'] = str(optimal_workers)
        os.environ['MKL_NUM_THREADS'] = str(optimal_workers)
        os.environ['OPENBLAS_NUM_THREADS'] = str(optimal_workers)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(optimal_workers)
        os.environ['NUMEXPR_NUM_THREADS'] = str(optimal_workers)
        
        # Log the optimization
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"🚀 CPU OPTIMIZATION: Detected {cpu_count} physical cores, {cpu_count_logical} logical cores")
        logger.info(f"⚡ Using {optimal_workers} cores for pipeline ({(optimal_workers/cpu_count)*100:.1f}% utilization)")
        logger.info(f"💡 Environment variables set for optimal threading")

    def _optimize_for_memory(self):
        """Dynamically optimize memory-related settings based on available RAM."""
        import psutil
        import os
        from pathlib import Path
        vm = psutil.virtual_memory()
        total_gb = max(1, int(vm.total / (1024**3)))
        avail_ratio = vm.available / vm.total
        
        # Adjust batch size heuristically
        if avail_ratio < 0.25:
            self.training.batch_size = max(8, int(self.training.batch_size * 0.5))
            # Reduce joblib workers slightly to relieve pressure
            self.training.joblib_n_jobs = max(1, int(self.training.joblib_n_jobs * 0.75))
            # Reduce SHAP background to save RAM
            self.shap.background_samples = min(self.shap.background_samples, 50)
        elif avail_ratio > 0.5:
            # Safely scale up a bit
            self.training.batch_size = min(256, max(self.training.batch_size, 64))
            self.shap.background_samples = min(self.shap.background_samples, 200)
        
        # Prefer float32 math in downstream code (advisory; actual casting done in feature engineering)
        os.environ.setdefault('RP_FLOAT_DTYPE', 'float32')
        
        # Ensure a fast temp directory for joblib to spill to disk instead of RAM
        joblib_tmp = Path(self.data.cache_dir) / 'joblib_tmp'
        joblib_tmp.mkdir(parents=True, exist_ok=True)
        os.environ['JOBLIB_TEMP_FOLDER'] = str(joblib_tmp)
        
        # Set a cap for OpenMP thread stack to avoid overhead
        os.environ.setdefault('OMP_STACKSIZE', '16M')


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