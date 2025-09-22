"""
Logging configuration for RiskPipeline.

This module provides granular control over logging levels for different components
to reduce noise while maintaining important information for debugging.
"""

import logging
from typing import Dict, Any

# Default logging configuration
DEFAULT_LOGGING_CONFIG = {
    'root_level': logging.INFO,
    'file_level': logging.INFO,
    'console_level': logging.INFO,
    
    # Component-specific logging levels
    'components': {
        'risk_pipeline': logging.INFO,
        'risk_pipeline.core': logging.INFO,
        'risk_pipeline.core.data_loader': logging.INFO,
        'risk_pipeline.core.feature_engineer': logging.INFO,
        'risk_pipeline.core.validator': logging.INFO,
        'risk_pipeline.core.results_manager': logging.INFO,
        'risk_pipeline.models': logging.INFO,
        'risk_pipeline.interpretability': logging.INFO,
        'risk_pipeline.visualization': logging.INFO,
        'risk_pipeline.utils': logging.INFO,
    },
    
    # Third-party library logging levels
    'third_party': {
        'yfinance': logging.WARNING,
        'peewee': logging.WARNING,
        'PIL': logging.WARNING,
        'matplotlib': logging.WARNING,
        'urllib3': logging.WARNING,
        'requests': logging.WARNING,
        # 'tensorflow' removed
        'h5py': logging.WARNING,
        'numba': logging.WARNING,
        'shap': logging.WARNING,
        'sklearn': logging.WARNING,
        'xgboost': logging.WARNING,
        'statsmodels': logging.WARNING,
        'pandas': logging.WARNING,
        'numpy': logging.WARNING,
    },
    
    # Verbose mode (for debugging)
    'verbose': False,
    
    # Log format
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S',
    
    # File rotation settings
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5,
}

def get_logging_config(verbose: bool = False, custom_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get logging configuration with optional customization.
    
    Args:
        verbose: Enable verbose logging (DEBUG level)
        custom_config: Custom configuration overrides
        
    Returns:
        Logging configuration dictionary
    """
    config = DEFAULT_LOGGING_CONFIG.copy()
    
    if verbose:
        config['root_level'] = logging.DEBUG
        config['file_level'] = logging.DEBUG
        config['console_level'] = logging.DEBUG
        config['verbose'] = True
        
        # Enable DEBUG for core components in verbose mode
        for component in config['components']:
            if 'core' in component or 'models' in component:
                config['components'][component] = logging.DEBUG
    
    if custom_config:
        config.update(custom_config)
    
    return config

def apply_logging_config(config: Dict[str, Any]) -> None:
    """
    Apply logging configuration to all loggers.
    
    Args:
        config: Logging configuration dictionary
    """
    # Apply component-specific levels
    for logger_name, level in config['components'].items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
    
    # Apply third-party library levels
    for logger_name, level in config['third_party'].items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
    
    # Set root logger level
    root_logger = logging.getLogger()
    root_logger.setLevel(config['root_level'])

def get_quiet_config() -> Dict[str, Any]:
    """Get minimal logging configuration for production use."""
    return {
        'root_level': logging.WARNING,
        'file_level': logging.INFO,
        'console_level': logging.WARNING,
        'components': {k: logging.WARNING for k in DEFAULT_LOGGING_CONFIG['components']},
        'third_party': {k: logging.ERROR for k in DEFAULT_LOGGING_CONFIG['third_party']},
        'verbose': False,
    }
