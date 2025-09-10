"""
Model factory for creating different types of models.
"""

import logging
from typing import Dict, Any, Type
from .base_model import BaseModel
from .arima_model import ARIMAModel
from .xgboost_model import XGBoostModel

# Make GARCH optional (arch package may be missing)
try:
    from .garch_model import GARCHModel
    GARCH_AVAILABLE = True
except Exception:
    GARCH_AVAILABLE = False
    GARCHModel = None  # type: ignore

logger = logging.getLogger(__name__)

# Check TensorFlow availability
try:
    from .lstm_model import LSTMModel
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

# Check StockMixer availability
try:
    # Use the StockMixer implementation in stockmixer_model.py
    from .stockmixer_model import StockMixerModel
    STOCKMIXER_AVAILABLE = True
except ImportError:
    STOCKMIXER_AVAILABLE = False


class ModelFactory:
    """Factory for creating different types of models."""
    
    _models: Dict[str, Type[BaseModel]] = {
        'arima': ARIMAModel,
        'xgboost': XGBoostModel,
        # Aliases
        'xgb': XGBoostModel,
    }

    if GARCH_AVAILABLE and GARCHModel is not None:
        _models['garch'] = GARCHModel
    
    if LSTM_AVAILABLE:
        _models['lstm'] = LSTMModel
    
    if STOCKMIXER_AVAILABLE:
        _models['stockmixer'] = StockMixerModel
    
    @classmethod
    def create_model(cls, model_type: str, task: str = 'regression', **kwargs) -> BaseModel:
        """Create a model instance of the specified type."""
        # Normalize aliases
        model_type = (model_type or '').lower()
        alias_map = {
            'xgb': 'xgboost',
        }
        model_type = alias_map.get(model_type, model_type)

        if model_type not in cls._models:
            available_models = list(cls._models.keys())
            raise ValueError(f"Unknown model type '{model_type}'. Available: {available_models}")
        
        model_class = cls._models[model_type]
        logger.info(f"Creating {model_type} model for {task} task with parameters: {kwargs}")
        
        # Handle model-specific parameters
        if model_type == 'arima':
            # ARIMA only supports regression
            if task != 'regression':
                logger.warning("ARIMA only supports regression tasks. Using regression.")
            return model_class(**kwargs)
        elif model_type == 'lstm':
            # LSTM supports both tasks
            return model_class(task=task, **kwargs)
        elif model_type == 'xgboost':
            # XGBoost supports both tasks
            # Log final params after any caller-side merges
            try:
                final_params = kwargs.copy()
                if task == 'classification':
                    logger.info(f"Final XGB CLASSIFICATION params: {final_params}")
                else:
                    logger.info(f"Final XGB params (after merge): {final_params}")
            except Exception:
                pass
            return model_class(task=task, **kwargs)
        elif model_type == 'garch':
            # GARCH supports regression and derived classification (via thresholding)
            return model_class(task=task, **kwargs)
        elif model_type == 'stockmixer':
            # StockMixer supports both tasks
            return model_class(task=task, **kwargs)
        else:
            # Default case
            return model_class(**kwargs)
    
    @classmethod
    def get_available_models(cls) -> list:
        """Get list of available model types."""
        return list(cls._models.keys())
    
    @classmethod
    def is_model_available(cls, model_type: str) -> bool:
        """Check if a model type is available."""
        return model_type in cls._models


# Convenience functions for direct model creation
def create_arima_model(**kwargs) -> ARIMAModel:
    """Create an ARIMA model."""
    return ARIMAModel(**kwargs)

def create_enhanced_arima_model(**kwargs) -> ARIMAModel:
    """[DEPRECATED] Enhanced ARIMA removed. Use create_arima_model instead."""
    return ARIMAModel(**kwargs)

def create_xgboost_model(**kwargs) -> XGBoostModel:
    """Create an XGBoost model."""
    return XGBoostModel(**kwargs)

def create_garch_model(**kwargs) -> GARCHModel:
    """Create a GARCH model."""
    return GARCHModel(**kwargs)

def create_lstm_model(**kwargs):
    """Create an LSTM model if available."""
    if not LSTM_AVAILABLE:
        raise ImportError("LSTM model not available. TensorFlow required.")
    return LSTMModel(**kwargs)

def create_stockmixer_model(**kwargs):
    """Create a StockMixer model if available."""
    if not STOCKMIXER_AVAILABLE:
        raise ImportError("StockMixer model not available. TensorFlow required.")
    return StockMixerModel(**kwargs) 