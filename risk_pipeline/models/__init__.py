"""
Model components for RiskPipeline modular architecture.
"""

from .base_model import BaseModel
from .model_factory import ModelFactory

# Import models conditionally to handle missing dependencies
try:
    from .arima_model import ARIMAModel
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    ARIMAModel = None

try:
    from .xgboost_model import XGBoostModel
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBoostModel = None

try:
    from .stockmixer_model import StockMixerModel
    STOCKMIXER_AVAILABLE = True
except ImportError:
    STOCKMIXER_AVAILABLE = False
    StockMixerModel = None

try:
    from .lstm_model import LSTMModel
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    LSTMModel = None

__all__ = ['BaseModel', 'ModelFactory']

# Add available models to exports
if ARIMA_AVAILABLE:
    __all__.append('ARIMAModel')
if XGBOOST_AVAILABLE:
    __all__.append('XGBoostModel')
if STOCKMIXER_AVAILABLE:
    __all__.append('StockMixerModel')
if LSTM_AVAILABLE:
    __all__.append('LSTMModel') 