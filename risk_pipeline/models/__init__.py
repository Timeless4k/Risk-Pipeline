"""
Model implementations for the RiskPipeline.
"""

from .base_model import BaseModel
from .arima_model import ARIMAModel
from .xgboost_model import XGBoostModel
from .enhanced_arima_model import EnhancedARIMAModel

# Check TensorFlow availability
try:
    from .lstm_model import LSTMModel
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

# Check StockMixer availability
try:
    from .stockmixer_model import StockMixerModel
    STOCKMIXER_AVAILABLE = True
except ImportError:
    STOCKMIXER_AVAILABLE = False

__all__ = ['BaseModel', 'ARIMAModel', 'XGBoostModel', 'EnhancedARIMAModel']

if STOCKMIXER_AVAILABLE:
    __all__.append('StockMixerModel')
if LSTM_AVAILABLE:
    __all__.append('LSTMModel') 