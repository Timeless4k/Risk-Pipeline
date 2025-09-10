"""
Model implementations for the RiskPipeline.
"""

from .base_model import BaseModel
from .arima_model import ARIMAModel
from .xgboost_model import XGBoostModel

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

# Check GARCH availability
try:
    from .garch_model import GARCHModel
    GARCH_AVAILABLE = True
except Exception:
    GARCH_AVAILABLE = False
    GARCHModel = None

__all__ = ['BaseModel', 'ARIMAModel', 'XGBoostModel']

if GARCH_AVAILABLE:
    __all__.append('GARCHModel')

if STOCKMIXER_AVAILABLE:
    __all__.append('StockMixerModel')
if LSTM_AVAILABLE:
    __all__.append('LSTMModel') 