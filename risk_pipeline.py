"""
RiskPipeline: Interpretable Machine Learning for Volatility Forecasting
A comprehensive framework for volatility prediction across US and Australian markets
Author: Gurudeep Singh Dhinjan
Student ID: 24555981
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Iterator
import joblib
from pathlib import Path
import logging
import sys
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from collections import Counter

# Suppress warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# =============================================================================
# LOGGING SETUP (REPLACES OLD SETUP)
# =============================================================================
def setup_logging(log_file_path: str = None, level: int = logging.INFO) -> logging.Logger:
    """Setup comprehensive logging configuration with third-party filtering"""
    # Clear any existing handlers to avoid conflicts
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create logs directory
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Generate log file path if not provided
    if log_file_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file_path = log_dir / f'pipeline_run_{timestamp}.log'
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler - captures ALL logs to file
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # Capture everything in file
    file_handler.setFormatter(formatter)
    
    # Console handler - less verbose for console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # ===== FIX: Reduce third-party library verbosity =====
    # Set specific loggers to higher levels to reduce noise
    third_party_loggers = {
        'yfinance': logging.WARNING,        # Reduce yfinance verbosity
        'peewee': logging.WARNING,          # Reduce database logs
        'PIL': logging.WARNING,             # Reduce image processing logs
        'matplotlib': logging.WARNING,      # Reduce matplotlib logs
        'urllib3': logging.WARNING,         # Reduce HTTP request logs
        'requests': logging.WARNING,        # Reduce requests logs
        'tensorflow': logging.ERROR,        # Only show TF errors
        'h5py': logging.WARNING,           # Reduce HDF5 logs
        'numba': logging.WARNING,          # Reduce numba compilation logs
    }
    
    for logger_name, log_level in third_party_loggers.items():
        third_party_logger = logging.getLogger(logger_name)
        third_party_logger.setLevel(log_level)
    
    # Create and return pipeline logger
    logger = logging.getLogger('risk_pipeline')
    logger.setLevel(logging.DEBUG)
    
    # Test the logging
    logger.info(f"Logging initialized. Log file: {log_file_path}")
    logger.debug("Debug logging is working")
    logger.info("Third-party logging optimized for readability")
    
    return logger

# =============================================================================
# FIX: Add performance timing to key methods
# =============================================================================
def log_execution_time(func):
    """Decorator to log function execution time"""
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        
        # Get logger from the instance if it's a method
        if len(args) > 0 and hasattr(args[0], 'logger'):
            logger = args[0].logger
        else:
            logger = logging.getLogger(f'{func.__module__}.{func.__name__}')
        
        logger.debug(f"⏱️ Starting {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ {func.__name__} completed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"❌ {func.__name__} failed after {execution_time:.2f} seconds: {e}")
            raise
    
    return wrapper

# Initialize the logger
logger = setup_logging()

@dataclass
class AssetConfig:
    """Configuration for assets used in the study"""
    US_ASSETS = ['AAPL', 'MSFT', '^GSPC']
    AU_ASSETS = ['IOZ.AX', 'CBA.AX', 'BHP.AX']
    ALL_ASSETS = US_ASSETS + AU_ASSETS
    
    # Date range as per thesis
    START_DATE = '2000-01-01'
    END_DATE = '2025-01-05'
    
    # Feature windows
    VOLATILITY_WINDOW = 5
    MA_SHORT = 10
    MA_LONG = 50
    CORRELATION_WINDOW = 30
    
    # Model parameters
    WALK_FORWARD_SPLITS = 5
    TEST_SIZE = 45  # ~1.5 months of trading days (reduced from 252)
    RANDOM_STATE = 42

class DataLoader:
    """Handles data downloading and caching"""
    
    def __init__(self, cache_dir: str = 'data_cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger('risk_pipeline.DataLoader')
        self.logger.info(f"DataLoader initialized with cache_dir: {cache_dir}")
        
    def download_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Download or load cached data for multiple symbols"""
        data = {}
        
        self.logger.info(f"Starting data download for {len(symbols)} symbols: {symbols}")
        self.logger.debug(f"Date range: {start_date} to {end_date}")
        
        for symbol in tqdm(symbols, desc="Downloading data"):
            cache_file = self.cache_dir / f"{symbol.replace('^', '')}_data.pkl"
            
            if cache_file.exists():
                self.logger.info(f"Loading cached data for {symbol}")
                try:
                    data[symbol] = pd.read_pickle(cache_file)
                    self.logger.debug(f"{symbol} - Loaded from cache, shape: {data[symbol].shape}")
                except Exception as e:
                    self.logger.error(f"Failed to load cached data for {symbol}: {e}")
                    continue
            else:
                self.logger.info(f"Downloading data for {symbol}")
                try:
                    df = yf.download(
                        tickers=symbol,
                        start=start_date,
                        end=end_date,
                        progress=False,
                        auto_adjust=False
                    )
                    self.logger.debug(f"{symbol} - Download successful, shape: {df.shape}")
                    self.logger.info(f"{symbol} - Successfully downloaded with shape {df.shape}")
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.droplevel(1)
                    if 'Close' not in df.columns and 'Adj Close' not in df.columns:
                        self.logger.error(f"{symbol} - Missing both 'Close' and 'Adj Close' columns")
                        continue
                    if df.empty or df.isna().all().all():
                        self.logger.warning(f"{symbol} - Downloaded data is empty or all NaN")
                        continue
                    df.to_pickle(cache_file)
                    data[symbol] = df
                except Exception as e:
                    self.logger.error(f"Error downloading {symbol}: {e}")
                    continue
        
        # Handle VIX separately (same logic)
        vix_cache = self.cache_dir / "VIX_data.pkl"
        if vix_cache.exists():
            data['VIX'] = pd.read_pickle(vix_cache)
        else:
            self.logger.info("Downloading VIX data")
            try:
                vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False)
                if isinstance(vix_data.columns, pd.MultiIndex):
                    vix_data.columns = vix_data.columns.droplevel(1)
                vix_data.to_pickle(vix_cache)
                data['VIX'] = vix_data
                self.logger.info("VIX data downloaded successfully")
            except Exception as e:
                self.logger.error(f"Error downloading VIX data: {e}")
        
        return data

class FeatureEngineer:
    """Handles all feature engineering as per thesis specification"""
    
    def __init__(self, config: AssetConfig):
        self.config = config
        self.logger = logging.getLogger('risk_pipeline.FeatureEngineer')
        self.logger.info("FeatureEngineer initialized")
        
    def calculate_log_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate log returns"""
        returns = np.log(prices / prices.shift(1))
        if returns.dropna().empty:
            self.logger.warning(f"⚠️ Log returns are empty or invalid — check input prices:\n{prices.head()}")
        return returns
    
    def calculate_volatility(self, returns: pd.Series, window: int) -> pd.Series:
        """Calculate rolling volatility (standard deviation)"""
        return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
    
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical features as specified in thesis"""
        # Use Adj Close if available, otherwise fall back to Close
        price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        df['Price'] = df[price_col]  # Changed from df['Adj Close'].fillna(df['Close'])
        
        # Calculate returns using the combined price column
        returns = self.calculate_log_returns(df['Price'])
        
        # Create features DataFrame
        features = pd.DataFrame(index=df.index)
        
        # Add lagged returns
        for lag in [1, 2, 3]:
            features[f'Lag{lag}'] = returns.shift(lag)
        
        # Add rate of change
        features['ROC5'] = df['Price'].pct_change(periods=5)
        
        # Add moving averages
        features['MA10'] = df['Price'].rolling(window=self.config.MA_SHORT, min_periods=1).mean()
        features['MA50'] = df['Price'].rolling(window=self.config.MA_LONG, min_periods=1).mean()
        
        # Add MA ratio
        features['MA_ratio'] = features['MA10'] / features['MA50']
        
        # Add rolling standard deviation
        features['RollingStd5'] = returns.rolling(window=5, min_periods=1).std()
        
        # Add correlation with MA10 and MA50
        features['Corr_MA10'] = returns.rolling(window=self.config.MA_SHORT, min_periods=1).corr(features['MA10'])
        features['Corr_MA50'] = returns.rolling(window=self.config.MA_LONG, min_periods=1).corr(features['MA50'])
        
        # Add RSI
        features['RSI'] = self.calculate_rsi(df['Price'])
        
        # Add MACD
        features['MACD'] = self.calculate_macd(df['Price'])
        
        # Add ATR
        features['ATR'] = self.calculate_atr(df)
        
        # Add Bollinger Bands
        bb_upper, bb_lower = self.calculate_bollinger(df['Price'])
        features['Bollinger_Upper'] = bb_upper
        features['Bollinger_Lower'] = bb_lower
        
        # Add time-based features
        features['DayOfWeek'] = df.index.dayofweek
        features['MonthOfYear'] = df.index.month
        features['Quarter'] = df.index.quarter
        
        # Add rolling statistics with shorter windows only (5D, 10D, 20D)
        for window in [5, 10, 20]:
            features[f'Volatility{window}D'] = self.calculate_volatility(returns, window)
            features[f'Skew{window}D'] = returns.rolling(window=window, min_periods=1).skew()
            features[f'Kurt{window}D'] = returns.rolling(window=window, min_periods=1).kurt()
        
        # Use interpolation for missing values
        features = features.interpolate(method='linear', limit_direction='forward', axis=0)
        
        return features
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        return macd
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period, min_periods=1).mean()
    
    def calculate_bollinger(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        ma = prices.rolling(window=period, min_periods=1).mean()
        std = prices.rolling(window=period, min_periods=1).std()
        upper_band = ma + (std * std_dev)
        lower_band = ma - (std * std_dev)
        return upper_band, lower_band
    
    def add_vix_features(self, features: pd.DataFrame, vix_data: pd.DataFrame) -> pd.DataFrame:
        """Add VIX-related features"""
        # Use Adj Close if available, otherwise fall back to Close
        vix_price_col = 'Adj Close' if 'Adj Close' in vix_data.columns else 'Close'
        
        # Align VIX data with features index
        vix_aligned = vix_data[vix_price_col].reindex(features.index, method='ffill')
        
        features['VIX'] = vix_aligned
        features['VIX_change'] = vix_aligned.pct_change()
        
        # Z-score normalize VIX_change
        features['VIX_change'] = (features['VIX_change'] - features['VIX_change'].mean()) / features['VIX_change'].std()
        
        return features
    
    def calculate_correlations(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate inter-asset correlations"""
        self.logger.info("Starting correlation calculation")
        correlations = pd.DataFrame()
        
        # Extract returns for correlation calculation
        returns = {}
        for symbol, df in data.items():
            if symbol != 'VIX':
                # Use fallback logic for price column
                price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'

                if price_col in df.columns and not df[price_col].dropna().empty:
                    log_ret = self.calculate_log_returns(df[price_col])
                    if isinstance(log_ret, pd.Series) and not log_ret.dropna().empty:
                        returns[symbol] = log_ret
                        self.logger.info(f"✅ {symbol} - Valid return series with {log_ret.dropna().shape[0]} non-NaN values")
                    else:
                        self.logger.warning(f"⚠️ {symbol} - Log returns are empty after calculation")
                else:
                    self.logger.warning(f"⚠️ {symbol} - Missing or empty price column '{price_col}'")
        
        if len(returns) < 2:
            self.logger.warning("Insufficient assets for correlation calculation. Skipping correlations.")
            return pd.DataFrame()  # Return empty DataFrame instead of raising error
        
        returns_df = pd.DataFrame(returns)
        
        # Calculate rolling correlations as specified in thesis
        if 'AAPL' in returns_df.columns and '^GSPC' in returns_df.columns:
            correlations['AAPL_GSPC_corr'] = returns_df['AAPL'].rolling(
                window=self.config.CORRELATION_WINDOW
            ).corr(returns_df['^GSPC'])
        if 'IOZ.AX' in returns_df.columns and 'CBA.AX' in returns_df.columns:
            correlations['IOZ_CBA_corr'] = returns_df['IOZ.AX'].rolling(
                window=self.config.CORRELATION_WINDOW
            ).corr(returns_df['CBA.AX'])
        if 'BHP.AX' in returns_df.columns and 'IOZ.AX' in returns_df.columns:
            correlations['BHP_IOZ_corr'] = returns_df['BHP.AX'].rolling(
                window=self.config.CORRELATION_WINDOW
            ).corr(returns_df['IOZ.AX'])
        
        self.logger.info(f"Created correlation features: {correlations.columns.tolist()}")
        return correlations
    
    def create_regime_labels(self, returns: pd.Series, window: int = 60) -> pd.Series:
        """Create market regime labels (Bull, Bear, Sideways)"""
        # Calculate rolling returns
        rolling_returns = returns.rolling(window=window).sum()
        
        # Define regime thresholds
        bull_threshold = 0.1  # 10% positive return
        bear_threshold = -0.1  # 10% negative return
        
        # Assign regimes
        regimes = pd.Series(index=returns.index, dtype='object')
        regimes[rolling_returns > bull_threshold] = 'Bull'
        regimes[rolling_returns < bear_threshold] = 'Bear'
        regimes[(rolling_returns >= bear_threshold) & (rolling_returns <= bull_threshold)] = 'Sideways'
        
        return regimes
    
    def create_volatility_labels(self, volatility: pd.Series) -> pd.Series:
        """Create volatility regime labels using quartile binning for more granular classification"""
        self.logger.debug(f"Creating volatility labels for {len(volatility)} samples")
        
        if len(volatility) == 0:
            self.logger.error("Cannot create volatility labels: empty volatility series")
            return pd.Series(dtype='object')
        
        if volatility.isna().all():
            self.logger.error("Cannot create volatility labels: all values are NaN")
            return pd.Series(dtype='object')
        
        # Remove any remaining NaN values
        clean_volatility = volatility.dropna()
        
        if len(clean_volatility) < 4:
            self.logger.warning(f"Insufficient data for quartile binning: {len(clean_volatility)} samples")
            # Create simple labels based on mean
            mean_vol = clean_volatility.mean()
            labels = pd.Series(index=clean_volatility.index, dtype='object')
            labels[clean_volatility <= mean_vol] = 'Q1'
            labels[clean_volatility > mean_vol] = 'Q4'
            return labels
        
        try:
            # Use quartiles to create four balanced classes (Q1-Q4)
            labels = pd.qcut(clean_volatility, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
            self.logger.debug(f"Quartile-based labels created: {labels.value_counts().to_dict()}")
            return labels
            
        except Exception as e:
            self.logger.warning(f"Quartile binning failed: {e}. Using percentile-based approach")
            
            # Fallback to percentile-based labeling
            p25 = clean_volatility.quantile(0.25)
            p50 = clean_volatility.quantile(0.50)
            p75 = clean_volatility.quantile(0.75)
            
            labels = pd.Series(index=clean_volatility.index, dtype='object')
            labels[clean_volatility <= p25] = 'Q1'
            labels[(clean_volatility > p25) & (clean_volatility <= p50)] = 'Q2'
            labels[(clean_volatility > p50) & (clean_volatility <= p75)] = 'Q3'
            labels[clean_volatility > p75] = 'Q4'
            
            self.logger.debug(f"Percentile-based labels created: {labels.value_counts().to_dict()}")
            return labels

class ModelFactory:
    """Factory class for creating models"""
    
    @staticmethod
    def create_lstm_regressor(input_shape: Tuple[int, int], 
                            units: List[int] = [50, 30], 
                            dropout: float = 0.2) -> tf.keras.Model:
        """Create LSTM model for regression"""
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(units[0], return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout))
        
        # Second LSTM layer
        model.add(LSTM(units[1], return_sequences=False))
        model.add(Dropout(dropout))
        
        # Dense layers
        model.add(Dense(25, activation='relu'))
        model.add(Dense(1))  # Single output for regression
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='mse', 
                     metrics=['mae'])
        
        return model
    
    @staticmethod
    def create_lstm_classifier(input_shape: Tuple[int, int], 
                             n_classes: int = 4,  # Updated to 4 classes for Q1-Q4
                             units: List[int] = [50, 30], 
                             dropout: float = 0.2) -> tf.keras.Model:
        """Create LSTM model for classification with 4 volatility classes (Q1-Q4)"""
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(units[0], return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout))
        
        # Second LSTM layer
        model.add(LSTM(units[1], return_sequences=False))
        model.add(Dropout(dropout))
        
        # Dense layers
        model.add(Dense(25, activation='relu'))
        model.add(Dense(n_classes, activation='softmax'))  # 4 output classes for Q1-Q4
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='sparse_categorical_crossentropy', 
                     metrics=['accuracy'])
        
        return model
    
    @staticmethod
    def create_stockmixer(input_shape: Tuple[int, int], 
                         task: str = 'regression') -> tf.keras.Model:
        """
        Create StockMixer architecture as described in thesis
        MLP-based model with parallel pathways for temporal, indicator, and cross-stock mixing
        """
        from tensorflow.keras.layers import Input, Dense, Concatenate, BatchNormalization
        from tensorflow.keras.models import Model
        
        inputs = Input(shape=input_shape)
        
        # Flatten input for MLP processing
        flattened = tf.keras.layers.Flatten()(inputs)
        
        # Temporal pathway
        temporal = Dense(64, activation='relu')(flattened)
        temporal = BatchNormalization()(temporal)
        temporal = Dense(32, activation='relu')(temporal)
        
        # Indicator pathway
        indicator = Dense(64, activation='relu')(flattened)
        indicator = BatchNormalization()(indicator)
        indicator = Dense(32, activation='relu')(indicator)
        
        # Cross-asset pathway
        cross_asset = Dense(64, activation='relu')(flattened)
        cross_asset = BatchNormalization()(cross_asset)
        cross_asset = Dense(32, activation='relu')(cross_asset)
        
        # Concatenate pathways
        merged = Concatenate()([temporal, indicator, cross_asset])
        
        # Final layers
        mixed = Dense(64, activation='relu')(merged)
        mixed = BatchNormalization()(mixed)
        mixed = Dropout(0.2)(mixed)
        mixed = Dense(32, activation='relu')(mixed)
        
        # Output layer
        if task == 'regression':
            output = Dense(1)(mixed)
            model = Model(inputs=inputs, outputs=output)
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        else:
            output = Dense(3, activation='softmax')(mixed)
            model = Model(inputs=inputs, outputs=output)
            model.compile(optimizer=Adam(learning_rate=0.001), 
                         loss='sparse_categorical_crossentropy', 
                         metrics=['accuracy'])
        
        return model

class WalkForwardValidator:
    """Implements walk-forward cross-validation with dynamic sizing"""
    
    def __init__(self, n_splits: int = 5, test_size: int = 252):
        self.n_splits = n_splits
        self.test_size = test_size
        self.logger = logging.getLogger('risk_pipeline.WalkForwardValidator')
        self.logger.info(f"WalkForwardValidator initialized: {n_splits} splits, test_size={test_size}")
        
    def split(self, X: pd.DataFrame) -> List[Tuple[pd.Index, pd.Index]]:
        """Generate train/test indices for walk-forward validation with adaptive sizing"""
        n_samples = len(X)
        splits = []
        
        self.logger.info(f"WalkForward: Total samples={n_samples}, requested_splits={self.n_splits}, requested_test_size={self.test_size}")
        
        # CRITICAL FIX: Calculate viable parameters
        min_train_size = 60  # Minimum for meaningful models
        min_test_size = 20   # Minimum for evaluation
        
        # Calculate maximum feasible test size
        max_test_size = min(self.test_size, n_samples - min_train_size)
        
        if max_test_size < min_test_size:
            self.logger.error(f"Dataset too small: {n_samples} samples insufficient for validation")
            return []
        
        # Use adaptive test size
        actual_test_size = max(min_test_size, max_test_size)
        self.logger.info(f"Adaptive test_size: {actual_test_size} (requested: {self.test_size})")
        
        # Calculate maximum number of splits possible
        available_for_training = n_samples - actual_test_size
        max_possible_splits = min(self.n_splits, max(1, available_for_training // min_train_size))
        
        self.logger.info(f"Maximum possible splits: {max_possible_splits} (requested: {self.n_splits})")
        
        if max_possible_splits == 0:
            self.logger.error("Cannot create any valid splits")
            return []
        
        # Generate splits with expanding window
        for i in range(max_possible_splits):
            if max_possible_splits == 1:
                # Single split: use most data for training
                train_end = n_samples - actual_test_size
            else:
                # Multiple splits: expanding window
                train_end = min_train_size + ((available_for_training - min_train_size) * (i + 1)) // max_possible_splits
                
            test_start = train_end
            test_end = min(test_start + actual_test_size, n_samples)
            
            # Ensure we have valid indices
            if train_end <= 0 or test_start >= n_samples or test_end > n_samples or test_start >= test_end:
                self.logger.warning(f"Split {i+1}: Invalid indices, skipping")
                continue
                
            train_idx = X.index[:train_end]
            test_idx = X.index[test_start:test_end]
            
            # Final validation
            if len(train_idx) >= min_train_size and len(test_idx) >= min_test_size:
                splits.append((train_idx, test_idx))
                self.logger.info(f"✅ Split {i+1}: Train={len(train_idx)}, Test={len(test_idx)}")
            else:
                self.logger.warning(f"Split {i+1}: Insufficient data (Train={len(train_idx)}, Test={len(test_idx)})")
        
        self.logger.info(f"Generated {len(splits)} valid splits")
        return splits

class RiskPipeline:
    """Main pipeline class orchestrating the entire workflow"""
    
    def __init__(self, config: AssetConfig = AssetConfig()):
        self.config = config
        self.logger = logging.getLogger('risk_pipeline.RiskPipeline')
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer(config)
        self.model_factory = ModelFactory()
        self.validator = WalkForwardValidator(
            n_splits=config.WALK_FORWARD_SPLITS,
            test_size=config.TEST_SIZE
        )
        self.results = {}
        self.models = {}
        self.scalers = {}
        self.logger.info("RiskPipeline initialized successfully")
        
    @log_execution_time
    def run_pipeline(self, assets: List[str] = None, skip_correlations: bool = False, debug: bool = False):
        """Execute the complete pipeline with enhanced timing"""
        try:
            start_time = datetime.now()  # Add start_time at the beginning
            
            # Step 1: Data loading
            step_start = datetime.now()
            self.logger.info("Step 1: Loading data...")
            
            raw_data = self.data_loader.download_data(
                symbols=self.config.ASSETS if assets is None else assets,
                start_date=self.config.START_DATE,
                end_date=self.config.END_DATE
            )
            
            if not raw_data:
                self.logger.error("No data downloaded. Exiting pipeline.")
                return
            
            step_time = (datetime.now() - step_start).total_seconds()
            self.logger.info(f"✅ Step 1 completed in {step_time:.2f} seconds")
            
            # Step 2: Feature engineering
            step_start = datetime.now()
            self.logger.info("Step 2: Engineering features...")
            
            processed_data = {}
            for asset, df in raw_data.items():
                if asset == '^VIX':
                    continue
                    
                asset_start = datetime.now()
                self.logger.info(f"Processing features for {asset}")
                
                # Create technical features
                features = self.feature_engineer.create_technical_features(raw_data[asset])
                self.logger.debug(f"{asset} - Technical features shape: {features.shape}")
                
                # Add VIX features
                if '^VIX' in raw_data:
                    features = self.feature_engineer.add_vix_features(features, raw_data['^VIX'])
                    self.logger.debug(f"{asset} - Added VIX features")
                
                # Add correlation features if not skipped
                if not skip_correlations:
                    try:
                        correlations = self.feature_engineer.calculate_correlations(raw_data)
                        for col in correlations.columns:
                            if col in correlations:
                                features[col] = correlations[col]
                        self.logger.debug(f"{asset} - Added correlation features: {correlations.columns.tolist()}")
                    except Exception as e:
                        self.logger.warning(f"Skipping correlation features for {asset}: {e}")
                
                # Create labels BEFORE cleaning data
                price_col = 'Adj Close' if 'Adj Close' in raw_data[asset].columns else 'Close'
                returns = self.feature_engineer.calculate_log_returns(raw_data[asset][price_col])
                
                # Create regime labels
                self.logger.debug(f"{asset} - Creating regime labels...")
                features['Regime'] = self.feature_engineer.create_regime_labels(returns)
                self.logger.debug(f"{asset} - Regime labels created")
                
                # Create volatility labels
                self.logger.debug(f"{asset} - Creating volatility labels...")
                features['VolatilityLabel'] = self.feature_engineer.create_volatility_labels(returns)
                self.logger.debug(f"{asset} - Volatility labels created")
                
                # Verify required columns exist
                required_columns = ['VolatilityLabel', 'Regime']
                missing_columns = [col for col in required_columns if col not in features.columns]
                
                if missing_columns:
                    self.logger.error(f"{asset} - Missing required columns: {missing_columns}")
                    continue
                
                # Enhanced data cleaning
                original_shape = features.shape
                
                # Separate categorical and numerical columns
                categorical_cols = ['Regime', 'VolatilityLabel']
                numerical_cols = [col for col in features.columns if col not in categorical_cols]
                
                # Enhanced NaN handling for numerical columns
                for col in numerical_cols:
                    # First try linear interpolation
                    features[col] = features[col].interpolate(method='linear', limit_direction='both')
                    # Then fill remaining NaNs with rolling mean
                    features[col] = features[col].fillna(features[col].rolling(window=5, min_periods=1).mean())
                    # Finally, fill any remaining NaNs with column mean
                    features[col] = features[col].fillna(features[col].mean())
                
                # Handle categorical columns with forward fill and backward fill
                features[categorical_cols] = features[categorical_cols].fillna(method='ffill').fillna(method='bfill')
                
                # Verify no NaNs remain
                if features.isna().any().any():
                    self.logger.warning(f"{asset} - NaN values remain after cleaning. Dropping rows with NaNs.")
                    features = features.dropna()
                
                self.logger.debug(f"{asset} - Data cleaned: {original_shape} -> {features.shape}")
                
                # Verify we still have data after cleaning
                if features.empty:
                    self.logger.error(f"{asset} - No data remaining after cleaning")
                    continue
                
                # Verify VolatilityLabel has valid values
                vol_label_counts = features['VolatilityLabel'].value_counts()
                if vol_label_counts.empty:
                    self.logger.error(f"{asset} - No valid VolatilityLabel values after cleaning")
                    continue
                
                self.logger.info(f"{asset} - Final processed data shape: {features.shape}")
                self.logger.debug(f"{asset} - VolatilityLabel distribution: {vol_label_counts.to_dict()}")
                
                processed_data[asset] = features
                
                asset_time = (datetime.now() - asset_start).total_seconds()
                self.logger.debug(f"{asset} feature engineering completed in {asset_time:.2f} seconds")

            step_time = (datetime.now() - step_start).total_seconds()
            self.logger.info(f"✅ Step 2 completed in {step_time:.2f} seconds")
            
            # Step 3: Model training and evaluation
            step_start = datetime.now()
            self.logger.info("Step 3: Training and evaluating models...")

            total_models_trained = 0
            for asset, data in processed_data.items():
                self.logger.info(f"\nProcessing {asset}...")
                self.results[asset] = {}
                
                # Prepare features and targets with validation
                feature_cols = ['Lag1', 'Lag2', 'Lag3', 'ROC5', 'MA10', 'MA50', 
                              'RollingStd5', 'MA_ratio', 'Corr_MA10', 'Corr_MA50']
                
                # Add correlation features if available
                for col in ['AAPL_GSPC_corr', 'IOZ_CBA_corr', 'BHP_IOZ_corr']:
                    if col in data.columns:
                        feature_cols.append(col)
                
                # Validate feature columns exist
                available_features = [col for col in feature_cols if col in data.columns]
                missing_features = [col for col in feature_cols if col not in data.columns]
                
                if missing_features:
                    self.logger.warning(f"{asset} - Missing features: {missing_features}")
                
                self.logger.debug(f"{asset} - Using features: {available_features}")
                
                # Validate required columns exist
                required_cols = ['VolatilityLabel', 'Regime']
                missing_required = [col for col in required_cols if col not in data.columns]
                
                if missing_required:
                    self.logger.error(f"{asset} - Missing required columns: {missing_required}. Skipping asset.")
                    continue
                
                try:
                    X = data[available_features]
                    y_reg = data['Volatility5D']
                    
                    # Validate VolatilityLabel before mapping
                    vol_labels = data['VolatilityLabel']
                    self.logger.debug(f"{asset} - VolatilityLabel unique values: {vol_labels.unique()}")
                    
                    # Check for valid label values
                    valid_labels = {'Q1', 'Q2', 'Q3', 'Q4'}
                    actual_labels = set(vol_labels.dropna().unique())
                    
                    if not actual_labels.issubset(valid_labels):
                        self.logger.error(f"{asset} - Invalid volatility labels: {actual_labels}. Expected: {valid_labels}")
                        continue
                    
                    # Map labels to numbers (0=Q1, 1=Q2, 2=Q3, 3=Q4)
                    y_clf = vol_labels.map({'Q1': 0, 'Q2': 1, 'Q3': 2, 'Q4': 3})
                    
                    # Validate we have valid data
                    if X.empty or y_reg.empty or y_clf.empty:
                        self.logger.error(f"{asset} - Empty data after processing. Skipping.")
                        continue
                    
                    # Check for NaN values
                    if X.isna().any().any():
                        self.logger.warning(f"{asset} - Features contain NaN values")
                        
                    if y_reg.isna().any():
                        self.logger.warning(f"{asset} - Regression target contains NaN values")
                        
                    if y_clf.isna().any():
                        self.logger.warning(f"{asset} - Classification target contains NaN values")
                    
                    self.logger.info(f"{asset} - Feature matrix shape: {X.shape}, Regression target: {y_reg.shape}, Classification target: {y_clf.shape}")
                    self.logger.debug(f"{asset} - Volatility quartile distribution: {vol_labels.value_counts().to_dict()}")
                    
                    # Run regression models
                    self.logger.info(f"Training regression models for {asset}...")
                    self.results[asset]['regression'] = self._run_regression_models(X, y_reg, asset)
                    
                    # Run classification models
                    self.logger.info(f"Training classification models for {asset}...")
                    self.results[asset]['classification'] = self._run_classification_models(X, y_clf, asset)
                    
                    # Count models trained
                    asset_models = len(self.results[asset]['regression']) + len(self.results[asset]['classification'])
                    total_models_trained += asset_models
                    self.logger.debug(f"{asset}: Trained {asset_models} models")
                    
                except Exception as e:
                    self.logger.error(f"{asset} - Model training failed: {e}", exc_info=True)
                    continue
            
            step_time = (datetime.now() - step_start).total_seconds()
            self.logger.info(f"✅ Step 3 completed in {step_time:.2f} seconds - Trained {total_models_trained} models")
            
            # Step 4: SHAP analysis
            step_start = datetime.now()
            self.logger.info("Step 4: Generating SHAP interpretability analysis...")
            self._generate_shap_analysis()
            step_time = (datetime.now() - step_start).total_seconds()
            self.logger.info(f"✅ Step 4 completed in {step_time:.2f} seconds")
            
            # Step 5: Save results
            step_start = datetime.now()
            self.logger.info("Step 5: Saving results...")
            self._save_results()
            step_time = (datetime.now() - step_start).total_seconds()
            self.logger.info(f"✅ Step 5 completed in {step_time:.2f} seconds")
            
            # Final summary
            total_time = (datetime.now() - start_time).total_seconds()
            self.logger.info("=" * 80)
            self.logger.info("PIPELINE EXECUTION SUMMARY")
            self.logger.info("=" * 80)
            self.logger.info(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
            self.logger.info(f"Assets processed: {len(processed_data)}")
            self.logger.info(f"Models trained: {total_models_trained}")
            self.logger.info(f"Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info("=" * 80)
            self.logger.info("✅ Pipeline execution completed successfully!")
            
        except Exception as e:
            total_time = (datetime.now() - start_time).total_seconds()
            self.logger.error("=" * 80)
            self.logger.error("PIPELINE EXECUTION FAILED")
            self.logger.error("=" * 80)
            self.logger.error(f"Error after {total_time:.2f} seconds: {e}")
            self.logger.error("=" * 80)
            raise
    
    def _run_regression_models(self, X: pd.DataFrame, y: pd.Series, asset: str) -> Dict:
        """Run all regression models with walk-forward validation"""
        results = {}
        
        # Models to evaluate
        models = {
            'Naive_MA': None,  # Special case
            'ARIMA': None,     # Special case
            'LSTM': 'lstm',
            'StockMixer': 'stockmixer'
        }
        
        for model_name, model_type in models.items():
            self.logger.info(f"  Training {model_name}...")
            
            if model_name == 'Naive_MA':
                # Naive baseline: use previous value
                results[model_name] = self._evaluate_naive_baseline(X, y)
                
            elif model_name == 'ARIMA':
                # ARIMA requires special handling
                results[model_name] = self._evaluate_arima(X, y)
                
            else:
                # Deep learning models
                results[model_name] = self._evaluate_dl_model(
                    X, y, model_type, task='regression', asset=asset
                )
                
        return results
    
    def _run_classification_models(self, X: pd.DataFrame, y: pd.Series, asset: str) -> Dict:
        """Run all classification models with walk-forward validation and SMOTE+Tomek for class balancing"""
        results = {}
        
        # ===== FIX: Log the exact features being used =====
        self.logger.info(f"{asset} - Training classification models with features: {X.columns.tolist()}")
        
        # Models to evaluate
        models = {
            'Random': DummyClassifier(strategy='stratified', random_state=self.config.RANDOM_STATE),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=self.config.RANDOM_STATE,
                use_label_encoder=False,
                eval_metric='mlogloss'
            ),
        }
        
        for model_name, model in models.items():
            self.logger.info(f"  Training {model_name}...")
            
            try:
                if isinstance(model, str):
                    results[model_name] = self._evaluate_dl_model(
                        X, y, model, task='classification', asset=asset
                    )
                else:
                    results[model_name] = self._evaluate_sklearn_model(X, y, model, asset, model_name)
                    
                if results[model_name] and 'Accuracy' in results[model_name]:
                    accuracy = results[model_name]['Accuracy']
                    if not np.isfinite(accuracy) or accuracy < 0 or accuracy > 1:
                        self.logger.warning(f"{model_name} produced invalid accuracy: {accuracy}")
                        results[model_name]['Accuracy'] = 0.0
                else:
                    self.logger.warning(f"{model_name} failed to produce valid results")
                    
            except Exception as e:
                self.logger.error(f"Model {model_name} failed: {e}")
                results[model_name] = {
                    'Accuracy': 0.0, 'F1': 0.0, 'Precision': 0.0, 'Recall': 0.0,
                    'predictions': [], 'actuals': []
                }
                
        return results
    
    def _evaluate_naive_baseline(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Evaluate naive baseline with comprehensive error handling"""
        try:
            splits = self.validator.split(X)
            
            if not splits:
                self.logger.error("No valid splits for naive baseline evaluation")
                return {
                    'RMSE': float('inf'),
                    'MAE': float('inf'),
                    'R2': -float('inf'),
                    'predictions': [],
                    'actuals': []
                }
            
            predictions = []
            actuals = []
            
            for train_idx, test_idx in splits:
                try:
                    y_train = y.loc[train_idx]
                    y_test = y.loc[test_idx]
                    
                    if len(y_train) == 0 or len(y_test) == 0:
                        self.logger.warning("Empty train or test set in naive baseline")
                        continue
                    
                    # Naive prediction: use last known value
                    last_value = y_train.iloc[-1] if len(y_train) > 0 else 0.1  # fallback
                    y_pred = pd.Series(index=test_idx, data=last_value)
                    
                    predictions.extend(y_pred.values)
                    actuals.extend(y_test.values)
                    
                except Exception as e:
                    self.logger.error(f"Error in naive baseline fold: {e}")
                    continue
            
            if not predictions or not actuals:
                self.logger.error("No predictions generated for naive baseline")
                return {
                    'RMSE': float('inf'),
                    'MAE': float('inf'), 
                    'R2': -float('inf'),
                    'predictions': predictions,
                    'actuals': actuals
                }
            
            # Calculate metrics with error handling
            try:
                rmse = np.sqrt(mean_squared_error(actuals, predictions))
                mae = mean_absolute_error(actuals, predictions)
                r2 = r2_score(actuals, predictions)
            except Exception as e:
                self.logger.error(f"Error calculating naive baseline metrics: {e}")
                return {
                    'RMSE': float('inf'),
                    'MAE': float('inf'),
                    'R2': -float('inf'),
                    'predictions': predictions,
                    'actuals': actuals
                }
            
            return {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'predictions': predictions,
                'actuals': actuals
            }
            
        except Exception as e:
            self.logger.error(f"Naive baseline evaluation failed: {e}")
            return {
                'RMSE': float('inf'),
                'MAE': float('inf'),
                'R2': -float('inf'),
                'predictions': [],
                'actuals': []
            }
    
    def _evaluate_arima(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Evaluate ARIMA model"""
        splits = self.validator.split(X)
        predictions = []
        actuals = []
        
        for train_idx, test_idx in splits:
            y_train = y.loc[train_idx]
            y_test = y.loc[test_idx]
            
            try:
                # Fit ARIMA model
                model = ARIMA(y_train, order=(1, 1, 1))
                fitted_model = model.fit()
                
                # Make predictions
                y_pred = fitted_model.forecast(steps=len(test_idx))
                
                predictions.extend(y_pred)
                actuals.extend(y_test.values)
                
            except Exception as e:
                self.logger.warning(f"ARIMA failed for fold: {e}")
                # Use naive fallback
                y_pred = [y_train.iloc[-1]] * len(test_idx)
                predictions.extend(y_pred)
                actuals.extend(y_test.values)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'predictions': predictions,
            'actuals': actuals
        }
    
    def _evaluate_dl_model(self, X: pd.DataFrame, y: pd.Series, 
                          model_type: str, task: str, asset: str) -> Dict:
        """Evaluate deep learning models with improved NaN handling"""
        splits = self.validator.split(X)
        predictions = []
        actuals = []
        
        # Prepare data for LSTM (requires 3D input)
        sequence_length = 15  # Look back 15 days
        
        self.logger.info(f"Starting DL model evaluation for {model_type} on {asset}")
        self.logger.info(f"Total data shape: X={X.shape}, y={y.shape}")
        self.logger.info(f"Number of CV splits: {len(splits)}")
        
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            self.logger.info(f"Processing fold {fold_idx + 1}/{len(splits)}")
            
            X_train, X_test = X.loc[train_idx], X.loc[test_idx]
            y_train, y_test = y.loc[train_idx], y.loc[test_idx]
            
            # Enhanced NaN handling for training data
            if X_train.isna().any().any() or y_train.isna().any():
                self.logger.warning(f"NaN values found in training data. Handling them...")
                # Enhanced NaN handling for features
                for col in X_train.columns:
                    # Try linear interpolation first
                    X_train[col] = X_train[col].interpolate(method='linear', limit_direction='both')
                    # Then use rolling mean
                    X_train[col] = X_train[col].fillna(X_train[col].rolling(window=5, min_periods=1).mean())
                    # Finally use column mean
                    X_train[col] = X_train[col].fillna(X_train[col].mean())
                
                # Handle target variable NaNs
                y_train = y_train.interpolate(method='linear', limit_direction='both')
                y_train = y_train.fillna(y_train.rolling(window=5, min_periods=1).mean())
                y_train = y_train.fillna(y_train.mean())
            
            # Enhanced NaN handling for test data
            if X_test.isna().any().any() or y_test.isna().any():
                self.logger.warning(f"NaN values found in test data. Handling them...")
                # Use the same approach as training data
                for col in X_test.columns:
                    X_test[col] = X_test[col].interpolate(method='linear', limit_direction='both')
                    X_test[col] = X_test[col].fillna(X_test[col].rolling(window=5, min_periods=1).mean())
                    X_test[col] = X_test[col].fillna(X_test[col].mean())
                
                y_test = y_test.interpolate(method='linear', limit_direction='both')
                y_test = y_test.fillna(y_test.rolling(window=5, min_periods=1).mean())
                y_test = y_test.fillna(y_test.mean())
            
            self.logger.info(f"Fold {fold_idx + 1}: Train shape={X_train.shape}, Test shape={X_test.shape}")
            
            # Verify no NaNs remain
            if X_train.isna().any().any() or y_train.isna().any():
                self.logger.error(f"Fold {fold_idx + 1}: NaN values remain in training data after preprocessing")
                continue
                
            if X_test.isna().any().any() or y_test.isna().any():
                self.logger.error(f"Fold {fold_idx + 1}: NaN values remain in test data after preprocessing")
                continue
            
            # Check if we have enough data
            if len(X_train) < sequence_length + 10:
                self.logger.warning(f"Fold {fold_idx + 1}: Insufficient training data ({len(X_train)} samples), skipping")
                continue
            
            if len(X_test) < sequence_length:
                self.logger.warning(f"Fold {fold_idx + 1}: Insufficient test data ({len(X_test)} samples), skipping")
                continue
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Create sequences with validation
            try:
                X_train_seq, y_train_seq = self._create_sequences(
                    X_train_scaled, y_train.values, sequence_length
                )
                X_test_seq, y_test_seq = self._create_sequences(
                    X_test_scaled, y_test.values, sequence_length
                )
                
                # Verify sequences
                if np.isnan(X_train_seq).any() or np.isnan(y_train_seq).any():
                    self.logger.error(f"Fold {fold_idx + 1}: NaN values in training sequences after preprocessing")
                    continue
                    
                if np.isnan(X_test_seq).any() or np.isnan(y_test_seq).any():
                    self.logger.error(f"Fold {fold_idx + 1}: NaN values in test sequences after preprocessing")
                    continue
                
                self.logger.info(f"Fold {fold_idx + 1}: Sequences created - Train: {X_train_seq.shape}, Test: {X_test_seq.shape}")
                
            except Exception as e:
                self.logger.error(f"Fold {fold_idx + 1}: Error creating sequences: {e}")
                continue
            
            # Create model
            input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
            
            try:
                if model_type == 'lstm':
                    if task == 'regression':
                        model = self.model_factory.create_lstm_regressor(input_shape)
                    else:
                        model = self.model_factory.create_lstm_classifier(input_shape)
                elif model_type == 'stockmixer':
                    model = self.model_factory.create_stockmixer(input_shape, task)
                elif model_type == 'lstm_clf':
                    model = self.model_factory.create_lstm_classifier(input_shape)
                
                self.logger.info(f"Fold {fold_idx + 1}: Model created successfully")
                
                # Train model with NaN handling
                early_stop = EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True,
                    mode='min'
                )
                reduce_lr = ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=0.00001,
                    mode='min'
                )
                
                # Set epochs to 100
                epochs = 100
                
                history = model.fit(
                    X_train_seq, y_train_seq,
                    epochs=epochs,
                    batch_size=16,
                    validation_split=0.2,
                    callbacks=[early_stop, reduce_lr],
                    verbose=0
                )
                
                self.logger.info(f"Fold {fold_idx + 1}: Model training completed")
                
                # Make predictions with validation
                y_pred = model.predict(X_test_seq, verbose=0)
                
                # Check for NaN in predictions
                if np.isnan(y_pred).any():
                    self.logger.error(f"Fold {fold_idx + 1}: NaN values in predictions")
                    continue
                
                if task == 'classification':
                    y_pred = np.argmax(y_pred, axis=1)
                else:
                    y_pred = y_pred.flatten()
                
                # Validate predictions and actuals
                if len(y_pred) != len(y_test_seq):
                    self.logger.error(f"Fold {fold_idx + 1}: Prediction length mismatch")
                    continue
                    
                if np.isnan(y_pred).any() or np.isnan(y_test_seq).any():
                    self.logger.error(f"Fold {fold_idx + 1}: NaN values in predictions or actuals")
                    continue
                
                self.logger.info(f"Fold {fold_idx + 1}: Predictions shape={y_pred.shape}, Test targets shape={y_test_seq.shape}")
                
                predictions.extend(y_pred)
                actuals.extend(y_test_seq)
                
                # Save model for last fold (for SHAP analysis)
                if fold_idx == len(splits) - 1:
                    model_key = f"{asset}_{model_type}_{task}"
                    self.models[model_key] = model
                    self.scalers[model_key] = scaler
                
            except Exception as e:
                self.logger.error(f"Fold {fold_idx + 1}: Model training failed: {e}")
                continue
        
        # Check if we have any predictions
        if len(predictions) == 0 or len(actuals) == 0:
            self.logger.error(f"No predictions generated for {model_type} on {asset}")
            # Return default metrics indicating failure
            if task == 'regression':
                return {
                    'RMSE': float('inf'),
                    'MAE': float('inf'),
                    'R2': -float('inf'),
                    'predictions': [],
                    'actuals': []
                }
            else:
                return {
                    'Accuracy': 0.0,
                    'F1': 0.0,
                    'Precision': 0.0,
                    'Recall': 0.0,
                    'predictions': [],
                    'actuals': []
                }
        
        # Convert to numpy arrays for final validation
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Final NaN check
        if np.isnan(predictions).any() or np.isnan(actuals).any():
            self.logger.error(f"NaN values found in final predictions or actuals")
            if task == 'regression':
                return {
                    'RMSE': float('inf'),
                    'MAE': float('inf'),
                    'R2': -float('inf'),
                    'predictions': predictions.tolist(),
                    'actuals': actuals.tolist()
                }
            else:
                return {
                    'Accuracy': 0.0,
                    'F1': 0.0,
                    'Precision': 0.0,
                    'Recall': 0.0,
                    'predictions': predictions.tolist(),
                    'actuals': actuals.tolist()
                }
        
        self.logger.info(f"Final results for {model_type}: {len(predictions)} predictions, {len(actuals)} actuals")
        
        # Calculate metrics
        try:
            if task == 'regression':
                rmse = np.sqrt(mean_squared_error(actuals, predictions))
                mae = mean_absolute_error(actuals, predictions)
                r2 = r2_score(actuals, predictions)
                
                return {
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2,
                    'predictions': predictions.tolist(),
                    'actuals': actuals.tolist()
                }
            else:
                accuracy = accuracy_score(actuals, predictions)
                f1 = f1_score(actuals, predictions, average='weighted', zero_division=0)
                precision = precision_score(actuals, predictions, average='weighted', zero_division=0)
                recall = recall_score(actuals, predictions, average='weighted', zero_division=0)
                
                return {
                    'Accuracy': accuracy,
                    'F1': f1,
                    'Precision': precision,
                    'Recall': recall,
                    'predictions': predictions.tolist(),
                    'actuals': actuals.tolist()
                }
        except Exception as e:
            self.logger.error(f"Metrics calculation failed for {model_type}: {e}")
            # Return safe default values
            if task == 'regression':
                return {
                    'RMSE': float('inf'),
                    'MAE': float('inf'),
                    'R2': -float('inf'),
                    'predictions': predictions.tolist(),
                    'actuals': actuals.tolist()
                }
            else:
                return {
                    'Accuracy': 0.0,
                    'F1': 0.0,
                    'Precision': 0.0,
                    'Recall': 0.0,
                    'predictions': predictions.tolist(),
                    'actuals': actuals.tolist()
                }
    
    def _evaluate_sklearn_model(self, X: pd.DataFrame, y: pd.Series, 
                               model, asset: str, model_name: str) -> Dict:
        """Evaluate sklearn-compatible models with SMOTE+Tomek for class balancing"""
        splits = self.validator.split(X)
        predictions = []
        actuals = []
        
        # ===== FIX: Store feature names for SHAP consistency =====
        feature_names_used = X.columns.tolist()
        self.logger.debug(f"{asset} - {model_name} using features: {feature_names_used}")
        
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X.loc[train_idx], X.loc[test_idx]
            y_train, y_test = y.loc[train_idx], y.loc[test_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Apply SMOTE+Tomek to training data only
            try:
                # Log original class distribution
                original_dist = Counter(y_train)
                self.logger.debug(f"Fold {fold_idx + 1} - Original class distribution: {dict(original_dist)}")
                
                # Apply SMOTE+Tomek with error handling
                smt = SMOTETomek(smote=SMOTE(k_neighbors=1), random_state=self.config.RANDOM_STATE)
                X_train_resampled, y_train_resampled = smt.fit_resample(X_train_scaled, y_train)
                
                # Log new class distribution
                new_dist = Counter(y_train_resampled)
                self.logger.debug(f"Fold {fold_idx + 1} - New class distribution: {dict(new_dist)}")
                
                # Optional: Visualize class balance for debugging
                if fold_idx == 0:  # Only for first fold
                    plt.figure(figsize=(10, 6))
                    sns.countplot(x=y_train_resampled)
                    plt.title(f"Class Distribution After SMOTE+Tomek - {asset}")
                    plt.savefig(f'class_balance_{asset}_{model_name}.png')
                    plt.close()
                
            except Exception as e:
                self.logger.warning(f"SMOTE+Tomek failed for fold {fold_idx + 1}: {e}. Using original data.")
                X_train_resampled, y_train_resampled = X_train_scaled, y_train
            
            # Train model
            model_copy = model.__class__(**model.get_params())
            model_copy.fit(X_train_resampled, y_train_resampled)
            
            # Make predictions
            y_pred = model_copy.predict(X_test_scaled)
            
            predictions.extend(y_pred)
            actuals.extend(y_test.values)
            
            # ===== FIX: Save model and scaler with feature names for last fold =====
            if fold_idx == len(splits) - 1 and model_name == 'XGBoost':
                model_key = f"{asset}_xgboost_classification"
                self.models[model_key] = model_copy
                
                # Store feature names in the scaler for SHAP consistency
                scaler.feature_names_in_ = feature_names_used
                self.scalers[model_key] = scaler
                
                self.logger.debug(f"Stored {model_name} model and scaler for {asset} with features: {feature_names_used}")
        
        # Calculate metrics
        accuracy = accuracy_score(actuals, predictions)
        f1 = f1_score(actuals, predictions, average='weighted', zero_division=0)
        precision = precision_score(actuals, predictions, average='weighted', zero_division=0)
        recall = recall_score(actuals, predictions, average='weighted', zero_division=0)
        
        return {
            'Accuracy': accuracy,
            'F1': f1,
            'Precision': precision,
            'Recall': recall,
            'predictions': predictions,
            'actuals': actuals
        }
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, 
                         sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM input with stride=1 for maximum training size"""
        X_seq, y_seq = [], []
        stride = 1  # Use stride=1 to maximize training size
        
        for i in range(0, len(X) - sequence_length, stride):
            X_seq.append(X[i:i+sequence_length])
            y_seq.append(y[i+sequence_length-1])  # Use the last value in the sequence
            
        return np.array(X_seq), np.array(y_seq)
    
    @log_execution_time
    def _generate_shap_analysis(self):
        """Generate SHAP interpretability analysis with proper feature alignment"""
        self.logger.info("Generating SHAP analysis...")
        
        shap_results = {}
        
        try:
            self.logger.info("Loading all data for SHAP analysis...")
            all_assets_data = self.data_loader.download_data(
                self.config.ALL_ASSETS + ['^VIX'], 
                self.config.START_DATE, 
                self.config.END_DATE
            )
            
            if not all_assets_data:
                self.logger.error("Failed to load data for SHAP analysis")
                return
            
            self.logger.info(f"Loaded data for {len(all_assets_data)} symbols")
            
        except Exception as e:
            self.logger.error(f"Failed to load data for SHAP analysis: {e}")
            return
        
        # Focus on XGBoost for classification (as it's most interpretable)
        for asset in self.results.keys():
            model_key = f"{asset}_xgboost_classification"
            
            if model_key in self.models:
                self.logger.info(f"Starting SHAP analysis for {asset}")
                try:
                    model = self.models[model_key]
                    scaler = self.scalers[model_key]
                    
                    # ===== FIX: Get the exact feature list used during training =====
                    if hasattr(scaler, 'feature_names_in_'):
                        training_feature_cols = list(scaler.feature_names_in_)
                        self.logger.info(f"Using stored training features for {asset}: {training_feature_cols}")
                    else:
                        # Fallback to the feature list without VIX features (as shown in the error)
                        training_feature_cols = ['Lag1', 'Lag2', 'Lag3', 'ROC5', 'MA10', 'MA50', 
                                               'RollingStd5', 'MA_ratio', 'AAPL_GSPC_corr', 'IOZ_CBA_corr', 'BHP_IOZ_corr']
                        self.logger.warning(f"No stored features found for {asset}, using fallback list: {training_feature_cols}")
                    
                    # ===== STEP 1: Reconstruct features using EXACT same process as training =====
                    if asset not in all_assets_data:
                        self.logger.warning(f"No data available for {asset} in SHAP analysis")
                        continue
                    
                    # Create technical features
                    self.logger.debug(f"{asset} - Creating technical features for SHAP...")
                    features = self.feature_engineer.create_technical_features(all_assets_data[asset])
                    
                    # ===== FIX: Only add VIX features if they were used during training =====
                    if 'VIX' in training_feature_cols and 'VIX_change' in training_feature_cols:
                        if '^VIX' in all_assets_data:
                            features = self.feature_engineer.add_vix_features(features, all_assets_data['^VIX'])
                            self.logger.debug(f"{asset} - VIX features added successfully")
                        else:
                            self.logger.warning(f"{asset} - VIX features required but VIX data not available")
                            continue
                    else:
                        self.logger.debug(f"{asset} - Skipping VIX features (not used during training)")
                    
                    # ===== STEP 2: Calculate correlation features only if used during training =====
                    correlation_features_needed = [f for f in training_feature_cols if '_corr' in f]
                    if correlation_features_needed:
                        self.logger.debug(f"{asset} - Calculating correlation features for SHAP...")
                        try:
                            correlations = self.feature_engineer.calculate_correlations(all_assets_data)
                            
                            # Add only the correlation features that were used during training
                            for corr_col in correlation_features_needed:
                                if corr_col in correlations.columns and not correlations[corr_col].empty:
                                    aligned_corr = correlations[corr_col].reindex(features.index, method='ffill')
                                    features[corr_col] = aligned_corr
                                    self.logger.debug(f"{asset} - Added correlation feature: {corr_col}")
                            
                            added_corr_features = [f for f in correlation_features_needed if f in features.columns]
                            self.logger.info(f"{asset} - Correlation features added: {added_corr_features}")
                            
                        except Exception as e:
                            self.logger.error(f"{asset} - Failed to calculate correlations for SHAP: {e}")
                            continue
                    
                    # ===== STEP 3: Extract data in EXACT same order as training =====
                    try:
                        # Validate all required features exist
                        missing_features = [col for col in training_feature_cols if col not in features.columns]
                        
                        if missing_features:
                            self.logger.error(f"Missing features for {asset} SHAP analysis: {missing_features}")
                            self.logger.info(f"Available features: {features.columns.tolist()}")
                            continue
                        
                        X = features[training_feature_cols].dropna()
                        
                        if X.empty:
                            self.logger.warning(f"No valid data for {asset} SHAP analysis after dropna()")
                            continue
                        
                        # Use last 100 samples for SHAP analysis
                        X = X.iloc[-100:] if len(X) > 100 else X
                        
                        self.logger.debug(f"{asset} - SHAP data shape: {X.shape}")
                        self.logger.debug(f"{asset} - SHAP feature columns: {X.columns.tolist()}")
                        
                        # ===== STEP 4: Scale using the same scaler from training =====
                        self.logger.debug(f"{asset} - Scaling features for SHAP analysis...")
                        X_scaled = scaler.transform(X)
                        self.logger.debug(f"{asset} - Features scaled successfully, shape: {X_scaled.shape}")
                        
                        # ===== STEP 5: Create SHAP explainer and calculate values =====
                        self.logger.info(f"{asset} - Creating SHAP explainer...")
                        explainer = shap.TreeExplainer(model)
                        self.logger.debug(f"{asset} - SHAP explainer created")
                        
                        # Calculate SHAP values
                        self.logger.info(f"{asset} - Computing SHAP values...")
                        shap_values = explainer.shap_values(X_scaled)
                        self.logger.debug(f"{asset} - SHAP values computed, type: {type(shap_values)}")
                        
                        if isinstance(shap_values, list):
                            self.logger.debug(f"{asset} - SHAP values shape: {[np.array(sv).shape for sv in shap_values]}")
                        else:
                            self.logger.debug(f"{asset} - SHAP values shape: {np.array(shap_values).shape}")
                        
                        # ===== STEP 6: Save SHAP plots =====
                        self.logger.info(f"{asset} - Generating SHAP plots...")
                        plot_files = self._save_shap_plots(shap_values, X, asset)
                        
                        # Store results
                        shap_results[asset] = {
                            'feature_count': len(X.columns),
                            'sample_count': len(X),
                            'plot_files': plot_files
                        }
                        
                        self.logger.info(f"✅ SHAP analysis completed for {asset}")
                        
                    except Exception as e:
                        self.logger.error(f"SHAP computation failed for {asset}: {e}", exc_info=True)
                        continue
                    
                except Exception as e:
                    self.logger.error(f"SHAP analysis failed for {asset}: {e}", exc_info=True)
                    continue
            else:
                self.logger.info(f"No XGBoost model available for {asset} - skipping SHAP analysis")
        
        # Summary of SHAP analysis
        if shap_results:
            self.logger.info("="*50)
            self.logger.info("SHAP ANALYSIS SUMMARY")
            self.logger.info("="*50)
            for asset, results in shap_results.items():
                self.logger.info(f"{asset}: {results['feature_count']} features, "
                               f"{results['sample_count']} samples, "
                               f"{len(results['plot_files'])} plots generated")
            self.logger.info("="*50)
        else:
            self.logger.warning("No SHAP analysis results generated")
    
    @log_execution_time
    def _save_shap_plots(self, shap_values: np.ndarray, X: pd.DataFrame, asset: str) -> list:
        """Save SHAP visualization plots with enhanced logging"""
        plot_files = []
        
        try:
            output_dir = Path('shap_plots')
            output_dir.mkdir(exist_ok=True)
            self.logger.debug(f"{asset} - SHAP plots directory: {output_dir}")
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list) and len(shap_values) > 1:
                # Multi-class case - use class 1 (Medium volatility)
                shap_vals_to_plot = shap_values[1]
                self.logger.debug(f"{asset} - Using SHAP values for class 1 (Medium volatility)")
            elif isinstance(shap_values, list):
                # Single class or binary case
                shap_vals_to_plot = shap_values[0]
                self.logger.debug(f"{asset} - Using SHAP values for single class")
            else:
                # Single array case
                shap_vals_to_plot = shap_values
                self.logger.debug(f"{asset} - Using direct SHAP values")
            
            # Create figure
            self.logger.debug(f"{asset} - Creating SHAP summary plot...")
            plt.figure(figsize=(10, 8))
            
            # Create summary plot
            shap.summary_plot(shap_vals_to_plot, X, show=False, plot_type='bar')
            plt.title(f'SHAP Feature Importance - {asset}', fontsize=16, pad=20)
            plt.xlabel('Mean |SHAP value|', fontsize=12)
            plt.tight_layout()
            
            # Save plot
            plot_path = output_dir / f'{asset}_shap_summary.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(plot_path.name)
            self.logger.debug(f"{asset} - Summary plot saved: {plot_path.name}")
            
            # Also create a detailed summary plot
            self.logger.debug(f"{asset} - Creating detailed SHAP plot...")
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_vals_to_plot, X, show=False)
            plt.title(f'SHAP Value Distribution - {asset}', fontsize=16, pad=20)
            plt.tight_layout()
            
            detailed_path = output_dir / f'{asset}_shap_detailed.png'
            plt.savefig(detailed_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(detailed_path.name)
            self.logger.debug(f"{asset} - Detailed plot saved: {detailed_path.name}")
            
            self.logger.info(f"SHAP plots saved for {asset}: {', '.join(plot_files)}")
            return plot_files
            
        except Exception as e:
            self.logger.error(f"Failed to save SHAP plots for {asset}: {e}", exc_info=True)
            return []
    
    def _save_results(self):
        """Save all results to files"""
        output_dir = Path('results')
        output_dir.mkdir(exist_ok=True)
        
        # Save results as CSV
        results_df = self._format_results_dataframe()
        
        # Check if results dataframe is empty
        if results_df.empty:
            self.logger.warning("No results to save - results dataframe is empty")
            # Create a minimal CSV with headers
            empty_df = pd.DataFrame(columns=['Asset', 'Task', 'Model', 'RMSE', 'MAE', 'R2', 'Accuracy', 'F1', 'Precision', 'Recall'])
            empty_df.to_csv(output_dir / 'model_performance.csv', index=False)
            
            # Create a basic summary
            with open(output_dir / 'summary_report.txt', 'w') as f:
                f.write("="*80 + "\n")
                f.write("VOLATILITY FORECASTING RESULTS SUMMARY\n")
                f.write("="*80 + "\n\n")
                f.write("No results were generated. This could be due to:\n")
                f.write("- Insufficient data for model training\n")
                f.write("- Model training failures\n")
                f.write("- Data quality issues\n\n")
                f.write("Check the log files for detailed error information.\n")
            
            return
        
        # Log summary of results
        self.logger.info(f"Saving results: {len(results_df)} model evaluations across {len(results_df['Asset'].unique())} assets")
        
        results_df.to_csv(output_dir / 'model_performance.csv', index=False)
        
        # Save detailed results
        joblib.dump(self.results, output_dir / 'detailed_results.pkl')
        
        # Generate summary report only if we have meaningful results
        valid_results = results_df[
            (results_df['R2'].notna() & (results_df['R2'] > -1)) |  # Valid R2 scores
            (results_df['Accuracy'].notna() & (results_df['Accuracy'] > 0))  # Valid accuracy scores
        ]
        
        if not valid_results.empty:
            self._generate_summary_report(results_df, output_dir)
        else:
            self.logger.warning("No meaningful results to generate summary report")
            # Create a basic summary anyway
            with open(output_dir / 'summary_report.txt', 'w') as f:
                f.write("="*80 + "\n")
                f.write("VOLATILITY FORECASTING RESULTS SUMMARY\n")
                f.write("="*80 + "\n\n")
                f.write("Models were trained but did not produce meaningful results.\n")
                f.write("This could indicate:\n")
                f.write("- Models need more training data\n")
                f.write("- Hyperparameter tuning required\n")
                f.write("- Feature engineering improvements needed\n\n")
                f.write("See model_performance.csv for raw results.\n")
        
        self.logger.info(f"Results saved to {output_dir}")
    
    def _format_results_dataframe(self) -> pd.DataFrame:
        """Format results into a summary dataframe"""
        rows = []
        
        for asset, asset_results in self.results.items():
            for task, task_results in asset_results.items():
                for model, metrics in task_results.items():
                    row = {
                        'Asset': asset,
                        'Task': task,
                        'Model': model
                    }
                    
                    # Add metrics (excluding predictions and actuals)
                    for metric, value in metrics.items():
                        if metric not in ['predictions', 'actuals']:
                            # Handle inf and -inf values
                            if isinstance(value, float):
                                if value == float('inf'):
                                    value = None  # Will become NaN in pandas
                                elif value == -float('inf'):
                                    value = None
                            row[metric] = value
                        
                    rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Log some statistics about the results
        if not df.empty:
            self.logger.info(f"Results summary: {len(df)} total evaluations")
            if 'R2' in df.columns:
                valid_r2 = df['R2'].dropna()
                if not valid_r2.empty:
                    self.logger.info(f"R2 scores - Mean: {valid_r2.mean():.3f}, Best: {valid_r2.max():.3f}")
            if 'Accuracy' in df.columns:
                valid_acc = df['Accuracy'].dropna()
                if not valid_acc.empty:
                    self.logger.info(f"Accuracy scores - Mean: {valid_acc.mean():.3f}, Best: {valid_acc.max():.3f}")
        
        return df
    
    def _generate_summary_report(self, results_df: pd.DataFrame, output_dir: Path):
        """Generate a summary report with visualizations"""
        # Check if results dataframe is empty
        if results_df.empty:
            self.logger.warning("Cannot generate summary report - results dataframe is empty")
            return
            
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Regression performance by model
        reg_results = results_df[results_df['Task'] == 'regression']
        if not reg_results.empty:
            reg_pivot = reg_results.pivot_table(
                index='Model', 
                values=['RMSE', 'MAE', 'R2'], 
                aggfunc='mean'
            )
            reg_pivot.plot(kind='bar', ax=axes[0, 0])
            axes[0, 0].set_title('Average Regression Performance by Model')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].legend(loc='best')
        
        # 2. Classification performance by model
        clf_results = results_df[results_df['Task'] == 'classification']
        if not clf_results.empty:
            clf_pivot = clf_results.pivot_table(
                index='Model', 
                values=['Accuracy', 'F1'], 
                aggfunc='mean'
            )
            clf_pivot.plot(kind='bar', ax=axes[0, 1])
            axes[0, 1].set_title('Average Classification Performance by Model')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].legend(loc='best')
        
        # 3. Performance by asset (regression)
        if not reg_results.empty:
            asset_reg = reg_results.pivot_table(
                index='Asset', 
                columns='Model', 
                values='R2', 
                aggfunc='mean'
            )
            asset_reg.plot(kind='bar', ax=axes[1, 0])
            axes[1, 0].set_title('R² Score by Asset and Model')
            axes[1, 0].set_ylabel('R² Score')
            axes[1, 0].legend(loc='best', bbox_to_anchor=(1.05, 1))
        
        # 4. US vs AU performance comparison
        results_df['Market'] = results_df['Asset'].apply(
            lambda x: 'US' if x in AssetConfig.US_ASSETS else 'AU'
        )
        market_comparison = results_df.groupby(['Market', 'Task']).agg({
            'RMSE': 'mean',
            'Accuracy': 'mean'
        }).reset_index()
        
        # Simple bar plot for market comparison
        if not market_comparison.empty:
            axes[1, 1].bar(['US', 'AU'], 
                          market_comparison[market_comparison['Task'] == 'regression']['RMSE'].fillna(0),
                          alpha=0.5, label='RMSE (Regression)')
            axes[1, 1].bar(['US', 'AU'], 
                          market_comparison[market_comparison['Task'] == 'classification']['Accuracy'].fillna(0),
                          alpha=0.5, label='Accuracy (Classification)')
            axes[1, 1].set_title('US vs AU Market Performance')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate text summary
        with open(output_dir / 'summary_report.txt', 'w') as f:
            f.write("="*80 + "\n")
            f.write("VOLATILITY FORECASTING RESULTS SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write("Study Period: {} to {}\n".format(
                self.config.START_DATE, self.config.END_DATE
            ))
            f.write("Assets Analyzed: {}\n".format(", ".join(self.config.ALL_ASSETS)))
            f.write("Walk-Forward Splits: {}\n\n".format(self.config.WALK_FORWARD_SPLITS))
            
            # Best performing models
            f.write("BEST PERFORMING MODELS:\n")
            f.write("-"*40 + "\n")
            
            # Regression
            if not reg_results.empty:
                best_reg = reg_results.loc[reg_results['RMSE'].idxmin()]
                f.write(f"Regression (lowest RMSE): {best_reg['Model']} on {best_reg['Asset']} (RMSE={best_reg['RMSE']:.4f})\n")
            
            # Classification
            if not clf_results.empty:
                best_clf = clf_results.loc[clf_results['F1'].idxmax()]
                f.write(f"Classification (highest F1): {best_clf['Model']} on {best_clf['Asset']} (F1={best_clf['F1']:.4f})\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("Detailed results saved in 'model_performance.csv'\n")
            f.write("SHAP interpretability plots saved in 'shap_plots/'\n")
        
        self.logger.info("Summary report generated")

def test_logging():
    """Test function to verify logging is working"""
    logger = logging.getLogger('test')
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    print("Logging test completed. Check the log file for all messages.")

def main():
    """Main execution function"""
    print("="*80)
    print("INTERPRETABLE MACHINE LEARNING FOR VOLATILITY FORECASTING")
    print("Author: Gurudeep Singh Dhinjan")
    print("="*80 + "\n")
    
    # Initialize pipeline
    pipeline = RiskPipeline()
    
    # Run complete pipeline
    try:
        pipeline.run_pipeline()
        print("\n✅ Pipeline completed successfully!")
        print("📊 Results saved to 'results/' directory")
        print("📈 SHAP plots saved to 'shap_plots/' directory")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    test_logging()
    # main()  # Uncomment this to run the main pipeline

def augment_time_series(X: np.ndarray, y: np.ndarray, n_augmentations: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic training data through time series augmentation"""
    augmented_X, augmented_y = [], []
    
    for _ in range(n_augmentations):
        # Add small random noise
        noise = np.random.normal(0, 0.01, X.shape)
        X_aug = X + noise
        
        # Time warping
        stretch_factor = np.random.uniform(0.9, 1.1)
        n_samples = len(X)
        new_n_samples = int(n_samples * stretch_factor)
        
        # Create warped indices
        indices = np.linspace(0, n_samples - 1, new_n_samples)
        indices = np.clip(indices, 0, n_samples - 1).astype(int)
        
        # Apply warping
        X_warped = X[indices]
        y_warped = y[indices]
        
        # Add to augmented data
        augmented_X.append(X_warped)
        augmented_y.append(y_warped)
    
    # Combine original and augmented data
    X_combined = np.vstack([X] + augmented_X)
    y_combined = np.hstack([y] + augmented_y)
    
    return X_combined, y_combined

def augment_time_shifts(X: np.ndarray, y: np.ndarray, max_shift: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic training data through time shifts.
    
    This function creates additional training samples by shifting the time series
    forward by 1 to max_shift days. This preserves the temporal structure while
    creating more training data.
    
    Args:
        X: Input features array of shape (n_samples, n_features)
        y: Target values array of shape (n_samples,)
        max_shift: Maximum number of days to shift forward (default: 2)
        
    Returns:
        Tuple of (X_combined, y_combined) containing original and shifted data
    """
    augmented_X, augmented_y = [], []
    
    # Keep original data
    augmented_X.append(X)
    augmented_y.append(y)
    
    # Create shifted versions
    for shift in range(1, max_shift + 1):
        # Shift features forward by 'shift' days
        X_shifted = X[shift:]
        y_shifted = y[shift:]
        
        # Pad the beginning with the first value to maintain sequence length
        X_padded = np.pad(X_shifted, ((shift, 0), (0, 0)), mode='edge')
        y_padded = np.pad(y_shifted, (shift, 0), mode='edge')
        
        augmented_X.append(X_padded)
        augmented_y.append(y_padded)
    
    # Combine all versions
    X_combined = np.vstack(augmented_X)
    y_combined = np.hstack(augmented_y)
    
    return X_combined, y_combined

def prepare_training_data(self, features: pd.DataFrame, target: pd.Series, 
                         sequence_length: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare training data with sequences and augmentation"""
    # Create sequences
    X, y = [], []
    for i in range(len(features) - sequence_length):
        X.append(features.iloc[i:(i + sequence_length)].values)
        y.append(target.iloc[i + sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    # Apply both types of augmentation
    X_aug1, y_aug1 = augment_time_series(X, y)
    X_aug2, y_aug2 = augment_time_shifts(X, y)
    
    # Combine all augmented data
    X_combined = np.vstack([X_aug1, X_aug2])
    y_combined = np.hstack([y_aug1, y_aug2])
    
    return X_combined, y_combined

def time_series_cv_split(X: pd.DataFrame, y: pd.Series, n_splits: int = 5, gap: int = 5) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Implement proper time series cross-validation with gap"""
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    
    for train_idx, test_idx in tscv.split(X):
        # Ensure minimum training size
        if len(train_idx) < 252:  # 1 year minimum
            continue
            
        yield train_idx, test_idx

def create_ensemble_predictions(self, models_predictions: List[np.ndarray], 
                              weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Combine predictions from multiple models"""
    if weights is None:
        # Equal weights
        weights = np.ones(len(models_predictions)) / len(models_predictions)
    
    ensemble_pred = np.zeros_like(models_predictions[0])
    for pred, weight in zip(models_predictions, weights):
        ensemble_pred += pred * weight
    
    return ensemble_pred

def _run_ensemble_models(self, X: pd.DataFrame, y: pd.Series, asset: str) -> Dict:
    """Run ensemble of models with proper cross-validation"""
    results = {}
    
    # Initialize models
    models = {
        'LSTM': create_lstm_regressor(input_shape=(X.shape[1], X.shape[2])),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=self.config.RANDOM_STATE
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=self.config.RANDOM_STATE
        )
    }
    
    # Get cross-validation splits
    splits = list(time_series_cv_split(X, y))
    
    # Store predictions for each model
    model_predictions = {name: [] for name in models.keys()}
    actuals = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        # Train and predict with each model
        for name, model in models.items():
            if name == 'LSTM':
                # Prepare sequences for LSTM
                X_train_seq = X_train_scaled
                X_test_seq = X_test_scaled
                
                # Train LSTM
                history = train_model(
                    model, X_train_seq, y_train,
                    X_test_seq, y_test,
                    asset, name, self.config
                )
                
                # Make predictions
                y_pred = model.predict(X_test_seq)
            else:
                # Reshape for sklearn models
                X_train_flat = X_train_scaled.reshape(X_train_scaled.shape[0], -1)
                X_test_flat = X_test_scaled.reshape(X_test_scaled.shape[0], -1)
                
                # Train and predict
                model.fit(X_train_flat, y_train)
                y_pred = model.predict(X_test_flat)
            
            model_predictions[name].extend(y_pred)
        
        actuals.extend(y_test)
    
    # Create ensemble predictions
    ensemble_pred = self.create_ensemble_predictions(
        [np.array(preds) for preds in model_predictions.values()]
    )
    
    # Calculate metrics for each model and ensemble
    for name, preds in model_predictions.items():
        results[name] = {
            'mse': mean_squared_error(actuals, preds),
            'mae': mean_absolute_error(actuals, preds),
            'r2': r2_score(actuals, preds)
        }
    
    results['Ensemble'] = {
        'mse': mean_squared_error(actuals, ensemble_pred),
        'mae': mean_absolute_error(actuals, ensemble_pred),
        'r2': r2_score(actuals, ensemble_pred)
    }
    
    return results
