"""
Feature engineering module for RiskPipeline.

This module provides comprehensive feature engineering capabilities including:
- Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Statistical features (volatility, skewness, kurtosis)
- Time-based features
- Correlation features
- Market regime classification
- Modular and pluggable feature creation
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from risk_pipeline.config.global_config import GlobalConfig
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import warnings

from ..utils.logging_utils import log_execution_time
from .config import PipelineConfig

logger = logging.getLogger(__name__)

@dataclass
class FeatureConfig:
    """Configuration for feature engineering parameters."""
    
    # Technical indicator parameters
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    bollinger_period: int = 20
    bollinger_std: int = 2
    
    # Moving average parameters
    ma_short: int = 10
    ma_long: int = 50
    
    # Volatility and correlation windows
    volatility_windows: List[int] = None
    correlation_window: int = 30
    
    # Regime classification
    regime_window: int = 60
    bull_threshold: float = 0.1
    bear_threshold: float = -0.1
    
    # Feature selection
    min_correlation_threshold: float = 0.01
    max_feature_correlation: float = 0.95
    
    def __post_init__(self):
        if self.volatility_windows is None:
            self.volatility_windows = [5, 10, 20]

class BaseFeatureModule(ABC):
    """Abstract base class for feature modules."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features from input data."""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Return list of feature names created by this module."""
        pass
    
    def validate_input(self, data: pd.DataFrame) -> bool:
        """Validate input data for feature creation."""
        if data.empty:
            self.logger.error("Input data is empty")
            return False
        
        required_columns = self.get_required_columns()
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        return True
    
    @abstractmethod
    def get_required_columns(self) -> List[str]:
        """Return list of required columns for this feature module."""
        pass

class TechnicalFeatureModule(BaseFeatureModule):
    """Technical indicator feature module."""
    
    def get_required_columns(self) -> List[str]:
        return ['Close', 'High', 'Low']
    
    def get_feature_names(self) -> List[str]:
        return [
            'RSI', 'MACD', 'ATR', 'Bollinger_Upper', 'Bollinger_Lower',
            'MA10', 'MA50', 'MA_ratio', 'ROC5', 'RollingStd5',
            'Corr_MA10', 'Corr_MA50'
        ]
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators."""
        if not self.validate_input(data):
            return pd.DataFrame()
        
        features = pd.DataFrame(index=data.index)
        
        # Use Adj Close if available, otherwise fall back to Close
        price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
        df = data.copy()
        df['Price'] = df[price_col]
        
        # Calculate returns
        returns = self._calculate_log_returns(df['Price'])
        
        # RSI
        features['RSI'] = self._calculate_rsi(df['Price'])
        
        # MACD
        features['MACD'] = self._calculate_macd(df['Price'])
        
        # ATR
        features['ATR'] = self._calculate_atr(df)
        
        # Bollinger Bands
        bb_upper, bb_lower = self._calculate_bollinger(df['Price'])
        features['Bollinger_Upper'] = bb_upper
        features['Bollinger_Lower'] = bb_lower
        
        # Moving Averages
        features['MA10'] = df['Price'].rolling(window=self.config.ma_short, min_periods=1).mean()
        features['MA50'] = df['Price'].rolling(window=self.config.ma_long, min_periods=1).mean()
        features['MA_ratio'] = features['MA10'] / features['MA50']
        
        # Rate of Change
        features['ROC5'] = df['Price'].pct_change(periods=5)
        
        # Rolling Standard Deviation
        features['RollingStd5'] = returns.rolling(window=5, min_periods=1).std()
        
        # Correlation with Moving Averages
        features['Corr_MA10'] = returns.rolling(window=self.config.ma_short, min_periods=1).corr(features['MA10'])
        features['Corr_MA50'] = returns.rolling(window=self.config.ma_long, min_periods=1).corr(features['MA50'])
        
        return features
    
    def _calculate_log_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate log returns."""
        returns = np.log(prices / prices.shift(1))
        if returns.dropna().empty:
            self.logger.warning("Log returns are empty or invalid")
        return returns
    
    def _calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.config.rsi_period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config.rsi_period, min_periods=1).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD."""
        exp1 = prices.ewm(span=self.config.macd_fast, adjust=False).mean()
        exp2 = prices.ewm(span=self.config.macd_slow, adjust=False).mean()
        return exp1 - exp2
    
    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range."""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=self.config.atr_period, min_periods=1).mean()
    
    def _calculate_bollinger(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        ma = prices.rolling(window=self.config.bollinger_period, min_periods=1).mean()
        std = prices.rolling(window=self.config.bollinger_period, min_periods=1).std()
        upper_band = ma + (std * self.config.bollinger_std)
        lower_band = ma - (std * self.config.bollinger_std)
        return upper_band, lower_band

class StatisticalFeatureModule(BaseFeatureModule):
    """Statistical feature module (volatility, skewness, kurtosis)."""
    
    def get_required_columns(self) -> List[str]:
        return ['Close']
    
    def get_feature_names(self) -> List[str]:
        names = []
        for window in self.config.volatility_windows:
            names.extend([
                f'Volatility{window}D',
                f'Skew{window}D',
                f'Kurt{window}D'
            ])
        return names
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features."""
        if not self.validate_input(data):
            return pd.DataFrame()
        
        features = pd.DataFrame(index=data.index)
        
        # Use Adj Close if available, otherwise fall back to Close
        price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
        returns = np.log(data[price_col] / data[price_col].shift(1))
        
        # Create features for each window
        for window in self.config.volatility_windows:
            features[f'Volatility{window}D'] = self._calculate_volatility(returns, window)
            features[f'Skew{window}D'] = returns.rolling(window=window, min_periods=1).skew()
            features[f'Kurt{window}D'] = returns.rolling(window=window, min_periods=1).kurt()
        
        return features
    
    def _calculate_volatility(self, returns: pd.Series, window: int) -> pd.Series:
        """Calculate rolling volatility (annualized)."""
        return returns.rolling(window=window).std() * np.sqrt(252)

class TimeFeatureModule(BaseFeatureModule):
    """Time-based feature module."""
    
    def get_required_columns(self) -> List[str]:
        return []  # No specific columns required, just needs datetime index
    
    def get_feature_names(self) -> List[str]:
        return ['DayOfWeek', 'MonthOfYear', 'Quarter', 'DayOfYear']
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        try:
            # Check if index is datetime using pandas version-compatible method
            if not isinstance(data.index, pd.DatetimeIndex):
                self.logger.error(f"Data index must be datetime for time features. Got: {type(data.index)}")
                self.logger.debug(f"Index sample: {data.index[:5] if len(data.index) > 0 else 'Empty index'}")
                # Try to convert index to datetime
                try:
                    data.index = pd.to_datetime(data.index)
                    self.logger.info("Successfully converted index to datetime")
                except Exception as conv_e:
                    self.logger.error(f"Failed to convert index to datetime: {conv_e}")
                    return pd.DataFrame()
            
            features = pd.DataFrame(index=data.index)
            
            features['DayOfWeek'] = data.index.dayofweek
            features['MonthOfYear'] = data.index.month
            features['Quarter'] = data.index.quarter
            features['DayOfYear'] = data.index.dayofyear
            
            return features
            
        except Exception as e:
            self.logger.error(f"Time feature creation failed: {e}")
            return pd.DataFrame()

class LagFeatureModule(BaseFeatureModule):
    """Lag feature module."""
    
    def __init__(self, config: FeatureConfig, lags: List[int] = None):
        super().__init__(config)
        self.lags = lags or [1, 2, 3, 5, 10]
    
    def get_required_columns(self) -> List[str]:
        return ['Close']
    
    def get_feature_names(self) -> List[str]:
        return [f'Lag{lag}' for lag in self.lags]
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create lag features."""
        if not self.validate_input(data):
            return pd.DataFrame()
        
        features = pd.DataFrame(index=data.index)
        
        # Use Adj Close if available, otherwise fall back to Close
        price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
        returns = np.log(data[price_col] / data[price_col].shift(1))
        
        # Add lagged returns
        for lag in self.lags:
            features[f'Lag{lag}'] = returns.shift(lag)
        
        return features

class CorrelationFeatureModule(BaseFeatureModule):
    """Correlation feature module."""
    
    def get_required_columns(self) -> List[str]:
        return ['Close']  # Will be applied to each asset
    
    def get_feature_names(self) -> List[str]:
        return ['AAPL_GSPC_corr', 'IOZ_CBA_corr', 'BHP_IOZ_corr']
    
    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate inter-asset correlations."""
        self.logger.info("Starting correlation calculation")
        correlations = pd.DataFrame()
        
        # Extract returns for correlation calculation
        returns = {}
        for symbol, df in data.items():
            if symbol != 'VIX':
                price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
                
                if price_col in df.columns and not df[price_col].dropna().empty:
                    log_ret = np.log(df[price_col] / df[price_col].shift(1))
                    if isinstance(log_ret, pd.Series) and not log_ret.dropna().empty:
                        returns[symbol] = log_ret
                        self.logger.info(f"✅ {symbol} - Valid return series with {log_ret.dropna().shape[0]} non-NaN values")
                    else:
                        self.logger.warning(f"⚠️ {symbol} - Log returns are empty after calculation")
                else:
                    self.logger.warning(f"⚠️ {symbol} - Missing or empty price column '{price_col}'")
        
        if len(returns) < 2:
            self.logger.warning("Insufficient assets for correlation calculation")
            return pd.DataFrame()
        
        returns_df = pd.DataFrame(returns)
        
        # Calculate rolling correlations
        if 'AAPL' in returns_df.columns and '^GSPC' in returns_df.columns:
            correlations['AAPL_GSPC_corr'] = returns_df['AAPL'].rolling(
                window=self.config.correlation_window
            ).corr(returns_df['^GSPC'])
        
        if 'IOZ.AX' in returns_df.columns and 'CBA.AX' in returns_df.columns:
            correlations['IOZ_CBA_corr'] = returns_df['IOZ.AX'].rolling(
                window=self.config.correlation_window
            ).corr(returns_df['CBA.AX'])
        
        if 'BHP.AX' in returns_df.columns and 'IOZ.AX' in returns_df.columns:
            correlations['BHP_IOZ_corr'] = returns_df['BHP.AX'].rolling(
                window=self.config.correlation_window
            ).corr(returns_df['IOZ.AX'])
        
        self.logger.info(f"Created correlation features: {correlations.columns.tolist()}")
        return correlations

class FeatureEngineer:
    """Comprehensive feature engineering with modular design."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.feature_config = FeatureConfig(
            ma_short=self.config.features.ma_short,
            ma_long=self.config.features.ma_long,
            correlation_window=self.config.features.correlation_window
        )
        
        # Initialize feature modules
        self.modules = {
            'technical': TechnicalFeatureModule(self.feature_config),
            'statistical': StatisticalFeatureModule(self.feature_config),
            'time': TimeFeatureModule(self.feature_config),
            'lag': LagFeatureModule(self.feature_config),
            'correlation': CorrelationFeatureModule(self.feature_config)
        }
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("FeatureEngineer initialized with modular architecture")
    
    @log_execution_time
    def create_all_features(self, data: Dict[str, pd.DataFrame], 
                          skip_correlations: bool = False) -> Dict[str, pd.DataFrame]:
        
        self.logger.info("Creating features for all assets")
        
        all_features = {}
        
        for asset, df in data.items():
            if asset == 'VIX':
                continue  # Skip VIX for now, will be added as features later
            
            self.logger.info(f"Creating features for {asset}")
            features = self.create_asset_features(df)
            
            if not features.empty:
                all_features[asset] = features
                self.logger.info(f"✅ {asset}: Created {len(features.columns)} features")
            else:
                self.logger.warning(f"⚠️ {asset}: No features created")
        
        # Add correlation features if requested
        if not skip_correlations and len(all_features) > 1:
            self.logger.info("Adding correlation features")
            correlations = self.modules['correlation'].create_features(data)
            
            if not correlations.empty:
                for asset in all_features.keys():
                    # Align correlation features with asset features
                    asset_correlations = correlations.reindex(all_features[asset].index, method='ffill')
                    all_features[asset] = pd.concat([all_features[asset], asset_correlations], axis=1)
        
        # Add VIX features if available
        if 'VIX' in data:
            self.logger.info("Adding VIX features")
            for asset in all_features.keys():
                vix_features = self.add_vix_features(all_features[asset], data['VIX'])
                all_features[asset] = vix_features
        
        return all_features

    def create_features(self, data: Dict[str, pd.DataFrame], skip_correlations: bool = False) -> Dict[str, Dict[str, Any]]:
        """Backward-compatible API expected by pipeline/tests.

        Returns a mapping per asset with keys: 'features' (np.ndarray or DataFrame),
        'volatility_target' (np.ndarray/Series), 'regime_target' (np.ndarray/Series),
        'feature_names' (List[str]), and optional 'scaler'.
        """
        all_feature_frames = self.create_all_features(data, skip_correlations=skip_correlations)
        structured: Dict[str, Dict[str, Any]] = {}
        for asset, feat_df in all_feature_frames.items():
            if feat_df.empty:
                continue
            
            # Clean features: handle NaN values
            feat_df_clean = self._clean_features(feat_df)
            if feat_df_clean.empty:
                self.logger.warning(f"Skipping {asset}: all features are NaN after cleaning")
                continue
            
            # Choose a volatility target proxy
            target_series = None
            for candidate in [
                'Volatility20D', 'Volatility10D', 'Volatility5D',
                'RollingStd5'
            ]:
                if candidate in feat_df_clean.columns:
                    target_series = feat_df_clean[candidate]
                    break
            if target_series is None:
                # Fallback: rolling std of first numeric column
                numeric_cols = feat_df_clean.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    target_series = feat_df_clean[numeric_cols[0]].rolling(window=5, min_periods=1).std().fillna(0)
                else:
                    continue
            
            # Ensure target has no NaN values
            target_series = target_series.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Binary regime by median split
            median_val = float(np.median(target_series.values)) if len(target_series) else 0.0
            regime = (target_series > median_val).astype(int)

            structured[asset] = {
                'features': feat_df_clean.copy(),
                'volatility_target': target_series.copy(),
                'regime_target': regime,
                'feature_names': feat_df_clean.columns.tolist(),
                'scaler': None,
            }
        return structured
    
    def _clean_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Clean features by handling NaN values and ensuring data quality."""
        if features_df.empty:
            return features_df
        
        # Count NaN values before cleaning
        nan_count_before = features_df.isna().sum().sum()
        if nan_count_before > 0:
            self.logger.info(f"Found {nan_count_before} NaN values in features, cleaning...")
        
        # Strategy 1: Forward fill then backward fill for time series
        features_clean = features_df.fillna(method='ffill').fillna(method='bfill')
        
        # Strategy 2: For remaining NaN values, use column median
        for col in features_clean.columns:
            if features_clean[col].isna().any():
                median_val = features_clean[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                features_clean[col] = features_clean[col].fillna(median_val)
        
        # Strategy 3: Remove rows that still have NaN values (should be very few)
        features_clean = features_clean.dropna()
        
        # Count NaN values after cleaning
        nan_count_after = features_clean.isna().sum().sum()
        if nan_count_after > 0:
            self.logger.warning(f"Still have {nan_count_after} NaN values after cleaning")
        else:
            self.logger.info(f"Successfully cleaned all NaN values")
        
        # Ensure we have enough data after cleaning
        if len(features_clean) < 100:
            self.logger.warning(f"Very few samples after cleaning: {len(features_clean)}")
        
        return features_clean

    # New canonical path for fairness framework
    def create_canonical_views(
        self,
        df: pd.DataFrame,
        y: pd.Series,
        cfg: GlobalConfig,
        train_slice: slice,
        val_slice: slice,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Produce canonical X_seq and X_flat from the same source with identical scaling.

        Returns: ((X_seq_train, X_flat_train, y_train), (X_seq_val, X_flat_val, y_val))
        """
        close = df["Close"].astype(float).values
        T = cfg.lookback_T
        n = len(close)
        if n < T + 1:
            raise ValueError("Not enough data for lookback window")

        def to_seq(slc: slice) -> Tuple[np.ndarray, np.ndarray]:
            idx = np.arange(slc.start, slc.stop)
            windows: List[np.ndarray] = []
            targets: List[float] = []
            for t in idx:
                start = t - T
                if start < 0 or t >= len(y):
                    continue
                windows.append(close[start:t])
                targets.append(float(y.iloc[t]))
            if not windows:
                return np.zeros((0, T, 1), dtype=float), np.zeros((0,), dtype=float)
            X_seq = np.array(windows, dtype=float)[:, :, None]  # [N, T, F=1]
            y_arr = np.array(targets, dtype=float)
            return X_seq, y_arr

        X_seq_train, y_train = to_seq(train_slice)
        X_seq_val, y_val = to_seq(val_slice)

        # scaling on train only
        scaler = None
        if cfg.scaling == "standard":
            scaler = StandardScaler()
        elif cfg.scaling == "minmax":
            scaler = MinMaxScaler()
        if scaler is not None and X_seq_train.size > 0:
            shp = X_seq_train.shape
            Xf = X_seq_train.reshape(shp[0], -1)
            Xf = scaler.fit_transform(Xf)
            X_seq_train = Xf.reshape(shp)
            if X_seq_val.size > 0:
                shp_v = X_seq_val.shape
                Xfv = X_seq_val.reshape(shp_v[0], -1)
                Xfv = scaler.transform(Xfv)
                X_seq_val = Xfv.reshape(shp_v)

        # flat views from the same scaled tensors
        X_flat_train = X_seq_train.reshape(X_seq_train.shape[0], -1)
        X_flat_val = X_seq_val.reshape(X_seq_val.shape[0], -1)

        return (X_seq_train, X_flat_train, y_train), (X_seq_val, X_flat_val, y_val)
    
    def create_asset_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for a single asset."""
        if df.empty:
            self.logger.error("Input DataFrame is empty")
            return pd.DataFrame()
        
        # Create features using each module
        feature_dfs = []
        
        for module_name, module in self.modules.items():
            if module_name == 'correlation':
                continue  # Handle correlations separately
            
            try:
                module_features = module.create_features(df)
                if not module_features.empty:
                    feature_dfs.append(module_features)
                    self.logger.debug(f"✅ {module_name} module: {len(module_features.columns)} features")
                else:
                    self.logger.warning(f"⚠️ {module_name} module: No features created")
            except Exception as e:
                self.logger.error(f"❌ {module_name} module failed: {str(e)}")
        
        if not feature_dfs:
            self.logger.error("No features created by any module")
            return pd.DataFrame()
        
        # Combine all features
        combined_features = pd.concat(feature_dfs, axis=1)
        
        # Handle missing values
        combined_features = self._handle_missing_values(combined_features)
        
        # Validate features
        if not self._validate_features(combined_features):
            self.logger.error("Feature validation failed")
            return pd.DataFrame()
        
        return combined_features
    
    def add_vix_features(self, features: pd.DataFrame, vix_data: pd.DataFrame) -> pd.DataFrame:
        """Add VIX-related features."""
        if vix_data.empty:
            self.logger.warning("VIX data is empty, skipping VIX features")
            return features
        
        # Use Adj Close if available, otherwise fall back to Close
        vix_price_col = 'Adj Close' if 'Adj Close' in vix_data.columns else 'Close'
        
        # Align VIX data with features index
        vix_aligned = vix_data[vix_price_col].reindex(features.index, method='ffill')
        
        features['VIX'] = vix_aligned
        features['VIX_change'] = vix_aligned.pct_change()
        
        # Z-score normalize VIX_change
        vix_change = features['VIX_change']
        if not vix_change.isna().all() and vix_change.std() > 0:
            features['VIX_change'] = (vix_change - vix_change.mean()) / vix_change.std()
        
        return features
    
    def create_regime_labels(self, returns: pd.Series, window: int = None) -> pd.Series:
        """Create market regime labels (Bull, Bear, Sideways)."""
        if window is None:
            window = self.feature_config.regime_window
        
        # Calculate rolling returns
        rolling_returns = returns.rolling(window=window).sum()
        
        # Define regime thresholds
        bull_threshold = self.feature_config.bull_threshold
        bear_threshold = self.feature_config.bear_threshold
        
        # Assign regimes
        regimes = pd.Series(index=returns.index, dtype='object')
        regimes[rolling_returns > bull_threshold] = 'Bull'
        regimes[rolling_returns < bear_threshold] = 'Bear'
        regimes[(rolling_returns >= bear_threshold) & (rolling_returns <= bull_threshold)] = 'Sideways'
        
        return regimes
    
    def create_volatility_labels(self, volatility: pd.Series) -> pd.Series:
        """Create volatility regime labels using quartile binning."""
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
    
    def select_features(self, features: pd.DataFrame, target: Optional[pd.Series] = None, 
                       method: str = 'correlation', **kwargs) -> pd.DataFrame:
        """Select features based on various criteria."""
        if method == 'correlation':
            if target is None:
                raise TypeError("target is required for correlation-based selection")
            return self._select_by_correlation(features, target, **kwargs)
        elif method == 'variance':
            return self._select_by_variance(features, **kwargs)
        else:
            self.logger.warning(f"Unknown feature selection method: {method}")
            return features
    
    def _select_by_correlation(self, features: pd.DataFrame, target: pd.Series, 
                             threshold: float = None) -> pd.DataFrame:
        """Select features based on correlation with target."""
        if threshold is None:
            threshold = self.feature_config.min_correlation_threshold
        
        # Calculate correlations
        correlations = features.corrwith(target).abs()
        
        # Select features above threshold
        selected_features = correlations[correlations >= threshold].index.tolist()
        
        self.logger.info(f"Selected {len(selected_features)} features by correlation (threshold={threshold})")
        return features[selected_features]
    
    def _select_by_variance(self, features: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
        """Select features based on variance threshold."""
        variances = features.var()
        selected_features = variances[variances >= threshold].index.tolist()
        
        self.logger.info(f"Selected {len(selected_features)} features by variance (threshold={threshold})")
        return features[selected_features]
    
    def _handle_missing_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features."""
        # Count missing values before handling
        missing_before = features.isna().sum().sum()
        
        # Use interpolation for missing values
        features = features.interpolate(method='linear', limit_direction='forward', axis=0)
        
        # Forward fill any remaining NaNs at the beginning
        features = features.fillna(method='ffill')
        
        # Backward fill any remaining NaNs at the end
        features = features.fillna(method='bfill')
        
        # Count missing values after handling
        missing_after = features.isna().sum().sum()
        
        if missing_before > 0:
            self.logger.info(f"Handled {missing_before - missing_after} missing values")
        
        if missing_after > 0:
            self.logger.warning(f"Still have {missing_after} missing values after handling")
        
        return features
    
    def _validate_features(self, features: pd.DataFrame) -> bool:
        """Validate feature quality."""
        if features.empty:
            self.logger.error("Features DataFrame is empty")
            return False
        
        # Check for infinite values
        inf_count = np.isinf(features.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            self.logger.warning(f"Found {inf_count} infinite values in features")
            # Replace infinite values with NaN
            features = features.replace([np.inf, -np.inf], np.nan)
        
        # Check for high correlation between features
        numeric_features = features.select_dtypes(include=[np.number])
        if len(numeric_features.columns) > 1:
            corr_matrix = numeric_features.corr().abs()
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > self.feature_config.max_feature_correlation:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
            
            if high_corr_pairs:
                self.logger.warning(f"Found {len(high_corr_pairs)} highly correlated feature pairs")
        
        return True
    
    def get_feature_summary(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics for features."""
        summary = {
            'total_features': len(features.columns),
            'total_samples': len(features),
            'missing_values': features.isna().sum().to_dict(),
            'feature_types': features.dtypes.value_counts().to_dict(),
            'numeric_features': len(features.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(features.select_dtypes(include=['object']).columns)
        }
        
        # Add statistics for numeric features
        numeric_features = features.select_dtypes(include=[np.number])
        if not numeric_features.empty:
            summary['numeric_stats'] = {
                'mean': numeric_features.mean().to_dict(),
                'std': numeric_features.std().to_dict(),
                'min': numeric_features.min().to_dict(),
                'max': numeric_features.max().to_dict()
            }
        
        return summary
    
    def add_custom_module(self, name: str, module: BaseFeatureModule):
        """Add a custom feature module."""
        if not isinstance(module, BaseFeatureModule):
            raise ValueError("Module must inherit from BaseFeatureModule")
        
        self.modules[name] = module
        self.logger.info(f"Added custom feature module: {name}")
    
    def remove_module(self, name: str):
        """Remove a feature module."""
        if name in self.modules:
            del self.modules[name]
            self.logger.info(f"Removed feature module: {name}")
        else:
            self.logger.warning(f"Module {name} not found")
    
    def list_modules(self) -> List[str]:
        """List all available feature modules."""
        return list(self.modules.keys()) 