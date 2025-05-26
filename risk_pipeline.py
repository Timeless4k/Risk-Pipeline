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
from typing import Dict, List, Tuple, Optional, Union
import joblib
from pathlib import Path
import logging
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

# Suppress warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class AssetConfig:
    """Configuration for assets used in the study"""
    US_ASSETS = ['AAPL', 'MSFT', '^GSPC']
    AU_ASSETS = ['IOZ.AX', 'CBA.AX', 'BHP.AX']
    ALL_ASSETS = US_ASSETS + AU_ASSETS
    
    # Date range as per thesis
    START_DATE = '2017-01-01'
    END_DATE = '2024-03-31'
    
    # Feature windows
    VOLATILITY_WINDOW = 5
    MA_SHORT = 10
    MA_LONG = 50
    CORRELATION_WINDOW = 30
    
    # Model parameters
    WALK_FORWARD_SPLITS = 5
    TEST_SIZE = 252  # ~1 year of trading days
    RANDOM_STATE = 42

class DataLoader:
    """Handles data downloading and caching"""
    
    def __init__(self, cache_dir: str = 'data_cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def download_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Download or load cached data for multiple symbols"""
        data = {}
        
        for symbol in tqdm(symbols, desc="Downloading data"):
            cache_file = self.cache_dir / f"{symbol.replace('^', '')}_data.pkl"
            
            if cache_file.exists():
                logger.info(f"Loading cached data for {symbol}")
                data[symbol] = pd.read_pickle(cache_file)
            else:
                logger.info(f"Downloading data for {symbol}")
                try:
                    # Download the data
                    df = yf.download(
                        tickers=symbol,
                        start=start_date,
                        end=end_date,
                        progress=False,
                        auto_adjust=False
                    )

                    # If columns are MultiIndex (as with single-ticker), fix them properly
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(1)  # ✅ This gives ['Open', 'High', ...]
                        logger.info(f"{symbol} - Cleaned MultiIndex columns: {df.columns.tolist()}")
                    else:
                        logger.info(f"{symbol} - Flat columns: {df.columns.tolist()}")
                        
                    df.to_pickle(cache_file)
                    data[symbol] = df
                    
                    # Debug statement to verify columns
                    logger.info(f"{symbol} - Final columns: {df.columns.tolist()}")
                except Exception as e:
                    logger.error(f"Error downloading {symbol}: {e}")
            
            # Log data shape and columns after download/load
            if symbol in data:
                df = data[symbol]
                logger.info(f"{symbol} - Downloaded shape: {df.shape}, Columns: {df.columns.tolist()}")
                if df.empty or df.isna().all().all():
                    logger.warning(f"⚠️ {symbol} - WARNING: Downloaded DataFrame is empty or all NaNs.")
                    
        # Download VIX data
        vix_cache = self.cache_dir / "VIX_data.pkl"
        if vix_cache.exists():
            data['VIX'] = pd.read_pickle(vix_cache)
        else:
            logger.info("Downloading VIX data")
            vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False)
            vix_data.to_pickle(vix_cache)
            data['VIX'] = vix_data
            
        return data

class FeatureEngineer:
    """Handles all feature engineering as per thesis specification"""
    
    def __init__(self, config: AssetConfig):
        self.config = config
        
    def calculate_log_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate log returns"""
        returns = np.log(prices / prices.shift(1))
        if returns.dropna().empty:
            logger.warning(f"⚠️ Log returns are empty or invalid — check input prices:\n{prices.head()}")
        return returns
    
    def calculate_volatility(self, returns: pd.Series, window: int) -> pd.Series:
        """Calculate rolling volatility (standard deviation)"""
        return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
    
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical features as specified in thesis"""
        logger.info(f"Creating technical features from {df.shape[0]} rows")
        features = pd.DataFrame(index=df.index)
        
        # Use Adj Close if available, otherwise fall back to Close
        price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        
        # Calculate returns
        returns = self.calculate_log_returns(df[price_col])
        
        # Target variable: 5-day volatility
        features['Volatility5D'] = self.calculate_volatility(returns, self.config.VOLATILITY_WINDOW)
        
        # Lagged returns (Lag1, Lag2, Lag3)
        for i in range(1, 4):
            features[f'Lag{i}'] = returns.shift(i)
        
        # Rate of Change over 5 days (ROC5)
        features['ROC5'] = (df[price_col] / df[price_col].shift(5) - 1) * 100
        
        # Moving Averages (MA10, MA50)
        features['MA10'] = df[price_col].rolling(window=self.config.MA_SHORT).mean()
        features['MA50'] = df[price_col].rolling(window=self.config.MA_LONG).mean()
        
        # Rolling Standard Deviation (RollingStd5)
        features['RollingStd5'] = returns.rolling(window=self.config.VOLATILITY_WINDOW).std()
        
        # MA Ratio (MA10/MA50)
        features['MA_ratio'] = features['MA10'] / features['MA50']
        
        logger.info(f"✅ Features created: {features.shape[1]} columns, {features.dropna().shape[0]} valid rows")
        logger.debug(f"Feature preview:\n{features.head(3)}")
        return features
    
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
        correlations = pd.DataFrame()
        
        # Extract returns for correlation calculation
        returns = {}
        for symbol, df in data.items():
            if symbol != 'VIX':
                price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'

                if price_col in df.columns and not df[price_col].dropna().empty:
                    log_ret = self.calculate_log_returns(df[price_col])
                    if isinstance(log_ret, pd.Series) and not log_ret.dropna().empty:
                        returns[symbol] = log_ret
                        logger.info(f"✅ {symbol} - Valid log return series with {log_ret.dropna().shape[0]} non-NaN values")
                    else:
                        logger.warning(f"⚠️ {symbol} - Log returns are empty after calculation")
                else:
                    logger.warning(f"⚠️ {symbol} - Missing or empty price column '{price_col}'")
        
        if not returns:
            raise ValueError("No valid return series found. Correlation calculation skipped.")
            
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
        """Create volatility regime labels using quantile binning"""
        # Use quantiles to create balanced classes
        labels = pd.qcut(volatility, q=3, labels=['Low', 'Medium', 'High'])
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
                             n_classes: int = 3,
                             units: List[int] = [50, 30], 
                             dropout: float = 0.2) -> tf.keras.Model:
        """Create LSTM model for classification"""
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(units[0], return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout))
        
        # Second LSTM layer
        model.add(LSTM(units[1], return_sequences=False))
        model.add(Dropout(dropout))
        
        # Dense layers
        model.add(Dense(25, activation='relu'))
        model.add(Dense(n_classes, activation='softmax'))
        
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
    """Implements walk-forward cross-validation"""
    
    def __init__(self, n_splits: int = 5, test_size: int = 252):
        self.n_splits = n_splits
        self.test_size = test_size
        
    def split(self, X: pd.DataFrame) -> List[Tuple[pd.Index, pd.Index]]:
        """Generate train/test indices for walk-forward validation"""
        n_samples = len(X)
        splits = []
        
        # Calculate size of each fold
        fold_size = (n_samples - self.test_size) // self.n_splits
        
        for i in range(self.n_splits):
            train_end = fold_size * (i + 1)
            test_start = train_end
            test_end = min(test_start + self.test_size, n_samples)
            
            if test_end > n_samples:
                break
                
            train_idx = X.index[:train_end]
            test_idx = X.index[test_start:test_end]
            
            splits.append((train_idx, test_idx))
            
        return splits

class RiskPipeline:
    """Main pipeline class orchestrating the entire workflow"""
    
    def __init__(self, config: AssetConfig = AssetConfig()):
        self.config = config
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
        
    def run_pipeline(self, assets: List[str] = None, skip_correlations: bool = False, debug: bool = False):
        """Execute the complete pipeline"""
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
        if assets is None:
            assets = self.config.ALL_ASSETS
        
        logger.info("Starting RiskPipeline execution...")
        
        # Step 1: Load data
        logger.info("Step 1: Loading data...")
        raw_data = self.data_loader.download_data(
            assets + ['^VIX'], 
            self.config.START_DATE, 
            self.config.END_DATE
        )
        
        # Step 2: Feature engineering for each asset
        logger.info("Step 2: Engineering features...")
        processed_data = {}
        
        for asset in assets:
            if asset in raw_data:
                logger.info(f"Processing features for {asset}")
                
                # Create technical features
                features = self.feature_engineer.create_technical_features(raw_data[asset])
                
                # Add VIX features
                if '^VIX' in raw_data:
                    features = self.feature_engineer.add_vix_features(features, raw_data['^VIX'])
                
                # Add correlation features if not skipped
                if not skip_correlations:
                    try:
                        correlations = self.feature_engineer.calculate_correlations(raw_data)
                        for col in correlations.columns:
                            if col in correlations:
                                features[col] = correlations[col]
                    except ValueError as e:
                        logger.warning(f"Skipping correlation features due to: {e}")
                
                # Create labels
                price_col = 'Adj Close' if 'Adj Close' in raw_data[asset].columns else 'Close'
                returns = self.feature_engineer.calculate_log_returns(raw_data[asset][price_col])
                features['Regime'] = self.feature_engineer.create_regime_labels(returns)
                features['VolatilityLabel'] = self.feature_engineer.create_volatility_labels(
                    features['Volatility5D'].dropna()
                )
                
                # Clean data
                features = features.dropna()
                processed_data[asset] = features
                
        # Step 3: Model training and evaluation
        logger.info("Step 3: Training and evaluating models...")
        
        for asset, data in processed_data.items():
            logger.info(f"\nProcessing {asset}...")
            self.results[asset] = {}
            
            # Prepare features and targets
            feature_cols = ['Lag1', 'Lag2', 'Lag3', 'ROC5', 'MA10', 'MA50', 
                          'RollingStd5', 'MA_ratio', 'VIX', 'VIX_change']
            
            # Add correlation features if available
            for col in ['AAPL_GSPC_corr', 'IOZ_CBA_corr', 'BHP_IOZ_corr']:
                if col in data.columns:
                    feature_cols.append(col)
            
            X = data[feature_cols]
            y_reg = data['Volatility5D']
            y_clf = data['VolatilityLabel'].map({'Low': 0, 'Medium': 1, 'High': 2})
            
            # Run regression models
            logger.info(f"Training regression models for {asset}...")
            self.results[asset]['regression'] = self._run_regression_models(X, y_reg, asset)
            
            # Run classification models
            logger.info(f"Training classification models for {asset}...")
            self.results[asset]['classification'] = self._run_classification_models(X, y_clf, asset)
            
        # Step 4: Generate interpretability analysis
        logger.info("Step 4: Generating SHAP interpretability analysis...")
        self._generate_shap_analysis()
        
        # Step 5: Save results
        logger.info("Step 5: Saving results...")
        self._save_results()
        
        logger.info("Pipeline execution completed successfully!")
        
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
            logger.info(f"  Training {model_name}...")
            
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
        """Run all classification models with walk-forward validation"""
        results = {}
        
        # Models to evaluate
        models = {
            'Random': DummyClassifier(strategy='stratified', random_state=self.config.RANDOM_STATE),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.config.RANDOM_STATE,
                use_label_encoder=False,
                eval_metric='mlogloss'
            ),
            'MLP': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                random_state=self.config.RANDOM_STATE,
                early_stopping=True
            ),
            'LSTM_Classifier': 'lstm_clf'
        }
        
        for model_name, model in models.items():
            logger.info(f"  Training {model_name}...")
            
            if isinstance(model, str):
                # Deep learning model
                results[model_name] = self._evaluate_dl_model(
                    X, y, model, task='classification', asset=asset
                )
            else:
                # Sklearn-compatible model
                results[model_name] = self._evaluate_sklearn_model(X, y, model, asset, model_name)
                
        return results
    
    def _evaluate_naive_baseline(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Evaluate naive baseline (previous value)"""
        splits = self.validator.split(X)
        predictions = []
        actuals = []
        
        for train_idx, test_idx in splits:
            # Use last training value as prediction for all test samples
            y_train = y.loc[train_idx]
            y_test = y.loc[test_idx]
            
            # Naive prediction: use last known value
            y_pred = pd.Series(index=test_idx, data=y_train.iloc[-1])
            
            predictions.extend(y_pred.values)
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
                logger.warning(f"ARIMA failed for fold: {e}")
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
        """Evaluate deep learning models"""
        splits = self.validator.split(X)
        predictions = []
        actuals = []
        
        # Prepare data for LSTM (requires 3D input)
        sequence_length = 20  # Look back 20 days
        
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X.loc[train_idx], X.loc[test_idx]
            y_train, y_test = y.loc[train_idx], y.loc[test_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Create sequences
            X_train_seq, y_train_seq = self._create_sequences(
                X_train_scaled, y_train.values, sequence_length
            )
            X_test_seq, y_test_seq = self._create_sequences(
                X_test_scaled, y_test.values, sequence_length
            )
            
            if len(X_train_seq) == 0 or len(X_test_seq) == 0:
                continue
            
            # Create model
            input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
            
            if model_type == 'lstm':
                if task == 'regression':
                    model = self.model_factory.create_lstm_regressor(input_shape)
                else:
                    model = self.model_factory.create_lstm_classifier(input_shape)
            elif model_type == 'stockmixer':
                model = self.model_factory.create_stockmixer(input_shape, task)
            elif model_type == 'lstm_clf':
                model = self.model_factory.create_lstm_classifier(input_shape)
            
            # Train model
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
            
            history = model.fit(
                X_train_seq, y_train_seq,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stop, reduce_lr],
                verbose=0
            )
            
            # Make predictions
            y_pred = model.predict(X_test_seq, verbose=0)
            
            if task == 'classification':
                y_pred = np.argmax(y_pred, axis=1)
            else:
                y_pred = y_pred.flatten()
            
            predictions.extend(y_pred)
            actuals.extend(y_test_seq)
            
            # Save model for last fold (for SHAP analysis)
            if fold_idx == len(splits) - 1:
                model_key = f"{asset}_{model_type}_{task}"
                self.models[model_key] = model
                self.scalers[model_key] = scaler
        
        # Calculate metrics
        if task == 'regression':
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
        else:
            accuracy = accuracy_score(actuals, predictions)
            f1 = f1_score(actuals, predictions, average='weighted')
            precision = precision_score(actuals, predictions, average='weighted')
            recall = recall_score(actuals, predictions, average='weighted')
            
            return {
                'Accuracy': accuracy,
                'F1': f1,
                'Precision': precision,
                'Recall': recall,
                'predictions': predictions,
                'actuals': actuals
            }
    
    def _evaluate_sklearn_model(self, X: pd.DataFrame, y: pd.Series, 
                               model, asset: str, model_name: str) -> Dict:
        """Evaluate sklearn-compatible models"""
        splits = self.validator.split(X)
        predictions = []
        actuals = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X.loc[train_idx], X.loc[test_idx]
            y_train, y_test = y.loc[train_idx], y.loc[test_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model_copy = model.__class__(**model.get_params())
            model_copy.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model_copy.predict(X_test_scaled)
            
            predictions.extend(y_pred)
            actuals.extend(y_test.values)
            
            # Save model for last fold (for SHAP analysis)
            if fold_idx == len(splits) - 1 and model_name == 'XGBoost':
                model_key = f"{asset}_xgboost_classification"
                self.models[model_key] = model_copy
                self.scalers[model_key] = scaler
        
        # Calculate metrics
        accuracy = accuracy_score(actuals, predictions)
        f1 = f1_score(actuals, predictions, average='weighted')
        precision = precision_score(actuals, predictions, average='weighted')
        recall = recall_score(actuals, predictions, average='weighted')
        
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
        """Create sequences for LSTM input"""
        X_seq, y_seq = [], []
        
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
            y_seq.append(y[i])
            
        return np.array(X_seq), np.array(y_seq)
    
    def _generate_shap_analysis(self):
        """Generate SHAP interpretability analysis"""
        logger.info("Generating SHAP analysis...")
        
        # Focus on XGBoost for classification (as it's most interpretable)
        for asset in self.config.ALL_ASSETS:
            model_key = f"{asset}_xgboost_classification"
            
            if model_key in self.models:
                model = self.models[model_key]
                scaler = self.scalers[model_key]
                
                # Get some test data
                data = self.data_loader.download_data([asset], 
                                                    self.config.START_DATE, 
                                                    self.config.END_DATE)
                features = self.feature_engineer.create_technical_features(data[asset])
                
                # Prepare features
                feature_cols = ['Lag1', 'Lag2', 'Lag3', 'ROC5', 'MA10', 'MA50', 
                              'RollingStd5', 'MA_ratio', 'VIX', 'VIX_change']
                
                X = features[feature_cols].dropna().iloc[-100:]  # Last 100 samples
                X_scaled = scaler.transform(X)
                
                # Create SHAP explainer
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_scaled)
                
                # Save SHAP plots
                self._save_shap_plots(shap_values, X, asset)
    
    def _save_shap_plots(self, shap_values: np.ndarray, X: pd.DataFrame, asset: str):
        """Save SHAP visualization plots"""
        output_dir = Path('shap_plots')
        output_dir.mkdir(exist_ok=True)
        
        # Summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values[1], X, show=False)  # Class 1 (Medium volatility)
        plt.title(f'SHAP Summary Plot - {asset}')
        plt.tight_layout()
        plt.savefig(output_dir / f'{asset}_shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"SHAP plots saved for {asset}")
    
    def _save_results(self):
        """Save all results to files"""
        output_dir = Path('results')
        output_dir.mkdir(exist_ok=True)
        
        # Save results as CSV
        results_df = self._format_results_dataframe()
        results_df.to_csv(output_dir / 'model_performance.csv')
        
        # Save detailed results
        joblib.dump(self.results, output_dir / 'detailed_results.pkl')
        
        # Generate summary report
        self._generate_summary_report(results_df, output_dir)
        
        logger.info(f"Results saved to {output_dir}")
    
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
                    
                    # Add metrics (excluding predictions)
                    for metric, value in metrics.items():
                        if metric not in ['predictions', 'actuals']:
                            row[metric] = value
                            
                    rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _generate_summary_report(self, results_df: pd.DataFrame, output_dir: Path):
        """Generate a summary report with visualizations"""
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
        
        logger.info("Summary report generated")

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
    main()
