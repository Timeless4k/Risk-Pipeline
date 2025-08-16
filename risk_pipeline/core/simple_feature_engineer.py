"""
Simplified Feature Engineering for RiskPipeline Testing.

This module provides minimal feature engineering with only price lags
to test the basic pipeline functionality without complex features.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class SimpleFeatureEngineer:
    """Simplified feature engineer that only creates price lag features."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize simple feature engineer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
        
        # Extract configuration
        self.price_lag_days = config.get('features', {}).get('price_lag_days', [30, 60, 90])
        self.use_only_price_lags = config.get('features', {}).get('use_only_price_lags', True)
        
        logger.info(f"SimpleFeatureEngineer initialized with lags: {self.price_lag_days}")
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create simple price lag features.
        
        Args:
            data: Input price data with 'Close' column
            
        Returns:
            DataFrame with features
        """
        if data.empty:
            raise ValueError("Input data is empty")
        
        if 'Close' not in data.columns:
            raise ValueError("Input data must contain 'Close' column")
        
        # Calculate returns
        returns = data['Close'].pct_change()
        
        # Create features DataFrame
        features = pd.DataFrame(index=data.index)
        
        # Add price lag features
        for lag in self.price_lag_days:
            if lag < len(data):
                # Return at t-lag
                features[f'return_t-{lag}'] = returns.shift(lag)
                
                # Price at t-lag (normalized)
                features[f'price_t-{lag}'] = data['Close'].shift(lag) / data['Close'].iloc[-1] - 1
        
        # Add basic moving averages if enabled
        if not self.use_only_price_lags:
            ma_short = self.config.get('features', {}).get('ma_short', 30)
            ma_long = self.config.get('features', {}).get('ma_long', 60)
            
            if ma_short < len(data):
                features[f'ma_{ma_short}'] = data['Close'].rolling(ma_short).mean() / data['Close'] - 1
            if ma_long < len(data):
                features[f'ma_{ma_long}'] = data['Close'].rolling(ma_long).mean() / data['Close'] - 1
        
        # Drop rows with NaN values
        features = features.dropna()
        
        # Store feature names
        self.feature_names = list(features.columns)
        
        logger.info(f"Created {len(self.feature_names)} features: {self.feature_names}")
        logger.info(f"Feature shape: {features.shape}")
        
        return features
    
    def create_target(self, data: pd.DataFrame, target_type: str = 'volatility', 
                     target_horizon: int = 5) -> pd.Series:
        """
        Create target variable with temporal separation.
        
        Args:
            data: Input price data
            target_type: Type of target ('volatility', 'return', 'direction')
            target_horizon: Days in the future for target
            
        Returns:
            Target series
        """
        if data.empty:
            raise ValueError("Input data is empty")
        
        if 'Close' not in data.columns:
            raise ValueError("Input data must contain 'Close' column")
        
        if target_type == 'volatility':
            # Calculate future volatility (rolling std of returns)
            returns = data['Close'].pct_change()
            future_vol = returns.rolling(5).std().shift(-target_horizon)
            target = future_vol
        elif target_type == 'return':
            # Calculate future return
            future_return = data['Close'].pct_change().shift(-target_horizon)
            target = future_return
        elif target_type == 'direction':
            # Calculate future price direction (1 if up, 0 if down)
            future_return = data['Close'].pct_change().shift(-target_horizon)
            target = (future_return > 0).astype(int)
        else:
            raise ValueError(f"Unsupported target type: {target_type}")
        
        # Drop NaN values
        target = target.dropna()
        
        logger.info(f"Created {target_type} target with horizon {target_horizon}")
        logger.info(f"Target shape: {target.shape}")
        
        return target
    
    def align_features_and_target(self, features: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Align features and target to ensure they have matching indices.
        
        Args:
            features: Feature DataFrame
            target: Target Series
            
        Returns:
            Tuple of aligned features and target
        """
        # Find common index
        common_index = features.index.intersection(target.index)
        
        if len(common_index) == 0:
            raise ValueError("No common timestamps between features and target")
        
        # Align data
        features_aligned = features.loc[common_index]
        target_aligned = target.loc[common_index]
        
        logger.info(f"Aligned features and target to {len(common_index)} samples")
        
        return features_aligned, target_aligned
    
    def fit_scaler(self, features: pd.DataFrame) -> 'SimpleFeatureEngineer':
        """
        Fit the scaler on training features.
        
        Args:
            features: Training features
            
        Returns:
            Self for chaining
        """
        self.scaler.fit(features)
        self.is_fitted = True
        logger.info("Scaler fitted successfully")
        return self
    
    def transform_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted scaler.
        
        Args:
            features: Features to transform
            
        Returns:
            Scaled features
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transforming features")
        
        scaled_features = self.scaler.transform(features)
        scaled_df = pd.DataFrame(scaled_features, index=features.index, columns=features.columns)
        
        logger.info("Features scaled successfully")
        return scaled_df
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.feature_names.copy()
    
    def get_feature_importance_placeholder(self) -> Dict[str, float]:
        """Return placeholder feature importance for testing."""
        importance = {}
        for feature in self.feature_names:
            importance[feature] = np.random.random()
        
        # Sort by importance
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        return sorted_importance
