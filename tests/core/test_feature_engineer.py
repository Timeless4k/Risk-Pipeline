"""
Unit tests for FeatureEngineer and related components.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
import sys

# Add the parent directory to the path to import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from risk_pipeline.core.feature_engineer import (
    FeatureEngineer, FeatureConfig, BaseFeatureModule,
    TechnicalFeatureModule, StatisticalFeatureModule, TimeFeatureModule,
    LagFeatureModule, CorrelationFeatureModule
)
from risk_pipeline.core.config import PipelineConfig

class TestFeatureConfig(unittest.TestCase):
    """Test FeatureConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = FeatureConfig()
        
        self.assertEqual(config.rsi_period, 14)
        self.assertEqual(config.macd_fast, 12)
        self.assertEqual(config.macd_slow, 26)
        self.assertEqual(config.ma_short, 10)
        self.assertEqual(config.ma_long, 50)
        self.assertEqual(config.correlation_window, 30)
        self.assertEqual(config.volatility_windows, [5, 10, 20])
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = FeatureConfig(
            rsi_period=21,
            ma_short=5,
            ma_long=20,
            volatility_windows=[10, 20, 30]
        )
        
        self.assertEqual(config.rsi_period, 21)
        self.assertEqual(config.ma_short, 5)
        self.assertEqual(config.ma_long, 20)
        self.assertEqual(config.volatility_windows, [10, 20, 30])

class TestBaseFeatureModule(unittest.TestCase):
    """Test BaseFeatureModule abstract class."""
    
    def test_abstract_methods(self):
        """Test that BaseFeatureModule cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            BaseFeatureModule(FeatureConfig())
    
    def test_validate_input_empty_data(self):
        """Test input validation with empty data."""
        class TestModule(BaseFeatureModule):
            def create_features(self, data):
                return pd.DataFrame()
            
            def get_feature_names(self):
                return []
            
            def get_required_columns(self):
                return []
        
        module = TestModule(FeatureConfig())
        empty_df = pd.DataFrame()
        
        self.assertFalse(module.validate_input(empty_df))
    
    def test_validate_input_missing_columns(self):
        """Test input validation with missing columns."""
        class TestModule(BaseFeatureModule):
            def create_features(self, data):
                return pd.DataFrame()
            
            def get_feature_names(self):
                return []
            
            def get_required_columns(self):
                return ['Close', 'High']
        
        module = TestModule(FeatureConfig())
        df = pd.DataFrame({'Close': [1, 2, 3]})  # Missing 'High'
        
        self.assertFalse(module.validate_input(df))

class TestTechnicalFeatureModule(unittest.TestCase):
    """Test TechnicalFeatureModule."""
    
    def setUp(self):
        """Set up test data."""
        self.config = FeatureConfig()
        self.module = TechnicalFeatureModule(self.config)
        
        # Create sample data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        self.sample_data = pd.DataFrame({
            'Close': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 102,
            'Low': np.random.randn(100).cumsum() + 98,
            'Adj Close': np.random.randn(100).cumsum() + 100
        }, index=dates)
    
    def test_required_columns(self):
        """Test required columns."""
        required = self.module.get_required_columns()
        self.assertIn('Close', required)
        self.assertIn('High', required)
        self.assertIn('Low', required)
    
    def test_feature_names(self):
        """Test feature names."""
        names = self.module.get_feature_names()
        expected_features = ['RSI', 'MACD', 'ATR', 'Bollinger_Upper', 'Bollinger_Lower']
        
        for feature in expected_features:
            self.assertIn(feature, names)
    
    def test_create_features(self):
        """Test feature creation."""
        features = self.module.create_features(self.sample_data)
        
        self.assertIsInstance(features, pd.DataFrame)
        self.assertGreater(len(features.columns), 0)
        self.assertEqual(len(features), len(self.sample_data))
        
        # Check that expected features are present
        expected_features = ['RSI', 'MACD', 'ATR', 'MA10', 'MA50']
        for feature in expected_features:
            self.assertIn(feature, features.columns)
    
    def test_create_features_missing_adj_close(self):
        """Test feature creation without Adj Close column."""
        data_no_adj = self.sample_data.drop(columns=['Adj Close'])
        features = self.module.create_features(data_no_adj)
        
        self.assertIsInstance(features, pd.DataFrame)
        self.assertGreater(len(features.columns), 0)
    
    def test_rsi_calculation(self):
        """Test RSI calculation."""
        prices = pd.Series([100, 101, 102, 101, 100, 99, 98, 97, 96, 95])
        rsi = self.module._calculate_rsi(prices)
        
        self.assertIsInstance(rsi, pd.Series)
        self.assertEqual(len(rsi), len(prices))
        self.assertTrue(all(0 <= val <= 100 for val in rsi.dropna()))
    
    def test_macd_calculation(self):
        """Test MACD calculation."""
        prices = pd.Series([100, 101, 102, 101, 100, 99, 98, 97, 96, 95])
        macd = self.module._calculate_macd(prices)
        
        self.assertIsInstance(macd, pd.Series)
        self.assertEqual(len(macd), len(prices))

class TestStatisticalFeatureModule(unittest.TestCase):
    """Test StatisticalFeatureModule."""
    
    def setUp(self):
        """Set up test data."""
        self.config = FeatureConfig(volatility_windows=[5, 10])
        self.module = StatisticalFeatureModule(self.config)
        
        # Create sample data
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        self.sample_data = pd.DataFrame({
            'Close': np.random.randn(50).cumsum() + 100,
            'Adj Close': np.random.randn(50).cumsum() + 100
        }, index=dates)
    
    def test_create_features(self):
        """Test statistical feature creation."""
        features = self.module.create_features(self.sample_data)
        
        self.assertIsInstance(features, pd.DataFrame)
        self.assertGreater(len(features.columns), 0)
        
        # Check for expected features
        expected_features = ['Volatility5D', 'Volatility10D', 'Skew5D', 'Skew10D', 'Kurt5D', 'Kurt10D']
        for feature in expected_features:
            self.assertIn(feature, features.columns)
    
    def test_volatility_calculation(self):
        """Test volatility calculation."""
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        volatility = self.module._calculate_volatility(returns, 3)
        
        self.assertIsInstance(volatility, pd.Series)
        self.assertEqual(len(volatility), len(returns))

class TestTimeFeatureModule(unittest.TestCase):
    """Test TimeFeatureModule."""
    
    def setUp(self):
        """Set up test data."""
        self.config = FeatureConfig()
        self.module = TimeFeatureModule(self.config)
        
        # Create sample data with datetime index
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        self.sample_data = pd.DataFrame({
            'Close': np.random.randn(100).cumsum() + 100
        }, index=dates)
    
    def test_create_features(self):
        """Test time feature creation."""
        features = self.module.create_features(self.sample_data)
        
        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(len(features.columns), 4)  # DayOfWeek, MonthOfYear, Quarter, DayOfYear
        
        # Check for expected features
        expected_features = ['DayOfWeek', 'MonthOfYear', 'Quarter', 'DayOfYear']
        for feature in expected_features:
            self.assertIn(feature, features.columns)
        
        # Check value ranges
        self.assertTrue(all(0 <= val <= 6 for val in features['DayOfWeek']))
        self.assertTrue(all(1 <= val <= 12 for val in features['MonthOfYear']))
        self.assertTrue(all(1 <= val <= 4 for val in features['Quarter']))
    
    def test_create_features_non_datetime_index(self):
        """Test time feature creation with non-datetime index."""
        data_no_datetime = pd.DataFrame({
            'Close': [1, 2, 3, 4, 5]
        }, index=[1, 2, 3, 4, 5])
        
        features = self.module.create_features(data_no_datetime)
        self.assertTrue(features.empty)

class TestLagFeatureModule(unittest.TestCase):
    """Test LagFeatureModule."""
    
    def setUp(self):
        """Set up test data."""
        self.config = FeatureConfig()
        self.module = LagFeatureModule(self.config, lags=[1, 2, 3])
        
        # Create sample data
        dates = pd.date_range('2020-01-01', periods=20, freq='D')
        self.sample_data = pd.DataFrame({
            'Close': np.random.randn(20).cumsum() + 100,
            'Adj Close': np.random.randn(20).cumsum() + 100
        }, index=dates)
    
    def test_create_features(self):
        """Test lag feature creation."""
        features = self.module.create_features(self.sample_data)
        
        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(len(features.columns), 3)  # Lag1, Lag2, Lag3
        
        # Check for expected features
        expected_features = ['Lag1', 'Lag2', 'Lag3']
        for feature in expected_features:
            self.assertIn(feature, features.columns)
        
        # Check that lag features have NaN values at the beginning
        self.assertTrue(features['Lag1'].iloc[0] != features['Lag1'].iloc[0])  # NaN check
        self.assertTrue(features['Lag2'].iloc[1] != features['Lag2'].iloc[1])  # NaN check

class TestCorrelationFeatureModule(unittest.TestCase):
    """Test CorrelationFeatureModule."""
    
    def setUp(self):
        """Set up test data."""
        self.config = FeatureConfig(correlation_window=5)
        self.module = CorrelationFeatureModule(self.config)
        
        # Create sample data for multiple assets
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        self.sample_data = {
            'AAPL': pd.DataFrame({
                'Close': np.random.randn(50).cumsum() + 100,
                'Adj Close': np.random.randn(50).cumsum() + 100
            }, index=dates),
            '^GSPC': pd.DataFrame({
                'Close': np.random.randn(50).cumsum() + 200,
                'Adj Close': np.random.randn(50).cumsum() + 200
            }, index=dates),
            'IOZ.AX': pd.DataFrame({
                'Close': np.random.randn(50).cumsum() + 50,
                'Adj Close': np.random.randn(50).cumsum() + 50
            }, index=dates),
            'CBA.AX': pd.DataFrame({
                'Close': np.random.randn(50).cumsum() + 60,
                'Adj Close': np.random.randn(50).cumsum() + 60
            }, index=dates)
        }
    
    def test_create_features(self):
        """Test correlation feature creation."""
        features = self.module.create_features(self.sample_data)
        
        self.assertIsInstance(features, pd.DataFrame)
        self.assertGreater(len(features.columns), 0)
        
        # Check for expected correlation features
        expected_features = ['AAPL_GSPC_corr', 'IOZ_CBA_corr']
        for feature in expected_features:
            self.assertIn(feature, features.columns)
    
    def test_create_features_insufficient_assets(self):
        """Test correlation feature creation with insufficient assets."""
        single_asset_data = {'AAPL': self.sample_data['AAPL']}
        features = self.module.create_features(single_asset_data)
        
        self.assertTrue(features.empty)

class TestFeatureEngineer(unittest.TestCase):
    """Test FeatureEngineer main class."""
    
    def setUp(self):
        """Set up test data."""
        self.config = PipelineConfig()
        self.engineer = FeatureEngineer(self.config)
        
        # Create sample data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        self.sample_data = {
            'AAPL': pd.DataFrame({
                'Close': np.random.randn(100).cumsum() + 100,
                'High': np.random.randn(100).cumsum() + 102,
                'Low': np.random.randn(100).cumsum() + 98,
                'Adj Close': np.random.randn(100).cumsum() + 100
            }, index=dates),
            '^GSPC': pd.DataFrame({
                'Close': np.random.randn(100).cumsum() + 200,
                'High': np.random.randn(100).cumsum() + 202,
                'Low': np.random.randn(100).cumsum() + 198,
                'Adj Close': np.random.randn(100).cumsum() + 200
            }, index=dates),
            'VIX': pd.DataFrame({
                'Close': np.random.randn(100).cumsum() + 20,
                'Adj Close': np.random.randn(100).cumsum() + 20
            }, index=dates)
        }
    
    def test_initialization(self):
        """Test FeatureEngineer initialization."""
        self.assertIsNotNone(self.engineer.config)
        self.assertIsNotNone(self.engineer.feature_config)
        self.assertGreater(len(self.engineer.modules), 0)
    
    def test_create_all_features(self):
        """Test creating all features for all assets."""
        features = self.engineer.create_all_features(self.sample_data)
        
        self.assertIsInstance(features, dict)
        self.assertIn('AAPL', features)
        self.assertIn('^GSPC', features)
        
        # Check that features were created
        for asset, asset_features in features.items():
            self.assertIsInstance(asset_features, pd.DataFrame)
            self.assertGreater(len(asset_features.columns), 0)
            self.assertEqual(len(asset_features), len(self.sample_data[asset]))
    
    def test_create_all_features_skip_correlations(self):
        """Test creating features without correlations."""
        features = self.engineer.create_all_features(self.sample_data, skip_correlations=True)
        
        self.assertIsInstance(features, dict)
        # Should still have features but without correlation features
        for asset, asset_features in features.items():
            self.assertIsInstance(asset_features, pd.DataFrame)
            self.assertGreater(len(asset_features.columns), 0)
    
    def test_create_asset_features(self):
        """Test creating features for a single asset."""
        features = self.engineer.create_asset_features(self.sample_data['AAPL'])
        
        self.assertIsInstance(features, pd.DataFrame)
        self.assertGreater(len(features.columns), 0)
        self.assertEqual(len(features), len(self.sample_data['AAPL']))
    
    def test_create_asset_features_empty_data(self):
        """Test creating features with empty data."""
        empty_df = pd.DataFrame()
        features = self.engineer.create_asset_features(empty_df)
        
        self.assertTrue(features.empty)
    
    def test_add_vix_features(self):
        """Test adding VIX features."""
        base_features = pd.DataFrame({
            'RSI': np.random.rand(50),
            'MACD': np.random.rand(50)
        }, index=pd.date_range('2020-01-01', periods=50, freq='D'))
        
        vix_data = pd.DataFrame({
            'Close': np.random.rand(50) * 20 + 15,
            'Adj Close': np.random.rand(50) * 20 + 15
        }, index=base_features.index)
        
        features_with_vix = self.engineer.add_vix_features(base_features, vix_data)
        
        self.assertIn('VIX', features_with_vix.columns)
        self.assertIn('VIX_change', features_with_vix.columns)
        self.assertEqual(len(features_with_vix), len(base_features))
    
    def test_create_regime_labels(self):
        """Test creating regime labels."""
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02] * 20)
        regimes = self.engineer.create_regime_labels(returns)
        
        self.assertIsInstance(regimes, pd.Series)
        self.assertEqual(len(regimes), len(returns))
        self.assertTrue(all(regime in ['Bull', 'Bear', 'Sideways'] for regime in regimes.dropna()))
    
    def test_create_volatility_labels(self):
        """Test creating volatility labels."""
        volatility = pd.Series(np.random.rand(100))
        labels = self.engineer.create_volatility_labels(volatility)
        
        self.assertIsInstance(labels, pd.Series)
        self.assertTrue(all(label in ['Q1', 'Q2', 'Q3', 'Q4'] for label in labels.dropna()))
    
    def test_select_features_correlation(self):
        """Test feature selection by correlation."""
        features = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'feature3': np.random.rand(100)
        })
        target = pd.Series(np.random.rand(100))
        
        selected = self.engineer.select_features(features, target, method='correlation', threshold=0.01)
        
        self.assertIsInstance(selected, pd.DataFrame)
        self.assertLessEqual(len(selected.columns), len(features.columns))
    
    def test_select_features_variance(self):
        """Test feature selection by variance."""
        features = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.zeros(100),  # Zero variance
            'feature3': np.random.rand(100)
        })
        
        selected = self.engineer.select_features(features, method='variance', threshold=0.01)
        
        self.assertIsInstance(selected, pd.DataFrame)
        self.assertNotIn('feature2', selected.columns)  # Should be removed
    
    def test_get_feature_summary(self):
        """Test getting feature summary."""
        features = pd.DataFrame({
            'numeric_feature': np.random.rand(100),
            'categorical_feature': ['A', 'B'] * 50
        })
        
        summary = self.engineer.get_feature_summary(features)
        
        self.assertIsInstance(summary, dict)
        self.assertIn('total_features', summary)
        self.assertIn('total_samples', summary)
        self.assertIn('numeric_features', summary)
        self.assertIn('categorical_features', summary)
    
    def test_add_custom_module(self):
        """Test adding custom feature module."""
        class CustomModule(BaseFeatureModule):
            def create_features(self, data):
                return pd.DataFrame({'custom': [1, 2, 3]})
            
            def get_feature_names(self):
                return ['custom']
            
            def get_required_columns(self):
                return []
        
        custom_module = CustomModule(self.engineer.feature_config)
        self.engineer.add_custom_module('custom', custom_module)
        
        self.assertIn('custom', self.engineer.modules)
        self.assertEqual(self.engineer.list_modules().count('custom'), 1)
    
    def test_remove_module(self):
        """Test removing feature module."""
        initial_modules = len(self.engineer.modules)
        self.engineer.remove_module('time')
        
        self.assertNotIn('time', self.engineer.modules)
        self.assertEqual(len(self.engineer.modules), initial_modules - 1)
    
    def test_list_modules(self):
        """Test listing available modules."""
        modules = self.engineer.list_modules()
        
        self.assertIsInstance(modules, list)
        self.assertGreater(len(modules), 0)
        expected_modules = ['technical', 'statistical', 'time', 'lag', 'correlation']
        for module in expected_modules:
            self.assertIn(module, modules)

if __name__ == '__main__':
    unittest.main() 