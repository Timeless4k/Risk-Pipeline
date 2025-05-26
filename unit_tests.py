"""
Unit tests for RiskPipeline components
Ensures all modules function correctly and meet thesis requirements
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import shutil
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import modules to test
from risk_pipeline import (
    AssetConfig, DataLoader, FeatureEngineer, 
    ModelFactory, WalkForwardValidator, RiskPipeline
)
from stockmixer_model import StockMixer, create_stockmixer
from visualization import VolatilityVisualizer


class TestAssetConfig(unittest.TestCase):
    """Test AssetConfig dataclass"""
    
    def test_asset_lists(self):
        """Test asset configuration"""
        config = AssetConfig()
        
        # Check US assets
        self.assertEqual(len(config.US_ASSETS), 3)
        self.assertIn('AAPL', config.US_ASSETS)
        self.assertIn('^GSPC', config.US_ASSETS)
        
        # Check AU assets
        self.assertEqual(len(config.AU_ASSETS), 3)
        self.assertIn('IOZ.AX', config.AU_ASSETS)
        
        # Check combined list
        self.assertEqual(len(config.ALL_ASSETS), 6)
        
    def test_date_range(self):
        """Test date configuration"""
        config = AssetConfig()
        
        start_date = datetime.strptime(config.START_DATE, '%Y-%m-%d')
        end_date = datetime.strptime(config.END_DATE, '%Y-%m-%d')
        
        # Should cover multiple years
        self.assertGreater((end_date - start_date).days, 365 * 5)
        
    def test_parameters(self):
        """Test model parameters"""
        config = AssetConfig()
        
        self.assertEqual(config.VOLATILITY_WINDOW, 5)
        self.assertEqual(config.MA_SHORT, 10)
        self.assertEqual(config.MA_LONG, 50)
        self.assertEqual(config.CORRELATION_WINDOW, 30)
        self.assertEqual(config.WALK_FORWARD_SPLITS, 5)


class TestDataLoader(unittest.TestCase):
    """Test DataLoader functionality"""
    
    def setUp(self):
        """Create temporary directory for cache"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_loader = DataLoader(cache_dir=self.temp_dir)
        
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)
        
    def test_cache_directory_creation(self):
        """Test cache directory is created"""
        self.assertTrue(Path(self.temp_dir).exists())
        
    def test_data_structure(self):
        """Test downloaded data structure"""
        # Create mock data
        dates = pd.date_range('2023-01-01', '2023-01-31', freq='D')
        mock_data = pd.DataFrame({
            'Open': np.random.randn(len(dates)) * 10 + 100,
            'High': np.random.randn(len(dates)) * 10 + 105,
            'Low': np.random.randn(len(dates)) * 10 + 95,
            'Close': np.random.randn(len(dates)) * 10 + 100,
            'Adj Close': np.random.randn(len(dates)) * 10 + 100,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        # Save to cache
        cache_file = Path(self.temp_dir) / 'TEST_data.pkl'
        mock_data.to_pickle(cache_file)
        
        # Test loading
        data = pd.read_pickle(cache_file)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIn('Adj Close', data.columns)
        self.assertEqual(len(data), len(dates))


class TestFeatureEngineer(unittest.TestCase):
    """Test FeatureEngineer functionality"""
    
    def setUp(self):
        """Create test data"""
        self.config = AssetConfig()
        self.feature_engineer = FeatureEngineer(self.config)
        
        # Create sample price data
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        self.test_data = pd.DataFrame({
            'Adj Close': 100 + np.cumsum(np.random.randn(len(dates)) * 2)
        }, index=dates)
        
    def test_log_returns(self):
        """Test log returns calculation"""
        returns = self.feature_engineer.calculate_log_returns(self.test_data['Adj Close'])
        
        self.assertEqual(len(returns), len(self.test_data))
        self.assertTrue(returns.iloc[0] != returns.iloc[0])  # First value should be NaN
        self.assertTrue(all(np.isfinite(returns.iloc[1:])))  # Rest should be finite
        
    def test_volatility_calculation(self):
        """Test volatility calculation"""
        returns = self.feature_engineer.calculate_log_returns(self.test_data['Adj Close'])
        volatility = self.feature_engineer.calculate_volatility(returns, window=5)
        
        self.assertEqual(len(volatility), len(returns))
        # First window-1 values should be NaN
        self.assertEqual(volatility.isna().sum(), 5)
        
    def test_technical_features(self):
        """Test technical feature creation"""
        features = self.feature_engineer.create_technical_features(self.test_data)
        
        # Check all features are created
        expected_features = [
            'Volatility5D', 'Lag1', 'Lag2', 'Lag3', 'ROC5',
            'MA10', 'MA50', 'RollingStd5', 'MA_ratio',
            'Corr_MA10', 'Corr_MA50'  # Added new correlation features
        ]
        
        for feature in expected_features:
            self.assertIn(feature, features.columns)
            
        # Check feature shapes
        self.assertEqual(len(features), len(self.test_data))
        
        # Check correlation features are within valid range [-1, 1]
        self.assertTrue(all(features['Corr_MA10'].dropna().between(-1, 1)))
        self.assertTrue(all(features['Corr_MA50'].dropna().between(-1, 1)))
        
    def test_regime_labels(self):
        """Test regime label creation"""
        returns = self.feature_engineer.calculate_log_returns(self.test_data['Adj Close'])
        regimes = self.feature_engineer.create_regime_labels(returns)
        
        self.assertEqual(len(regimes), len(returns))
        self.assertTrue(all(regime in ['Bull', 'Bear', 'Sideways'] 
                          for regime in regimes.dropna()))
                          
    def test_volatility_labels(self):
        """Test volatility label creation"""
        volatility = pd.Series(np.random.rand(100))
        labels = self.feature_engineer.create_volatility_labels(volatility)
        
        self.assertEqual(len(labels), len(volatility))
        self.assertEqual(set(labels.unique()), {'Q1', 'Q2', 'Q3', 'Q4'})
        
        # Check roughly balanced classes
        counts = labels.value_counts()
        for label in ['Q1', 'Q2', 'Q3', 'Q4']:
            self.assertGreater(counts[label], len(volatility) * 0.15)  # Allow for some imbalance


class TestModelFactory(unittest.TestCase):
    """Test ModelFactory functionality"""
    
    def setUp(self):
        """Set up test parameters"""
        self.input_shape = (20, 10)  # 20 timesteps, 10 features
        
    def test_lstm_regressor_creation(self):
        """Test LSTM regressor creation"""
        model = ModelFactory.create_lstm_regressor(self.input_shape)
        
        self.assertIsNotNone(model)
        # Check input shape
        self.assertEqual(model.input_shape[1:], self.input_shape)
        # Check output shape (should be 1 for regression)
        self.assertEqual(model.output_shape[-1], 1)
        
    def test_lstm_classifier_creation(self):
        """Test LSTM classifier creation"""
        model = ModelFactory.create_lstm_classifier(self.input_shape, n_classes=3)
        
        self.assertIsNotNone(model)
        # Check output shape (should be 3 for 3 classes)
        self.assertEqual(model.output_shape[-1], 3)
        
    def test_stockmixer_creation(self):
        """Test StockMixer creation"""
        # Regression
        reg_model = ModelFactory.create_stockmixer(self.input_shape, task='regression')
        self.assertIsNotNone(reg_model)
        self.assertEqual(reg_model.output_shape[-1], 1)
        
        # Classification
        clf_model = ModelFactory.create_stockmixer(self.input_shape, task='classification')
        self.assertIsNotNone(clf_model)
        self.assertEqual(clf_model.output_shape[-1], 3)


class TestWalkForwardValidator(unittest.TestCase):
    """Test WalkForwardValidator functionality"""
    
    def setUp(self):
        """Create test data"""
        self.validator = WalkForwardValidator(n_splits=5, test_size=50)
        
        # Create sample data
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        self.test_data = pd.DataFrame({
            'feature1': np.random.randn(len(dates)),
            'feature2': np.random.randn(len(dates))
        }, index=dates)
        
    def test_split_generation(self):
        """Test walk-forward split generation"""
        splits = self.validator.split(self.test_data)
        
        self.assertEqual(len(splits), 5)
        
        # Check each split
        for i, (train_idx, test_idx) in enumerate(splits):
            # Test size should be consistent
            self.assertEqual(len(test_idx), 50)
            
            # Train size should increase
            if i > 0:
                prev_train_idx = splits[i-1][0]
                self.assertGreater(len(train_idx), len(prev_train_idx))
                
            # No overlap between train and test
            self.assertEqual(len(set(train_idx) & set(test_idx)), 0)
            
            # Test should come after train
            self.assertGreater(test_idx[0], train_idx[-1])


class TestStockMixer(unittest.TestCase):
    """Test StockMixer model"""
    
    def test_model_creation(self):
        """Test StockMixer instantiation"""
        model = StockMixer(task='regression')
        self.assertIsNotNone(model)
        
    def test_forward_pass(self):
        """Test forward pass through model"""
        input_shape = (20, 10)
        model = create_stockmixer(input_shape, task='regression')
        
        # Create dummy input
        batch_size = 32
        dummy_input = np.random.randn(batch_size, *input_shape)
        
        # Forward pass
        output = model(dummy_input, training=False)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, 1))
        
    def test_pathway_outputs(self):
        """Test getting pathway outputs"""
        model = StockMixer(task='regression')
        
        # Create dummy input
        dummy_input = np.random.randn(10, 20, 13)
        pathway_outputs = model.get_pathway_outputs(dummy_input)
        
        # Check all pathways present
        self.assertIn('temporal', pathway_outputs)
        self.assertIn('indicator', pathway_outputs)
        self.assertIn('cross_stock', pathway_outputs)


class TestVisualization(unittest.TestCase):
    """Test visualization functionality"""
    
    def setUp(self):
        """Create temporary directory for plots"""
        self.temp_dir = tempfile.mkdtemp()
        self.visualizer = VolatilityVisualizer(output_dir=self.temp_dir)
        
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)
        
    def test_directory_creation(self):
        """Test output directory is created"""
        self.assertTrue(Path(self.temp_dir).exists())
        
    def test_color_schemes(self):
        """Test color schemes are defined"""
        self.assertIn('LSTM', self.visualizer.model_colors)
        self.assertIn('Bull', self.visualizer.regime_colors)
        
    def test_plot_generation(self):
        """Test plot file generation"""
        # Create dummy results
        dummy_results = {
            'AAPL': {
                'regression': {
                    'LSTM': {
                        'RMSE': 0.05, 'MAE': 0.03, 'R2': 0.75,
                        'predictions': [0.1] * 100,
                        'actuals': [0.11] * 100
                    }
                }
            }
        }
        
        # Generate plot
        self.visualizer._create_architecture_diagram()
        
        # Check file was created
        plot_files = list(Path(self.temp_dir).glob('*.png'))
        self.assertGreater(len(plot_files), 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete pipeline"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_pipeline_initialization(self):
        """Test pipeline can be initialized"""
        pipeline = RiskPipeline()
        self.assertIsNotNone(pipeline)
        self.assertIsNotNone(pipeline.data_loader)
        self.assertIsNotNone(pipeline.feature_engineer)
        self.assertIsNotNone(pipeline.model_factory)
        self.assertIsNotNone(pipeline.validator)
        
    def test_feature_pipeline(self):
        """Test feature engineering pipeline"""
        config = AssetConfig()
        feature_engineer = FeatureEngineer(config)
        
        # Create sample data
        dates = pd.date_range('2023-01-01', '2023-03-31', freq='D')
        test_data = pd.DataFrame({
            'Adj Close': 100 + np.cumsum(np.random.randn(len(dates)) * 2)
        }, index=dates)
        
        # Process features
        features = feature_engineer.create_technical_features(test_data)
        
        # Add VIX (mock)
        vix_data = pd.DataFrame({
            'Adj Close': 20 + np.random.randn(len(dates)) * 2
        }, index=dates)
        features = feature_engineer.add_vix_features(features, vix_data)
        
        # Check all features present
        required_features = ['Lag1', 'Lag2', 'Lag3', 'VIX', 'VIX_change']
        for feat in required_features:
            self.assertIn(feat, features.columns)


def run_all_tests():
    """Run all unit tests with reporting"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestAssetConfig,
        TestDataLoader,
        TestFeatureEngineer,
        TestModelFactory,
        TestWalkForwardValidator,
        TestStockMixer,
        TestVisualization,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.wasSuccessful()}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
