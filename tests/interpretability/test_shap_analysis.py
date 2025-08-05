"""
Comprehensive test suite for SHAP analysis system.

Tests all components of the SHAP analysis system including:
- ExplainerFactory
- InterpretationUtils
- SHAPAnalyzer
- SHAPVisualizer
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import warnings

# Suppress warnings for testing
warnings.filterwarnings('ignore')

# Import the modules to test
from risk_pipeline.interpretability.explainer_factory import (
    ExplainerFactory, ARIMAExplainer, StockMixerExplainer
)
from risk_pipeline.interpretability.interpretation_utils import InterpretationUtils
from risk_pipeline.interpretability.shap_analyzer import SHAPAnalyzer
from risk_pipeline.visualization.shap_visualizer import SHAPVisualizer


class TestExplainerFactory(unittest.TestCase):
    """Test the ExplainerFactory class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Mock()
        self.config.shap = Mock()
        self.config.shap.background_samples = 50
        self.config.output = Mock()
        self.config.output.shap_dir = "test_shap_dir"
        
        self.factory = ExplainerFactory(self.config)
        
        # Mock data
        self.X = np.random.randn(100, 10)
        self.feature_names = [f'feature_{i}' for i in range(10)]
    
    def test_init(self):
        """Test ExplainerFactory initialization."""
        self.assertIsNotNone(self.factory)
        self.assertEqual(self.factory.config, self.config)
        self.assertEqual(len(self.factory._explainers), 0)
        self.assertEqual(len(self.factory._background_data), 0)
    
    def test_create_xgboost_explainer(self):
        """Test XGBoost explainer creation."""
        # Mock XGBoost model
        mock_model = Mock()
        mock_model.__class__.__name__ = 'XGBRegressor'
        
        explainer = self.factory._create_xgboost_explainer(
            model=mock_model,
            X=self.X,
            task='regression'
        )
        
        self.assertIsNotNone(explainer)
        self.assertIn('xgboost_regression', self.factory._explainers)
    
    def test_create_lstm_explainer(self):
        """Test LSTM explainer creation."""
        # Mock LSTM model
        mock_model = Mock()
        mock_model.__class__.__name__ = 'Sequential'
        
        explainer = self.factory._create_lstm_explainer(
            model=mock_model,
            X=self.X,
            task='regression'
        )
        
        self.assertIsNotNone(explainer)
        self.assertIn('lstm_regression', self.factory._explainers)
        self.assertIn('lstm_regression', self.factory._background_data)
    
    def test_create_arima_explainer(self):
        """Test ARIMA explainer creation."""
        # Mock ARIMA model
        mock_model = Mock()
        
        explainer = self.factory._create_arima_explainer(
            model=mock_model,
            X=self.X,
            task='regression'
        )
        
        self.assertIsInstance(explainer, ARIMAExplainer)
        self.assertEqual(explainer.model, mock_model)
        self.assertEqual(explainer.task, 'regression')
    
    def test_create_stockmixer_explainer(self):
        """Test StockMixer explainer creation."""
        # Mock StockMixer model
        mock_model = Mock()
        mock_model.__class__.__name__ = 'StockMixer'
        
        explainer = self.factory._create_stockmixer_explainer(
            model=mock_model,
            X=self.X,
            task='regression'
        )
        
        self.assertIsInstance(explainer, StockMixerExplainer)
        self.assertEqual(explainer.model, mock_model)
        self.assertEqual(explainer.task, 'regression')
    
    def test_prepare_deep_background_data(self):
        """Test background data preparation for deep learning models."""
        # Test LSTM
        background_data = self.factory._prepare_deep_background_data(
            self.X, model_type='lstm'
        )
        self.assertEqual(background_data.shape[0], 50)  # background_samples
        self.assertEqual(background_data.shape[1], 1)   # time dimension
        self.assertEqual(background_data.shape[2], 10)  # features
        
        # Test StockMixer
        background_data = self.factory._prepare_deep_background_data(
            self.X, model_type='stockmixer'
        )
        self.assertEqual(background_data.shape[0], 50)
        self.assertEqual(background_data.shape[1], 1)
        self.assertEqual(background_data.shape[2], 10)
    
    def test_get_explainer(self):
        """Test getting stored explainer."""
        # Add a mock explainer
        mock_explainer = Mock()
        self.factory._explainers['test_regression'] = mock_explainer
        
        retrieved = self.factory.get_explainer('test', 'regression')
        self.assertEqual(retrieved, mock_explainer)
        
        # Test non-existent explainer
        retrieved = self.factory.get_explainer('nonexistent', 'regression')
        self.assertIsNone(retrieved)


class TestARIMAExplainer(unittest.TestCase):
    """Test the ARIMAExplainer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Mock()
        self.model = Mock()
        self.X = np.random.randn(100, 10)
        self.task = 'regression'
        
        # Mock fitted model
        self.fitted_model = Mock()
        self.fitted_model.params = pd.Series([0.1, 0.2, 0.3], index=['param1', 'param2', 'param3'])
        self.fitted_model.aic = 100.0
        self.fitted_model.bic = 110.0
        self.fitted_model.resid = pd.Series(np.random.randn(100))
        
        self.model.fit.return_value = self.fitted_model
        
        self.explainer = ARIMAExplainer(self.model, self.X, self.task, self.config)
    
    def test_init(self):
        """Test ARIMAExplainer initialization."""
        self.assertEqual(self.explainer.model, self.model)
        self.assertEqual(self.explainer.X, self.X)
        self.assertEqual(self.explainer.task, self.task)
        self.assertEqual(self.explainer.config, self.config)
        self.assertEqual(self.explainer.fitted_model, self.fitted_model)
    
    def test_analyze_coefficients(self):
        """Test coefficient analysis."""
        result = self.explainer._analyze_coefficients()
        
        self.assertIn('parameters', result)
        self.assertIn('aic', result)
        self.assertIn('bic', result)
        self.assertEqual(result['aic'], 100.0)
        self.assertEqual(result['bic'], 110.0)
    
    def test_analyze_residuals(self):
        """Test residual analysis."""
        result = self.explainer._analyze_residuals()
        
        self.assertIn('residuals', result)
        self.assertIn('mean', result)
        self.assertIn('std', result)
        self.assertIn('skewness', result)
        self.assertIn('kurtosis', result)
    
    def test_shap_values(self):
        """Test SHAP-like values generation."""
        shap_values = self.explainer.shap_values(self.X)
        
        self.assertEqual(shap_values.shape[0], len(self.X))
        self.assertEqual(shap_values.shape[1], len(self.fitted_model.params))
        self.assertTrue(np.allclose(np.sum(shap_values, axis=1), 1.0))


class TestStockMixerExplainer(unittest.TestCase):
    """Test the StockMixerExplainer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Mock()
        self.config.shap = Mock()
        self.config.shap.background_samples = 50
        
        self.model = Mock()
        self.X = np.random.randn(100, 10)
        self.task = 'regression'
        
        # Mock pathway outputs
        self.pathway_outputs = {
            'temporal': np.random.randn(100, 32),
            'indicator': np.random.randn(100, 32),
            'cross_stock': np.random.randn(100, 32)
        }
        self.model.get_pathway_outputs.return_value = self.pathway_outputs
        
        self.explainer = StockMixerExplainer(self.model, self.X, self.task, self.config)
    
    def test_init(self):
        """Test StockMixerExplainer initialization."""
        self.assertEqual(self.explainer.model, self.model)
        self.assertEqual(self.explainer.X, self.X)
        self.assertEqual(self.explainer.task, self.task)
        self.assertEqual(self.explainer.config, self.config)
        self.assertIsNotNone(self.explainer.deep_explainer)
    
    def test_prepare_background_data(self):
        """Test background data preparation."""
        background_data = self.explainer._prepare_background_data(self.X)
        
        self.assertEqual(background_data.shape[0], 50)
        self.assertEqual(background_data.shape[1], 1)
        self.assertEqual(background_data.shape[2], 10)
    
    def test_analyze_pathways(self):
        """Test pathway analysis."""
        result = self.explainer._analyze_pathways(self.X)
        
        self.assertIn('temporal', result)
        self.assertIn('indicator', result)
        self.assertIn('cross_stock', result)
        
        for pathway_name, pathway_data in result.items():
            self.assertIn('output_shape', pathway_data)
            self.assertIn('mean_activation', pathway_data)
            self.assertIn('std_activation', pathway_data)


class TestInterpretationUtils(unittest.TestCase):
    """Test the InterpretationUtils class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Mock()
        self.config.output = Mock()
        self.config.output.shap_dir = "test_shap_dir"
        
        self.utils = InterpretationUtils(self.config)
        
        # Test data
        self.shap_values = np.random.randn(100, 10)
        self.X = np.random.randn(100, 10)
        self.feature_names = [f'feature_{i}' for i in range(10)]
    
    def test_init(self):
        """Test InterpretationUtils initialization."""
        self.assertEqual(self.utils.config, self.config)
        self.assertTrue(self.utils.shap_data_dir.exists())
    
    def test_analyze_time_series_shap(self):
        """Test time-series SHAP analysis."""
        result = self.utils.analyze_time_series_shap(
            shap_values=self.shap_values,
            X=self.X,
            feature_names=self.feature_names,
            window_size=30
        )
        
        self.assertIn('rolling_stats', result)
        self.assertIn('temporal_importance', result)
        self.assertIn('regime_changes', result)
        self.assertIn('seasonality', result)
    
    def test_analyze_feature_interactions(self):
        """Test feature interaction analysis."""
        result = self.utils.analyze_feature_interactions(
            shap_values=self.shap_values,
            X=self.X,
            feature_names=self.feature_names,
            top_k=5
        )
        
        self.assertIn('pairwise_interactions', result)
        self.assertIn('top_interactions', result)
        self.assertIn('interaction_patterns', result)
    
    def test_calculate_rolling_shap_stats(self):
        """Test rolling SHAP statistics calculation."""
        result = self.utils._calculate_rolling_shap_stats(
            self.shap_values, pd.DataFrame(self.X), window_size=30
        )
        
        self.assertIn('rolling_mean', result)
        self.assertIn('rolling_std', result)
        self.assertIn('rolling_max', result)
        self.assertIn('rolling_min', result)
    
    def test_analyze_temporal_importance(self):
        """Test temporal importance analysis."""
        result = self.utils._analyze_temporal_importance(
            self.shap_values, pd.DataFrame(self.X), self.feature_names
        )
        
        for feature in self.feature_names:
            self.assertIn(feature, result)
            feature_data = result[feature]
            self.assertIn('mean', feature_data)
            self.assertIn('std', feature_data)
            self.assertIn('trend', feature_data)
            self.assertIn('volatility', feature_data)
    
    def test_detect_regime_changes(self):
        """Test regime change detection."""
        result = self.utils._detect_regime_changes(
            self.shap_values, pd.DataFrame(self.X), window_size=30
        )
        
        self.assertIn('change_points', result)
        self.assertIn('rolling_importance', result)
        self.assertIn('threshold', result)
    
    def test_save_and_load_shap_data(self):
        """Test SHAP data persistence."""
        metadata = {
            'asset': 'AAPL',
            'model_type': 'xgboost',
            'task': 'regression',
            'feature_names': self.feature_names
        }
        
        # Test saving
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / 'test_shap'
            
            success = self.utils.save_shap_data(
                shap_values=self.shap_values,
                metadata=metadata,
                filepath=filepath
            )
            
            self.assertTrue(success)
            
            # Test loading
            loaded_shap, loaded_metadata = self.utils.load_shap_data(filepath)
            
            self.assertIsNotNone(loaded_shap)
            self.assertIsNotNone(loaded_metadata)
            np.testing.assert_array_equal(loaded_shap, self.shap_values)
            self.assertEqual(loaded_metadata['asset'], metadata['asset'])
    
    def test_generate_individual_explanation(self):
        """Test individual prediction explanation."""
        # Mock explainer
        mock_explainer = Mock()
        mock_explainer.shap_values.return_value = self.shap_values[:1]
        mock_explainer.expected_value = 0.5
        
        result = self.utils.generate_individual_explanation(
            explainer=mock_explainer,
            instance=self.X[:1],
            feature_names=self.feature_names,
            instance_index=0
        )
        
        self.assertIn('instance_index', result)
        self.assertIn('feature_contributions', result)
        self.assertIn('total_contribution', result)
        self.assertIn('base_value', result)
    
    def test_create_comparison_analysis(self):
        """Test comparison analysis creation."""
        # Mock SHAP results
        shap_results = {
            'AAPL': {
                'regression': {
                    'xgboost': {
                        'feature_importance': {
                            'feature_1': 0.5,
                            'feature_2': 0.3
                        }
                    }
                }
            }
        }
        
        result = self.utils.create_comparison_analysis(
            shap_results=shap_results,
            assets=['AAPL'],
            model_types=['xgboost'],
            task='regression'
        )
        
        self.assertIn('feature_importance_comparison', result)
        self.assertIn('performance_patterns', result)
        self.assertIn('summary_statistics', result)


class TestSHAPAnalyzer(unittest.TestCase):
    """Test the SHAPAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Mock()
        self.config.shap = Mock()
        self.config.shap.background_samples = 50
        self.config.output = Mock()
        self.config.output.shap_dir = "test_shap_dir"
        
        self.results_manager = Mock()
        self.analyzer = SHAPAnalyzer(self.config, self.results_manager)
        
        # Test data
        self.shap_values = np.random.randn(100, 10)
        self.X = np.random.randn(100, 10)
        self.feature_names = [f'feature_{i}' for i in range(10)]
    
    def test_init(self):
        """Test SHAPAnalyzer initialization."""
        self.assertEqual(self.analyzer.config, self.config)
        self.assertEqual(self.analyzer.results_manager, self.results_manager)
        self.assertIsNotNone(self.analyzer.explainer_factory)
        self.assertIsNotNone(self.analyzer.interpretation_utils)
    
    def test_analyze_single_model(self):
        """Test single model analysis."""
        # Mock model and explainer
        mock_model = Mock()
        mock_explainer = Mock()
        mock_explainer.shap_values.return_value = self.shap_values
        
        self.analyzer.explainer_factory.create_explainer.return_value = mock_explainer
        
        result = self.analyzer._analyze_single_model(
            model=mock_model,
            X=self.X,
            feature_names=self.feature_names,
            asset='AAPL',
            model_type='xgboost',
            task='regression'
        )
        
        self.assertIn('shap_values', result)
        self.assertIn('feature_importance', result)
        self.assertIn('explainer', result)
        self.assertIn('plots', result)
        self.assertEqual(result['model_type'], 'xgboost')
        self.assertEqual(result['task'], 'regression')
        self.assertEqual(result['asset'], 'AAPL')
    
    def test_calculate_feature_importance(self):
        """Test feature importance calculation."""
        result = self.analyzer._calculate_feature_importance(
            shap_values=self.shap_values,
            feature_names=self.feature_names
        )
        
        self.assertEqual(len(result), len(self.feature_names))
        
        # Check that importance is sorted in descending order
        importance_values = list(result.values())
        self.assertEqual(importance_values, sorted(importance_values, reverse=True))
    
    def test_get_feature_importance(self):
        """Test getting feature importance."""
        # Mock stored results
        result_key = "AAPL_xgboost_regression"
        self.analyzer._shap_values[result_key] = self.shap_values
        
        mock_shap_results = {
            'regression': {
                'xgboost': {
                    'feature_importance': {
                        'feature_1': 0.5,
                        'feature_2': 0.3,
                        'feature_3': 0.2
                    }
                }
            }
        }
        self.analyzer.results_manager.get_shap_results.return_value = mock_shap_results
        
        result = self.analyzer.get_feature_importance(
            asset='AAPL',
            model_type='xgboost',
            task='regression',
            top_n=2
        )
        
        self.assertEqual(len(result), 2)
        self.assertIn('feature_1', result)
        self.assertIn('feature_2', result)
    
    def test_get_shap_values(self):
        """Test getting SHAP values."""
        result_key = "AAPL_xgboost_regression"
        self.analyzer._shap_values[result_key] = self.shap_values
        
        retrieved = self.analyzer.get_shap_values(
            asset='AAPL',
            model_type='xgboost',
            task='regression'
        )
        
        np.testing.assert_array_equal(retrieved, self.shap_values)
    
    def test_compare_feature_importance(self):
        """Test feature importance comparison."""
        mock_shap_results = {
            'regression': {
                'xgboost': {
                    'feature_importance': {
                        'feature_1': 0.5,
                        'feature_2': 0.3
                    }
                },
                'lstm': {
                    'feature_importance': {
                        'feature_1': 0.4,
                        'feature_2': 0.4
                    }
                }
            }
        }
        self.analyzer.results_manager.get_shap_results.return_value = mock_shap_results
        
        result = self.analyzer.compare_feature_importance(
            asset='AAPL',
            task='regression',
            model_types=['xgboost', 'lstm']
        )
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 4)  # 2 features * 2 models
    
    def test_explain_prediction(self):
        """Test individual prediction explanation."""
        # Mock explainer
        mock_explainer = Mock()
        self.analyzer._explainers["AAPL_xgboost_regression"] = mock_explainer
        
        # Mock model
        mock_model = Mock()
        self.analyzer.results_manager.get_model.return_value = mock_model
        
        # Mock interpretation utils
        mock_explanation = {
            'instance_index': 0,
            'feature_contributions': {'feature_1': 0.5},
            'total_contribution': 0.5,
            'base_value': 0.0
        }
        self.analyzer.interpretation_utils.generate_individual_explanation.return_value = mock_explanation
        
        result = self.analyzer.explain_prediction(
            asset='AAPL',
            model_type='xgboost',
            task='regression',
            instance=self.X[:1],
            feature_names=self.feature_names,
            instance_index=0
        )
        
        self.assertEqual(result, mock_explanation)
    
    def test_analyze_feature_interactions(self):
        """Test feature interaction analysis."""
        # Mock SHAP values
        result_key = "AAPL_xgboost_regression"
        self.analyzer._shap_values[result_key] = self.shap_values
        
        # Mock features
        mock_features = {
            'features': self.X,
            'feature_names': self.feature_names
        }
        self.analyzer.results_manager.get_features.return_value = mock_features
        
        # Mock interpretation utils
        mock_interactions = {
            'pairwise_interactions': {},
            'top_interactions': [],
            'interaction_patterns': {}
        }
        self.analyzer.interpretation_utils.analyze_feature_interactions.return_value = mock_interactions
        
        result = self.analyzer.analyze_feature_interactions(
            asset='AAPL',
            model_type='xgboost',
            task='regression',
            top_k=5
        )
        
        self.assertEqual(result, mock_interactions)
    
    def test_generate_time_series_shap(self):
        """Test time-series SHAP analysis."""
        # Mock SHAP values
        result_key = "AAPL_xgboost_regression"
        self.analyzer._shap_values[result_key] = self.shap_values
        
        # Mock features
        mock_features = {
            'features': self.X,
            'feature_names': self.feature_names,
            'time_index': pd.date_range('2020-01-01', periods=100, freq='D')
        }
        self.analyzer.results_manager.get_features.return_value = mock_features
        
        # Mock interpretation utils
        mock_time_series = {
            'rolling_stats': {},
            'temporal_importance': {},
            'regime_changes': {},
            'seasonality': {}
        }
        self.analyzer.interpretation_utils.analyze_time_series_shap.return_value = mock_time_series
        
        result = self.analyzer.generate_time_series_shap(
            asset='AAPL',
            model_type='xgboost',
            task='regression',
            window_size=30
        )
        
        self.assertEqual(result, mock_time_series)
    
    def test_save_and_load_shap_data(self):
        """Test SHAP data persistence."""
        # Mock SHAP values
        result_key = "AAPL_xgboost_regression"
        self.analyzer._shap_values[result_key] = self.shap_values
        
        # Mock features
        mock_features = {
            'feature_names': self.feature_names
        }
        self.analyzer.results_manager.get_features.return_value = mock_features
        
        # Mock interpretation utils
        self.analyzer.interpretation_utils.save_shap_data.return_value = True
        self.analyzer.interpretation_utils.load_shap_data.return_value = (self.shap_values, {})
        
        # Test saving
        success = self.analyzer.save_shap_data(
            asset='AAPL',
            model_type='xgboost',
            task='regression'
        )
        
        self.assertTrue(success)
        
        # Test loading
        success = self.analyzer.load_shap_data(
            asset='AAPL',
            model_type='xgboost',
            task='regression'
        )
        
        self.assertTrue(success)


class TestSHAPVisualizer(unittest.TestCase):
    """Test the SHAPVisualizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Mock()
        self.config.output = Mock()
        self.config.output.shap_dir = "test_shap_plots"
        
        self.visualizer = SHAPVisualizer(self.config)
        
        # Test data
        self.shap_values = np.random.randn(100, 10)
        self.X = np.random.randn(100, 10)
        self.feature_names = [f'feature_{i}' for i in range(10)]
    
    def test_init(self):
        """Test SHAPVisualizer initialization."""
        self.assertEqual(self.visualizer.config, self.config)
        self.assertTrue(self.visualizer.output_dir.exists())
    
    def test_create_basic_shap_plots(self):
        """Test basic SHAP plot creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            plots = self.visualizer._create_basic_shap_plots(
                shap_values=self.shap_values,
                X=self.X,
                feature_names=self.feature_names,
                output_dir=output_dir,
                asset='AAPL',
                model_type='xgboost',
                task='regression'
            )
            
            self.assertIn('summary', plots)
            self.assertIn('beeswarm', plots)
            self.assertIn('waterfall', plots)
            
            # Check that plot files were created
            for plot_path in plots.values():
                self.assertTrue(Path(plot_path).exists())
    
    def test_create_arima_plots(self):
        """Test ARIMA-specific plot creation."""
        # Mock explainer with explanations
        mock_explainer = Mock()
        mock_explainer.explain.return_value = {
            'residuals': {
                'residuals': np.random.randn(100)
            },
            'decomposition': {
                'trend': np.random.randn(100),
                'seasonal': np.random.randn(100),
                'residual': np.random.randn(100)
            },
            'forecast_intervals': {
                'forecast': np.random.randn(10),
                'confidence_intervals': {
                    'lower': np.random.randn(10),
                    'upper': np.random.randn(10)
                }
            }
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            plots = self.visualizer._create_arima_plots(
                explainer=mock_explainer,
                X=self.X,
                feature_names=self.feature_names,
                output_dir=output_dir,
                asset='AAPL',
                task='regression'
            )
            
            self.assertIn('residuals', plots)
            self.assertIn('decomposition', plots)
            self.assertIn('forecast', plots)
    
    def test_create_stockmixer_plots(self):
        """Test StockMixer-specific plot creation."""
        # Mock explainer with explanations
        mock_explainer = Mock()
        mock_explainer.explain.return_value = {
            'pathways': {
                'temporal': {
                    'mean_activation': 0.5,
                    'std_activation': 0.1
                },
                'indicator': {
                    'mean_activation': 0.4,
                    'std_activation': 0.2
                },
                'cross_stock': {
                    'mean_activation': 0.3,
                    'std_activation': 0.15
                }
            },
            'feature_mixing': {
                'temporal_activations': {
                    'mean': 0.5,
                    'std': 0.1
                },
                'indicator_activations': {
                    'mean': 0.4,
                    'std': 0.2
                },
                'cross_stock_activations': {
                    'mean': 0.3,
                    'std': 0.15
                }
            }
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            plots = self.visualizer._create_stockmixer_plots(
                explainer=mock_explainer,
                X=self.X,
                feature_names=self.feature_names,
                output_dir=output_dir,
                asset='AAPL',
                task='regression'
            )
            
            self.assertIn('pathways', plots)
            self.assertIn('mixing', plots)
    
    def test_create_lstm_plots(self):
        """Test LSTM-specific plot creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            plots = self.visualizer._create_lstm_plots(
                shap_values=self.shap_values,
                X=self.X,
                feature_names=self.feature_names,
                output_dir=output_dir,
                asset='AAPL',
                task='regression'
            )
            
            self.assertIn('temporal', plots)
            self.assertIn('importance_time', plots)
    
    def test_create_xgboost_plots(self):
        """Test XGBoost-specific plot creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            plots = self.visualizer._create_xgboost_plots(
                shap_values=self.shap_values,
                X=self.X,
                feature_names=self.feature_names,
                output_dir=output_dir,
                asset='AAPL',
                task='regression'
            )
            
            self.assertIn('dependence', plots)
            self.assertIn('interactions', plots)
    
    def test_create_time_series_plots(self):
        """Test time-series specific plot creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            plots = self.visualizer._create_time_series_plots(
                shap_values=self.shap_values,
                X=self.X,
                feature_names=self.feature_names,
                output_dir=output_dir,
                asset='AAPL',
                model_type='xgboost',
                task='regression'
            )
            
            self.assertIn('rolling', plots)
            self.assertIn('regime', plots)
    
    def test_create_interaction_plots(self):
        """Test feature interaction plot creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            plots = self.visualizer._create_interaction_plots(
                shap_values=self.shap_values,
                X=self.X,
                feature_names=self.feature_names,
                output_dir=output_dir,
                asset='AAPL',
                model_type='xgboost',
                task='regression'
            )
            
            self.assertIn('interaction_matrix', plots)
            self.assertIn('top_interactions', plots)
    
    def test_create_comparison_plots(self):
        """Test comparison plot creation."""
        # Mock SHAP results
        shap_results = {
            'AAPL': {
                'regression': {
                    'xgboost': {
                        'feature_importance': {
                            'feature_1': 0.5,
                            'feature_2': 0.3
                        }
                    },
                    'lstm': {
                        'feature_importance': {
                            'feature_1': 0.4,
                            'feature_2': 0.4
                        }
                    }
                }
            }
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            plots = self.visualizer.create_comparison_plots(
                shap_results=shap_results,
                assets=['AAPL'],
                model_types=['xgboost', 'lstm'],
                task='regression'
            )
            
            self.assertIn('importance_comparison', plots)
            self.assertIn('performance_comparison', plots)
            self.assertIn('asset_comparison', plots)
    
    def test_get_top_features(self):
        """Test getting top features."""
        top_features = self.visualizer._get_top_features(
            self.shap_values, self.feature_names, top_k=5
        )
        
        self.assertEqual(len(top_features), 5)
        self.assertTrue(all(feature in self.feature_names for feature in top_features))
    
    def test_get_top_interactions(self):
        """Test getting top interactions."""
        interaction_matrix = np.random.rand(10, 10)
        interaction_matrix = (interaction_matrix + interaction_matrix.T) / 2  # Make symmetric
        
        top_interactions = self.visualizer._get_top_interactions(
            interaction_matrix, self.feature_names, top_k=5
        )
        
        self.assertEqual(len(top_interactions), 5)
        for interaction in top_interactions:
            self.assertEqual(len(interaction), 3)  # feature1, feature2, strength


if __name__ == '__main__':
    # Create temporary directories for testing
    temp_dirs = []
    
    try:
        # Run tests
        unittest.main(verbosity=2)
    finally:
        # Clean up temporary directories
        for temp_dir in temp_dirs:
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir) 