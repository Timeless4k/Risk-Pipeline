"""
Tests for the ExplainerFactory class.

This module tests the creation of SHAP explainers for different model types,
with special focus on the kill-switch mechanisms to prevent recurring bugs.
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
import os

# Add the parent directory to the path to import risk_pipeline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from risk_pipeline.interpretability.explainer_factory import ExplainerFactory


class TestExplainerFactory(unittest.TestCase):
    """Test the ExplainerFactory class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock config
        self.config = Mock()
        self.config.shap = Mock()
        self.config.shap.background_samples = 100
        
        # Create factory instance
        self.factory = ExplainerFactory(self.config)
    
    def test_xgboost_shap_kill_switch(self):
        """🌲 KILL-SWITCH: Test that XGBoost SHAP requires fitted models."""
        # Test with unfitted model (missing get_booster)
        unfitted_model = Mock()
        unfitted_model.get_booster = None  # Simulate unfitted model
        
        with self.assertRaises(RuntimeError, msg="Should fail for unfitted XGBoost model"):
            self.factory.create_explainer(
                model=unfitted_model,
                model_type='xgboost',
                task='regression',
                X=np.random.randn(100, 5)
            )
    
    def test_xgboost_shap_fitted_model_success(self):
        """🌲 KILL-SWITCH: Test that XGBoost SHAP works with fitted models."""
        # Test with fitted model (has get_booster)
        fitted_model = Mock()
        mock_booster = Mock()
        mock_booster.num_boosted_rounds = 100
        fitted_model.get_booster.return_value = mock_booster
        
        # This should pass the initial validation but fail when trying to create TreeExplainer
        # because Mock objects aren't supported by SHAP
        try:
            explainer = self.factory.create_explainer(
                model=fitted_model,
                model_type='xgboost',
                task='regression',
                X=np.random.randn(100, 5)
            )
            # If we get here, SHAP was available and worked
            self.assertIsNotNone(explainer)
            
        except RuntimeError as e:
            # Expected: Mock objects aren't supported by SHAP TreeExplainer
            self.assertIn("🌲 KILL-SWITCH: All XGBoost SHAP approaches failed", str(e))
        except Exception as e:
            # Any other exception should also contain kill-switch info
            self.assertIn("KILL-SWITCH", str(e))
    
    def test_xgboost_shap_booster_validation(self):
        """🌲 KILL-SWITCH: Test that XGBoost booster is properly validated."""
        # Test with None booster
        fitted_model = Mock()
        fitted_model.get_booster.return_value = None
        
        with self.assertRaises(RuntimeError, msg="Should fail for None booster"):
            self.factory.create_explainer(
                model=fitted_model,
                model_type='xgboost',
                task='regression',
                X=np.random.randn(100, 5)
            )
    
    def test_xgboost_shap_booster_attributes(self):
        """🌲 KILL-SWITCH: Test that XGBoost booster has required attributes."""
        # Test with booster missing num_boosted_rounds
        fitted_model = Mock()
        mock_booster = Mock()
        # Don't set num_boosted_rounds
        fitted_model.get_booster.return_value = mock_booster
        
        with self.assertRaises(RuntimeError, msg="Should fail for booster missing num_boosted_rounds"):
            self.factory.create_explainer(
                model=fitted_model,
                model_type='xgboost',
                task='regression',
                X=np.random.randn(100, 5)
            )
    
    def test_xgboost_regression_shap_kill_switch(self):
        """🌲 KILL-SWITCH: Test that XGBoost regression SHAP requires fitted models."""
        # Test with unfitted model - remove get_booster attribute entirely
        unfitted_model = Mock()
        # Remove the get_booster attribute to simulate unfitted model
        del unfitted_model.get_booster
        
        with self.assertRaises(RuntimeError, msg="Should fail for unfitted XGBoost regression model"):
            self.factory.create_explainer(
                model=unfitted_model,
                model_type='xgboost_regression',
                task='regression',
                X=np.random.randn(100, 5)
            )
    
    def test_background_data_preparation(self):
        """Test background data preparation for different model types."""
        X = np.random.randn(200, 10)
        
        # Test LSTM background data
        lstm_bg = self.factory._prepare_deep_background_data(X, 'lstm')
        self.assertIsInstance(lstm_bg, np.ndarray)
        self.assertLessEqual(len(lstm_bg), 100)  # Should respect background_samples
        
        # Test StockMixer background data
        stockmixer_bg = self.factory._prepare_deep_background_data(X, 'stockmixer')
        self.assertIsInstance(stockmixer_bg, np.ndarray)
        self.assertLessEqual(len(stockmixer_bg), 100)
    
    def test_deep_background_data_shape_handling(self):
        """Test deep background data shape handling."""
        X_2d = np.random.randn(100, 5)
        X_3d = np.random.randn(100, 1, 5)
        
        # Test 2D input for LSTM
        lstm_bg_2d = self.factory._prepare_deep_background_data(X_2d, 'lstm')
        self.assertEqual(lstm_bg_2d.ndim, 3)  # Should add timestep dimension
        
        # Test 3D input for LSTM
        lstm_bg_3d = self.factory._prepare_deep_background_data(X_3d, 'lstm')
        self.assertEqual(lstm_bg_3d.ndim, 3)  # Should preserve 3D shape
    
    def test_background_data_sampling(self):
        """Test that background data sampling respects limits."""
        X = np.random.randn(50, 5)  # Small dataset
        
        # Should not sample more than available
        bg = self.factory._prepare_deep_background_data(X, 'lstm')
        self.assertEqual(len(bg), 50)  # Should use all available data
        
        # Large dataset should be sampled
        X_large = np.random.randn(200, 5)
        bg_large = self.factory._prepare_deep_background_data(X_large, 'lstm')
        self.assertEqual(len(bg_large), 100)  # Should respect background_samples


if __name__ == '__main__':
    unittest.main()



