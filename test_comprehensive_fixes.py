#!/usr/bin/env python3
"""
Comprehensive test script to verify all RiskPipeline fixes.
"""

import sys
import os
import logging
import numpy as np
import pandas as pd

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_datetime_conversion_fixes():
    """Test the datetime conversion fixes in data loader."""
    print("Testing datetime conversion fixes...")
    
    try:
        from risk_pipeline.core.data_loader import DataLoader
        
        # Test with timezone-aware datetime
        dates = pd.date_range('2020-01-01', periods=100, freq='D', tz='UTC')
        test_data = pd.DataFrame({
            'Open': range(100),
            'High': range(100, 200),
            'Low': range(50, 150),
            'Close': range(75, 175),
            'Volume': range(1000, 1100)
        }, index=dates)
        
        # Test data cleaning
        data_loader = DataLoader()
        cleaned_data = data_loader._clean_data(test_data)
        
        if not cleaned_data.empty and isinstance(cleaned_data.index, pd.DatetimeIndex):
            print(f"✅ Timezone-aware datetime conversion working: {type(cleaned_data.index)}")
            return True
        else:
            print(f"❌ Timezone-aware datetime conversion failed: {type(cleaned_data.index)}")
            return False
            
    except Exception as e:
        print(f"❌ Datetime conversion test failed: {e}")
        return False

def test_feature_cleaning():
    """Test the feature cleaning functionality."""
    print("Testing feature cleaning...")
    
    try:
        from risk_pipeline.core.feature_engineer import FeatureEngineer
        
        # Create test data with NaN values
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        test_data = pd.DataFrame({
            'feature1': range(100),
            'feature2': [i if i % 10 != 0 else np.nan for i in range(100)],
            'feature3': [i if i % 15 != 0 else np.nan for i in range(100)]
        }, index=dates)
        
        # Test feature cleaning
        fe = FeatureEngineer()
        cleaned_features = fe._clean_features(test_data)
        
        if not cleaned_features.empty and cleaned_features.isna().sum().sum() == 0:
            print(f"✅ Feature cleaning working: {len(cleaned_features)} rows, no NaN values")
            return True
        else:
            print(f"❌ Feature cleaning failed: {cleaned_features.isna().sum().sum()} NaN values remaining")
            return False
            
    except Exception as e:
        print(f"❌ Feature cleaning test failed: {e}")
        return False

def test_arima_model_fix():
    """Test the ARIMA model fix for missing Close column."""
    print("Testing ARIMA model fix...")
    
    try:
        from risk_pipeline.models.arima_model import ARIMAModel
        
        # Create test data without Close column
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        X = pd.DataFrame({
            'feature1': range(100),
            'feature2': range(100, 200)
        }, index=dates)
        
        # Create target series (volatility)
        y = pd.Series(np.random.randn(100).cumsum(), index=dates)
        
        # Test ARIMA model
        model = ARIMAModel()
        model.build_model((100, 2))
        
        # This should not raise an error about missing Close column
        result = model.train(X, y)
        
        if 'error' not in result:
            print("✅ ARIMA model fix working: no Close column error")
            return True
        else:
            print(f"❌ ARIMA model still has issues: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ ARIMA model test failed: {e}")
        return False

def test_model_building():
    """Test that models are properly built before training."""
    print("Testing model building...")
    
    try:
        from risk_pipeline.models.lstm_model import LSTMModel
        from risk_pipeline.models.stockmixer_model import StockMixerModel
        
        # Test LSTM model building
        lstm_model = LSTMModel(task='regression')
        built_lstm = lstm_model.build_model((100, 10))
        
        if built_lstm.model is not None:
            print("✅ LSTM model building working")
        else:
            print("❌ LSTM model building failed")
            return False
        
        # Test StockMixer model building
        stockmixer_model = StockMixerModel(task='regression')
        built_stockmixer = stockmixer_model.build_model((100, 10))
        
        if built_stockmixer.model is not None:
            print("✅ StockMixer model building working")
        else:
            print("❌ StockMixer model building failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Model building test failed: {e}")
        return False

def test_validator_fixes():
    """Test the validator fixes."""
    print("Testing validator fixes...")
    
    try:
        from risk_pipeline.core.validator import WalkForwardValidator
        
        # Create test data
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        test_data = pd.DataFrame({
            'feature1': range(500),
            'feature2': range(500, 1000)
        }, index=dates)
        
        # Test validator
        validator = WalkForwardValidator(n_splits=3, test_size=50)
        
        # Test data quality validation
        quality_report = validator.validate_data_quality(test_data)
        
        if quality_report['is_valid']:
            print("✅ Validator data quality validation working")
        else:
            print(f"⚠️ Validator data quality issues: {quality_report['issues']}")
        
        # Test split generation
        splits = validator.split(test_data)
        
        if len(splits) > 0:
            print(f"✅ Validator split generation working: {len(splits)} splits created")
            return True
        else:
            print("❌ Validator split generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Validator test failed: {e}")
        return False

def test_explainer_fixes():
    """Test the explainer factory fixes."""
    print("Testing explainer fixes...")
    
    try:
        from risk_pipeline.interpretability.explainer_factory import ExplainerFactory
        from unittest.mock import Mock
        
        # Create mock config
        mock_config = Mock()
        mock_config.shap = Mock()
        mock_config.shap.background_samples = 50
        
        # Test explainer factory
        factory = ExplainerFactory(mock_config)
        
        # Test XGBoost explainer creation
        mock_xgb_model = Mock()
        mock_xgb_model.get_booster.return_value = Mock()
        
        try:
            xgb_explainer = factory._create_xgboost_explainer(mock_xgb_model, np.random.randn(100, 5), 'regression')
            print("✅ XGBoost explainer creation working")
        except Exception as e:
            print(f"❌ XGBoost explainer creation failed: {e}")
            return False
        
        # Test LSTM explainer creation
        try:
            lstm_explainer = factory._create_lstm_explainer(mock_xgb_model, np.random.randn(100, 5), 'regression')
            print("✅ LSTM explainer creation working")
        except Exception as e:
            print(f"❌ LSTM explainer creation failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Explainer test failed: {e}")
        return False

def test_pipeline_integration():
    """Test the complete pipeline integration."""
    print("Testing pipeline integration...")
    
    try:
        from risk_pipeline import RiskPipeline
        
        # Test pipeline initialization
        pipeline = RiskPipeline()
        print("✅ Pipeline initialization working")
        
        # Test all components are initialized
        components = [
            'feature_engineer',
            'validator', 
            'model_factory',
            'shap_analyzer',
            'results_manager'
        ]
        
        for component in components:
            if hasattr(pipeline, component):
                print(f"✅ {component} initialized")
            else:
                print(f"❌ {component} not initialized")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline integration test failed: {e}")
        return False

def main():
    """Run all comprehensive tests."""
    print("🧪 Running Comprehensive RiskPipeline Fix Tests\n")
    
    tests = [
        ("Datetime Conversion Fixes", test_datetime_conversion_fixes),
        ("Feature Cleaning", test_feature_cleaning),
        ("ARIMA Model Fix", test_arima_model_fix),
        ("Model Building", test_model_building),
        ("Validator Fixes", test_validator_fixes),
        ("Explainer Fixes", test_explainer_fixes),
        ("Pipeline Integration", test_pipeline_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}")
        print("-" * 50)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"💥 {test_name} ERROR: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! All fixes are working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
