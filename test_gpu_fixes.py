#!/usr/bin/env python3
"""
Test script to verify GPU fallback and SHAP fixes.
"""

import sys
import os
import logging
import numpy as np
import pandas as pd

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_tensorflow_utils():
    """Test the enhanced TensorFlow utilities."""
    print("Testing enhanced TensorFlow utilities...")
    
    try:
        from risk_pipeline.utils.tensorflow_utils import (
            check_tensorflow_compatibility,
            configure_tensorflow_memory,
            get_optimal_device,
            force_cpu_mode
        )
        
        # Test compatibility check
        compat_info = check_tensorflow_compatibility()
        print(f"✅ TensorFlow compatibility: {compat_info['compatible']}")
        print(f"   Version: {compat_info['version']}")
        print(f"   GPU devices: {compat_info['gpu_devices']}")
        print(f"   GPU working: {compat_info['gpu_working']}")
        
        # Test memory configuration
        device = configure_tensorflow_memory(gpu_memory_growth=True, force_cpu=False)
        print(f"✅ Configured device: {device}")
        
        # Test optimal device selection
        optimal_device = get_optimal_device(prefer_gpu=True)
        print(f"✅ Optimal device: {optimal_device}")
        
        # Test CPU forcing
        force_cpu_mode()
        print("✅ CPU mode forced successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ TensorFlow utilities test failed: {e}")
        return False

def test_lstm_gpu_fallback():
    """Test LSTM model GPU fallback."""
    print("Testing LSTM GPU fallback...")
    
    try:
        from risk_pipeline.models.lstm_model import LSTMModel
        
        # Create LSTM model
        lstm_model = LSTMModel(task='regression')
        
        # Test model building with GPU fallback
        try:
            built_model = lstm_model.build_model((100, 10))
            if built_model.model is not None:
                print("✅ LSTM model built successfully with GPU fallback")
                return True
            else:
                print("❌ LSTM model not built")
                return False
        except Exception as build_error:
            print(f"❌ LSTM model building failed: {build_error}")
            return False
            
    except Exception as e:
        print(f"❌ LSTM GPU fallback test failed: {e}")
        return False

def test_stockmixer_gpu_fallback():
    """Test StockMixer model GPU fallback."""
    print("Testing StockMixer GPU fallback...")
    
    try:
        from risk_pipeline.models.stockmixer_model import StockMixerModel
        
        # Create StockMixer model
        stockmixer_model = StockMixerModel(task='regression')
        
        # Test model building with GPU fallback
        try:
            built_model = stockmixer_model.build_model((100, 10))
            if built_model.model is not None:
                print("✅ StockMixer model built successfully with GPU fallback")
                return True
            else:
                print("❌ StockMixer model not built")
                return False
        except Exception as build_error:
            print(f"❌ StockMixer model building failed: {build_error}")
            return False
            
    except Exception as e:
        print(f"❌ StockMixer GPU fallback test failed: {e}")
        return False

def test_explainer_factory():
    """Test the fixed explainer factory."""
    print("Testing explainer factory fixes...")
    
    try:
        from risk_pipeline.interpretability.explainer_factory import ExplainerFactory
        from unittest.mock import Mock
        
        # Create mock config
        mock_config = Mock()
        mock_config.shap = Mock()
        mock_config.shap.background_samples = 50
        
        # Test explainer factory
        factory = ExplainerFactory(mock_config)
        
        # Test model type detection
        mock_lstm = Mock()
        mock_lstm.__class__.__name__ = 'LSTMModel'
        
        detected_type = factory._detect_model_type(mock_lstm)
        if detected_type == 'lstm':
            print("✅ Model type detection working")
        else:
            print(f"❌ Model type detection failed: expected 'lstm', got '{detected_type}'")
            return False
        
        # Test background data preparation
        test_data = pd.DataFrame(np.random.randn(100, 33))
        background_data = factory._prepare_deep_background_data(test_data, 'lstm')
        
        if background_data.shape == (100, 1, 33):
            print("✅ Background data preparation working")
        else:
            print(f"❌ Background data preparation failed: expected (100, 1, 33), got {background_data.shape}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Explainer factory test failed: {e}")
        return False

def test_arima_explainer_fix():
    """Test the ARIMA explainer index bounds fix."""
    print("Testing ARIMA explainer index fix...")
    
    try:
        from risk_pipeline.interpretability.explainer_factory import ARIMAExplainer
        from unittest.mock import Mock
        
        # Create mock ARIMA model
        mock_model = Mock()
        mock_model.fit.return_value = mock_model
        mock_model.params = pd.Series([0.1, 0.2, 0.3], index=['param1', 'param2', 'param3'])
        mock_model.aic = 100.0
        mock_model.bic = 110.0
        mock_model.resid = pd.Series(np.random.randn(100))
        
        # Create mock config
        mock_config = Mock()
        
        # Create ARIMA explainer
        explainer = ARIMAExplainer(mock_model, np.random.randn(100, 5), 'regression', mock_config)
        
        # Test SHAP values generation
        test_X = pd.DataFrame(np.random.randn(10, 33))
        shap_values = explainer.shap_values(test_X)
        
        if shap_values.shape == (10, 33):
            print("✅ ARIMA explainer index fix working")
            return True
        else:
            print(f"❌ ARIMA explainer shape wrong: expected (10, 33), got {shap_values.shape}")
            return False
            
    except Exception as e:
        print(f"❌ ARIMA explainer test failed: {e}")
        return False

def main():
    """Run all GPU and SHAP fix tests."""
    print("🧪 Testing GPU Fallback and SHAP Fixes\n")
    
    tests = [
        ("TensorFlow Utilities", test_tensorflow_utils),
        ("LSTM GPU Fallback", test_lstm_gpu_fallback),
        ("StockMixer GPU Fallback", test_stockmixer_gpu_fallback),
        ("Explainer Factory", test_explainer_factory),
        ("ARIMA Explainer Fix", test_arima_explainer_fix)
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
        print("🎉 All tests passed! GPU fallback and SHAP fixes are working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
