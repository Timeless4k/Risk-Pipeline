#!/usr/bin/env python3
"""
Test script to verify the fixes for the RiskPipeline issues.
"""

import sys
import os
import logging
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_logging_config():
    """Test the new logging configuration system."""
    print("Testing logging configuration...")
    
    try:
        from risk_pipeline.config.logging_config import get_logging_config, apply_logging_config
        
        # Test default config
        config = get_logging_config()
        print(f"✅ Default config loaded: {len(config['components'])} components configured")
        
        # Test verbose config
        verbose_config = get_logging_config(verbose=True)
        print(f"✅ Verbose config loaded: root_level={verbose_config['root_level']}")
        
        # Test quiet config
        quiet_config = get_logging_config(quiet=True)
        print(f"✅ Quiet config loaded: root_level={quiet_config['root_level']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Logging config test failed: {e}")
        return False

def test_feature_engineer_fix():
    """Test the time feature module fix."""
    print("Testing feature engineer fixes...")
    
    try:
        import pandas as pd
        from risk_pipeline.core.feature_engineer import TimeFeatureModule, FeatureConfig
        
        # Create test data with datetime index
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        test_data = pd.DataFrame({'Close': range(100)}, index=dates)
        
        # Test time feature module
        config = FeatureConfig()
        time_module = TimeFeatureModule(config)
        features = time_module.create_features(test_data)
        
        if not features.empty and len(features.columns) == 4:
            print(f"✅ Time feature module working: {len(features.columns)} features created")
            return True
        else:
            print(f"❌ Time feature module failed: {len(features.columns)} features created")
            return False
            
    except Exception as e:
        print(f"❌ Feature engineer test failed: {e}")
        return False

def test_xgboost_fix():
    """Test the XGBoost input validation fix."""
    print("Testing XGBoost fixes...")
    
    try:
        import numpy as np
        from risk_pipeline.models.xgboost_model import XGBoostModel
        
        # Test with 1D input (should be reshaped to 2D)
        X_1d = np.random.randn(100)
        y = np.random.randn(100)
        
        model = XGBoostModel(task='regression')
        model.build_model(input_shape=(1,))
        
        # This should not raise an error now
        model.train(X_1d, y)
        print("✅ XGBoost 1D input handling working")
        
        # Test prediction with 1D input
        preds = model.predict(X_1d[:10])
        if preds.shape == (10,):
            print("✅ XGBoost 1D prediction working")
            return True
        else:
            print(f"❌ XGBoost prediction shape incorrect: {preds.shape}")
            return False
            
    except Exception as e:
        print(f"❌ XGBoost test failed: {e}")
        return False

def test_logging_reduction():
    """Test that logging verbosity is reduced."""
    print("Testing logging reduction...")
    
    try:
        from risk_pipeline.utils.logging_utils import setup_logging
        
        # Test quiet logging
        logger = setup_logging(quiet=True)
        
        # Check that root logger level is WARNING
        root_logger = logging.getLogger()
        if root_logger.level == logging.WARNING:
            print("✅ Quiet logging working: root level is WARNING")
            return True
        else:
            print(f"❌ Quiet logging failed: root level is {root_logger.level}")
            return False
            
    except Exception as e:
        print(f"❌ Logging reduction test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Running RiskPipeline Fix Tests\n")
    
    tests = [
        test_logging_config,
        test_feature_engineer_fix,
        test_xgboost_fix,
        test_logging_reduction,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}\n")
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The fixes are working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
