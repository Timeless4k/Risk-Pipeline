#!/usr/bin/env python3
"""
Test script to verify fixes for the RiskPipeline issues.
"""

import sys
import os
import logging

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_logging_config():
    """Test logging configuration."""
    print("Testing logging configuration...")
    
    try:
        from risk_pipeline.utils.logging_utils import setup_logging
        
        # Test logging setup
        setup_logging(level=logging.INFO)
        logger = logging.getLogger("test")
        logger.info("Test log message")
        
        print("✅ Logging config test passed")
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

def test_validator_fix():
    """Test the validator fix for is_all_dates."""
    print("Testing validator fixes...")
    
    try:
        import pandas as pd
        from risk_pipeline.core.validator import WalkForwardValidator
        
        # Create test data with datetime index
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

def test_data_loader_fix():
    """Test the data loader datetime index fix."""
    print("Testing data loader fixes...")
    
    try:
        import pandas as pd
        from risk_pipeline.core.data_loader import DataLoader
        
        # Create test data with string index (simulating loaded data)
        test_data = pd.DataFrame({
            'Open': range(100),
            'High': range(100, 200),
            'Low': range(50, 150),
            'Close': range(75, 175),
            'Volume': range(1000, 1100)
        }, index=[f'2020-01-{i+1:02d}' for i in range(100)])
        
        # Test data cleaning
        data_loader = DataLoader()
        cleaned_data = data_loader._clean_data(test_data)
        
        if not cleaned_data.empty and isinstance(cleaned_data.index, pd.DatetimeIndex):
            print(f"✅ Data loader datetime conversion working: {type(cleaned_data.index)}")
            return True
        else:
            print(f"❌ Data loader datetime conversion failed: {type(cleaned_data.index)}")
            return False
            
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        return False

def test_pipeline_integration():
    """Test the pipeline integration fixes."""
    print("Testing pipeline integration fixes...")
    
    try:
        from risk_pipeline import RiskPipeline
        
        # Test pipeline initialization
        pipeline = RiskPipeline()
        print("✅ Pipeline initialization working")
        
        # Test feature engineer initialization
        if hasattr(pipeline, 'feature_engineer'):
            print("✅ Feature engineer initialized")
        else:
            print("❌ Feature engineer not initialized")
            return False
        
        # Test validator initialization
        if hasattr(pipeline, 'validator'):
            print("✅ Validator initialized")
        else:
            print("❌ Validator not initialized")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Running RiskPipeline Fix Tests\n")
    
    tests = [
        ("Logging Configuration", test_logging_config),
        ("Feature Engineer Fixes", test_feature_engineer_fix),
        ("Validator Fixes", test_validator_fix),
        ("Data Loader Fixes", test_data_loader_fix),
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
        print("🎉 All tests passed! The fixes are working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
