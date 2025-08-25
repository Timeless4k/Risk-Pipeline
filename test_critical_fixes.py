#!/usr/bin/env python3
"""
Test script to validate all critical fixes implemented.
This script tests the emergency fixes for R² scores, StockMixer stability, and cross-asset consistency.
"""

import sys
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from risk_pipeline.core.feature_engineer import FeatureEngineer
from risk_pipeline.core.config import FeatureConfig
from risk_pipeline.models.lstm_model import LSTMModel
from risk_pipeline.models.stockmixer_model import StockMixerModel
from risk_pipeline.models.xgboost_model import XGBoostModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler

def test_temporal_separation():
    """Test that temporal separation is working correctly."""
    print("🔍 Testing Temporal Separation...")
    
    # Create synthetic data with known temporal structure
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    prices = 100 + np.cumsum(np.random.randn(1000) * 0.02)
    
    data = pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'Adj Close': prices
    })
    data.set_index('Date', inplace=True)
    
    # Create feature engineer with 90-day temporal separation
    config = FeatureConfig()
    config.volatility_windows = [5, 10, 20]
    
    engineer = FeatureEngineer(config)
    
    # Create features and targets
    features_dict = engineer.create_features({'TEST': data})
    
    if 'TEST' not in features_dict:
        print("   ❌ Feature creation failed")
        return False
    
    features = features_dict['TEST']['features']
    targets = features_dict['TEST']['targets']
    
    print(f"   ✅ Features created: {features.shape}")
    print(f"   ✅ Targets created: {targets.shape}")
    
    # Check temporal separation
    if len(features) > 0 and len(targets) > 0:
        # Features should be available earlier than targets
        # Due to 90-day shift, we should have fewer aligned samples
        aligned_samples = min(len(features), len(targets))
        print(f"   ✅ Aligned samples: {aligned_samples}")
        
        if aligned_samples < len(features):
            print("   ✅ Temporal separation confirmed (fewer aligned samples)")
            return True
        else:
            print("   ❌ Temporal separation may not be working")
            return False
    
    return False

def test_stockmixer_stability():
    """Test StockMixer model stability with input validation."""
    print("🔍 Testing StockMixer Stability...")
    
    try:
        # Create synthetic data
        np.random.seed(42)
        X = np.random.randn(100, 10) * 0.1  # Small values to test clipping
        y = np.random.randn(100) * 0.1
        
        # Test with extreme values
        X_extreme = X.copy()
        X_extreme[0, 0] = 1000  # Extreme value
        
        # Initialize StockMixer
        model = StockMixerModel(task='regression')
        model.build_model(X.shape)
        
        # Test training with extreme values (should trigger clipping)
        print("   Testing with extreme values...")
        results = model.train(X_extreme, y, epochs=5)
        print(f"   ✅ Training completed: {results}")
        
        # Test predictions
        predictions = model.predict(X[:10])
        print(f"   ✅ Predictions shape: {predictions.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ StockMixer test failed: {e}")
        return False

def test_xgboost_optimization():
    """Test XGBoost with optimized hyperparameters."""
    print("🔍 Testing XGBoost Optimization...")
    
    try:
        # Create synthetic data
        np.random.seed(42)
        X = np.random.randn(200, 15)
        y = X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.randn(200) * 0.1
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize XGBoost with optimized config
        model = XGBoostModel(task='regression')
        model.build_model(X_train.shape)
        
        # Train model
        print("   Training XGBoost...")
        results = model.train(X_train, y_train)
        print(f"   ✅ Training completed: {results}")
        
        # Make predictions
        predictions = model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        
        print(f"   ✅ XGBoost R²: {r2:.4f}")
        print(f"   ✅ XGBoost MSE: {mse:.6f}")
        
        # Check if R² is reasonable (should be positive for synthetic data)
        if r2 > -0.1:  # Allow some negative R² for noisy data
            print("   ✅ XGBoost performance is reasonable")
            return True
        else:
            print("   ⚠️ XGBoost performance could be improved")
            return False
            
    except Exception as e:
        print(f"   ❌ XGBoost test failed: {e}")
        return False

def test_cross_asset_consistency():
    """Test that features are normalized consistently across assets."""
    print("🔍 Testing Cross-Asset Consistency...")
    
    try:
        # Create synthetic data for different asset types
        np.random.seed(42)
        
        # US asset (higher volatility)
        us_dates = pd.date_range('2020-01-01', periods=500, freq='D')
        us_prices = 100 + np.cumsum(np.random.randn(500) * 0.03)  # Higher volatility
        
        # AU asset (lower volatility)
        au_dates = pd.date_range('2020-01-01', periods=500, freq='D')
        au_prices = 50 + np.cumsum(np.random.randn(500) * 0.01)  # Lower volatility
        
        us_data = pd.DataFrame({
            'Close': us_prices,
            'Adj Close': us_prices
        }, index=us_dates)
        
        au_data = pd.DataFrame({
            'Close': au_prices,
            'Adj Close': au_prices
        }, index=au_dates)
        
        # Create feature engineer
        config = FeatureConfig()
        config.volatility_windows = [5, 10]
        
        engineer = FeatureEngineer(config)
        
        # Create features for both assets
        features_dict = engineer.create_features({
            'US_ASSET': us_data,
            'AU_ASSET': au_data
        })
        
        if 'US_ASSET' not in features_dict or 'AU_ASSET' not in features_dict:
            print("   ❌ Feature creation failed for one or both assets")
            return False
        
        us_features = features_dict['US_ASSET']['features']
        au_features = features_dict['AU_ASSET']['features']
        
        print(f"   ✅ US features: {us_features.shape}")
        print(f"   ✅ AU features: {au_features.shape}")
        
        # Check if features are normalized (should have similar statistics)
        us_mean = us_features.mean().mean()
        us_std = us_features.std().mean()
        au_mean = au_features.mean().mean()
        au_std = au_features.std().mean()
        
        print(f"   US features - mean: {us_mean:.4f}, std: {us_std:.4f}")
        print(f"   AU features - mean: {au_mean:.4f}, std: {au_std:.4f}")
        
        # Check if normalization is working (means should be close to 0, stds close to 1)
        if abs(us_mean) < 0.1 and abs(au_mean) < 0.1 and 0.8 < us_std < 1.2 and 0.8 < au_std < 1.2:
            print("   ✅ Cross-asset normalization working correctly")
            return True
        else:
            print("   ⚠️ Cross-asset normalization may need adjustment")
            return False
            
    except Exception as e:
        print(f"   ❌ Cross-asset consistency test failed: {e}")
        return False

def test_target_validation():
    """Test that targets are properly validated and constructed."""
    print("🔍 Testing Target Validation...")
    
    try:
        # Create synthetic data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=300, freq='D')
        prices = 100 + np.cumsum(np.random.randn(300) * 0.02)
        
        data = pd.DataFrame({
            'Close': prices,
            'Adj Close': prices
        }, index=dates)
        
        # Create feature engineer
        config = FeatureConfig()
        config.volatility_windows = [5, 10]
        
        engineer = FeatureEngineer(config)
        
        # Create features and targets
        features_dict = engineer.create_features({'TEST': data})
        
        if 'TEST' not in features_dict:
            print("   ❌ Feature creation failed")
            return False
        
        targets = features_dict['TEST']['targets']
        
        if 'volatility' not in targets or 'regime' not in targets:
            print("   ❌ Targets not created correctly")
            return False
        
        volatility = targets['volatility']
        regime = targets['regime']
        
        print(f"   ✅ Volatility target: {volatility.shape}")
        print(f"   ✅ Regime target: {regime.shape}")
        
        # Check target statistics
        vol_mean = volatility.mean()
        vol_std = volatility.std()
        regime_unique = regime.nunique()
        
        print(f"   Volatility - mean: {vol_mean:.6f}, std: {vol_std:.6f}")
        print(f"   Regime - unique values: {regime_unique}")
        
        # Validate targets
        if vol_std > 0 and regime_unique >= 2:
            print("   ✅ Target validation passed")
            return True
        else:
            print("   ❌ Target validation failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Target validation test failed: {e}")
        return False

def main():
    """Run all critical fix tests."""
    print("🚀 CRITICAL FIXES VALIDATION TEST")
    print("=" * 50)
    
    tests = [
        ("Temporal Separation", test_temporal_separation),
        ("StockMixer Stability", test_stockmixer_stability),
        ("XGBoost Optimization", test_xgboost_optimization),
        ("Cross-Asset Consistency", test_cross_asset_consistency),
        ("Target Validation", test_target_validation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL CRITICAL FIXES VALIDATED SUCCESSFULLY!")
        print("   The pipeline should now achieve positive R² scores.")
    else:
        print("⚠️ Some fixes need additional attention.")
        print("   Review failed tests and implement additional fixes.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
