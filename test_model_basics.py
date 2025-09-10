#!/usr/bin/env python3
"""
Test basic model functionality to isolate the issue.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_arima_basic():
    """Test basic ARIMA functionality."""
    try:
        from risk_pipeline.models.arima_model import ARIMAModel
        
        print("🧪 Testing ARIMA model...")
        
        # Create simple test data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 0.02)
        
        df = pd.DataFrame({
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # Create target (volatility)
        df['volatility'] = df['Close'].pct_change().rolling(5).std()
        
        # Remove NaNs
        df = df.dropna()
        
        print(f"Test data shape: {df.shape}")
        print(f"Features: {df.columns.tolist()}")
        
        # Create model
        model = ARIMAModel(
            input_shape=(len(df), len(df.columns) - 1),  # Exclude target
            task='regression'
        )
        
        print(f"Model created: {type(model)}")
        print(f"Model has fit method: {hasattr(model, 'fit')}")
        print(f"Model has predict method: {hasattr(model, 'predict')}")
        
        # Prepare data
        X = df.drop('volatility', axis=1)
        y = df['volatility']
        
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        
        # Test fitting
        print("Testing model fitting...")
        model.fit(X, y)
        print("✅ Model fitting successful")
        
        # Test prediction
        print("Testing model prediction...")
        y_pred = model.predict(X)
        print(f"✅ Prediction successful, shape: {y_pred.shape}")
        print(f"Prediction range: {y_pred.min():.6f} to {y_pred.max():.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ ARIMA test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_xgboost_basic():
    """Test basic XGBoost functionality."""
    try:
        from risk_pipeline.models.xgboost_model import XGBoostModel
        
        print("\n🧪 Testing XGBoost model...")
        
        # Create simple test data
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        print(f"Test data shape: X={X.shape}, y={y.shape}")
        
        # Create model
        model = XGBoostModel(
            input_shape=(100, 5),
            task='regression'
        )
        
        print(f"Model created: {type(model)}")
        print(f"Model has fit method: {hasattr(model, 'fit')}")
        print(f"Model has predict method: {hasattr(model, 'predict')}")
        
        # Test fitting
        print("Testing model fitting...")
        model.fit(X, y)
        print("✅ Model fitting successful")
        
        # Test prediction
        print("Testing model prediction...")
        y_pred = model.predict(X)
        print(f"✅ Prediction successful, shape: {y_pred.shape}")
        print(f"Prediction range: {y_pred.min():.6f} to {y_pred.max():.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ XGBoost test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_lstm_basic():
    """Test basic LSTM functionality."""
    try:
        from risk_pipeline.models.lstm_model import LSTMModel
        
        print("\n🧪 Testing LSTM model...")
        
        # Create simple test data
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        print(f"Test data shape: X={X.shape}, y={y.shape}")
        
        # Create model
        model = LSTMModel(
            input_shape=(100, 5),
            task='regression'
        )
        
        print(f"Model created: {type(model)}")
        print(f"Model has fit method: {hasattr(model, 'fit')}")
        print(f"Model has predict method: {hasattr(model, 'predict')}")
        
        # Test fitting
        print("Testing model fitting...")
        model.fit(X, y)
        print("✅ Model fitting successful")
        
        # Test prediction
        print("Testing model prediction...")
        y_pred = model.predict(X)
        print(f"✅ Prediction successful, shape: {y_pred.shape}")
        print(f"Prediction range: {y_pred.min():.6f} to {y_pred.max():.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ LSTM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all basic model tests."""
    print("🔍 Testing basic model functionality...")
    
    results = {}
    
    # Test ARIMA
    results['ARIMA'] = test_arima_basic()
    
    # Test XGBoost
    results['XGBoost'] = test_xgboost_basic()
    
    # Test LSTM
    results['LSTM'] = test_lstm_basic()
    
    # Summary
    print("\n📊 Test Results Summary:")
    for model, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {model}: {status}")
    
    passed = sum(results.values())
    total = len(results)
    print(f"\nOverall: {passed}/{total} models passed basic tests")
    
    if passed == total:
        print("🎉 All models passed basic functionality tests!")
    else:
        print("⚠️  Some models failed basic tests - this may explain the validation issues")

if __name__ == "__main__":
    main()
