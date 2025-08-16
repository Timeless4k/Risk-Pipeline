#!/usr/bin/env python3
"""
Simple Test Pipeline for RiskPipeline
Tests the basic functionality with simplified features and single asset.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from risk_pipeline.core.simple_feature_engineer import SimpleFeatureEngineer
    from risk_pipeline.models.lstm_model import LSTMModel
    from risk_pipeline.models.stockmixer_model import StockMixerModel
    from risk_pipeline.models.xgboost_model import XGBoostModel
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.model_selection import train_test_split
    print("✅ All modules imported successfully!")
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("Please ensure you're running this from the project root directory")
    sys.exit(1)

def create_dummy_data(n_samples: int = 1000) -> Tuple[pd.DataFrame, pd.Series]:
    """Create dummy price data for testing."""
    print("📊 Creating dummy price data...")
    
    # Generate synthetic price series
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # Random walk price series
    returns = np.random.normal(0.001, 0.02, n_samples)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Create DataFrame
    data = pd.DataFrame({
        'Close': prices,
        'High': prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples))),
        'Volume': np.random.randint(1000000, 10000000, n_samples)
    }, index=dates)
    
    print(f"✅ Created dummy data with {len(data)} samples")
    return data

def test_simple_features(data: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, pd.Series]:
    """Test simplified feature engineering."""
    print("🔧 Testing simplified feature engineering...")
    
    try:
        # Initialize feature engineer
        feature_engineer = SimpleFeatureEngineer(config)
        
        # Create features
        features = feature_engineer.create_features(data)
        print(f"✅ Features created: {features.shape}")
        print(f"   Feature names: {feature_engineer.get_feature_names()}")
        
        # Create target (future volatility)
        target = feature_engineer.create_target(data, target_type='volatility', target_horizon=5)
        print(f"✅ Target created: {target.shape}")
        
        # Align features and target
        features_aligned, target_aligned = feature_engineer.align_features_and_target(features, target)
        print(f"✅ Data aligned: {features_aligned.shape}, {target_aligned.shape}")
        
        # Fit scaler and transform features
        feature_engineer.fit_scaler(features_aligned)
        features_scaled = feature_engineer.transform_features(features_aligned)
        print(f"✅ Features scaled: {features_scaled.shape}")
        
        return features_scaled, target_aligned
        
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        raise

def test_baseline_model(X: pd.DataFrame, y: pd.Series) -> float:
    """Test baseline linear regression model."""
    print("📈 Testing baseline linear regression...")
    
    try:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train baseline model
        baseline = LinearRegression()
        baseline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = baseline.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        print(f"✅ Baseline model trained successfully")
        print(f"   R² Score: {r2:.4f}")
        print(f"   MSE: {mse:.6f}")
        
        return r2
        
    except Exception as e:
        print(f"❌ Baseline model failed: {e}")
        raise

def test_lstm_model(X: pd.DataFrame, y: pd.Series) -> bool:
    """Test LSTM model with simplified features."""
    print("🧠 Testing LSTM model...")
    
    try:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize LSTM model
        lstm = LSTMModel(task='regression')
        
        # Build model
        print(f"   Building LSTM model with input shape: {X_train.shape}")
        lstm.build_model(X_train.shape)
        print(f"   ✅ LSTM model built successfully")
        
        # Train model
        print(f"   Training LSTM model...")
        training_results = lstm.train(X_train, y_train)
        print(f"   ✅ LSTM training completed: {training_results}")
        
        # Make predictions
        predictions = lstm.predict(X_test)
        print(f"   ✅ LSTM predictions made: {predictions.shape}")
        
        # Calculate metrics
        r2 = r2_score(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        print(f"   LSTM R² Score: {r2:.4f}")
        print(f"   LSTM MSE: {mse:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ LSTM model failed: {e}")
        return False

def test_stockmixer_model(X: pd.DataFrame, y: pd.Series) -> bool:
    """Test StockMixer model with simplified features."""
    print("🎯 Testing StockMixer model...")
    
    try:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize StockMixer model
        stockmixer = StockMixerModel(task='regression')
        
        # Build model
        print(f"   Building StockMixer model with input shape: {X_train.shape}")
        stockmixer.build_model(X_train.shape)
        print(f"   ✅ StockMixer model built successfully")
        
        # Train model
        print(f"   Training StockMixer model...")
        training_results = stockmixer.train(X_train, y_train)
        print(f"   ✅ StockMixer training completed: {training_results}")
        
        # Make predictions
        predictions = stockmixer.predict(X_test)
        print(f"   ✅ StockMixer predictions made: {predictions.shape}")
        
        # Calculate metrics
        r2 = r2_score(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        print(f"   StockMixer R² Score: {r2:.4f}")
        print(f"   StockMixer MSE: {mse:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ StockMixer model failed: {e}")
        return False

def test_xgboost_model(X: pd.DataFrame, y: pd.Series) -> bool:
    """Test XGBoost model with simplified features."""
    print("🌳 Testing XGBoost model...")
    
    try:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize XGBoost model
        xgb_model = XGBoostModel(task='regression')
        
        # Build model
        print(f"   Building XGBoost model with input shape: {X_train.shape}")
        xgb_model.build_model(X_train.shape)
        print(f"   ✅ XGBoost model built successfully")
        
        # Train model
        print(f"   Training XGBoost model...")
        training_results = xgb_model.train(X_train, y_train)
        print(f"   ✅ XGBoost training completed: {training_results}")
        
        # Make predictions
        predictions = xgb_model.predict(X_test)
        print(f"   ✅ XGBoost predictions made: {predictions.shape}")
        
        # Calculate metrics
        r2 = r2_score(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        print(f"   XGBoost R² Score: {r2:.4f}")
        print(f"   XGBoost MSE: {mse:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ XGBoost model failed: {e}")
        return False

def main():
    """Run the simple test pipeline."""
    print("🚀 Starting Simple Test Pipeline!")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Load configuration
        config_path = Path("configs/simple_test_config.json")
        if not config_path.exists():
            print(f"❌ Configuration file not found: {config_path}")
            sys.exit(1)
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("✅ Configuration loaded successfully")
        
        # Create dummy data
        data = create_dummy_data(n_samples=1000)
        
        # Test feature engineering
        X, y = test_simple_features(data, config)
        
        # Test baseline model
        baseline_r2 = test_baseline_model(X, y)
        
        # Test deep learning models
        print("\n🤖 Testing Deep Learning Models...")
        print("-" * 40)
        
        lstm_success = test_lstm_model(X, y)
        stockmixer_success = test_stockmixer_model(X, y)
        xgboost_success = test_xgboost_model(X, y)
        
        # Summary
        print("\n" + "=" * 60)
        print("🎉 TEST PIPELINE COMPLETED!")
        print("=" * 60)
        
        print(f"✅ Baseline R² Score: {baseline_r2:.4f}")
        print(f"✅ LSTM Model: {'SUCCESS' if lstm_success else 'FAILED'}")
        print(f"✅ StockMixer Model: {'SUCCESS' if stockmixer_success else 'FAILED'}")
        print(f"✅ XGBoost Model: {'SUCCESS' if xgboost_success else 'FAILED'}")
        
        # Calculate execution time
        execution_time = time.time() - start_time
        print(f"⏱️  Total Execution Time: {execution_time:.1f} seconds")
        
        # Success criteria
        if baseline_r2 > 0.05:
            print("🎯 SUCCESS: Baseline model achieved R² > 0.05")
        else:
            print("⚠️  WARNING: Baseline model R² below 0.05")
        
        if lstm_success and stockmixer_success:
            print("🎯 SUCCESS: All deep learning models trained successfully")
        else:
            print("⚠️  WARNING: Some deep learning models failed")
        
        print("\n🚀 Simple test pipeline completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
