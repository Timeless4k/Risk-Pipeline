#!/usr/bin/env python3
"""
Test script to verify PyTorch installation and test LSTM and StockMixer models.
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, Any

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_pytorch_installation():
    """Test PyTorch installation and CUDA availability."""
    print("=" * 60)
    print("TESTING PYTORCH INSTALLATION")
    print("=" * 60)
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"✓ CUDA available: {cuda_available}")
        
        if cuda_available:
            print(f"✓ CUDA device count: {torch.cuda.device_count()}")
            print(f"✓ Current CUDA device: {torch.cuda.current_device()}")
            print(f"✓ CUDA device name: {torch.cuda.get_device_name(0)}")
            print(f"✓ CUDA capability: {torch.cuda.get_device_capability(0)}")
        else:
            print("ℹ CUDA not available, using CPU")
            
        # Test basic tensor operations
        device = torch.device('cuda' if cuda_available else 'cpu')
        x = torch.randn(2, 3, device=device)
        y = torch.randn(2, 3, device=device)
        z = torch.mm(x, y.t())
        print(f"✓ Basic tensor operations work on {device}")
        
        return True, device
        
    except ImportError as e:
        print(f"✗ PyTorch not installed: {e}")
        return False, None
    except Exception as e:
        print(f"✗ PyTorch error: {e}")
        return False, None

def test_lstm_model():
    """Test LSTM model functionality."""
    print("\n" + "=" * 60)
    print("TESTING LSTM MODEL")
    print("=" * 60)
    
    try:
        from risk_pipeline.models.lstm_model import LSTMModel
        from risk_pipeline.utils.torch_utils import get_torch_device
        
        # Test device detection
        device = get_torch_device(prefer_gpu=True)
        print(f"✓ Device detected: {device}")
        
        # Create sample data
        np.random.seed(42)
        n_samples, n_features = 100, 5
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)
        
        # Test regression model
        print("Testing LSTM regression...")
        lstm_reg = LSTMModel(task='regression', units=[32, 16], epochs=5, batch_size=16)
        lstm_reg.build_model(X.shape)
        print(f"✓ Model built with input shape: {X.shape}")
        
        # Train model
        history = lstm_reg.train(X, y)
        print(f"✓ Model trained - Final loss: {history['train_loss']:.4f}")
        
        # Test prediction
        predictions = lstm_reg.predict(X[:10])
        print(f"✓ Predictions generated: {predictions.shape}")
        
        # Test evaluation
        metrics = lstm_reg.evaluate(X[:20], y[:20])
        print(f"✓ Evaluation metrics: {metrics}")
        
        # Test classification model
        print("\nTesting LSTM classification...")
        y_class = np.random.randint(0, 3, n_samples)
        lstm_cls = LSTMModel(task='classification', units=[32, 16], epochs=5, batch_size=16, num_classes=3)
        lstm_cls.build_model(X.shape)
        lstm_cls.train(X, y_class)
        predictions_cls = lstm_cls.predict(X[:10])
        print(f"✓ Classification predictions: {predictions_cls.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ LSTM model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_stockmixer_model():
    """Test StockMixer model functionality."""
    print("\n" + "=" * 60)
    print("TESTING STOCKMIXER MODEL")
    print("=" * 60)
    
    try:
        from risk_pipeline.models.stockmixer_model import StockMixerModel
        from risk_pipeline.utils.torch_utils import get_torch_device
        
        # Test device detection
        device = get_torch_device(prefer_gpu=True)
        print(f"✓ Device detected: {device}")
        
        # Create sample data for different input shapes
        np.random.seed(42)
        
        # Test 2D input (N, F)
        print("Testing StockMixer with 2D input...")
        X_2d = np.random.randn(50, 4)
        y_2d = np.random.randn(50)
        
        mixer_2d = StockMixerModel(task='regression', epochs=3, batch_size=8)
        mixer_2d.build_model(X_2d.shape)
        print(f"✓ Model built with 2D input shape: {X_2d.shape}")
        
        mixer_2d.train(X_2d, y_2d)
        predictions_2d = mixer_2d.predict(X_2d[:10])
        print(f"✓ 2D predictions: {predictions_2d.shape}")
        
        # Test 3D input (N, T, F)
        print("\nTesting StockMixer with 3D input...")
        X_3d = np.random.randn(30, 10, 4)
        y_3d = np.random.randn(30)
        
        mixer_3d = StockMixerModel(task='regression', epochs=3, batch_size=8)
        mixer_3d.build_model(X_3d.shape)
        print(f"✓ Model built with 3D input shape: {X_3d.shape}")
        
        mixer_3d.train(X_3d, y_3d)
        predictions_3d = mixer_3d.predict(X_3d[:10])
        print(f"✓ 3D predictions: {predictions_3d.shape}")
        
        # Test 4D input (N, S, T, F)
        print("\nTesting StockMixer with 4D input...")
        X_4d = np.random.randn(20, 2, 5, 3)
        y_4d = np.random.randn(20)
        
        mixer_4d = StockMixerModel(task='regression', epochs=3, batch_size=8)
        mixer_4d.build_model(X_4d.shape)
        print(f"✓ Model built with 4D input shape: {X_4d.shape}")
        
        mixer_4d.train(X_4d, y_4d)
        predictions_4d = mixer_4d.predict(X_4d[:10])
        print(f"✓ 4D predictions: {predictions_4d.shape}")
        
        # Test classification
        print("\nTesting StockMixer classification...")
        y_class = np.random.randint(0, 2, 30)
        mixer_cls = StockMixerModel(task='classification', epochs=3, batch_size=8, num_classes=2)
        mixer_cls.build_model(X_3d.shape)
        mixer_cls.train(X_3d, y_class)
        predictions_cls = mixer_cls.predict(X_3d[:10])
        print(f"✓ Classification predictions: {predictions_cls.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ StockMixer model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_factory():
    """Test model factory integration."""
    print("\n" + "=" * 60)
    print("TESTING MODEL FACTORY INTEGRATION")
    print("=" * 60)
    
    try:
        from risk_pipeline.models.model_factory import ModelFactory
        
        # Test LSTM creation
        lstm = ModelFactory.create_model('lstm', task='regression')
        print(f"✓ LSTM created via factory: {lstm.name}")
        
        # Test StockMixer creation
        mixer = ModelFactory.create_model('stockmixer', task='regression')
        print(f"✓ StockMixer created via factory: {mixer.name}")
        
        # Test with parameters
        lstm_custom = ModelFactory.create_model('lstm', task='classification', 
                                               units=[64, 32], dropout=0.3)
        print(f"✓ Custom LSTM created: {lstm_custom.name}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model factory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("PYTORCH AND MODEL TESTING SUITE")
    print("=" * 60)
    
    # Test PyTorch installation
    pytorch_ok, device = test_pytorch_installation()
    
    if not pytorch_ok:
        print("\n❌ PyTorch installation failed. Please install PyTorch first.")
        return False
    
    # Test models
    lstm_ok = test_lstm_model()
    mixer_ok = test_stockmixer_model()
    factory_ok = test_model_factory()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"PyTorch Installation: {'✓ PASS' if pytorch_ok else '✗ FAIL'}")
    print(f"LSTM Model: {'✓ PASS' if lstm_ok else '✗ FAIL'}")
    print(f"StockMixer Model: {'✓ PASS' if mixer_ok else '✗ FAIL'}")
    print(f"Model Factory: {'✓ PASS' if factory_ok else '✗ FAIL'}")
    
    all_passed = pytorch_ok and lstm_ok and mixer_ok and factory_ok
    print(f"\nOverall Result: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🎉 PyTorch, LSTM, and StockMixer models are working correctly!")
        print(f"Device being used: {device}")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
