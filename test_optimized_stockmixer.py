#!/usr/bin/env python3
"""
Test script for optimized StockMixer model with 12GB VRAM utilization.
"""

import numpy as np
import pandas as pd
import sys
import os
import torch

# Add the project root to the path
sys.path.insert(0, os.path.abspath('.'))

from risk_pipeline.models.stockmixer_model import StockMixerModel

def test_optimized_stockmixer():
    """Test the optimized StockMixer model."""
    print("Testing optimized StockMixer model for 12GB VRAM...")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("❌ No GPU available, using CPU")
    
    # Create sample data
    n_samples = 200
    n_features = 42
    n_time_steps = 30
    
    # Create 3D data [B, T, C] for time series
    X = np.random.randn(n_samples, n_time_steps, n_features)
    y = np.random.randn(n_samples)
    
    print(f"Data shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    try:
        # Create optimized model
        model = StockMixerModel(
            task='regression',
            stocks=1,
            time_steps=n_time_steps,
            channels=n_features,
            market=64,  # Increased for 12GB VRAM
            scale=5,    # Increased scale count
            dropout=0.15,
            batch_size=128,  # Larger batch size
            epochs=5,        # Just a few epochs for testing
            learning_rate=5e-4,
            mixed_precision=True,
            gradient_accumulation_steps=2
        )
        
        print("✅ Model created successfully")
        
        # Build model
        model.build_model(X.shape)
        print(f"✅ Model built with input shape: {X.shape}")
        
        # Test training
        print("Testing training...")
        result = model.train(X, y)
        print(f"✅ Training completed: {result}")
        
        # Test prediction
        print("Testing prediction...")
        predictions = model.predict(X[:10])
        print(f"✅ Prediction completed. Shape: {predictions.shape}")
        print(f"✅ Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
        
        # Check GPU memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1e9
            memory_reserved = torch.cuda.memory_reserved(0) / 1e9
            print(f"✅ GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
        
        print("🎉 Optimized StockMixer model is working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_optimized_stockmixer()
    sys.exit(0 if success else 1)
