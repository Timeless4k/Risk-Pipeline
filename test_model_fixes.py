#!/usr/bin/env python3
"""
Quick test script to verify the model fixes work correctly.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from risk_pipeline.models.model_factory import ModelFactory

def test_lstm_fix():
    """Test that LSTM model can be created and trained without verbose error."""
    print("Testing LSTM model fix...")
    
    try:
        # Create LSTM model
        model = ModelFactory.create_model(
            model_type='lstm',
            task_type='regression',
            input_shape=(100, 42),  # Small test shape
            n_classes=None
        )
        
        # Create dummy data
        X = torch.randn(32, 100, 42)  # batch_size=32, seq_len=100, features=42
        y = torch.randn(32)  # regression target
        
        # Test training for a few steps
        model.train(X, y, epochs=1, verbose=False)
        print("✅ LSTM model fix successful!")
        return True
        
    except Exception as e:
        print(f"❌ LSTM model test failed: {e}")
        return False

def test_stockmixer_fix():
    """Test that StockMixer model can handle small input dimensions."""
    print("Testing StockMixer model fix...")
    
    try:
        # Create StockMixer model
        model = ModelFactory.create_model(
            model_type='stockmixer',
            task_type='regression',
            input_shape=(100, 42),  # Small test shape
            n_classes=None
        )
        
        # Create dummy data with small temporal dimension
        X = torch.randn(32, 1, 42)  # batch_size=32, seq_len=1, features=42 (very small!)
        y = torch.randn(32)  # regression target
        
        # Test forward pass
        with torch.no_grad():
            output = model.model(X)
            print(f"StockMixer output shape: {output.shape}")
        
        print("✅ StockMixer model fix successful!")
        return True
        
    except Exception as e:
        print(f"❌ StockMixer model test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running model fix tests...\n")
    
    lstm_success = test_lstm_fix()
    print()
    stockmixer_success = test_stockmixer_fix()
    print()
    
    if lstm_success and stockmixer_success:
        print("🎉 All model fixes are working correctly!")
        return 0
    else:
        print("❌ Some model fixes failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
