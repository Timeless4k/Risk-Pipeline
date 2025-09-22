#!/usr/bin/env python3
"""
Simple PyTorch test script.
"""

print("Testing PyTorch installation...")

try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA device count: {torch.cuda.device_count()}")
        print(f"✓ Current device: {torch.cuda.current_device()}")
        print(f"✓ Device name: {torch.cuda.get_device_name(0)}")
    
    # Test basic operations
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)
    z = torch.mm(x, y.t())
    print(f"✓ Basic tensor operations work")
    
    print("\n✓ PyTorch is working correctly!")
    
except ImportError as e:
    print(f"✗ PyTorch not installed: {e}")
except Exception as e:
    print(f"✗ PyTorch error: {e}")
    import traceback
    traceback.print_exc()
