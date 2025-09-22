#!/usr/bin/env python3
"""
GPU-accelerated SHAP installation script for RiskPipeline.

This script installs the necessary dependencies for GPU-accelerated SHAP analysis.
"""

import subprocess
import sys
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_cuda_availability():
    """Check if CUDA is available on the system."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"✅ CUDA available! GPU: {gpu_name}, Memory: {gpu_memory:.1f} GB")
            return True
        else:
            logger.warning("❌ CUDA not available. GPU acceleration will be disabled.")
            return False
    except ImportError:
        logger.warning("❌ PyTorch not installed. Please install PyTorch first.")
        return False


def install_gpu_shap():
    """Install GPU-accelerated SHAP."""
    logger.info("Installing GPU-accelerated SHAP...")
    
    try:
        # Install SHAP with GPU support
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "shap[gpu]", "--upgrade"
        ])
        logger.info("✅ GPU-accelerated SHAP installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install GPU-accelerated SHAP: {e}")
        return False


def install_cpu_shap():
    """Install CPU-only SHAP as fallback."""
    logger.info("Installing CPU-only SHAP...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "shap==0.41.0", "--upgrade"
        ])
        logger.info("✅ CPU-only SHAP installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install CPU-only SHAP: {e}")
        return False


def main():
    """Main installation function."""
    logger.info("🚀 RiskPipeline GPU SHAP Installation")
    logger.info("=" * 50)
    
    # Check CUDA availability
    cuda_available = check_cuda_availability()
    
    if cuda_available:
        # Try to install GPU-accelerated SHAP
        if install_gpu_shap():
            logger.info("🎉 GPU-accelerated SHAP installation complete!")
            logger.info("You can now use GPU acceleration for SHAP analysis.")
        else:
            logger.warning("⚠️ GPU SHAP installation failed, falling back to CPU version.")
            install_cpu_shap()
    else:
        # Install CPU-only version
        logger.info("Installing CPU-only SHAP...")
        install_cpu_shap()
    
    # Test installation
    try:
        import shap
        logger.info(f"✅ SHAP version: {shap.__version__}")
        
        # Check for GPU support
        if hasattr(shap, 'GPUTreeExplainer'):
            logger.info("✅ GPU TreeExplainer available!")
        else:
            logger.info("ℹ️ GPU TreeExplainer not available (CPU-only SHAP)")
            
    except ImportError as e:
        logger.error(f"❌ SHAP import failed: {e}")
        return False
    
    logger.info("🎉 Installation complete!")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
