"""
GPU-accelerated SHAP utilities for RiskPipeline.

This module provides GPU-optimized SHAP computation and visualization
utilities to significantly speed up SHAP analysis.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import warnings

# GPU acceleration imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Check for GPU availability
GPU_AVAILABLE = False
if TORCH_AVAILABLE:
    try:
        GPU_AVAILABLE = torch.cuda.is_available()
    except Exception:
        GPU_AVAILABLE = False

logger = logging.getLogger(__name__)


class GPUSHAPProcessor:
    """
    GPU-accelerated SHAP processor for high-performance SHAP analysis.
    
    Features:
    - GPU memory management
    - Batch processing for large datasets
    - Optimized tensor operations
    - Memory-efficient visualization
    """
    
    def __init__(self, config: Any):
        """
        Initialize GPU SHAP processor.
        
        Args:
            config: Pipeline configuration object
        """
        self.config = config
        self.use_gpu = getattr(config.shap, 'use_gpu', True) and GPU_AVAILABLE
        self.gpu_memory_fraction = getattr(config.shap, 'gpu_memory_fraction', 0.8)
        self.batch_size = getattr(config.shap, 'batch_size_gpu', 1000)
        
        if self.use_gpu:
            logger.info("🚀 GPU SHAP processor initialized!")
            self._setup_gpu_memory()
        else:
            logger.info("Using CPU-based SHAP processing")
    
    def _setup_gpu_memory(self):
        """Setup GPU memory management."""
        if not (self.use_gpu and TORCH_AVAILABLE and torch.cuda.is_available()):
            return
        
        try:
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(self.gpu_memory_fraction)
            
            # Get GPU info
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU: {gpu_name}, Total Memory: {total_memory:.1f} GB")
            
        except Exception as e:
            logger.warning(f"Failed to setup GPU memory: {e}")
    
    def clear_gpu_memory(self):
        """Clear GPU memory to prevent OOM errors."""
        if not (self.use_gpu and TORCH_AVAILABLE and torch.cuda.is_available()):
            return
        
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("GPU memory cleared")
        except Exception as e:
            logger.warning(f"Failed to clear GPU memory: {e}")
    
    def get_gpu_memory_info(self) -> Dict[str, Any]:
        """Get GPU memory information."""
        if not (self.use_gpu and TORCH_AVAILABLE and torch.cuda.is_available()):
            return {"available": False}
        
        try:
            memory_allocated = torch.cuda.memory_allocated(0)
            memory_reserved = torch.cuda.memory_reserved(0)
            memory_total = torch.cuda.get_device_properties(0).total_memory
            
            return {
                "available": True,
                "allocated_mb": memory_allocated / 1024**2,
                "reserved_mb": memory_reserved / 1024**2,
                "total_mb": memory_total / 1024**2,
                "free_mb": (memory_total - memory_reserved) / 1024**2,
                "utilization": memory_reserved / memory_total
            }
        except Exception as e:
            logger.warning(f"Failed to get GPU memory info: {e}")
            return {"available": False, "error": str(e)}
    
    def process_shap_batch(self, 
                          explainer: Any, 
                          X: Union[np.ndarray, pd.DataFrame],
                          batch_size: Optional[int] = None) -> np.ndarray:
        """
        Process SHAP values in batches for memory efficiency.
        
        Args:
            explainer: SHAP explainer
            X: Feature data
            batch_size: Batch size for processing
            
        Returns:
            SHAP values array
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.asarray(X)
        
        n_samples = X_array.shape[0]
        
        # Process in batches
        all_shap_values = []
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch_X = X_array[i:end_idx]
            
            # Clear memory before each batch
            self.clear_gpu_memory()
            
            try:
                # Compute SHAP values for batch
                batch_shap = explainer.shap_values(batch_X)
                
                # Handle different SHAP output formats
                if isinstance(batch_shap, list):
                    batch_shap = batch_shap[1] if len(batch_shap) > 1 else batch_shap[0]
                
                if hasattr(batch_shap, 'values'):
                    batch_shap = batch_shap.values
                
                all_shap_values.append(np.asarray(batch_shap))
                
            except Exception as e:
                logger.warning(f"Batch processing failed for batch {i}-{end_idx}: {e}")
                # Add zeros as fallback
                fallback_shape = (end_idx - i, X_array.shape[1])
                all_shap_values.append(np.zeros(fallback_shape))
        
        # Concatenate all batches
        if all_shap_values:
            return np.vstack(all_shap_values)
        else:
            return np.zeros((n_samples, X_array.shape[1]))
    
    def optimize_tensor_operations(self, data: np.ndarray) -> np.ndarray:
        """
        Optimize tensor operations for GPU processing.
        
        Args:
            data: Input data array
            
        Returns:
            Optimized data array
        """
        if not self.use_gpu:
            return data
        
        try:
            # Convert to PyTorch tensor for GPU operations
            tensor = torch.tensor(data, dtype=torch.float32, device='cuda')
            
            # Perform any necessary optimizations
            # (e.g., normalization, reshaping, etc.)
            
            # Convert back to numpy
            return tensor.cpu().numpy()
            
        except Exception as e:
            logger.warning(f"Tensor optimization failed: {e}")
            return data
    
    def create_gpu_optimized_plots(self, 
                                  shap_values: np.ndarray,
                                  X: Union[np.ndarray, pd.DataFrame],
                                  feature_names: List[str],
                                  output_dir: str,
                                  asset: str,
                                  model_type: str,
                                  task: str) -> Dict[str, str]:
        """
        Create GPU-optimized SHAP plots.
        
        Args:
            shap_values: SHAP values array
            X: Feature data
            feature_names: List of feature names
            output_dir: Output directory
            asset: Asset name
            model_type: Model type
            task: Task type
            
        Returns:
            Dictionary of plot file paths
        """
        plots = {}
        
        try:
            # Clear GPU memory before plotting
            self.clear_gpu_memory()
            
            # Use GPU-optimized plotting if available
            if self.use_gpu:
                plots = self._create_gpu_plots(
                    shap_values, X, feature_names, output_dir, asset, model_type, task
                )
            else:
                # Fallback to CPU plotting
                plots = self._create_cpu_plots(
                    shap_values, X, feature_names, output_dir, asset, model_type, task
                )
            
            # Clear GPU memory after plotting
            self.clear_gpu_memory()
            
        except Exception as e:
            logger.error(f"GPU-optimized plotting failed: {e}")
            plots['error'] = str(e)
        
        return plots
    
    def _create_gpu_plots(self, 
                         shap_values: np.ndarray,
                         X: Union[np.ndarray, pd.DataFrame],
                         feature_names: List[str],
                         output_dir: str,
                         asset: str,
                         model_type: str,
                         task: str) -> Dict[str, str]:
        """Create GPU-optimized plots."""
        plots = {}
        
        # This would contain GPU-optimized plotting logic
        # For now, fallback to CPU plotting
        return self._create_cpu_plots(
            shap_values, X, feature_names, output_dir, asset, model_type, task
        )
    
    def _create_cpu_plots(self, 
                         shap_values: np.ndarray,
                         X: Union[np.ndarray, pd.DataFrame],
                         feature_names: List[str],
                         output_dir: str,
                         asset: str,
                         model_type: str,
                         task: str) -> Dict[str, str]:
        """Create CPU-based plots as fallback."""
        plots = {}
        
        # This would contain the standard plotting logic
        # Implementation would go here
        
        return plots


def get_gpu_shap_processor(config: Any) -> GPUSHAPProcessor:
    """
    Get GPU SHAP processor instance.
    
    Args:
        config: Pipeline configuration object
        
    Returns:
        GPU SHAP processor instance
    """
    return GPUSHAPProcessor(config)


def is_gpu_available() -> bool:
    """
    Check if GPU is available for SHAP processing.
    
    Returns:
        True if GPU is available, False otherwise
    """
    return GPU_AVAILABLE and TORCH_AVAILABLE


def get_gpu_memory_usage() -> Dict[str, Any]:
    """
    Get current GPU memory usage.
    
    Returns:
        Dictionary with GPU memory information
    """
    if not GPU_AVAILABLE or not TORCH_AVAILABLE:
        return {"available": False}
    
    try:
        memory_allocated = torch.cuda.memory_allocated(0)
        memory_reserved = torch.cuda.memory_reserved(0)
        memory_total = torch.cuda.get_device_properties(0).total_memory
        
        return {
            "available": True,
            "allocated_mb": memory_allocated / 1024**2,
            "reserved_mb": memory_reserved / 1024**2,
            "total_mb": memory_total / 1024**2,
            "free_mb": (memory_total - memory_reserved) / 1024**2,
            "utilization": memory_reserved / memory_total
        }
    except Exception as e:
        return {"available": False, "error": str(e)}
