"""
TensorFlow utilities for RiskPipeline.

This module handles TensorFlow compatibility, memory management, and GPU/CPU fallback.
"""

import os
import logging
from typing import Optional, List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

def check_tensorflow_compatibility() -> Dict[str, Any]:
    """
    Check TensorFlow installation and compatibility.
    
    Returns:
        Dictionary with compatibility information
    """
    try:
        import tensorflow as tf
        
        # Check TensorFlow version
        tf_version = tf.__version__
        logger.info(f"TensorFlow version: {tf_version}")
        
        # Check GPU availability
        gpu_devices = tf.config.list_physical_devices('GPU')
        cpu_devices = tf.config.list_physical_devices('CPU')
        
        logger.info(f"GPU devices: {len(gpu_devices)}")
        logger.info(f"CPU devices: {len(cpu_devices)}")
        
        # Check CUDA compatibility
        cuda_available = False
        try:
            cuda_available = tf.test.is_built_with_cuda()
        except:
            pass
        
        # Check memory growth
        memory_growth_enabled = False
        if gpu_devices:
            try:
                for gpu in gpu_devices:
                    tf.config.experimental.set_memory_growth(gpu, True)
                memory_growth_enabled = True
                logger.info("GPU memory growth enabled")
            except Exception as e:
                logger.warning(f"Failed to enable GPU memory growth: {e}")
        
        return {
            'version': tf_version,
            'gpu_devices': len(gpu_devices),
            'cpu_devices': len(cpu_devices),
            'cuda_available': cuda_available,
            'memory_growth_enabled': memory_growth_enabled,
            'compatible': True
        }
        
    except ImportError as e:
        logger.error(f"TensorFlow not available: {e}")
        return {
            'compatible': False,
            'error': str(e)
        }
    except Exception as e:
        logger.error(f"TensorFlow compatibility check failed: {e}")
        return {
            'compatible': False,
            'error': str(e)
        }

def configure_tensorflow_memory() -> None:
    """
    Configure TensorFlow memory settings for optimal performance.
    """
    try:
        import tensorflow as tf
        
        # Set memory growth to prevent GPU memory allocation issues
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info("GPU memory growth enabled")
            except RuntimeError as e:
                logger.warning(f"GPU memory growth setup failed: {e}")
        
        # Set mixed precision if available
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("Mixed precision enabled")
        except Exception as e:
            logger.debug(f"Mixed precision not available: {e}")
        
        # Set thread parallelism
        try:
            tf.config.threading.set_inter_op_parallelism_threads(4)
            tf.config.threading.set_intra_op_parallelism_threads(4)
            logger.info("Thread parallelism configured")
        except Exception as e:
            logger.debug(f"Thread parallelism setup failed: {e}")
            
    except Exception as e:
        logger.warning(f"TensorFlow memory configuration failed: {e}")

def get_optimal_device() -> str:
    """
    Get the optimal device for TensorFlow operations.
    
    Returns:
        Device string ('GPU:0' or '/CPU:0')
    """
    try:
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # Check if GPU is actually usable
            try:
                with tf.device('/GPU:0'):
                    # Test GPU with a simple operation
                    test_tensor = tf.constant([1.0])
                    _ = tf.reduce_sum(test_tensor)
                logger.info("GPU is available and functional")
                return '/GPU:0'
            except Exception as e:
                logger.warning(f"GPU test failed, falling back to CPU: {e}")
                return '/CPU:0'
        else:
            logger.info("No GPU devices found, using CPU")
            return '/CPU:0'
            
    except Exception as e:
        logger.warning(f"Device detection failed: {e}")
        return '/CPU:0'

def safe_tensorflow_operation(operation_func, *args, **kwargs):
    """
    Safely execute a TensorFlow operation with fallback to CPU if needed.
    
    Args:
        operation_func: Function to execute
        *args: Arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        Result of the operation
    """
    try:
        # Try with default device (GPU if available)
        return operation_func(*args, **kwargs)
    except Exception as e:
        logger.warning(f"Operation failed on default device: {e}")
        
        # Fallback to CPU
        try:
            import tensorflow as tf
            with tf.device('/CPU:0'):
                logger.info("Retrying operation on CPU")
                return operation_func(*args, **kwargs)
        except Exception as cpu_error:
            logger.error(f"Operation failed on CPU as well: {cpu_error}")
            raise cpu_error

def cleanup_tensorflow_memory() -> None:
    """
    Clean up TensorFlow memory and clear GPU memory.
    """
    try:
        import tensorflow as tf
        
        # Clear GPU memory
        if tf.config.list_physical_devices('GPU'):
            tf.keras.backend.clear_session()
            logger.info("TensorFlow GPU memory cleared")
        
        # Force garbage collection
        import gc
        gc.collect()
        
    except Exception as e:
        logger.warning(f"TensorFlow memory cleanup failed: {e}")

def validate_model_input(X: np.ndarray, expected_shape: tuple) -> np.ndarray:
    """
    Validate and reshape model input to expected shape.
    
    Args:
        X: Input data
        expected_shape: Expected shape (batch_size, ...)
        
    Returns:
        Validated and reshaped input
    """
    if X.ndim != len(expected_shape):
        raise ValueError(f"Expected {len(expected_shape)}D input, got {X.ndim}D")
    
    # Handle batch dimension
    if X.shape[0] != expected_shape[0]:
        if expected_shape[0] is None:  # Variable batch size
            pass
        else:
            raise ValueError(f"Expected batch size {expected_shape[0]}, got {X.shape[0]}")
    
    # Ensure other dimensions match
    for i, (actual, expected) in enumerate(zip(X.shape[1:], expected_shape[1:])):
        if expected is not None and actual != expected:
            raise ValueError(f"Dimension {i+1} mismatch: expected {expected}, got {actual}")
    
    return X
