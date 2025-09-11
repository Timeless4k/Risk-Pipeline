"""
TensorFlow utilities for RiskPipeline with enhanced GPU/CPU fallback handling.
"""

import logging
import os
import warnings
import time
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

def check_tensorflow_compatibility() -> Dict[str, Any]:
    """
    Check TensorFlow compatibility and available devices.
    
    Returns:
        Dictionary with compatibility information
    """
    try:
        import tensorflow as tf
        
        # Check version
        version = tf.__version__
        
        # Check GPU availability
        gpu_devices = tf.config.list_physical_devices('GPU')
        cpu_devices = tf.config.list_physical_devices('CPU')
        
        # Test GPU functionality
        gpu_working = False
        if gpu_devices:
            try:
                # Try to create a simple tensor on GPU
                with tf.device('/GPU:0'):
                    test_tensor = tf.constant([1.0, 2.0, 3.0])
                    gpu_working = True
            except Exception as e:
                logger.warning(f"GPU test failed: {e}")
                gpu_working = False
        
        return {
            'compatible': True,
            'version': version,
            'gpu_devices': len(gpu_devices),
            'cpu_devices': len(cpu_devices),
            'gpu_working': gpu_working,
            'gpu_available': len(gpu_devices) > 0
        }
        
    except ImportError:
        return {
            'compatible': False,
            'version': None,
            'gpu_devices': 0,
            'cpu_devices': 0,
            'gpu_working': False,
            'gpu_available': False
        }
    except Exception as e:
        logger.error(f"TensorFlow compatibility check failed: {e}")
        return {
            'compatible': False,
            'version': 'unknown',
            'gpu_devices': 0,
            'cpu_devices': 0,
            'gpu_working': False,
            'gpu_available': False
        }

def configure_tensorflow_memory(gpu_memory_growth: bool = True, 
                               gpu_memory_limit: Optional[int] = None,
                               force_cpu: bool = False) -> str:
    """
    Configure TensorFlow memory and device settings with enhanced fallback.
    
    Args:
        gpu_memory_growth: Whether to allow GPU memory growth
        gpu_memory_limit: GPU memory limit in MB (None for no limit)
        force_cpu: Force CPU usage even if GPU is available
        
    Returns:
        Device string that was configured
    """
    try:
        import tensorflow as tf
        
        # Check if we should force CPU
        if force_cpu:
            logger.info("Forcing CPU usage as requested")
            # Environment guard for any future TF imports
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            try:
                # Hide all GPUs from the current TF runtime as well
                tf.config.set_visible_devices([], 'GPU')
            except Exception:
                pass
            # Ensure no mixed precision policies that could trigger GPU-only kernels
            try:
                from tensorflow.keras import mixed_precision as _mp
                _mp.set_global_policy('float32')
                logger.info("Disabled TensorFlow mixed precision policy (set to float32)")
            except Exception:
                pass
            return '/CPU:0'
        
        # Check GPU availability
        gpu_devices = tf.config.list_physical_devices('GPU')
        
        if not gpu_devices:
            logger.info("No GPU devices available, using CPU")
            return '/CPU:0'
        
        # Test GPU functionality
        try:
            # Configure GPU memory growth
            for gpu in gpu_devices:
                if gpu_memory_growth:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"Enabled memory growth for {gpu}")
                
                if gpu_memory_limit:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=gpu_memory_limit)]
                    )
                    logger.info(f"Set memory limit to {gpu_memory_limit}MB for {gpu}")
            
            # Test GPU with a set of ops that commonly trigger driver issues
            with tf.device('/GPU:0'):
                # Simple constant + reduction
                test_tensor = tf.constant([1.0, 2.0, 3.0])
                test_result = tf.reduce_sum(test_tensor).numpy()
                # Small matmul
                a = tf.random.uniform((32, 32), dtype=tf.float32)
                _ = tf.matmul(a, tf.transpose(a)).numpy()
                # Cast kernel (often triggers CUDA handle issues with drivers)
                _ = tf.cast(a, tf.float16).numpy()
                logger.info(f"GPU test successful: {test_result}")
            
            logger.info(f"GPU configuration successful, using {len(gpu_devices)} GPU(s)")
            return '/GPU:0'
            
        except Exception as e:
            logger.warning(f"GPU configuration failed: {e}, falling back to CPU")
            # Disable GPU and fall back to CPU
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            try:
                tf.config.set_visible_devices([], 'GPU')
            except Exception:
                pass
            try:
                from tensorflow.keras import mixed_precision as _mp
                _mp.set_global_policy('float32')
            except Exception:
                pass
            return '/CPU:0'
            
    except ImportError:
        logger.warning("TensorFlow not available, using CPU fallback")
        return '/CPU:0'
    except Exception as e:
        logger.error(f"TensorFlow configuration failed: {e}, using CPU fallback")
        return '/CPU:0'

def get_optimal_device(prefer_gpu: bool = True, 
                      gpu_memory_threshold: int = 1000) -> str:
    """
    Get the optimal device for TensorFlow operations with smart fallback.
    
    Args:
        prefer_gpu: Whether to prefer GPU over CPU
        gpu_memory_threshold: Minimum GPU memory in MB to consider GPU viable
        
    Returns:
        Optimal device string
    """
    try:
        import tensorflow as tf
        # Honor explicit CPU-only environment overrides
        if os.environ.get('RISKPIPELINE_FORCE_CPU', '').lower() in ('1', 'true', 'yes') or os.environ.get('CUDA_VISIBLE_DEVICES') == '-1':
            logger.info("Environment enforces CPU-only mode; selecting CPU device")
            try:
                tf.config.set_visible_devices([], 'GPU')
            except Exception:
                pass
            try:
                from tensorflow.keras import mixed_precision as _mp
                _mp.set_global_policy('float32')
            except Exception:
                pass
            return '/CPU:0'
        
        if not prefer_gpu:
            logger.info("CPU preferred, using CPU")
            return '/CPU:0'
        
        # Check GPU availability and functionality
        compat_info = check_tensorflow_compatibility()
        
        if not compat_info['gpu_available']:
            logger.info("No GPU available, using CPU")
            return '/CPU:0'
        
        if not compat_info['gpu_working']:
            logger.warning("GPU available but not working, using CPU")
            return '/CPU:0'
        
        # Check GPU memory
        try:
            gpu_devices = tf.config.list_physical_devices('GPU')
            if gpu_devices:
                # Get GPU memory info
                gpu_details = tf.config.experimental.get_device_details(gpu_devices[0])
                if 'device_name' in gpu_details:
                    logger.info(f"Using GPU: {gpu_details['device_name']}")
                    return '/GPU:0'
        except Exception as e:
            logger.warning(f"Could not get GPU details: {e}")
        
        # If we get here, GPU should be working
        logger.info("GPU available and working, using GPU")
        return '/GPU:0'
        
    except Exception as e:
        logger.error(f"Device selection failed: {e}, using CPU")
        return '/CPU:0'

def safe_tensorflow_operation(operation_func, 
                            fallback_device: str = '/CPU:0',
                            max_retries: int = 2) -> Any:
    """
    Execute TensorFlow operation with automatic fallback on failure.
    
    Args:
        operation_func: Function that performs the TensorFlow operation
        fallback_device: Device to fall back to on failure
        max_retries: Maximum number of retry attempts
        
    Returns:
        Result of the operation
    """
    for attempt in range(max_retries + 1):
        try:
            if attempt == 0:
                # First attempt: try with current device
                result = operation_func()
                return result
            else:
                # Retry with fallback device
                logger.warning(f"Operation failed, retrying with {fallback_device} (attempt {attempt + 1})")
                try:
                    import tensorflow as tf
                    with tf.device(fallback_device):
                        result = operation_func()
                        logger.info(f"Operation successful on {fallback_device}")
                        return result
                except ImportError:
                    logger.error("TensorFlow not available for fallback")
                    raise
                    
        except Exception as e:
            if attempt == max_retries:
                logger.error(f"Operation failed after {max_retries + 1} attempts: {e}")
                raise
            else:
                logger.warning(f"Operation attempt {attempt + 1} failed: {e}")
                continue

def cleanup_tensorflow_memory():
    """Clean up TensorFlow memory and reset GPU state."""
    try:
        import tensorflow as tf
        
        # Clear GPU memory
        if tf.config.list_physical_devices('GPU'):
            tf.keras.backend.clear_session()
            logger.info("Cleared TensorFlow GPU memory")
        
        # Force garbage collection
        import gc
        gc.collect()
        
    except Exception as e:
        logger.warning(f"Memory cleanup failed: {e}")

def force_cpu_mode():
    """Force TensorFlow to use CPU only."""
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    try:
        import tensorflow as tf
        try:
            tf.config.set_visible_devices([], 'GPU')
        except Exception:
            pass
        try:
            from tensorflow.keras import mixed_precision as _mp
            _mp.set_global_policy('float32')
        except Exception:
            pass
    except Exception:
        pass
    logger.info("Forced CPU-only mode for TensorFlow")

def reset_gpu_state():
    """Reset GPU state and clear memory."""
    try:
        import tensorflow as tf
        
        # Clear all GPU memory
        tf.keras.backend.clear_session()
        
        # Reset GPU devices
        gpu_devices = tf.config.list_physical_devices('GPU')
        for gpu in gpu_devices:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as e:
                logger.warning(f"Could not reset GPU {gpu}: {e}")
        
        logger.info("GPU state reset completed")
        
    except Exception as e:
        logger.error(f"GPU state reset failed: {e}")

def warm_up_gpu(jit_intensive: bool = False) -> Tuple[bool, str]:
    """Trigger light GPU work so CUDA kernels JIT-compile once.
    
    Args:
        jit_intensive: If True, run a heavier matmul to JIT more kernels.
    
    Returns:
        Tuple of (used_gpu, message)
    """
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            return False, "No GPU available for warm-up"
        start = time.time()
        logger.info("Starting TensorFlow GPU warm-up (jit_intensive=%s)", jit_intensive)
        # Keep memory growth sane to avoid large pre-allocations
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logger.info("Enabled memory growth for %s", gpu)
        except Exception:
            pass
        # Minimal warm-up: small random op on GPU:0
        with tf.device('/GPU:0'):
            a = tf.random.normal([256, 256])
            b = tf.random.normal([256, 256])
            _ = tf.linalg.matmul(a, b)
            if jit_intensive:
                # A bit heavier pass to JIT more kernels once
                x = tf.random.normal([1024, 1024])
                y = tf.random.normal([1024, 1024])
                _ = tf.linalg.matmul(x, y)
        elapsed = time.time() - start
        msg = f"GPU warm-up complete in {elapsed:.2f}s"
        logger.info(msg)
        return True, msg
    except Exception as e:
        logger.warning(f"GPU warm-up skipped: {e}")
        return False, f"GPU warm-up skipped: {e}"