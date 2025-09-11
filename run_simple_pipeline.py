#!/usr/bin/env python3
"""
Simple RiskPipeline Runner
Runs the complete pipeline with maximum performance settings.
No CLI, no menus - just runs everything!
"""

import os
import sys
import time
from pathlib import Path
import logging

# Silence TensorFlow/absl/CUDA noise before any heavy imports
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')  # 0=all,1=INFO,2=WARNING,3=ERROR
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('ABSL_LOGLEVEL', '3')  # absl to ERROR
os.environ.setdefault('NVIDIA_TF32_OVERRIDE', '0')

"""
GPU notes:
- We keep TensorFlow 2.20.0 and accept PTX JIT on first run for RTX 50xx.
- We no longer force CPU-only; instead we warm up the GPU once at startup.
"""
if 'TF_XLA_FLAGS' in os.environ:
    os.environ.pop('TF_XLA_FLAGS', None)

# Control GPU warm-up via env flag (default: on). Set RISKPIPELINE_SKIP_TF_WARMUP=1 to disable.
_SKIP_TF_WARMUP = os.environ.get('RISKPIPELINE_SKIP_TF_WARMUP', '').lower() in ('1', 'true', 'yes')

# Always attempt a minimal GPU warm-up before importing the package (unless skipped)
_EARLY_WARMUP_MSG = None
_SECONDARY_WARMUP_MSG = None
_TF_DEVICE_STR = None


def _inline_gpu_warmup():
    global _EARLY_WARMUP_MSG
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            _EARLY_WARMUP_MSG = "TensorFlow warm-up: No GPU available"
            print(_EARLY_WARMUP_MSG)
            return _EARLY_WARMUP_MSG
        try:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:
                    pass
        except Exception:
            pass
        import time as _t
        t0 = _t.time()
        with tf.device('/GPU:0'):
            a = tf.random.normal([256, 256])
            b = tf.random.normal([256, 256])
            _ = tf.linalg.matmul(a, b)
        dt = _t.time() - t0
        _EARLY_WARMUP_MSG = f"TensorFlow warm-up: GPU warmed in {dt:.2f}s"
        print(_EARLY_WARMUP_MSG)
        return _EARLY_WARMUP_MSG
    except Exception as e:
        _EARLY_WARMUP_MSG = f"TensorFlow warm-up skipped: {e}"
        print(_EARLY_WARMUP_MSG)
        return _EARLY_WARMUP_MSG

if not _SKIP_TF_WARMUP:
    _inline_gpu_warmup()
else:
    _EARLY_WARMUP_MSG = "TensorFlow warm-up skipped by env flag"
    print(_EARLY_WARMUP_MSG)

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from risk_pipeline import RiskPipeline
    # Configure TF after warm-up for consistent device state
    try:
        from risk_pipeline.utils.tensorflow_utils import configure_tensorflow_memory, warm_up_gpu
        _TF_DEVICE_STR = configure_tensorflow_memory(
            gpu_memory_growth=True,
            gpu_memory_limit=None,
            force_cpu=(os.environ.get('RISKPIPELINE_FORCE_CPU','').lower() in ('1','true','yes'))
        )
        if not _SKIP_TF_WARMUP:
            used_gpu, _SECONDARY_WARMUP_MSG = warm_up_gpu(jit_intensive=False)
            print(f"TensorFlow device: {_TF_DEVICE_STR} | Secondary warm-up: {_SECONDARY_WARMUP_MSG}")
        else:
            _SECONDARY_WARMUP_MSG = "GPU warm-up skipped by env flag"
            print(f"TensorFlow device: {_TF_DEVICE_STR} | Secondary warm-up: {_SECONDARY_WARMUP_MSG}")
    except Exception as _e:
        print(f"TensorFlow setup/warm-up skipped: {_e}")
    print("RiskPipeline imported successfully")
except ImportError as e:
    print(f"❌ Error importing RiskPipeline: {e}")
    print("Please ensure you're running this from the project root directory")
    sys.exit(1)

def main():
    """Run the complete RiskPipeline with maximum performance."""
    print("Starting RiskPipeline with Maximum Performance")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Initialize pipeline with config file
        print("Initializing RiskPipeline...")
        experiment_name = f"simple_run_{int(time.time())}"
        config_path = str(project_root / 'configs' / 'pipeline_config.json')
        pipeline = RiskPipeline(config_path=config_path, experiment_name=experiment_name)
        # Now that logging is configured by RiskPipeline, mirror any early warm-up info into the log file
        try:
            _logger = logging.getLogger(__name__)
            if _TF_DEVICE_STR is not None:
                _logger.info(f"TensorFlow device configured: {_TF_DEVICE_STR}")
            if _EARLY_WARMUP_MSG:
                _logger.info(_EARLY_WARMUP_MSG)
            if _SECONDARY_WARMUP_MSG:
                _logger.info(f"Secondary warm-up: {_SECONDARY_WARMUP_MSG}")
        except Exception:
            pass
        print("Pipeline initialized successfully")
        
        # Define assets and models from config
        assets = pipeline.config.data.all_assets
        # Preferred run order: run regression-first models then ML
        default_order = ['arima', 'garch', 'xgboost', 'lstm', 'stockmixer']
        models_cfg = list(getattr(pipeline.config, 'models_to_run', []))
        models = models_cfg if models_cfg else default_order
        # Normalize aliases (e.g., xgb -> xgboost) and preserve order
        alias = {'xgb': 'xgboost'}
        models = [alias.get(m, m) for m in models]
        
        # Minimal, concise output
        print(f"Assets: {', '.join(assets)}")
        print(f"Models: {', '.join(models)}")
        
        print("Starting pipeline execution...")
        print("-" * 60)
        
        # Run the complete pipeline
        results = pipeline.run_complete_pipeline(
            assets=assets,
            models=models,
            save_models=True,
            run_shap=True,
            run_cross_transfer=True
        )
        
        # Calculate execution time
        execution_time = time.time() - start_time
        execution_minutes = execution_time / 60
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        # Show results summary
        if results:
            asset_count = len(results)
            total_models = 0
            
            for asset, asset_results in results.items():
                if isinstance(asset_results, dict):
                    for task, task_results in asset_results.items():
                        if isinstance(task_results, dict):
                            total_models += len(task_results)
            
            print(f"Assets Processed: {asset_count}")
            print(f"Total Models Trained: {total_models}")
        
        print(f"Total Execution Time: {execution_minutes:.1f} minutes")
        print(f"Results saved to: {pipeline.results_manager.base_dir}")
        
        print("\nPipeline Summary:")
        print("  - All models trained successfully")
        print("  - SHAP analysis completed")
        print("  - Models saved for future use")
        print("  - Results exported and organized")
        print("  - Visualizations generated")
        
        print("\nYour risk analysis pipeline is ready!")
        
    except Exception as e:
        print(f"\nPipeline execution failed: {str(e)}")
        print("Please check the error details above and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
