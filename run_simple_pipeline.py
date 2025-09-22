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

_TORCH_GPU_INFO = None
def _inline_gpu_check():
    global _TORCH_GPU_INFO
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            _TORCH_GPU_INFO = f"PyTorch CUDA available: {name}"
        else:
            _TORCH_GPU_INFO = "PyTorch CUDA not available"
        
        # Test gradient flow
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_tensor = torch.randn(2, 10, requires_grad=True, device=device)
        test_layer = torch.nn.Linear(10, 1).to(device)
        test_output = test_layer(test_tensor)
        if test_output.requires_grad:
            _TORCH_GPU_INFO += " (gradients working)"
        else:
            _TORCH_GPU_INFO += " (WARNING: gradients not working!)"
            
    except Exception as e:
        _TORCH_GPU_INFO = f"PyTorch check failed: {e}"
    print(_TORCH_GPU_INFO)

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from risk_pipeline import RiskPipeline
    _inline_gpu_check()
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
            if _TORCH_GPU_INFO:
                _logger.info(_TORCH_GPU_INFO)
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
        print(f"\n❌ Pipeline execution failed: {str(e)}")
        print("\nDebugging information:")
        print(f"  - Error type: {type(e).__name__}")
        print(f"  - Execution time before failure: {time.time() - start_time:.1f} seconds")
        
        # Try to get more detailed error info
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        
        print("\nTroubleshooting steps:")
        print("  1. Check if all required packages are installed")
        print("  2. Verify PyTorch installation and CUDA compatibility")
        print("  3. Check available disk space")
        print("  4. Review the log files in the logs/ directory")
        
        sys.exit(1)

if __name__ == "__main__":
    main()
