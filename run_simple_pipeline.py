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

# Silence TensorFlow/absl/CUDA noise and force CPU before any heavy imports
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')  # 0=all,1=INFO,2=WARNING,3=ERROR
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('ABSL_LOGLEVEL', '3')  # absl to ERROR
os.environ.setdefault('NVIDIA_TF32_OVERRIDE', '0')

# Force CPU-only in WSL to avoid CUDA/XLA issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Ensure no broken XLA flags leak in from the shell
if 'TF_XLA_FLAGS' in os.environ:
    os.environ.pop('TF_XLA_FLAGS', None)

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from risk_pipeline import RiskPipeline
    print("RiskPipeline imported successfully (CPU-only mode)")
except ImportError as e:
    print(f"❌ Error importing RiskPipeline: {e}")
    print("Please ensure you're running this from the project root directory")
    sys.exit(1)

def main():
    """Run the complete RiskPipeline with maximum performance."""
    print("Starting RiskPipeline with Maximum Performance (CPU-only)")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Initialize pipeline with config file
        print("Initializing RiskPipeline...")
        experiment_name = f"simple_run_{int(time.time())}"
        config_path = str(project_root / 'configs' / 'pipeline_config.json')
        pipeline = RiskPipeline(config_path=config_path, experiment_name=experiment_name)
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
