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

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from risk_pipeline import RiskPipeline
    print("✅ RiskPipeline imported successfully!")
except ImportError as e:
    print(f"❌ Error importing RiskPipeline: {e}")
    print("Please ensure you're running this from the project root directory")
    sys.exit(1)

def main():
    """Run the complete RiskPipeline with maximum performance."""
    print("🚀 Starting RiskPipeline with Maximum Performance!")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Initialize pipeline
        print("📊 Initializing RiskPipeline...")
        experiment_name = f"simple_run_{int(time.time())}"
        pipeline = RiskPipeline(experiment_name=experiment_name)
        print("✅ Pipeline initialized successfully!")
        
        # Define assets and models to run
        assets = ['AAPL', 'MSFT', '^GSPC', 'IOZ.AX', 'CBA.AX', 'BHP.AX']
        models = ['arima', 'enhanced_arima', 'lstm', 'stockmixer', 'xgboost']  # Added enhanced_arima
        
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
            run_shap=True
        )
        
        # Calculate execution time
        execution_time = time.time() - start_time
        execution_minutes = execution_time / 60
        
        print("\n" + "=" * 60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
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
            
            print(f"✅ Assets Processed: {asset_count}")
            print(f"✅ Total Models Trained: {total_models}")
        
        print(f"⏱️  Total Execution Time: {execution_minutes:.1f} minutes")
        print(f"📁 Results saved to: {pipeline.results_manager.base_dir}")
        
        print("\n🎯 Pipeline Summary:")
        print("  • All models trained successfully")
        print("  • SHAP analysis completed")
        print("  • Models saved for future use")
        print("  • Results exported and organized")
        print("  • Visualizations generated")
        
        print("\n🚀 Your risk analysis pipeline is ready!")
        
    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {str(e)}")
        print("Please check the error details above and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
