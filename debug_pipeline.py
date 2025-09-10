#!/usr/bin/env python3
"""
Debug script to investigate "No valid evaluations" issue in RiskPipeline.
"""

import logging
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from risk_pipeline import RiskPipeline
from risk_pipeline.config.logging_config import get_logging_config, apply_logging_config

def main():
    """Run minimal pipeline with debug logging."""
    
    # Enable debug logging
    debug_config = get_logging_config(verbose=True)
    apply_logging_config(debug_config)
    
    # Set console logging to DEBUG for immediate feedback
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add debug handler to validator logger
    validator_logger = logging.getLogger('risk_pipeline.core.validator')
    validator_logger.addHandler(console_handler)
    validator_logger.setLevel(logging.DEBUG)
    
    print("🔍 Starting debug pipeline with verbose logging...")
    
    try:
        # Initialize pipeline with minimal config
        pipeline = RiskPipeline(
            config_path="configs/high_performance_config.json"
        )
        
        # Run only AAPL regression to isolate the issue
        print("📊 Running regression models only...")
        
        # Check if AAPL data is available in config
        config_assets = pipeline.config.data.all_assets
        print(f"Configured assets: {config_assets}")
        
        if 'AAPL' not in config_assets:
            print("❌ AAPL not in configured assets, using first available asset...")
            test_asset = config_assets[0] if config_assets else '^GSPC'
        else:
            test_asset = 'AAPL'
        
        print(f"Testing with asset: {test_asset}")
        
        # Run only regression models for the test asset
        print("🚀 Running regression models...")
        results = pipeline.run_complete_pipeline(
            assets=[test_asset],
            models=['arima', 'enhanced_arima'],  # Start with simple models
            save_models=True,
            run_shap=False  # Skip SHAP for now to isolate the issue
        )
        
        print(f"✅ Pipeline completed. Results: {len(results)} assets")
        
        # Check results
        for asset_name, asset_results in results.items():
            print(f"\n📈 Asset: {asset_name}")
            for task_name, task_results in asset_results.items():
                print(f"  Task: {task_name}")
                for model_name, result in task_results.items():
                    print(f"    Model: {model_name}")
                    if isinstance(result, dict):
                        if 'error' in result:
                            print(f"      ❌ Error: {result['error']}")
                        else:
                            print(f"      ✅ Success: {len(result)} keys")
                            # Show key metrics if available
                            for key in ['mse', 'rmse', 'mae', 'r2']:
                                if key in result:
                                    print(f"        {key}: {result[key]}")
                    else:
                        print(f"      ❓ Unexpected result type: {type(result)}")
            
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to get more specific error info
        try:
            root_logger = logging.getLogger()
            print(f"Root logger level: {root_logger.level}")
            print(f"Root logger handlers: {len(root_logger.handlers)}")
        except Exception as log_e:
            print(f"Could not check logging: {log_e}")

if __name__ == "__main__":
    main()
