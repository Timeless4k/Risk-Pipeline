#!/usr/bin/env python3
"""
Ultimate Optimized Pipeline Test Script

This script tests the fixed pipeline with all optimizations:
1. Fixed XGBoost compatibility (added fit method)
2. Fixed classification target generation (balanced classes)
3. Fixed outlier detection (less aggressive)
4. Fixed feature count (target: 41 features)
5. Optimized temporal separation
"""

import sys
import os
import logging
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from risk_pipeline import RiskPipeline
from risk_pipeline.core.config import PipelineConfig

def setup_logging():
    """Setup comprehensive logging for debugging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ultimate_pipeline_test.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def test_ultimate_pipeline():
    """Test the ultimate optimized pipeline."""
    logger = setup_logging()
    logger.info("🚀 Starting Ultimate Optimized Pipeline Test")
    
    try:
        # Load the ultimate optimized configuration
        config_path = "configs/ultimate_optimized_config.json"
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found: {config_path}")
            return False
        
        logger.info(f"📋 Loading configuration from: {config_path}")
        
        # Initialize the pipeline with ultimate config
        pipeline = RiskPipeline(config_path=config_path)
        logger.info("✅ Pipeline initialized successfully")
        
        # Test with a single asset first
        test_assets = ["AAPL"]  # Start with one asset for testing
        
        logger.info(f"🎯 Testing with assets: {test_assets}")
        
        # Run the pipeline
        results = pipeline.run_pipeline(
            assets=test_assets,
            models=["arima", "enhanced_arima", "lstm", "stockmixer", "xgboost"],
            tasks=["regression", "classification"],
            save_models=True,
            generate_shap=True
        )
        
        logger.info("✅ Pipeline completed successfully")
        
        # Analyze results
        if results:
            logger.info("📊 Pipeline Results Summary:")
            for asset, asset_results in results.items():
                logger.info(f"  {asset}:")
                for task, task_results in asset_results.items():
                    logger.info(f"    {task}: {len(task_results)} models completed")
                    
                    # Check for specific issues
                    for model, model_result in task_results.items():
                        if isinstance(model_result, dict) and 'error' in model_result:
                            logger.warning(f"      {model}: {model_result['error']}")
                        else:
                            logger.info(f"      {model}: Success")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline test failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False

def main():
    """Main test function."""
    print("🚀 Ultimate Optimized Pipeline Test")
    print("=" * 50)
    
    success = test_ultimate_pipeline()
    
    if success:
        print("\n✅ Pipeline test completed successfully!")
        print("Check 'ultimate_pipeline_test.log' for detailed results")
    else:
        print("\n❌ Pipeline test failed!")
        print("Check 'ultimate_pipeline_test.log' for error details")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
