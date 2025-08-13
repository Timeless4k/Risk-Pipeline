# NOTE: This is a developer integration script, not part of unit test suite.
# Skip collection under pytest to avoid fixture errors.
try:
    import pytest  # type: ignore
    pytestmark = pytest.mark.skip(reason="Integration script; excluded from unit tests")
except Exception:
    pass
#!/usr/bin/env python3
"""
Integration verification script for RiskPipeline core components.

This script tests the integration of the newly extracted core components:
- DataLoader
- FeatureEngineer
- WalkForwardValidator
- Configuration management
- Results management

Run this script to verify that all components work together correctly.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(__file__))

from risk_pipeline.core import (
    PipelineConfig, DataLoader, FeatureEngineer, 
    WalkForwardValidator, ResultsManager
)
from risk_pipeline.utils.logging_utils import setup_logging

def setup_test_environment():
    """Set up test environment and logging."""
    # Create test directories
    test_dirs = ['test_output', 'test_cache', 'test_models']
    for dir_name in test_dirs:
        Path(dir_name).mkdir(exist_ok=True)
    
    # Setup logging
    setup_logging(log_file_path='test_output/integration_test.log', level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("RISK PIPELINE INTEGRATION TEST")
    logger.info("=" * 80)
    logger.info(f"Test started at: {datetime.now()}")
    
    return logger

def test_configuration():
    """Test configuration management."""
    logger = logging.getLogger(__name__)
    logger.info("\n" + "="*50)
    logger.info("TESTING CONFIGURATION MANAGEMENT")
    logger.info("="*50)
    
    try:
        # Test default configuration
        config = PipelineConfig()
        logger.info("✅ Default configuration created successfully")
        
        # Test configuration from file
        config_path = "configs/pipeline_config.json"
        if os.path.exists(config_path):
            config_from_file = PipelineConfig.from_file(config_path)
            logger.info("✅ Configuration loaded from file successfully")
        else:
            logger.warning("⚠️ Configuration file not found, skipping file loading test")
        
        # Test configuration updates
        config.update({
            'data': {
                'start_date': '2020-01-01',
                'end_date': '2023-12-31'
            }
        })
        logger.info("✅ Configuration updated successfully")
        
        # Test validation
        if config.validate():
            logger.info("✅ Configuration validation passed")
        else:
            logger.error("❌ Configuration validation failed")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {str(e)}")
        return False

def test_data_loading():
    """Test data loading functionality."""
    logger = logging.getLogger(__name__)
    logger.info("\n" + "="*50)
    logger.info("TESTING DATA LOADING")
    logger.info("="*50)
    
    try:
        # Initialize data loader
        data_loader = DataLoader(cache_dir='test_cache')
        logger.info("✅ DataLoader initialized successfully")
        
        # Test with a small set of symbols and date range
        symbols = ['AAPL', 'MSFT']
        start_date = '2023-01-01'
        end_date = '2023-12-31'
        
        logger.info(f"Downloading data for {symbols} from {start_date} to {end_date}")
        data = data_loader.download_data(symbols, start_date, end_date, force_download=False)
        
        if data:
            logger.info(f"✅ Data downloaded successfully for {len(data)} symbols")
            
            # Check data quality
            for symbol, df in data.items():
                logger.info(f"  {symbol}: {len(df)} rows, {len(df.columns)} columns")
                if 'Adj Close' in df.columns:
                    logger.info(f"    Price range: {df['Adj Close'].min():.2f} - {df['Adj Close'].max():.2f}")
            
            return data
        else:
            logger.error("❌ No data downloaded")
            return None
            
    except Exception as e:
        logger.error(f"❌ Data loading test failed: {str(e)}")
        return None

def test_feature_engineering(data):
    """Test feature engineering functionality."""
    logger = logging.getLogger(__name__)
    logger.info("\n" + "="*50)
    logger.info("TESTING FEATURE ENGINEERING")
    logger.info("="*50)
    
    try:
        # Initialize feature engineer
        config = PipelineConfig()
        feature_engineer = FeatureEngineer(config)
        logger.info("✅ FeatureEngineer initialized successfully")
        
        # Test feature creation
        logger.info("Creating features for all assets...")
        features = feature_engineer.create_all_features(data, skip_correlations=False)
        
        if features:
            logger.info(f"✅ Features created successfully for {len(features)} assets")
            
            # Analyze features for each asset
            for asset, asset_features in features.items():
                logger.info(f"\n  {asset}:")
                logger.info(f"    Total features: {len(asset_features.columns)}")
                logger.info(f"    Samples: {len(asset_features)}")
                
                # Get feature summary
                summary = feature_engineer.get_feature_summary(asset_features)
                logger.info(f"    Numeric features: {summary['numeric_features']}")
                logger.info(f"    Missing values: {sum(asset_features.isna().sum())}")
                
                # Test feature selection
                if len(asset_features.columns) > 5:
                    # Create a dummy target for testing
                    target = asset_features.iloc[:, 0]  # Use first feature as target
                    selected_features = feature_engineer.select_features(
                        asset_features, target, method='correlation', threshold=0.01
                    )
                    logger.info(f"    Selected features: {len(selected_features.columns)}")
            
            return features
        else:
            logger.error("❌ No features created")
            return None
            
    except Exception as e:
        logger.error(f"❌ Feature engineering test failed: {str(e)}")
        return None

def test_walk_forward_validation(features):
    """Test walk-forward validation functionality."""
    logger = logging.getLogger(__name__)
    logger.info("\n" + "="*50)
    logger.info("TESTING WALK-FORWARD VALIDATION")
    logger.info("="*50)
    
    try:
        # Initialize validator
        validator = WalkForwardValidator(
            n_splits=3,
            test_size=30,
            min_train_size=50,
            min_test_size=10
        )
        logger.info("✅ WalkForwardValidator initialized successfully")
        
        # Test with first asset's features
        if features:
            asset_name = list(features.keys())[0]
            asset_features = features[asset_name]
            
            logger.info(f"Testing validation with {asset_name} features: {len(asset_features)} samples")
            
            # Generate splits
            splits = validator.split(asset_features)
            logger.info(f"✅ Generated {len(splits)} splits")
            
            if splits:
                # Get split information
                split_info = validator.get_split_info(splits)
                logger.info(f"  Train sizes: {split_info['train_sizes']['min']}-{split_info['train_sizes']['max']}")
                logger.info(f"  Test sizes: {split_info['test_sizes']['min']}-{split_info['test_sizes']['max']}")
                
                # Test data quality validation
                quality_report = validator.validate_data_quality(asset_features)
                logger.info(f"  Data quality: {quality_report['is_valid']}")
                if not quality_report['is_valid']:
                    logger.warning(f"  Issues: {quality_report['issues']}")
                
                # Test time series split iterator
                target = asset_features.iloc[:, 0]  # Use first feature as target
                split_count = 0
                for X_train, X_test, y_train, y_test in validator.create_time_series_split(asset_features, target):
                    split_count += 1
                    logger.info(f"  Split {split_count}: Train={X_train.shape}, Test={X_test.shape}")
                
                # Get validation summary
                summary = validator.get_validation_summary(asset_features, target)
                logger.info(f"  Validation status: {summary['validation_status']}")
                
                return True
            else:
                logger.error("❌ No splits generated")
                return False
        else:
            logger.error("❌ No features available for validation test")
            return False
            
    except Exception as e:
        logger.error(f"❌ Walk-forward validation test failed: {str(e)}")
        return False

def test_results_management():
    """Test results management functionality."""
    logger = logging.getLogger(__name__)
    logger.info("\n" + "="*50)
    logger.info("TESTING RESULTS MANAGEMENT")
    logger.info("="*50)
    
    try:
        # Initialize results manager
        results_manager = ResultsManager()
        logger.info("✅ ResultsManager initialized successfully")
        
        # Test storing results
        test_results = {
            'test_metric': 0.85,
            'test_predictions': np.random.rand(100),
            'test_model': 'test_model_name'
        }
        
        results_manager.store_results(test_results, asset='AAPL')
        logger.info("✅ Results stored successfully")
        
        # Test retrieving results
        retrieved_results = results_manager.get_results('AAPL')
        if retrieved_results:
            logger.info("✅ Results retrieved successfully")
            logger.info(f"  Stored metrics: {list(retrieved_results.keys())}")
        else:
            logger.error("❌ Failed to retrieve results")
            return False
        
        # Test metrics aggregation
        # Add more test results
        results_manager.store_results({'metric2': 0.92}, asset='MSFT')
        results_manager.store_results({'metric3': 0.78}, asset='GOOGL')
        
        all_metrics = results_manager.get_all_metrics()
        if not all_metrics.empty:
            logger.info("✅ Metrics aggregation successful")
            logger.info(f"  Total metrics: {len(all_metrics)}")
        else:
            logger.warning("⚠️ No metrics to aggregate")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Results management test failed: {str(e)}")
        return False

def test_integration_workflow():
    """Test the complete integration workflow."""
    logger = logging.getLogger(__name__)
    logger.info("\n" + "="*50)
    logger.info("TESTING COMPLETE INTEGRATION WORKFLOW")
    logger.info("="*50)
    
    try:
        # Step 1: Configuration
        if not test_configuration():
            return False
        
        # Step 2: Data Loading
        data = test_data_loading()
        if data is None:
            return False
        
        # Step 3: Feature Engineering
        features = test_feature_engineering(data)
        if features is None:
            return False
        
        # Step 4: Walk-Forward Validation
        if not test_walk_forward_validation(features):
            return False
        
        # Step 5: Results Management
        if not test_results_management():
            return False
        
        logger.info("\n" + "="*50)
        logger.info("🎉 ALL INTEGRATION TESTS PASSED!")
        logger.info("="*50)
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration workflow test failed: {str(e)}")
        return False

def cleanup_test_environment():
    """Clean up test environment."""
    logger = logging.getLogger(__name__)
    logger.info("\n" + "="*50)
    logger.info("CLEANING UP TEST ENVIRONMENT")
    logger.info("="*50)
    
    try:
        import shutil
        
        # Remove test directories
        test_dirs = ['test_output', 'test_cache', 'test_models']
        for dir_name in test_dirs:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
                logger.info(f"✅ Removed {dir_name}")
        
        logger.info("✅ Cleanup completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {str(e)}")

def main():
    """Main integration test function."""
    logger = setup_test_environment()
    
    try:
        # Run integration tests
        success = test_integration_workflow()
        
        if success:
            logger.info("\n🎉 INTEGRATION TEST SUMMARY: SUCCESS")
            logger.info("All core components are working correctly together.")
        else:
            logger.error("\n❌ INTEGRATION TEST SUMMARY: FAILED")
            logger.error("Some components failed to work together.")
            return 1
        
        # Ask user if they want to clean up
        cleanup = input("\nDo you want to clean up test files? (y/n): ").lower().strip()
        if cleanup == 'y':
            cleanup_test_environment()
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Integration test interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\n❌ Integration test failed with unexpected error: {str(e)}")
        return 1
    finally:
        logger.info(f"\nTest completed at: {datetime.now()}")

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 