#!/usr/bin/env python3
"""
Example usage of the RiskPipeline modular architecture.

This script demonstrates how to use the new modular architecture
for volatility forecasting with comprehensive SHAP analysis.
"""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from risk_pipeline import RiskPipeline, create_pipeline
from risk_pipeline.core.config import PipelineConfig


def example_basic_usage():
    """Example of basic pipeline usage."""
    print("=== Basic Pipeline Usage ===")
    
    # Create pipeline with default configuration
    pipeline = create_pipeline()
    
    # Run pipeline for specific assets
    results = pipeline.run_pipeline(
        assets=['AAPL', 'MSFT'],
        skip_correlations=False,
        debug=True
    )
    
    print(f"Pipeline completed. Results for {len(results)} assets.")
    return results


def example_custom_configuration():
    """Example with custom configuration."""
    print("\n=== Custom Configuration Example ===")
    
    # Create custom configuration
    custom_config = {
        'data': {
            'start_date': '2020-01-01',
            'end_date': '2023-12-31',
            'us_assets': ['AAPL', 'MSFT', 'GOOGL'],
            'au_assets': ['CBA.AX', 'BHP.AX']
        },
        'training': {
            'walk_forward_splits': 3,
            'test_size': 30,
            'epochs': 50
        },
        'models': {
            'lstm_units': [64, 32],
            'xgboost_n_estimators': 50
        },
        'shap': {
            'background_samples': 50,
            'plot_type': 'bar'
        }
    }
    
    # Create pipeline with custom config
    config = PipelineConfig(custom_config)
    pipeline = RiskPipeline(config=config)
    
    # Run pipeline
    results = pipeline.run_pipeline()
    
    print(f"Custom pipeline completed with {len(results)} assets.")
    return results


def example_model_management():
    """Example of model management features."""
    print("\n=== Model Management Example ===")
    
    pipeline = create_pipeline()
    
    # Get results manager
    results_manager = pipeline.results_manager
    
    # List all assets with results
    assets = results_manager.list_assets()
    print(f"Assets with results: {assets}")
    
    # Get best model for a specific asset and task
    if assets:
        asset = assets[0]
        best_model = results_manager.get_best_model(asset, 'regression', 'mse')
        if best_model:
            print(f"Best model for {asset}: {best_model['model_type']}")
            print(f"Metrics: {best_model['metrics']}")
    
    # Get all metrics as DataFrame
    metrics_df = results_manager.get_all_metrics()
    if not metrics_df.empty:
        print(f"\nAll metrics summary:")
        print(metrics_df.head())
    
    return results_manager


def example_shap_analysis():
    """Example of SHAP analysis features."""
    print("\n=== SHAP Analysis Example ===")
    
    pipeline = create_pipeline()
    
    # Get SHAP analyzer
    shap_analyzer = pipeline.shap_analyzer
    
    # Get feature importance for a specific model
    # (This would work after running the pipeline)
    try:
        importance = shap_analyzer.get_feature_importance(
            asset='AAPL',
            model_type='lstm',
            task='regression',
            top_n=10
        )
        print(f"Top 10 features for AAPL LSTM model:")
        for feature, score in importance.items():
            print(f"  {feature}: {score:.4f}")
    except Exception as e:
        print(f"SHAP analysis not available yet: {e}")
    
    return shap_analyzer


def example_model_persistence():
    """Example of model persistence features."""
    print("\n=== Model Persistence Example ===")
    
    pipeline = create_pipeline()
    
    # Get model persistence utility
    model_persistence = pipeline.model_persistence
    
    # List all saved models
    models_info = model_persistence.list_models()
    print(f"Saved models: {len(models_info)} assets")
    
    # Get storage information
    storage_info = model_persistence.get_storage_info()
    print(f"Storage info:")
    print(f"  Total models: {storage_info['total_models']}")
    print(f"  Total size: {storage_info['total_size_mb']:.2f} MB")
    
    return model_persistence


def example_configuration_management():
    """Example of configuration management features."""
    print("\n=== Configuration Management Example ===")
    
    # Create configuration from file
    try:
        config = PipelineConfig.from_file('configs/pipeline_config.json')
        print("Loaded configuration from file")
    except FileNotFoundError:
        config = PipelineConfig()
        print("Using default configuration")
    
    # Update configuration
    config.update({
        'training': {'epochs': 100},
        'models': {'lstm_dropout': 0.3}
    })
    
    # Get model-specific configuration
    lstm_config = config.get_model_config('lstm')
    print(f"LSTM configuration: {lstm_config}")
    
    # Validate configuration
    is_valid = config.validate()
    print(f"Configuration valid: {is_valid}")
    
    return config


def main():
    """Main example function."""
    print("RiskPipeline Modular Architecture - Usage Examples")
    print("=" * 60)
    
    try:
        # Run examples
        results = example_basic_usage()
        
        config = example_configuration_management()
        
        results_manager = example_model_management()
        
        shap_analyzer = example_shap_analysis()
        
        model_persistence = example_model_persistence()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("\nKey benefits of the modular architecture:")
        print("✅ Centralized configuration management")
        print("✅ Comprehensive results management")
        print("✅ Standardized model interfaces")
        print("✅ Advanced SHAP analysis capabilities")
        print("✅ Robust model persistence")
        print("✅ Easy extensibility and testing")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("This is expected if the pipeline hasn't been run yet.")
        print("The examples demonstrate the architecture structure.")


if __name__ == "__main__":
    main() 