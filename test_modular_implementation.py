"""
Test script to verify the modular RiskPipeline implementation.
"""

import sys
import json
from pathlib import Path

def test_imports():
    """Test that all modules can be imported successfully."""
    print("🔍 Testing module imports...")
    
    try:
        # Test core modules
        from risk_pipeline.core import config, data_loader, feature_engineer, results_manager, validator
        print("✅ Core modules imported successfully")
        
        # Test models
        from risk_pipeline.models import BaseModel, ARIMAModel, LSTMModel, XGBoostModel, StockMixerModel, ModelFactory
        print("✅ Model modules imported successfully")
        
        # Test interpretability
        from risk_pipeline.interpretability import explainer_factory, interpretation_utils, shap_analyzer
        print("✅ Interpretability modules imported successfully")
        
        # Test utils
        from risk_pipeline.utils import experiment_tracking, file_utils, logging_utils, metrics, model_persistence
        print("✅ Utils modules imported successfully")
        
        # Test visualization
        from risk_pipeline.visualization import VolatilityVisualizer, SHAPVisualizer
        print("✅ Visualization modules imported successfully")
        
        # Test main pipeline
        from risk_pipeline import RiskPipeline
        print("✅ Main RiskPipeline imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_configuration():
    """Test configuration file loading."""
    print("\n🔍 Testing configuration files...")
    
    config_files = [
        'configs/pipeline_config.json',
        'configs/quick_test_config.json',
        'configs/full_pipeline_config.json'
    ]
    
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Check required sections
            required_sections = ['data', 'features', 'models', 'training', 'output']
            for section in required_sections:
                if section not in config:
                    print(f"❌ Missing section '{section}' in {config_file}")
                    return False
            
            print(f"✅ {config_file} loaded successfully")
            
        except Exception as e:
            print(f"❌ Failed to load {config_file}: {e}")
            return False
    
    return True

def test_model_factory():
    """Test model factory functionality."""
    print("\n🔍 Testing model factory...")
    
    try:
        from risk_pipeline.models import ModelFactory
        
        # Load config
        with open('configs/pipeline_config.json', 'r') as f:
            config = json.load(f)
        
        # Initialize factory
        factory = ModelFactory(config=config)
        
        # Test available models
        available_models = factory.get_available_models()
        expected_models = ['arima', 'lstm', 'xgboost', 'stockmixer']
        
        for model in expected_models:
            if model not in available_models:
                print(f"❌ Model '{model}' not available in factory")
                return False
        
        print("✅ Model factory initialized successfully")
        print(f"✅ Available models: {available_models}")
        
        # Test model creation
        for model_type in ['lstm', 'xgboost']:
            model = factory.create_model(model_type, task='regression')
            print(f"✅ Created {model_type} model: {model}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model factory test failed: {e}")
        return False

def test_visualization():
    """Test visualization components."""
    print("\n🔍 Testing visualization components...")
    
    try:
        from risk_pipeline.visualization import VolatilityVisualizer, SHAPVisualizer
        from risk_pipeline.core.config import PipelineConfig
        
        # Test VolatilityVisualizer
        viz = VolatilityVisualizer()
        print("✅ VolatilityVisualizer created successfully")
        
        # Test SHAPVisualizer with config
        config = PipelineConfig()
        shap_viz = SHAPVisualizer(config)
        print("✅ SHAPVisualizer created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False

def test_pipeline_initialization():
    """Test pipeline initialization."""
    print("\n🔍 Testing pipeline initialization...")
    
    try:
        from risk_pipeline import RiskPipeline
        
        # Create a temporary config file with proper logging setup
        temp_config = {
            "data": {
                "start_date": "2020-01-01",
                "end_date": "2024-01-01",
                "us_assets": ["AAPL"],
                "au_assets": ["CBA.AX"],
                "cache_dir": "data_cache"
            },
            "features": {
                "volatility_window": 5,
                "ma_short": 10,
                "ma_long": 50,
                "correlation_window": 30,
                "sequence_length": 15
            },
            "models": {
                "lstm_units": [30, 20],
                "lstm_dropout": 0.2,
                "stockmixer_temporal_units": 32,
                "stockmixer_indicator_units": 32,
                "stockmixer_cross_stock_units": 32,
                "stockmixer_fusion_units": 64,
                "xgboost_n_estimators": 50,
                "xgboost_max_depth": 3,
                "xgboost_learning_rate": 0.1,
                "arima_order": [1, 1, 1]
            },
            "training": {
                "walk_forward_splits": 2,
                "test_size": 50,
                "batch_size": 16,
                "epochs": 10,
                "early_stopping_patience": 3,
                "reduce_lr_patience": 3,
                "random_state": 42
            },
            "output": {
                "results_dir": "results",
                "plots_dir": "visualizations",
                "shap_dir": "shap_plots",
                "models_dir": "models",
                "log_dir": "logs/test_pipeline.log"
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "date_format": "%Y-%m-%d %H:%M:%S"
            },
            "shap": {
                "background_samples": 50,
                "max_display": 10,
                "plot_type": "bar",
                "save_plots": True
            }
        }
        
        # Write temporary config file
        temp_config_path = "temp_test_config.json"
        with open(temp_config_path, 'w') as f:
            json.dump(temp_config, f, indent=4)
        
        try:
            # Initialize pipeline with temporary config
            pipeline = RiskPipeline(config_path=temp_config_path)
            print("✅ RiskPipeline initialized successfully")
            
            # Clean up
            import os
            os.remove(temp_config_path)
            
            return True
            
        except Exception as e:
            # Clean up on error
            import os
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
            raise e
        
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        return False

def test_directory_structure():
    """Test that all required directories exist."""
    print("\n🔍 Testing directory structure...")
    
    required_dirs = [
        'risk_pipeline',
        'risk_pipeline/core',
        'risk_pipeline/models',
        'risk_pipeline/interpretability',
        'risk_pipeline/utils',
        'risk_pipeline/visualization',
        'configs'
    ]
    
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            print(f"❌ Directory missing: {dir_path}")
            return False
        else:
            print(f"✅ Directory exists: {dir_path}")
    
    return True

def test_file_structure():
    """Test that all required files exist."""
    print("\n🔍 Testing file structure...")
    
    required_files = [
        'risk_pipeline/__init__.py',
        'risk_pipeline/core/__init__.py',
        'risk_pipeline/core/config.py',
        'risk_pipeline/core/data_loader.py',
        'risk_pipeline/core/feature_engineer.py',
        'risk_pipeline/core/results_manager.py',
        'risk_pipeline/core/validator.py',
        'risk_pipeline/models/__init__.py',
        'risk_pipeline/models/base_model.py',
        'risk_pipeline/models/arima_model.py',
        'risk_pipeline/models/lstm_model.py',
        'risk_pipeline/models/xgboost_model.py',
        'risk_pipeline/models/stockmixer_model.py',
        'risk_pipeline/models/model_factory.py',
        'risk_pipeline/interpretability/__init__.py',
        'risk_pipeline/interpretability/explainer_factory.py',
        'risk_pipeline/interpretability/interpretation_utils.py',
        'risk_pipeline/interpretability/shap_analyzer.py',
        'risk_pipeline/utils/__init__.py',
        'risk_pipeline/utils/experiment_tracking.py',
        'risk_pipeline/utils/file_utils.py',
        'risk_pipeline/utils/logging_utils.py',
        'risk_pipeline/utils/metrics.py',
        'risk_pipeline/utils/model_persistence.py',
        'risk_pipeline/visualization/__init__.py',
        'risk_pipeline/visualization/shap_visualizer.py',
        'risk_pipeline/visualization/volatility_visualizer.py',
        'configs/pipeline_config.json',
        'configs/quick_test_config.json',
        'configs/full_pipeline_config.json',
        'main.py',
        'cli.py',
        'requirements.txt'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ File exists: {file_path}")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("🚀 Testing RiskPipeline Modular Implementation")
    print("=" * 60)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("File Structure", test_file_structure),
        ("Configuration Files", test_configuration),
        ("Module Imports", test_imports),
        ("Model Factory", test_model_factory),
        ("Visualization Components", test_visualization),
        ("Pipeline Initialization", test_pipeline_initialization)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Modular implementation is ready.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 