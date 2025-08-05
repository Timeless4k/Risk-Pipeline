# RiskPipeline Orchestration & Integration Guide

## Overview

The RiskPipeline orchestrator provides a comprehensive, modular framework for volatility forecasting with advanced features including experiment management, SHAP analysis, model persistence, and full reproducibility. This guide covers the new enhanced RiskPipeline class and its integration capabilities.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [New RiskPipeline Features](#new-riskpipeline-features)
3. [CLI Interface](#cli-interface)
4. [Experiment Management](#experiment-management)
5. [Advanced Usage](#advanced-usage)
6. [Integration Examples](#integration-examples)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting](#troubleshooting)

## Architecture Overview

The new RiskPipeline orchestrator coordinates all modular components:

```
RiskPipeline Orchestrator
├── Core Components
│   ├── DataLoader (data acquisition)
│   ├── FeatureEngineer (feature creation)
│   ├── WalkForwardValidator (model evaluation)
│   └── ResultsManager (experiment tracking)
├── Model Components
│   ├── ModelFactory (model creation)
│   ├── BaseModel (unified interface)
│   └── ModelPersistence (artifact management)
├── Interpretability Components
│   ├── SHAPAnalyzer (SHAP analysis)
│   ├── ExplainerFactory (explainer creation)
│   └── SHAPVisualizer (SHAP plots)
└── Utility Components
    ├── ExperimentTracker (versioning)
    ├── VolatilityVisualizer (standard plots)
    └── LoggingUtils (logging management)
```

## New RiskPipeline Features

### 1. Experiment Management

The orchestrator now supports comprehensive experiment tracking:

```python
from risk_pipeline import RiskPipeline

# Initialize with experiment tracking
pipeline = RiskPipeline(
    config_path="configs/pipeline_config.json",
    experiment_name="thesis_experiment_final"
)

# Run complete pipeline with experiment management
results = pipeline.run_complete_pipeline(
    assets=['AAPL', 'MSFT', '^GSPC'],
    models=['arima', 'lstm', 'stockmixer', 'xgboost'],
    save_models=True,
    run_shap=True,
    description="Final thesis experiment with all models and SHAP analysis"
)
```

### 2. Advanced Pipeline Modes

#### Quick Test Mode
```python
# Quick test for development and validation
results = pipeline.run_quick_test()
```

#### Models-Only Training
```python
# Train models without SHAP analysis
results = pipeline.train_models_only(
    assets=['AAPL', 'MSFT'],
    models=['xgboost', 'lstm'],
    save=True
)
```

#### Saved Model Analysis
```python
# Analyze previously saved models
results = pipeline.analyze_saved_models(
    experiment_id="experiment_20250805_143022",
    run_additional_shap=True
)
```

#### Experiment Comparison
```python
# Compare multiple experiments
comparison = pipeline.compare_experiments([
    "experiment_20250805_143022",
    "experiment_20250806_091530"
])
```

#### Best Model Retrieval
```python
# Get best performing models across experiments
best_models = pipeline.get_best_models(
    metric="R2",
    task="regression"
)
```

### 3. Memory and Performance Tracking

The orchestrator includes built-in performance monitoring:

```python
# Memory usage is automatically tracked
pipeline._track_memory_usage("Pipeline start")
pipeline._track_memory_usage("Data loading complete")
pipeline._track_memory_usage("Model training complete")

# Access memory usage history
print(f"Peak memory: {max(pipeline.memory_usage):.1f} MB")
```

## CLI Interface

The enhanced CLI supports both backward compatibility and new advanced features:

### Basic Usage (Backward Compatible)

```bash
# Quick test
python run_pipeline.py --quick

# Full pipeline
python run_pipeline.py --full

# Single asset
python run_pipeline.py --asset AAPL

# US markets only
python run_pipeline.py --full --us-only

# Australian markets only
python run_pipeline.py --full --au-only
```

### Advanced Usage (New Features)

```bash
# Complete pipeline with experiment management
python run_pipeline.py --full \
    --save-models \
    --run-shap \
    --experiment-name "thesis_final" \
    --assets AAPL,MSFT,^GSPC \
    --models arima,lstm,stockmixer,xgboost

# Models-only training
python run_pipeline.py --models-only \
    --assets AAPL,MSFT \
    --models xgboost,lstm \
    --no-save-models

# Experiment analysis
python run_pipeline.py --analyze-experiment experiment_20250805_143022 \
    --run-additional-shap

# Experiment comparison
python run_pipeline.py --compare-experiments \
    experiment_20250805_143022,experiment_20250806_091530

# Get best models
python run_pipeline.py --get-best-models \
    --metric R2 \
    --task regression
```

### CLI Options Reference

| Option | Description | Example |
|--------|-------------|---------|
| `--quick` | Quick test mode | `--quick` |
| `--full` | Full pipeline | `--full` |
| `--asset` | Single asset | `--asset AAPL` |
| `--models-only` | Models-only training | `--models-only` |
| `--analyze-experiment` | Analyze saved experiment | `--analyze-experiment exp123` |
| `--compare-experiments` | Compare experiments | `--compare-experiments exp1,exp2` |
| `--get-best-models` | Get best models | `--get-best-models` |
| `--assets` | Asset list | `--assets AAPL,MSFT,^GSPC` |
| `--models` | Model list | `--models arima,lstm,xgboost` |
| `--experiment-name` | Experiment name | `--experiment-name "my_experiment"` |
| `--save-models` | Save models (default) | `--save-models` |
| `--no-save-models` | Don't save models | `--no-save-models` |
| `--run-shap` | Run SHAP analysis (default) | `--run-shap` |
| `--no-shap` | Don't run SHAP | `--no-shap` |
| `--run-additional-shap` | Additional SHAP for analysis | `--run-additional-shap` |
| `--metric` | Metric for best models | `--metric R2` |
| `--task` | Task type | `--task regression` |
| `--config` | Config file path | `--config my_config.json` |

## Experiment Management

### Experiment Structure

Experiments are organized in a structured directory:

```
experiments/
├── experiment_20250805_143022_abc123/
│   ├── config.json                    # Complete configuration
│   ├── metadata.json                  # Experiment metadata
│   ├── results_summary.csv            # Performance metrics
│   ├── models/
│   │   ├── AAPL_lstm_regression/
│   │   │   ├── model.h5               # Model weights
│   │   │   ├── scaler.pkl             # Feature scaler
│   │   │   ├── feature_names.json     # Feature list
│   │   │   ├── config.json            # Model hyperparameters
│   │   │   ├── metrics.json           # Performance metrics
│   │   │   └── predictions.csv        # Actual vs predicted
│   │   └── AAPL_xgboost_classification/
│   │       ├── model.pkl
│   │       ├── scaler.pkl
│   │       ├── feature_names.json
│   │       └── ...
│   ├── shap_data/
│   │   ├── AAPL_lstm_regression/
│   │   │   ├── shap_values.pkl        # Raw SHAP values
│   │   │   ├── background_data.pkl    # SHAP background
│   │   │   ├── explainer_config.json  # Explainer settings
│   │   │   └── feature_importance.csv # Summary importance
│   │   └── ...
│   ├── visualizations/                 # All plots and charts
│   └── logs/                          # Experiment logs
└── experiment_index.json              # Index of all experiments
```

### Experiment Lifecycle

```python
# 1. Start experiment
experiment_id = results_manager.start_experiment(
    name="my_experiment",
    config=pipeline_config,
    description="Experiment description"
)

# 2. Save model results
results_manager.save_model_results(
    asset="AAPL",
    model_name="lstm",
    task="regression",
    metrics={"R2": 0.85, "MAE": 0.12},
    predictions={"actual": y_true, "predicted": y_pred},
    model=trained_model,
    scaler=feature_scaler,
    feature_names=feature_names,
    config=model_config
)

# 3. Save SHAP results
results_manager.save_shap_results(
    asset="AAPL",
    model_name="lstm",
    shap_values=shap_values,
    explainer_metadata=explainer_config,
    feature_importance=feature_importance
)

# 4. Save experiment metadata
results_manager.save_experiment_metadata({
    'assets_processed': 3,
    'models_run': ['arima', 'lstm', 'xgboost'],
    'shap_analysis': True,
    'execution_time_minutes': 45.2,
    'peak_memory_mb': 2048.5
})

# 5. Load experiment for analysis
experiment_data = results_manager.load_experiment(experiment_id)
```

## Advanced Usage

### Custom Configuration

```python
# Create custom configuration
config = {
    "data": {
        "start_date": "2017-01-01",
        "end_date": "2024-03-31",
        "all_assets": ["AAPL", "MSFT", "^GSPC", "IOZ.AX", "CBA.AX"],
        "cache_dir": "data/cache"
    },
    "training": {
        "walk_forward_splits": 5,
        "test_size": 252,
        "random_state": 42
    },
    "models": {
        "enabled": ["arima", "lstm", "stockmixer", "xgboost"],
        "hyperparameters": {
            "lstm": {"units": [50, 30], "dropout": 0.2},
            "stockmixer": {"temporal_units": 64}
        }
    },
    "shap": {
        "enabled": True,
        "background_samples": 100,
        "save_raw_values": True,
        "generate_interactions": True
    }
}

# Save configuration
with open("my_config.json", "w") as f:
    json.dump(config, f, indent=2)

# Use custom configuration
pipeline = RiskPipeline(config_path="my_config.json")
```

### Batch Processing

```python
# Process multiple asset groups
asset_groups = [
    ["AAPL", "MSFT", "GOOGL"],  # Tech stocks
    ["^GSPC", "^VIX", "^TNX"],   # Market indices
    ["IOZ.AX", "CBA.AX", "BHP.AX"]  # Australian stocks
]

for i, assets in enumerate(asset_groups):
    experiment_name = f"batch_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    pipeline = RiskPipeline(experiment_name=experiment_name)
    
    results = pipeline.run_complete_pipeline(
        assets=assets,
        models=['xgboost', 'lstm'],
        save_models=True,
        run_shap=True
    )
```

### Parallel Processing

```python
import multiprocessing as mp
from functools import partial

def run_asset_pipeline(asset, config_path):
    pipeline = RiskPipeline(config_path=config_path)
    return pipeline.run_complete_pipeline(
        assets=[asset],
        models=['xgboost', 'lstm'],
        save_models=True,
        run_shap=False  # Disable SHAP for parallel processing
    )

# Run assets in parallel
assets = ["AAPL", "MSFT", "GOOGL", "AMZN"]
with mp.Pool(processes=4) as pool:
    results = pool.map(
        partial(run_asset_pipeline, config_path="config.json"),
        assets
    )
```

## Integration Examples

### Integration with External Tools

```python
# Integration with MLflow
import mlflow

def run_with_mlflow():
    mlflow.set_experiment("volatility_forecasting")
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "assets": ["AAPL", "MSFT"],
            "models": ["xgboost", "lstm"],
            "walk_forward_splits": 5
        })
        
        # Run pipeline
        pipeline = RiskPipeline(experiment_name="mlflow_integration")
        results = pipeline.run_complete_pipeline(
            assets=["AAPL", "MSFT"],
            models=["xgboost", "lstm"]
        )
        
        # Log metrics
        for asset, asset_results in results.items():
            for task, task_results in asset_results.items():
                for model, model_results in task_results.items():
                    if 'metrics' in model_results:
                        for metric, value in model_results['metrics'].items():
                            mlflow.log_metric(f"{asset}_{model}_{task}_{metric}", value)
        
        # Log artifacts
        mlflow.log_artifacts("experiments/")
```

### Custom Model Integration

```python
# Custom model integration
class CustomModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None
    
    def fit(self, X, y):
        # Custom training logic
        self.model = CustomAlgorithm(**self.config)
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_feature_importance(self):
        return self.model.feature_importances_

# Register custom model
from risk_pipeline.models.model_factory import ModelFactory
ModelFactory.register_model("custom", CustomModel)

# Use custom model
pipeline = RiskPipeline()
results = pipeline.run_complete_pipeline(
    assets=["AAPL"],
    models=["custom", "xgboost"]
)
```

## Performance Optimization

### Memory Management

```python
# Monitor memory usage
import psutil
import gc

def monitor_memory():
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")

# Clean up memory between assets
for asset in assets:
    # Run pipeline for asset
    results = pipeline.run_complete_pipeline(assets=[asset])
    
    # Clean up
    gc.collect()
    monitor_memory()
```

### Caching Strategies

```python
# Enable data caching
pipeline = RiskPipeline()
pipeline.data_loader.enable_caching = True
pipeline.data_loader.cache_dir = "data/cache"

# Cache SHAP results
pipeline.shap_analyzer.cache_explainers = True
pipeline.shap_analyzer.cache_dir = "shap_cache"
```

### Batch Processing for Large Datasets

```python
# Process large datasets in batches
def process_large_dataset(assets, batch_size=5):
    for i in range(0, len(assets), batch_size):
        batch_assets = assets[i:i+batch_size]
        
        pipeline = RiskPipeline(
            experiment_name=f"batch_{i//batch_size + 1}"
        )
        
        results = pipeline.run_complete_pipeline(
            assets=batch_assets,
            models=['xgboost', 'lstm'],
            save_models=True,
            run_shap=False  # Disable SHAP for large batches
        )
        
        # Clean up memory
        gc.collect()
```

## Troubleshooting

### Common Issues

#### 1. Memory Issues

**Problem**: Out of memory errors during large experiments.

**Solution**:
```python
# Reduce batch size
pipeline.run_complete_pipeline(
    assets=assets[:5],  # Process fewer assets
    models=['xgboost'],  # Use fewer models
    run_shap=False  # Disable SHAP analysis
)

# Monitor memory usage
pipeline._track_memory_usage("Checkpoint")
print(f"Memory: {pipeline.memory_usage[-1]:.1f} MB")
```

#### 2. Experiment Loading Issues

**Problem**: Cannot load saved experiments.

**Solution**:
```python
# Check experiment integrity
experiment_id = "experiment_20250805_143022"
experiment_path = Path(f"experiments/{experiment_id}")

if experiment_path.exists():
    # Verify files
    required_files = ["config.json", "metadata.json", "results_summary.csv"]
    for file in required_files:
        if not (experiment_path / file).exists():
            print(f"Missing file: {file}")
else:
    print(f"Experiment not found: {experiment_id}")
```

#### 3. SHAP Analysis Failures

**Problem**: SHAP analysis fails for certain models.

**Solution**:
```python
# Use model-specific explainers
pipeline.explainer_factory.set_explainer_config({
    'lstm': {'background_samples': 50},
    'stockmixer': {'background_samples': 100},
    'xgboost': {'tree_limit': 100}
})

# Disable SHAP for problematic models
results = pipeline.run_complete_pipeline(
    assets=assets,
    models=['xgboost', 'lstm'],  # Skip problematic models
    run_shap=True
)
```

#### 4. Configuration Issues

**Problem**: Configuration validation errors.

**Solution**:
```python
# Validate configuration
from risk_pipeline.core.config import PipelineConfig

try:
    config = PipelineConfig.from_file("config.json")
    config.validate()
except Exception as e:
    print(f"Configuration error: {e}")
    
    # Use default configuration
    config = PipelineConfig()
    pipeline = RiskPipeline(config=config)
```

### Debug Mode

Enable debug mode for detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

pipeline = RiskPipeline()
pipeline.run_complete_pipeline(
    assets=['AAPL'],
    models=['xgboost'],
    debug=True
)
```

### Performance Profiling

```python
import cProfile
import pstats

def profile_pipeline():
    profiler = cProfile.Profile()
    profiler.enable()
    
    pipeline = RiskPipeline()
    results = pipeline.run_complete_pipeline(
        assets=['AAPL'],
        models=['xgboost']
    )
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions

profile_pipeline()
```

## Best Practices

1. **Experiment Naming**: Use descriptive experiment names with timestamps
2. **Configuration Management**: Version control your configuration files
3. **Resource Monitoring**: Monitor memory and CPU usage during large experiments
4. **Incremental Processing**: Process assets in batches for large datasets
5. **Error Handling**: Implement proper error handling and recovery
6. **Documentation**: Document experiment parameters and results
7. **Backup**: Regularly backup experiment results and models
8. **Validation**: Validate results and check for anomalies

## Conclusion

The new RiskPipeline orchestrator provides a comprehensive, production-ready framework for volatility forecasting with advanced features for experiment management, SHAP analysis, and model persistence. The modular architecture ensures maintainability and extensibility while maintaining backward compatibility with existing code.

For more information, see the individual component documentation:
- [Advanced SHAP Analysis](ADVANCED_SHAP_ANALYSIS.md)
- [Model Architecture](MODULAR_ARCHITECTURE_SUMMARY.md)
- [Testing Strategy](TESTING_STRATEGY.md) 