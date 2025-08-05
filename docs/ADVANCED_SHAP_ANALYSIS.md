# Advanced SHAP Analysis for RiskPipeline

## Overview

The Advanced SHAP Analysis system provides comprehensive model interpretability capabilities for all model types in the RiskPipeline. This system supports ARIMA (statistical interpretability), LSTM (sequence-based SHAP), StockMixer (pathway analysis), and XGBoost (tree-based SHAP) models with advanced features including time-series analysis, feature interactions, and data persistence.

## Architecture

```
Advanced SHAP Analysis System
├── ExplainerFactory
│   ├── ARIMAExplainer (Statistical interpretability)
│   ├── StockMixerExplainer (Pathway analysis)
│   ├── LSTM Explainer (DeepExplainer)
│   └── XGBoost Explainer (TreeExplainer)
├── InterpretationUtils
│   ├── Time-series SHAP analysis
│   ├── Feature interaction detection
│   ├── Data persistence utilities
│   └── Statistical analysis tools
├── SHAPAnalyzer
│   ├── Model-specific analysis
│   ├── Individual prediction explanations
│   ├── Comparison analysis
│   └── Integration with pipeline
└── SHAPVisualizer
    ├── Model-specific visualizations
    ├── Time-series plots
    ├── Comparison plots
    └── Interactive visualizations
```

## Components

### 1. ExplainerFactory

The `ExplainerFactory` creates appropriate SHAP explainers for different model types.

#### Features:
- **Model-specific explainers**: Automatic selection based on model type
- **Background data management**: Optimized for deep learning models
- **Error handling**: Graceful fallbacks for incompatible models
- **Memory efficiency**: Subsampling for large datasets

#### Usage:
```python
from risk_pipeline.interpretability.explainer_factory import ExplainerFactory

# Create factory
factory = ExplainerFactory(config)

# Create explainer for XGBoost
explainer = factory.create_explainer(
    model=xgb_model,
    model_type='xgboost',
    task='regression',
    X=feature_data
)

# Create explainer for LSTM
explainer = factory.create_explainer(
    model=lstm_model,
    model_type='lstm',
    task='regression',
    X=feature_data
)
```

#### Model-Specific Explainers:

##### ARIMAExplainer
- **Statistical interpretability**: Coefficient analysis, residuals, diagnostics
- **Time series decomposition**: Trend, seasonal, residual components
- **Forecast confidence intervals**: Uncertainty quantification
- **Model diagnostics**: Ljung-Box test, autocorrelation analysis

##### StockMixerExplainer
- **Pathway analysis**: Temporal, indicator, cross-stock pathway contributions
- **Feature mixing interpretability**: How features interact across pathways
- **DeepExplainer integration**: Neural network interpretability
- **Custom visualizations**: Pathway-specific plots

##### LSTM Explainer
- **DeepExplainer**: Gradient-based SHAP for neural networks
- **Sequence analysis**: Time-step importance analysis
- **Temporal attention**: Which time steps matter most
- **Memory optimization**: Background data subsampling

##### XGBoost Explainer
- **TreeExplainer**: Exact SHAP values for tree-based models
- **Feature interactions**: Tree-specific interpretability
- **Ensemble analysis**: Contribution from individual trees
- **Fast computation**: Optimized for large datasets

### 2. InterpretationUtils

The `InterpretationUtils` provides advanced analysis capabilities beyond basic SHAP values.

#### Features:
- **Time-series SHAP analysis**: Rolling statistics, regime detection
- **Feature interaction analysis**: Pairwise interactions, clustering
- **Data persistence**: Save/load SHAP values and metadata
- **Statistical analysis**: Trend analysis, seasonality detection

#### Usage:
```python
from risk_pipeline.interpretability.interpretation_utils import InterpretationUtils

# Create utils
utils = InterpretationUtils(config)

# Time-series analysis
time_series_results = utils.analyze_time_series_shap(
    shap_values=shap_values,
    X=feature_data,
    feature_names=feature_names,
    time_index=time_index,
    window_size=30
)

# Feature interaction analysis
interaction_results = utils.analyze_feature_interactions(
    shap_values=shap_values,
    X=feature_data,
    feature_names=feature_names,
    top_k=10
)

# Data persistence
utils.save_shap_data(
    shap_values=shap_values,
    metadata=metadata,
    filepath='path/to/save'
)
```

#### Time-Series Analysis:
- **Rolling statistics**: Mean, std, max, min over time windows
- **Temporal importance**: How feature importance changes over time
- **Regime detection**: Automatic detection of structural breaks
- **Seasonality analysis**: Periodic patterns in SHAP values

#### Feature Interaction Analysis:
- **Pairwise interactions**: Correlation between SHAP values
- **Interaction strength**: Quantified interaction metrics
- **Feature clustering**: Hierarchical clustering of related features
- **Top interactions**: Most important feature pairs

### 3. SHAPAnalyzer

The `SHAPAnalyzer` orchestrates comprehensive SHAP analysis for the entire pipeline.

#### Features:
- **Multi-model analysis**: Analyze all models in the pipeline
- **Individual explanations**: Single prediction interpretability
- **Comparison analysis**: Cross-model and cross-asset comparisons
- **Integration**: Seamless integration with existing pipeline

#### Usage:
```python
from risk_pipeline.interpretability.shap_analyzer import SHAPAnalyzer

# Create analyzer
analyzer = SHAPAnalyzer(config, results_manager)

# Analyze all models
shap_results = analyzer.analyze_all_models(features, results)

# Individual prediction explanation
explanation = analyzer.explain_prediction(
    asset='AAPL',
    model_type='xgboost',
    task='regression',
    instance=single_instance,
    feature_names=feature_names
)

# Feature interaction analysis
interactions = analyzer.analyze_feature_interactions(
    asset='AAPL',
    model_type='xgboost',
    task='regression',
    top_k=10
)

# Time-series analysis
time_series = analyzer.generate_time_series_shap(
    asset='AAPL',
    model_type='xgboost',
    task='regression',
    window_size=30
)
```

#### Analysis Capabilities:
- **Global interpretability**: Overall feature importance across dataset
- **Local interpretability**: Individual prediction explanations
- **Temporal analysis**: Time-varying feature importance
- **Cross-model comparison**: Feature importance across different models
- **Asset comparison**: Feature importance across different assets

### 4. SHAPVisualizer

The `SHAPVisualizer` creates comprehensive visualizations for all analysis types.

#### Features:
- **Model-specific plots**: Customized for each model type
- **Time-series visualizations**: Temporal patterns and regime changes
- **Comparison plots**: Cross-model and cross-asset comparisons
- **Interactive plots**: High-quality static and interactive visualizations

#### Usage:
```python
from risk_pipeline.visualization.shap_visualizer import SHAPVisualizer

# Create visualizer
visualizer = SHAPVisualizer(config)

# Comprehensive plots for a model
plots = visualizer.create_comprehensive_plots(
    shap_values=shap_values,
    X=feature_data,
    feature_names=feature_names,
    asset='AAPL',
    model_type='xgboost',
    task='regression'
)

# Comparison plots
comparison_plots = visualizer.create_comparison_plots(
    shap_results=shap_results,
    assets=['AAPL', 'MSFT'],
    model_types=['xgboost', 'lstm'],
    task='regression'
)
```

#### Visualization Types:

##### Basic SHAP Plots:
- **Summary plots**: Bar charts of feature importance
- **Beeswarm plots**: Distribution of SHAP values
- **Waterfall plots**: Individual prediction breakdowns
- **Dependence plots**: Feature value vs SHAP value relationships

##### Model-Specific Plots:
- **ARIMA**: Residuals, decomposition, forecast intervals
- **StockMixer**: Pathway activation, feature mixing
- **LSTM**: Temporal heatmaps, importance over time
- **XGBoost**: Dependence plots, interaction matrices

##### Time-Series Plots:
- **Rolling statistics**: Moving averages of SHAP values
- **Regime detection**: Structural break identification
- **Seasonality**: Periodic pattern visualization
- **Temporal importance**: Feature importance evolution

##### Comparison Plots:
- **Feature importance comparison**: Across models and assets
- **Performance comparison**: Model interpretability metrics
- **Asset comparison**: Cross-asset feature importance
- **Interaction comparison**: Feature interaction patterns

## Integration with RiskPipeline

### Configuration

Add SHAP analysis configuration to your pipeline config:

```python
from risk_pipeline.core.config import PipelineConfig

config = PipelineConfig()

# SHAP analysis settings
config.shap.background_samples = 100
config.shap.max_display = 20
config.shap.plot_types = ['summary', 'beeswarm', 'waterfall', 'heatmap']

# Output settings
config.output.shap_dir = 'shap_analysis'
config.output.plot_format = 'png'
config.output.plot_dpi = 300
```

### Pipeline Integration

```python
from risk_pipeline import RiskPipeline

# Initialize pipeline
pipeline = RiskPipeline(config)

# Run pipeline with SHAP analysis
results = pipeline.run_pipeline(
    assets=['AAPL', 'MSFT'],
    include_shap_analysis=True
)

# Access SHAP results
shap_results = results['shap_analysis']

# Get feature importance for specific model
importance = pipeline.get_feature_importance(
    asset='AAPL',
    model_type='xgboost',
    task='regression',
    top_n=10
)

# Generate individual explanation
explanation = pipeline.explain_prediction(
    asset='AAPL',
    model_type='xgboost',
    task='regression',
    instance=test_instance
)
```

### Data Storage

SHAP analysis results are stored in a structured format:

```
shap_analysis/
├── AAPL/
│   ├── xgboost/
│   │   ├── regression/
│   │   │   ├── shap_values.pkl
│   │   │   ├── metadata.json
│   │   │   └── plots/
│   │   │       ├── summary.png
│   │   │       ├── beeswarm.png
│   │   │       └── waterfall.png
│   │   └── classification/
│   └── lstm/
├── MSFT/
└── comparisons/
    ├── feature_importance_comparison.png
    ├── performance_comparison.png
    └── asset_comparison.png
```

## Advanced Features

### 1. Time-Series SHAP Analysis

```python
# Analyze temporal patterns in feature importance
time_series_results = analyzer.generate_time_series_shap(
    asset='AAPL',
    model_type='xgboost',
    task='regression',
    window_size=30
)

# Access rolling statistics
rolling_mean = time_series_results['rolling_stats']['rolling_mean']
rolling_std = time_series_results['rolling_stats']['rolling_std']

# Detect regime changes
change_points = time_series_results['regime_changes']['change_points']
rolling_importance = time_series_results['regime_changes']['rolling_importance']

# Analyze seasonality
trend = time_series_results['seasonality']['trend']
seasonal = time_series_results['seasonality']['seasonal']
```

### 2. Feature Interaction Analysis

```python
# Analyze feature interactions
interactions = analyzer.analyze_feature_interactions(
    asset='AAPL',
    model_type='xgboost',
    task='regression',
    top_k=10
)

# Get top interactions
top_interactions = interactions['top_interactions']
for feature1, feature2, strength in top_interactions:
    print(f"{feature1} ↔ {feature2}: {strength:.3f}")

# Get feature clusters
clusters = interactions['interaction_patterns']['feature_clusters']
for i, cluster in enumerate(clusters):
    print(f"Cluster {i}: {cluster}")
```

### 3. Individual Prediction Explanations

```python
# Explain a single prediction
explanation = analyzer.explain_prediction(
    asset='AAPL',
    model_type='xgboost',
    task='regression',
    instance=test_instance,
    feature_names=feature_names,
    instance_index=0
)

# Access feature contributions
contributions = explanation['feature_contributions']
for feature, contribution in contributions.items():
    print(f"{feature}: {contribution:.4f}")

# Get total contribution
total_contribution = explanation['total_contribution']
base_value = explanation['base_value']
```

### 4. Cross-Model Comparison

```python
# Compare feature importance across models
comparison = analyzer.compare_feature_importance(
    asset='AAPL',
    task='regression',
    model_types=['xgboost', 'lstm', 'stockmixer']
)

# Convert to DataFrame for analysis
comparison_df = comparison.pivot_table(
    values='importance',
    index='feature',
    columns='model_type'
)

# Find consistent features
consistent_features = comparison_df.std(axis=1).sort_values()
```

## Performance Optimization

### Memory Management

```python
# Configure background sample size for deep learning models
config.shap.background_samples = 50  # Reduce for memory constraints

# Use subsampling for large datasets
config.shap.subsample_size = 1000  # Maximum samples for analysis

# Enable memory-efficient processing
config.shap.memory_efficient = True
```

### Parallel Processing

```python
# Enable parallel SHAP computation
config.shap.parallel = True
config.shap.n_jobs = -1  # Use all available cores

# Batch processing for multiple assets
config.shap.batch_size = 10  # Process 10 assets at a time
```

### Caching

```python
# Enable SHAP value caching
config.shap.cache_results = True
config.shap.cache_dir = 'shap_cache'

# Cache explainers for reuse
config.shap.cache_explainers = True
```

## Best Practices

### 1. Model-Specific Considerations

#### ARIMA Models:
- Use statistical interpretability for coefficient analysis
- Focus on residual diagnostics and model assumptions
- Consider time series decomposition for trend analysis

#### LSTM Models:
- Use appropriate background data sampling
- Consider sequence length for temporal analysis
- Monitor memory usage for large models

#### StockMixer Models:
- Leverage pathway analysis for interpretability
- Analyze feature mixing patterns
- Use custom visualizations for pathway contributions

#### XGBoost Models:
- Use TreeExplainer for exact SHAP values
- Analyze feature interactions through tree structure
- Consider feature importance stability

### 2. Time-Series Analysis

```python
# Choose appropriate window size
window_size = min(30, len(data) // 10)  # Adaptive window size

# Handle seasonality
seasonal_period = 252  # Daily data, annual seasonality

# Detect regime changes
change_threshold = 2.0  # Standard deviations for regime detection
```

### 3. Feature Engineering

```python
# Ensure feature names are meaningful
feature_names = [
    'returns_lag_1', 'volatility_lag_1', 'vix',
    'correlation_sp500', 'market_regime'
]

# Handle categorical features
categorical_features = ['market_regime', 'sector']
numerical_features = [f for f in feature_names if f not in categorical_features]
```

### 4. Visualization

```python
# Choose appropriate plot types
plot_types = ['summary', 'beeswarm']  # Start with basic plots

# Add advanced plots as needed
if config.shap.advanced_plots:
    plot_types.extend(['waterfall', 'dependence', 'interaction'])

# Use consistent styling
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
```

## Troubleshooting

### Common Issues

#### Memory Errors:
```python
# Reduce background sample size
config.shap.background_samples = 25

# Use subsampling
config.shap.subsample_size = 500

# Enable memory-efficient processing
config.shap.memory_efficient = True
```

#### Slow Computation:
```python
# Enable parallel processing
config.shap.parallel = True
config.shap.n_jobs = 4

# Use caching
config.shap.cache_results = True

# Reduce analysis scope
config.shap.max_display = 10
```

#### Visualization Issues:
```python
# Check matplotlib backend
import matplotlib
matplotlib.use('Agg')  # For headless environments

# Increase DPI for better quality
config.output.plot_dpi = 300

# Use different plot format
config.output.plot_format = 'svg'
```

### Debug Mode

```python
# Enable debug logging
import logging
logging.getLogger('risk_pipeline.interpretability').setLevel(logging.DEBUG)

# Enable verbose output
config.shap.verbose = True

# Save intermediate results
config.shap.save_intermediate = True
```

## Examples

See the `examples/advanced_shap_analysis.py` file for comprehensive usage examples demonstrating all features of the Advanced SHAP Analysis system.

## API Reference

For detailed API documentation, see the docstrings in each module:

- `risk_pipeline.interpretability.explainer_factory`
- `risk_pipeline.interpretability.interpretation_utils`
- `risk_pipeline.interpretability.shap_analyzer`
- `risk_pipeline.visualization.shap_visualizer`

## Contributing

To extend the Advanced SHAP Analysis system:

1. Add new explainer types to `ExplainerFactory`
2. Implement model-specific analysis in `InterpretationUtils`
3. Add visualization methods to `SHAPVisualizer`
4. Update tests in `tests/interpretability/test_shap_analysis.py`
5. Update documentation and examples

## License

This Advanced SHAP Analysis system is part of the RiskPipeline project and follows the same license terms. 