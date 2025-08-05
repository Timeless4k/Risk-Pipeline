# RiskPipeline Migration Guide - Phase 2: Core Components

## Overview

This document outlines the migration of core data handling components from the monolithic `risk_pipeline.py` to the new modular architecture. Phase 2 focuses on extracting and enhancing the `FeatureEngineer` and `WalkForwardValidator` classes.

## Components Migrated

### 1. FeatureEngineer (`risk_pipeline/core/feature_engineer.py`)

**Original Location**: `risk_pipeline.py` lines 238-480

**New Location**: `risk_pipeline/core/feature_engineer.py`

**Enhancements**:
- **Modular Design**: Split into specialized feature modules (`TechnicalFeatureModule`, `StatisticalFeatureModule`, `TimeFeatureModule`, `LagFeatureModule`, `CorrelationFeatureModule`)
- **Pluggable Architecture**: Easy to add custom feature modules via `BaseFeatureModule`
- **Enhanced Configuration**: `FeatureConfig` dataclass for fine-grained control
- **Feature Selection**: Built-in correlation and variance-based feature selection
- **Quality Validation**: Comprehensive feature validation and quality checks
- **Missing Value Handling**: Robust interpolation and forward/backward filling
- **Feature Summary**: Detailed feature statistics and analysis

**Key Features**:
- Technical indicators (RSI, MACD, Bollinger Bands, ATR)
- Statistical features (volatility, skewness, kurtosis)
- Time-based features (day of week, month, quarter)
- Lag features (configurable lag periods)
- Correlation features (inter-asset correlations)
- VIX integration
- Market regime classification
- Volatility regime labeling

### 2. WalkForwardValidator (`risk_pipeline/core/validator.py`)

**Original Location**: `risk_pipeline.py` lines 588-658

**New Location**: `risk_pipeline/core/validator.py`

**Enhancements**:
- **Adaptive Sizing**: Automatically adjusts test size and number of splits based on data availability
- **Configuration Validation**: `ValidationConfig` with parameter validation
- **Multiple Window Types**: Support for both expanding and sliding windows
- **Gap Support**: Configurable gap between train and test sets
- **Data Quality Checks**: Comprehensive data validation
- **Split Analysis**: Detailed split information and overlap analysis
- **Visualization**: Built-in split visualization capabilities
- **Iterator Interface**: Easy-to-use time series split iterator

**Key Features**:
- Dynamic split generation based on data size
- Minimum train/test size enforcement
- Overlap analysis for train and test sets
- Data quality validation
- Comprehensive split information
- Time series split iterator
- Split visualization

## New Configuration Classes

### FeatureConfig
```python
@dataclass
class FeatureConfig:
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    ma_short: int = 10
    ma_long: int = 50
    volatility_windows: List[int] = None
    correlation_window: int = 30
    regime_window: int = 60
    bull_threshold: float = 0.1
    bear_threshold: float = -0.1
    min_correlation_threshold: float = 0.01
    max_feature_correlation: float = 0.95
```

### ValidationConfig
```python
@dataclass
class ValidationConfig:
    n_splits: int = 5
    test_size: int = 252
    min_train_size: int = 60
    min_test_size: int = 20
    gap: int = 0
    expanding_window: bool = True
```

## Migration Steps

### Step 1: Update Imports

**Before**:
```python
from risk_pipeline import FeatureEngineer, WalkForwardValidator
```

**After**:
```python
from risk_pipeline.core import FeatureEngineer, WalkForwardValidator, FeatureConfig, ValidationConfig
```

### Step 2: Update FeatureEngineer Usage

**Before**:
```python
# Old monolithic usage
feature_engineer = FeatureEngineer(config)
features = feature_engineer.create_technical_features(df)
correlations = feature_engineer.calculate_correlations(data)
```

**After**:
```python
# New modular usage
feature_engineer = FeatureEngineer(config)
features = feature_engineer.create_all_features(data, skip_correlations=False)

# Or create features for individual assets
asset_features = feature_engineer.create_asset_features(df)

# Add custom feature modules
class CustomFeatureModule(BaseFeatureModule):
    def create_features(self, data):
        return pd.DataFrame({'custom': [1, 2, 3]})
    
    def get_feature_names(self):
        return ['custom']
    
    def get_required_columns(self):
        return []

custom_module = CustomFeatureModule(feature_engineer.feature_config)
feature_engineer.add_custom_module('custom', custom_module)
```

### Step 3: Update WalkForwardValidator Usage

**Before**:
```python
# Old usage
validator = WalkForwardValidator(n_splits=5, test_size=252)
splits = validator.split(X)
```

**After**:
```python
# New enhanced usage
validator = WalkForwardValidator(
    n_splits=5,
    test_size=252,
    min_train_size=60,
    min_test_size=20,
    gap=0,
    expanding_window=True
)

# Generate splits
splits = validator.split(X)

# Get split information
split_info = validator.get_split_info(splits)

# Validate data quality
quality_report = validator.validate_data_quality(X, y)

# Use time series iterator
for X_train, X_test, y_train, y_test in validator.create_time_series_split(X, y):
    # Train and evaluate model
    pass

# Get comprehensive summary
summary = validator.get_validation_summary(X, y)
```

### Step 4: Update Configuration

**Before**:
```python
# Old configuration
config = AssetConfig()
```

**After**:
```python
# New configuration
config = PipelineConfig()
feature_config = FeatureConfig(
    rsi_period=21,
    ma_short=5,
    ma_long=20,
    volatility_windows=[10, 20, 30]
)
validation_config = ValidationConfig(
    n_splits=3,
    test_size=100,
    gap=5
)
```

## Testing

### Unit Tests
Run the unit tests for the new components:

```bash
# Test FeatureEngineer
python -m pytest tests/core/test_feature_engineer.py -v

# Test WalkForwardValidator
python -m pytest tests/core/test_validator.py -v
```

### Integration Tests
Run the integration verification script:

```bash
python test_integration.py
```

## Backward Compatibility

### FeatureEngineer Compatibility Layer
The new `FeatureEngineer` maintains backward compatibility for most methods:

```python
# These still work as before
feature_engineer.calculate_log_returns(prices)
feature_engineer.calculate_volatility(returns, window)
feature_engineer.calculate_rsi(prices)
feature_engineer.calculate_macd(prices)
feature_engineer.calculate_atr(df)
feature_engineer.calculate_bollinger(prices)
feature_engineer.add_vix_features(features, vix_data)
feature_engineer.calculate_correlations(data)
feature_engineer.create_regime_labels(returns)
feature_engineer.create_volatility_labels(volatility)
```

### WalkForwardValidator Compatibility Layer
The new `WalkForwardValidator` maintains the core interface:

```python
# These still work as before
validator.split(X)  # Returns List[Tuple[pd.Index, pd.Index]]
```

## New Features and Capabilities

### FeatureEngineer Enhancements

1. **Modular Feature Creation**:
   ```python
   # Access individual modules
   technical_features = feature_engineer.modules['technical'].create_features(df)
   statistical_features = feature_engineer.modules['statistical'].create_features(df)
   time_features = feature_engineer.modules['time'].create_features(df)
   ```

2. **Feature Selection**:
   ```python
   # Select features by correlation
   selected_features = feature_engineer.select_features(
       features, target, method='correlation', threshold=0.01
   )
   
   # Select features by variance
   selected_features = feature_engineer.select_features(
       features, method='variance', threshold=0.01
   )
   ```

3. **Feature Summary**:
   ```python
   summary = feature_engineer.get_feature_summary(features)
   print(f"Total features: {summary['total_features']}")
   print(f"Numeric features: {summary['numeric_features']}")
   ```

4. **Custom Feature Modules**:
   ```python
   # Add custom feature module
   feature_engineer.add_custom_module('my_module', my_custom_module)
   
   # Remove module
   feature_engineer.remove_module('time')
   
   # List available modules
   modules = feature_engineer.list_modules()
   ```

### WalkForwardValidator Enhancements

1. **Adaptive Configuration**:
   ```python
   # Automatically adapts to data size
   validator = WalkForwardValidator(n_splits=10, test_size=100)
   splits = validator.split(small_dataset)  # Will generate fewer splits
   ```

2. **Data Quality Validation**:
   ```python
   quality_report = validator.validate_data_quality(X, y)
   if not quality_report['is_valid']:
       print(f"Issues: {quality_report['issues']}")
   ```

3. **Split Analysis**:
   ```python
   split_info = validator.get_split_info(splits)
   print(f"Train sizes: {split_info['train_sizes']}")
   print(f"Test sizes: {split_info['test_sizes']}")
   print(f"Overlap: {split_info['overlap']}")
   ```

4. **Time Series Iterator**:
   ```python
   for X_train, X_test, y_train, y_test in validator.create_time_series_split(X, y):
       # Train model on X_train, y_train
       # Evaluate on X_test, y_test
       pass
   ```

5. **Visualization**:
   ```python
   validator.plot_splits(X, splits, save_path='splits.png')
   ```

## Performance Improvements

### FeatureEngineer
- **Modular Processing**: Only create features you need
- **Efficient Memory Usage**: Process features in chunks
- **Caching**: Reuse calculated features where possible
- **Parallel Processing**: Support for parallel feature creation (future enhancement)

### WalkForwardValidator
- **Adaptive Sizing**: No wasted computation on impossible splits
- **Early Validation**: Catch data issues before processing
- **Efficient Iterators**: Memory-efficient split iteration
- **Optimized Algorithms**: Improved split generation algorithms

## Error Handling

### FeatureEngineer
- **Input Validation**: Comprehensive validation of input data
- **Graceful Degradation**: Continue processing even if some modules fail
- **Detailed Logging**: Extensive logging for debugging
- **Error Recovery**: Automatic handling of missing data and edge cases

### WalkForwardValidator
- **Configuration Validation**: Validate parameters before use
- **Data Quality Checks**: Detect and report data issues
- **Adaptive Behavior**: Automatically adjust to data constraints
- **Comprehensive Error Messages**: Clear error messages for debugging

## Migration Checklist

- [ ] Update imports to use new modular components
- [ ] Replace old `FeatureEngineer` usage with new modular approach
- [ ] Update `WalkForwardValidator` initialization with new parameters
- [ ] Test feature creation with new modular design
- [ ] Verify walk-forward validation works with new adaptive sizing
- [ ] Run unit tests for new components
- [ ] Run integration tests
- [ ] Update configuration files if needed
- [ ] Test backward compatibility layer
- [ ] Update documentation

## Next Steps

After completing Phase 2 migration:

1. **Phase 3**: Migrate model implementations (LSTM, XGBoost, ARIMA, StockMixer)
2. **Phase 4**: Complete interpretability components (ExplainerFactory, InterpretationUtils)
3. **Phase 5**: Integration and end-to-end testing
4. **Phase 6**: Performance optimization and final validation

## Support

For issues during migration:
1. Check the unit tests for usage examples
2. Run the integration test script
3. Review the backward compatibility layer
4. Check the logging output for detailed error messages
5. Refer to the original monolithic code for reference

## Files Modified/Created

### New Files
- `risk_pipeline/core/feature_engineer.py`
- `risk_pipeline/core/validator.py`
- `tests/core/test_feature_engineer.py`
- `tests/core/test_validator.py`
- `test_integration.py`
- `MIGRATION_PHASE_2.md`

### Modified Files
- `risk_pipeline/core/__init__.py` (updated imports)

### Dependencies
- All existing dependencies from the original codebase
- No new external dependencies added 