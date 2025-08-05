"""
Advanced SHAP Analysis Example for RiskPipeline.

This example demonstrates the comprehensive SHAP analysis capabilities
for all model types in the RiskPipeline.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Import RiskPipeline components
from risk_pipeline.interpretability.explainer_factory import ExplainerFactory
from risk_pipeline.interpretability.interpretation_utils import InterpretationUtils
from risk_pipeline.interpretability.shap_analyzer import SHAPAnalyzer
from risk_pipeline.visualization.shap_visualizer import SHAPVisualizer
from risk_pipeline.core.config import PipelineConfig


def create_mock_data():
    """Create mock data for demonstration."""
    np.random.seed(42)
    
    # Create time series data
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    
    # Create features
    n_features = 15
    feature_names = [
        'returns_lag_1', 'returns_lag_2', 'returns_lag_3',
        'volatility_lag_1', 'volatility_lag_2', 'volatility_lag_3',
        'vix', 'vix_change', 'vix_volatility',
        'correlation_sp500', 'correlation_nasdaq', 'correlation_dow',
        'market_regime', 'sector_rotation', 'macro_indicator'
    ]
    
    # Generate realistic financial data
    returns = np.random.normal(0.001, 0.02, 1000)
    volatility = np.abs(returns) + np.random.normal(0, 0.005, 1000)
    vix = 20 + 10 * np.sin(np.arange(1000) * 2 * np.pi / 252) + np.random.normal(0, 2, 1000)
    
    # Create feature matrix
    X = np.zeros((1000, n_features))
    
    # Lagged returns
    for i in range(3):
        X[:, i] = np.roll(returns, i + 1)
    
    # Lagged volatility
    for i in range(3):
        X[:, i + 3] = np.roll(volatility, i + 1)
    
    # VIX features
    X[:, 6] = vix
    X[:, 7] = np.diff(vix, prepend=vix[0])
    X[:, 8] = np.rolling_std(vix, window=20, min_periods=1)
    
    # Correlation features
    X[:, 9] = np.random.uniform(0.7, 0.9, 1000)  # S&P 500 correlation
    X[:, 10] = np.random.uniform(0.6, 0.8, 1000)  # NASDAQ correlation
    X[:, 11] = np.random.uniform(0.5, 0.7, 1000)  # DOW correlation
    
    # Market regime (0: bear, 1: neutral, 2: bull)
    X[:, 12] = np.random.choice([0, 1, 2], 1000, p=[0.2, 0.6, 0.2])
    
    # Sector rotation
    X[:, 13] = np.random.uniform(-1, 1, 1000)
    
    # Macro indicator
    X[:, 14] = np.random.normal(0, 1, 1000)
    
    # Create target (volatility)
    y = np.sqrt(np.sum(X[:, :6]**2, axis=1)) + np.random.normal(0, 0.01, 1000)
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names, index=dates)
    df['volatility'] = y
    
    return df, feature_names


def create_mock_models():
    """Create mock models for demonstration."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    import xgboost as xgb
    
    # Create mock data
    df, feature_names = create_mock_data()
    X = df[feature_names].values
    y = df['volatility'].values
    
    # Split data
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train models
    models = {}
    
    # XGBoost model
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    models['xgboost'] = xgb_model
    
    # Random Forest model (as proxy for tree-based)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    models['random_forest'] = rf_model
    
    # Linear model (as proxy for ARIMA-like)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    models['linear'] = lr_model
    
    return models, X_test, y_test, feature_names


def demonstrate_explainer_factory():
    """Demonstrate the ExplainerFactory capabilities."""
    print("=== ExplainerFactory Demonstration ===")
    
    # Create configuration
    config = PipelineConfig()
    config.shap.background_samples = 100
    
    # Create factory
    factory = ExplainerFactory(config)
    
    # Create mock models and data
    models, X_test, y_test, feature_names = create_mock_models()
    
    print(f"Created {len(models)} mock models")
    
    # Test explainer creation for each model type
    for model_name, model in models.items():
        print(f"\nCreating explainer for {model_name}...")
        
        try:
            if model_name == 'xgboost':
                explainer = factory.create_explainer(
                    model=model,
                    model_type='xgboost',
                    task='regression',
                    X=X_test
                )
                print(f"✓ XGBoost explainer created successfully")
                
            elif model_name == 'random_forest':
                explainer = factory.create_explainer(
                    model=model,
                    model_type='xgboost',  # Use XGBoost explainer for RF
                    task='regression',
                    X=X_test
                )
                print(f"✓ Random Forest explainer created successfully")
                
            elif model_name == 'linear':
                explainer = factory.create_explainer(
                    model=model,
                    model_type='arima',  # Use ARIMA explainer for linear
                    task='regression',
                    X=X_test
                )
                print(f"✓ Linear model explainer created successfully")
                
        except Exception as e:
            print(f"✗ Failed to create explainer for {model_name}: {str(e)}")
    
    print("\nExplainerFactory demonstration completed!")


def demonstrate_interpretation_utils():
    """Demonstrate the InterpretationUtils capabilities."""
    print("\n=== InterpretationUtils Demonstration ===")
    
    # Create configuration
    config = PipelineConfig()
    config.output.shap_dir = "temp_shap_data"
    
    # Create utils
    utils = InterpretationUtils(config)
    
    # Create mock data
    df, feature_names = create_mock_data()
    X = df[feature_names].values
    y = df['volatility'].values
    
    # Generate mock SHAP values
    np.random.seed(42)
    shap_values = np.random.normal(0, 0.1, (len(X), len(feature_names)))
    
    print("Testing time-series SHAP analysis...")
    
    # Time-series SHAP analysis
    time_series_results = utils.analyze_time_series_shap(
        shap_values=shap_values,
        X=X,
        feature_names=feature_names,
        time_index=df.index,
        window_size=30
    )
    
    print(f"✓ Time-series analysis completed")
    print(f"  - Rolling stats: {len(time_series_results['rolling_stats'])} metrics")
    print(f"  - Temporal importance: {len(time_series_results['temporal_importance'])} features")
    print(f"  - Regime changes: {len(time_series_results['regime_changes']['change_points'])} detected")
    
    print("\nTesting feature interaction analysis...")
    
    # Feature interaction analysis
    interaction_results = utils.analyze_feature_interactions(
        shap_values=shap_values,
        X=X,
        feature_names=feature_names,
        top_k=5
    )
    
    print(f"✓ Feature interaction analysis completed")
    print(f"  - Pairwise interactions: {len(interaction_results['pairwise_interactions'])} pairs")
    print(f"  - Top interactions: {len(interaction_results['top_interactions'])} pairs")
    print(f"  - Feature clusters: {len(interaction_results['interaction_patterns']['feature_clusters'])} clusters")
    
    print("\nTesting SHAP data persistence...")
    
    # Test data persistence
    metadata = {
        'asset': 'AAPL',
        'model_type': 'xgboost',
        'task': 'regression',
        'feature_names': feature_names,
        'timestamp': '2024-01-01T00:00:00'
    }
    
    # Save data
    success = utils.save_shap_data(
        shap_values=shap_values,
        metadata=metadata,
        filepath='temp_shap_data/test_shap'
    )
    
    if success:
        print("✓ SHAP data saved successfully")
        
        # Load data
        loaded_shap, loaded_metadata = utils.load_shap_data('temp_shap_data/test_shap')
        
        if loaded_shap is not None:
            print("✓ SHAP data loaded successfully")
            print(f"  - Shape: {loaded_shap.shape}")
            print(f"  - Asset: {loaded_metadata['asset']}")
        else:
            print("✗ Failed to load SHAP data")
    else:
        print("✗ Failed to save SHAP data")
    
    print("\nInterpretationUtils demonstration completed!")


def demonstrate_shap_analyzer():
    """Demonstrate the SHAPAnalyzer capabilities."""
    print("\n=== SHAPAnalyzer Demonstration ===")
    
    # Create configuration
    config = PipelineConfig()
    config.shap.background_samples = 100
    config.output.shap_dir = "temp_shap_analysis"
    
    # Create mock results manager
    class MockResultsManager:
        def get_model(self, asset, model_type, task):
            models, _, _, _ = create_mock_models()
            return models.get(model_type)
        
        def get_features(self, asset):
            df, feature_names = create_mock_data()
            return {
                'features': df[feature_names].values,
                'feature_names': feature_names,
                'time_index': df.index
            }
        
        def get_shap_results(self, asset=None):
            return {}
        
        def store_shap_results(self, results):
            pass
    
    results_manager = MockResultsManager()
    
    # Create analyzer
    analyzer = SHAPAnalyzer(config, results_manager)
    
    # Create mock data
    df, feature_names = create_mock_data()
    X = df[feature_names].values
    y = df['volatility'].values
    
    print("Testing individual prediction explanation...")
    
    # Test individual prediction explanation
    explanation = analyzer.explain_prediction(
        asset='AAPL',
        model_type='xgboost',
        task='regression',
        instance=X[:1],
        feature_names=feature_names,
        instance_index=0
    )
    
    if 'error' not in explanation:
        print("✓ Individual prediction explanation completed")
        print(f"  - Top feature: {list(explanation['feature_contributions'].keys())[0]}")
        print(f"  - Total contribution: {explanation['total_contribution']:.4f}")
    else:
        print(f"✗ Individual prediction explanation failed: {explanation['error']}")
    
    print("\nTesting feature interaction analysis...")
    
    # Test feature interaction analysis
    interactions = analyzer.analyze_feature_interactions(
        asset='AAPL',
        model_type='xgboost',
        task='regression',
        top_k=5
    )
    
    if 'error' not in interactions:
        print("✓ Feature interaction analysis completed")
        print(f"  - Top interactions: {len(interactions['top_interactions'])} pairs")
    else:
        print(f"✗ Feature interaction analysis failed: {interactions['error']}")
    
    print("\nTesting time-series SHAP analysis...")
    
    # Test time-series SHAP analysis
    time_series = analyzer.generate_time_series_shap(
        asset='AAPL',
        model_type='xgboost',
        task='regression',
        window_size=30
    )
    
    if 'error' not in time_series:
        print("✓ Time-series SHAP analysis completed")
        print(f"  - Rolling stats: {len(time_series['rolling_stats'])} metrics")
        print(f"  - Temporal importance: {len(time_series['temporal_importance'])} features")
    else:
        print(f"✗ Time-series SHAP analysis failed: {time_series['error']}")
    
    print("\nSHAPAnalyzer demonstration completed!")


def demonstrate_shap_visualizer():
    """Demonstrate the SHAPVisualizer capabilities."""
    print("\n=== SHAPVisualizer Demonstration ===")
    
    # Create configuration
    config = PipelineConfig()
    config.output.shap_dir = "temp_shap_plots"
    
    # Create visualizer
    visualizer = SHAPVisualizer(config)
    
    # Create mock data
    df, feature_names = create_mock_data()
    X = df[feature_names].values
    y = df['volatility'].values
    
    # Generate mock SHAP values
    np.random.seed(42)
    shap_values = np.random.normal(0, 0.1, (len(X), len(feature_names)))
    
    print("Creating comprehensive SHAP plots...")
    
    # Create comprehensive plots for different model types
    model_types = ['xgboost', 'lstm', 'stockmixer', 'arima']
    
    for model_type in model_types:
        print(f"\nCreating plots for {model_type}...")
        
        try:
            plots = visualizer.create_comprehensive_plots(
                shap_values=shap_values,
                X=X,
                feature_names=feature_names,
                asset='AAPL',
                model_type=model_type,
                task='regression'
            )
            
            if 'error' not in plots:
                print(f"✓ {model_type} plots created successfully")
                print(f"  - Generated {len(plots)} plot types")
                for plot_type, plot_path in plots.items():
                    if plot_type != 'error':
                        print(f"    - {plot_type}: {Path(plot_path).name}")
            else:
                print(f"✗ {model_type} plots failed: {plots['error']}")
                
        except Exception as e:
            print(f"✗ {model_type} plots failed: {str(e)}")
    
    print("\nCreating comparison plots...")
    
    # Create comparison plots
    shap_results = {
        'AAPL': {
            'regression': {
                'xgboost': {
                    'feature_importance': {
                        'returns_lag_1': 0.15,
                        'volatility_lag_1': 0.12,
                        'vix': 0.10,
                        'correlation_sp500': 0.08,
                        'market_regime': 0.06
                    }
                },
                'lstm': {
                    'feature_importance': {
                        'returns_lag_1': 0.14,
                        'volatility_lag_1': 0.11,
                        'vix': 0.09,
                        'correlation_sp500': 0.07,
                        'market_regime': 0.05
                    }
                }
            }
        }
    }
    
    try:
        comparison_plots = visualizer.create_comparison_plots(
            shap_results=shap_results,
            assets=['AAPL'],
            model_types=['xgboost', 'lstm'],
            task='regression'
        )
        
        if 'error' not in comparison_plots:
            print("✓ Comparison plots created successfully")
            for plot_type, plot_path in comparison_plots.items():
                if plot_type != 'error':
                    print(f"  - {plot_type}: {Path(plot_path).name}")
        else:
            print(f"✗ Comparison plots failed: {comparison_plots['error']}")
            
    except Exception as e:
        print(f"✗ Comparison plots failed: {str(e)}")
    
    print("\nSHAPVisualizer demonstration completed!")


def demonstrate_integration():
    """Demonstrate integration of all components."""
    print("\n=== Integration Demonstration ===")
    
    # Create configuration
    config = PipelineConfig()
    config.shap.background_samples = 100
    config.output.shap_dir = "temp_integration"
    
    # Create all components
    factory = ExplainerFactory(config)
    utils = InterpretationUtils(config)
    visualizer = SHAPVisualizer(config)
    
    # Mock results manager
    class MockResultsManager:
        def get_model(self, asset, model_type, task):
            models, _, _, _ = create_mock_models()
            return models.get(model_type)
        
        def get_features(self, asset):
            df, feature_names = create_mock_data()
            return {
                'features': df[feature_names].values,
                'feature_names': feature_names,
                'time_index': df.index
            }
        
        def get_shap_results(self, asset=None):
            return {}
        
        def store_shap_results(self, results):
            pass
    
    results_manager = MockResultsManager()
    analyzer = SHAPAnalyzer(config, results_manager)
    
    # Create mock data
    df, feature_names = create_mock_data()
    X = df[feature_names].values
    y = df['volatility'].values
    
    # Generate mock SHAP values
    np.random.seed(42)
    shap_values = np.random.normal(0, 0.1, (len(X), len(feature_names)))
    
    print("Running integrated SHAP analysis pipeline...")
    
    # Step 1: Create explainer
    models, _, _, _ = create_mock_models()
    xgb_model = models['xgboost']
    
    explainer = factory.create_explainer(
        model=xgb_model,
        model_type='xgboost',
        task='regression',
        X=X
    )
    
    print("✓ Explainer created")
    
    # Step 2: Perform time-series analysis
    time_series_results = utils.analyze_time_series_shap(
        shap_values=shap_values,
        X=X,
        feature_names=feature_names,
        time_index=df.index,
        window_size=30
    )
    
    print("✓ Time-series analysis completed")
    
    # Step 3: Perform feature interaction analysis
    interaction_results = utils.analyze_feature_interactions(
        shap_values=shap_values,
        X=X,
        feature_names=feature_names,
        top_k=5
    )
    
    print("✓ Feature interaction analysis completed")
    
    # Step 4: Generate individual explanation
    individual_explanation = utils.generate_individual_explanation(
        explainer=explainer,
        instance=X[:1],
        feature_names=feature_names,
        instance_index=0
    )
    
    print("✓ Individual explanation generated")
    
    # Step 5: Create visualizations
    plots = visualizer.create_comprehensive_plots(
        shap_values=shap_values,
        X=X,
        feature_names=feature_names,
        asset='AAPL',
        model_type='xgboost',
        task='regression',
        explainer=explainer
    )
    
    print("✓ Visualizations created")
    
    # Step 6: Save results
    metadata = {
        'asset': 'AAPL',
        'model_type': 'xgboost',
        'task': 'regression',
        'feature_names': feature_names,
        'time_series_analysis': time_series_results,
        'interaction_analysis': interaction_results,
        'individual_explanation': individual_explanation
    }
    
    success = utils.save_shap_data(
        shap_values=shap_values,
        metadata=metadata,
        filepath='temp_integration/integrated_analysis'
    )
    
    if success:
        print("✓ Results saved successfully")
    else:
        print("✗ Failed to save results")
    
    print("\nIntegration demonstration completed!")
    print("\nSummary of generated files:")
    
    # List generated files
    temp_dirs = ['temp_shap_data', 'temp_shap_analysis', 'temp_shap_plots', 'temp_integration']
    
    for temp_dir in temp_dirs:
        if Path(temp_dir).exists():
            print(f"\n{temp_dir}/")
            for file_path in Path(temp_dir).rglob('*'):
                if file_path.is_file():
                    print(f"  - {file_path.name}")


def main():
    """Run the complete demonstration."""
    print("Advanced SHAP Analysis Demonstration for RiskPipeline")
    print("=" * 60)
    
    try:
        # Run all demonstrations
        demonstrate_explainer_factory()
        demonstrate_interpretation_utils()
        demonstrate_shap_analyzer()
        demonstrate_shap_visualizer()
        demonstrate_integration()
        
        print("\n" + "=" * 60)
        print("Demonstration completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✓ Model-specific explainers (ARIMA, LSTM, StockMixer, XGBoost)")
        print("✓ Time-series SHAP analysis with regime detection")
        print("✓ Feature interaction analysis and clustering")
        print("✓ Individual prediction explanations")
        print("✓ Comprehensive visualization suite")
        print("✓ Data persistence and retrieval")
        print("✓ Integration across all components")
        
        print("\nGenerated files are available in:")
        print("- temp_shap_data/ (SHAP data files)")
        print("- temp_shap_analysis/ (Analysis results)")
        print("- temp_shap_plots/ (Visualization plots)")
        print("- temp_integration/ (Integrated analysis)")
        
    except Exception as e:
        print(f"\nDemonstration failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 