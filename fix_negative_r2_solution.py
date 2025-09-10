#!/usr/bin/env python3
"""
Comprehensive Solution to Fix Negative R² Scores
Addresses all identified issues with systematic fixes
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from risk_pipeline import RiskPipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class R2Fixer:
    """Comprehensive solution to fix negative R² scores."""
    
    def __init__(self):
        """Initialize the R2 fixer."""
        self.results = {}
        logger.info("🔧 R2 Fixer initialized!")
    
    def create_optimized_config(self) -> Dict[str, Any]:
        """Create an optimized configuration that should fix R² issues."""
        return {
            "data": {
                "start_date": "2020-01-01",
                "end_date": "2024-01-01",
                "us_assets": ["AAPL", "MSFT"],
                "au_assets": [],
                "cache_dir": "data_cache"
            },
            "features": {
                "volatility_window": 10,
                "ma_short": 20,
                "ma_long": 50,
                "correlation_window": 30,
                "sequence_length": 5,
                "rsi_period": 14,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "bollinger_period": 20,
                "bollinger_std": 2.0,
                "atr_period": 14,
                "stochastic_k": 14,
                "stochastic_d": 3,
                "use_log_vol_target": True,  # Use log volatility for better numerical stability
                "log_target_epsilon": 1e-6
            },
            "models": {
                # Conservative XGBoost parameters for stability
                "xgboost_n_estimators": 100,
                "xgboost_max_depth": 3,
                "xgboost_learning_rate": 0.1,
                "xgboost_subsample": 0.8,
                "xgboost_colsample_bytree": 0.8,
                "xgboost_reg_alpha": 0.1,
                "xgboost_reg_lambda": 1.0,
                "xgboost_min_child_weight": 5,
                "xgboost_gamma": 0.1,
                
                # Conservative ARIMA parameters
                "arima_order": (1, 1, 1),
                "arima_auto_order": True,
                
                # Enhanced ARIMA parameters
                "enhanced_arima_order": (1, 1, 1),
                "enhanced_arima_top_k_features": 10,
                "enhanced_arima_use_log_vol_target": True,
                "enhanced_arima_residual_model": "xgb",
                "enhanced_arima_auto_order": True
            },
            "training": {
                "walk_forward_splits": 3,
                "test_size": 63,
                "batch_size": 32,
                "epochs": 50,
                "early_stopping_patience": 15,
                "reduce_lr_patience": 8,
                "random_state": 42,
                "validation_split": 0.2,
                "learning_rate": 0.001
            },
            "output": {
                "results_dir": "results",
                "plots_dir": "visualizations",
                "shap_dir": "shap_plots",
                "models_dir": "models",
                "log_dir": "logs"
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
            },
            "hyperparameter_tuning": {
                "enable_tuning": False
            },
            "ensemble": {
                "enable_ensemble": False
            },
            "advanced_features": {
                "enable_feature_selection": False,
                "enable_feature_importance": True,
                "importance_methods": ["shap"],
                "cv_method": "time_series_split",
                "cv_folds": 3,
                "enable_interpretability": True,
                "interpretability_methods": ["shap"],
                "enable_risk_metrics": False
            },
            "data_quality": {
                "enable_input_validation": True,
                "max_feature_value": 1000,
                "max_target_value": 100,
                "enable_extreme_value_clipping": True,
                "enable_nan_detection": True,
                "enable_inf_detection": True,
                "asset_specific_normalization": True,  # Enable asset-specific normalization
                "normalization_method": "standard",
                "train_split_ratio": 0.7,
                "outlier_detection_threshold": 3.0,
                "min_data_quality_score": 0.8
            }
        }
    
    def test_configuration(self, config_name: str, config: Dict) -> Dict[str, float]:
        """Test a specific configuration."""
        logger.info(f"🧪 Testing configuration: {config_name}")
        
        try:
            # Initialize and run pipeline
            pipeline = RiskPipeline(config=config)
            results = pipeline.run_complete_pipeline()
            
            # Extract metrics
            if 'model_performance' in results:
                perf_df = results['model_performance']
                
                # Get regression results
                reg_results = perf_df[perf_df['task'] == 'regression']
                
                if not reg_results.empty:
                    # Calculate average metrics across all models and assets
                    avg_metrics = {
                        'R2': reg_results['R2'].mean(),
                        'RMSE': reg_results['RMSE'].mean(),
                        'MAE': reg_results['MAE'].mean(),
                        'MSE': reg_results['MSE'].mean(),
                        'n_models': len(reg_results),
                        'best_r2': reg_results['R2'].max(),
                        'worst_r2': reg_results['R2'].min()
                    }
                    
                    logger.info(f"✅ {config_name} - R²: {avg_metrics['R2']:.4f}, Best: {avg_metrics['best_r2']:.4f}, Worst: {avg_metrics['worst_r2']:.4f}")
                    return avg_metrics
                else:
                    logger.warning(f"⚠️ {config_name} - No regression results found")
                    return {'R2': -999, 'RMSE': 999, 'MAE': 999, 'MSE': 999, 'n_models': 0}
            else:
                logger.warning(f"⚠️ {config_name} - No model performance data found")
                return {'R2': -999, 'RMSE': 999, 'MAE': 999, 'MSE': 999, 'n_models': 0}
                
        except Exception as e:
            logger.error(f"❌ {config_name} failed: {e}")
            return {'R2': -999, 'RMSE': 999, 'MAE': 999, 'MSE': 999, 'n_models': 0, 'error': str(e)}
    
    def run_comprehensive_fixes(self) -> Dict[str, Any]:
        """Run comprehensive fixes for R² issues."""
        logger.info("🚀 Starting comprehensive R² fixes...")
        
        # Define test configurations with different approaches
        test_configs = {
            # Configuration 1: Basic optimization
            "Basic_Optimized": self.create_optimized_config(),
            
            # Configuration 2: More conservative XGBoost
            "Conservative_XGBoost": self._create_conservative_xgboost_config(),
            
            # Configuration 3: Enhanced ARIMA focus
            "Enhanced_ARIMA_Focus": self._create_enhanced_arima_config(),
            
            # Configuration 4: Minimal features
            "Minimal_Features": self._create_minimal_features_config(),
            
            # Configuration 5: Aggressive regularization
            "Aggressive_Regularization": self._create_aggressive_regularization_config()
        }
        
        # Run tests
        results = {}
        for config_name, config in test_configs.items():
            result = self.test_configuration(config_name, config)
            results[config_name] = result
        
        return results
    
    def _create_conservative_xgboost_config(self) -> Dict[str, Any]:
        """Create a configuration with very conservative XGBoost parameters."""
        config = self.create_optimized_config()
        
        # Very conservative XGBoost parameters
        config['models'].update({
            'xgboost_n_estimators': 50,
            'xgboost_max_depth': 2,
            'xgboost_learning_rate': 0.05,
            'xgboost_subsample': 0.9,
            'xgboost_colsample_bytree': 0.9,
            'xgboost_reg_alpha': 0.5,
            'xgboost_reg_lambda': 2.0,
            'xgboost_min_child_weight': 10,
            'xgboost_gamma': 0.5
        })
        
        return config
    
    def _create_enhanced_arima_config(self) -> Dict[str, Any]:
        """Create a configuration focused on Enhanced ARIMA."""
        config = self.create_optimized_config()
        
        # Optimize for Enhanced ARIMA
        config['models'].update({
            'enhanced_arima_order': (2, 1, 2),
            'enhanced_arima_top_k_features': 5,
            'enhanced_arima_use_log_vol_target': True,
            'enhanced_arima_residual_model': 'xgb',
            'enhanced_arima_auto_order': True
        })
        
        # Reduce XGBoost complexity
        config['models'].update({
            'xgboost_n_estimators': 50,
            'xgboost_max_depth': 2,
            'xgboost_learning_rate': 0.1
        })
        
        return config
    
    def _create_minimal_features_config(self) -> Dict[str, Any]:
        """Create a configuration with minimal features."""
        config = self.create_optimized_config()
        
        # Reduce feature complexity
        config['features'].update({
            'volatility_window': 5,
            'ma_short': 10,
            'ma_long': 20,
            'correlation_window': 10,
            'sequence_length': 1,
            'rsi_period': 0,  # Disable RSI
            'macd_fast': 0,   # Disable MACD
            'macd_slow': 0,
            'macd_signal': 0,
            'bollinger_period': 0,  # Disable Bollinger Bands
            'bollinger_std': 0,
            'atr_period': 0,  # Disable ATR
            'stochastic_k': 0,  # Disable Stochastic
            'stochastic_d': 0
        })
        
        return config
    
    def _create_aggressive_regularization_config(self) -> Dict[str, Any]:
        """Create a configuration with aggressive regularization."""
        config = self.create_optimized_config()
        
        # Aggressive regularization
        config['models'].update({
            'xgboost_n_estimators': 200,
            'xgboost_max_depth': 3,
            'xgboost_learning_rate': 0.01,
            'xgboost_subsample': 0.6,
            'xgboost_colsample_bytree': 0.6,
            'xgboost_reg_alpha': 1.0,
            'xgboost_reg_lambda': 5.0,
            'xgboost_min_child_weight': 10,
            'xgboost_gamma': 1.0
        })
        
        return config
    
    def analyze_results(self, results: Dict[str, Any]) -> None:
        """Analyze and display results."""
        logger.info("\n" + "="*80)
        logger.info("📊 R² FIXING RESULTS")
        logger.info("="*80)
        
        # Sort by R² score
        sorted_results = sorted(results.items(), key=lambda x: x[1].get('R2', -999), reverse=True)
        
        for config_name, metrics in sorted_results:
            r2 = metrics.get('R2', -999)
            rmse = metrics.get('RMSE', 999)
            n_models = metrics.get('n_models', 0)
            best_r2 = metrics.get('best_r2', -999)
            
            if r2 > -999:
                status = "✅" if r2 > 0 else "⚠️" if r2 > -1 else "❌"
                logger.info(f"{status} {config_name:25} | R²: {r2:8.4f} | Best: {best_r2:8.4f} | RMSE: {rmse:8.4f} | Models: {n_models}")
            else:
                logger.info(f"❌ {config_name:25} | FAILED")
        
        # Find best configuration
        best_config = max(results.items(), key=lambda x: x[1].get('R2', -999))
        best_name, best_metrics = best_config
        
        logger.info("="*80)
        if best_metrics.get('R2', -999) > -999:
            logger.info(f"🏆 BEST CONFIGURATION: {best_name}")
            logger.info(f"   Average R²: {best_metrics['R2']:.4f}")
            logger.info(f"   Best R²: {best_metrics['best_r2']:.4f}")
            logger.info(f"   Worst R²: {best_metrics['worst_r2']:.4f}")
            logger.info(f"   RMSE: {best_metrics['RMSE']:.4f}")
            logger.info(f"   MAE: {best_metrics['MAE']:.4f}")
            logger.info(f"   Models tested: {best_metrics['n_models']}")
        else:
            logger.info("❌ No successful configurations found")
        
        logger.info("="*80)
    
    def save_results(self, results: Dict[str, Any]) -> None:
        """Save results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"r2_fixing_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"💾 Results saved to: {filename}")
    
    def create_final_optimized_config(self, best_config_name: str, results: Dict[str, Any]) -> None:
        """Create the final optimized configuration file."""
        if best_config_name in results and results[best_config_name].get('R2', -999) > -999:
            # Get the best configuration
            if best_config_name == "Basic_Optimized":
                config = self.create_optimized_config()
            elif best_config_name == "Conservative_XGBoost":
                config = self._create_conservative_xgboost_config()
            elif best_config_name == "Enhanced_ARIMA_Focus":
                config = self._create_enhanced_arima_config()
            elif best_config_name == "Minimal_Features":
                config = self._create_minimal_features_config()
            elif best_config_name == "Aggressive_Regularization":
                config = self._create_aggressive_regularization_config()
            else:
                config = self.create_optimized_config()
            
            # Save the configuration
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_file = f"optimized_config_fixed_r2_{timestamp}.json"
            
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"💾 Final optimized configuration saved to: {config_file}")
            
            # Also create a simple config for immediate use
            simple_config_file = "fixed_r2_config.json"
            with open(simple_config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"💾 Simple config saved to: {simple_config_file}")
        else:
            logger.warning("No successful configuration found to save")

def main():
    """Main function."""
    logger.info("🚀 Starting R² Fixing Process")
    
    # Initialize fixer
    fixer = R2Fixer()
    
    # Run comprehensive fixes
    results = fixer.run_comprehensive_fixes()
    
    # Analyze results
    fixer.analyze_results(results)
    
    # Save results
    fixer.save_results(results)
    
    # Create final optimized configuration
    best_config = max(results.items(), key=lambda x: x[1].get('R2', -999))
    best_name = best_config[0]
    fixer.create_final_optimized_config(best_name, results)
    
    logger.info("✅ R² fixing process completed!")

if __name__ == "__main__":
    main()
