#!/usr/bin/env python3
"""
Quick Hyperparameter Testing Script
Tests specific parameter combinations to fix negative R² scores
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

class QuickHyperparameterTester:
    """Quick hyperparameter testing for immediate R² improvements."""
    
    def __init__(self):
        """Initialize the tester."""
        self.results = {}
        logger.info("🚀 Quick Hyperparameter Tester initialized!")
    
    def create_test_config(self, model_params: Dict) -> Dict:
        """Create a test configuration with specific parameters."""
        # Base configuration for quick testing
        config = {
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
                "stochastic_d": 3
            },
            "models": {
                "lstm_units": model_params.get('lstm_units', [64, 32]),
                "lstm_dropout": model_params.get('lstm_dropout', 0.2),
                "lstm_recurrent_dropout": 0.1,
                "lstm_bidirectional": False,
                "lstm_attention": False,
                "stockmixer_temporal_units": model_params.get('stockmixer_temporal_units', 64),
                "stockmixer_indicator_units": model_params.get('stockmixer_indicator_units', 64),
                "stockmixer_cross_stock_units": model_params.get('stockmixer_cross_stock_units', 64),
                "stockmixer_fusion_units": model_params.get('stockmixer_fusion_units', 128),
                "stockmixer_dropout": model_params.get('stockmixer_dropout', 0.2),
                "xgboost_n_estimators": model_params.get('xgboost_n_estimators', 200),
                "xgboost_max_depth": model_params.get('xgboost_max_depth', 4),
                "xgboost_learning_rate": model_params.get('xgboost_learning_rate', 0.1),
                "xgboost_subsample": model_params.get('xgboost_subsample', 0.8),
                "xgboost_colsample_bytree": model_params.get('xgboost_colsample_bytree', 0.8),
                "xgboost_reg_alpha": model_params.get('xgboost_reg_alpha', 0.1),
                "xgboost_reg_lambda": model_params.get('xgboost_reg_lambda', 1.0),
                "xgboost_min_child_weight": model_params.get('xgboost_min_child_weight', 3),
                "xgboost_gamma": model_params.get('xgboost_gamma', 0.1)
            },
            "training": {
                "walk_forward_splits": 3,
                "test_size": 63,
                "batch_size": model_params.get('batch_size', 32),
                "epochs": model_params.get('epochs', 50),
                "early_stopping_patience": 15,
                "reduce_lr_patience": 8,
                "random_state": 42,
                "validation_split": 0.2,
                "learning_rate": model_params.get('learning_rate', 0.001)
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
            }
        }
        
        return config
    
    def test_configuration(self, config_name: str, model_params: Dict) -> Dict[str, float]:
        """Test a specific configuration."""
        logger.info(f"🧪 Testing configuration: {config_name}")
        
        try:
            # Create configuration
            config = self.create_test_config(model_params)
            
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
                        'n_models': len(reg_results)
                    }
                    
                    logger.info(f"✅ {config_name} - R²: {avg_metrics['R2']:.4f}, RMSE: {avg_metrics['RMSE']:.4f}")
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
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive hyperparameter tests."""
        logger.info("🚀 Starting comprehensive hyperparameter tests...")
        
        # Define test configurations
        test_configs = {
            # Conservative XGBoost (most likely to work)
            "XGBoost_Conservative": {
                'xgboost_n_estimators': 100,
                'xgboost_max_depth': 3,
                'xgboost_learning_rate': 0.1,
                'xgboost_subsample': 0.8,
                'xgboost_colsample_bytree': 0.8,
                'xgboost_reg_alpha': 0.1,
                'xgboost_reg_lambda': 1.0,
                'xgboost_min_child_weight': 5,
                'xgboost_gamma': 0.1
            },
            
            # More aggressive XGBoost
            "XGBoost_Aggressive": {
                'xgboost_n_estimators': 500,
                'xgboost_max_depth': 6,
                'xgboost_learning_rate': 0.05,
                'xgboost_subsample': 0.7,
                'xgboost_colsample_bytree': 0.7,
                'xgboost_reg_alpha': 0.5,
                'xgboost_reg_lambda': 2.0,
                'xgboost_min_child_weight': 3,
                'xgboost_gamma': 0.2
            },
            
            # Conservative LSTM
            "LSTM_Conservative": {
                'lstm_units': [32, 16],
                'lstm_dropout': 0.1,
                'batch_size': 16,
                'epochs': 30,
                'learning_rate': 0.001
            },
            
            # More complex LSTM
            "LSTM_Complex": {
                'lstm_units': [128, 64, 32],
                'lstm_dropout': 0.2,
                'batch_size': 32,
                'epochs': 50,
                'learning_rate': 0.0005
            },
            
            # Conservative StockMixer
            "StockMixer_Conservative": {
                'stockmixer_temporal_units': 32,
                'stockmixer_indicator_units': 32,
                'stockmixer_cross_stock_units': 32,
                'stockmixer_fusion_units': 64,
                'stockmixer_dropout': 0.1,
                'batch_size': 16,
                'epochs': 30,
                'learning_rate': 0.001
            },
            
            # More complex StockMixer
            "StockMixer_Complex": {
                'stockmixer_temporal_units': 128,
                'stockmixer_indicator_units': 128,
                'stockmixer_cross_stock_units': 128,
                'stockmixer_fusion_units': 256,
                'stockmixer_dropout': 0.2,
                'batch_size': 32,
                'epochs': 50,
                'learning_rate': 0.0005
            },
            
            # Balanced configuration
            "Balanced_Config": {
                'lstm_units': [64, 32],
                'lstm_dropout': 0.2,
                'stockmixer_temporal_units': 64,
                'stockmixer_indicator_units': 64,
                'stockmixer_cross_stock_units': 64,
                'stockmixer_fusion_units': 128,
                'stockmixer_dropout': 0.2,
                'xgboost_n_estimators': 200,
                'xgboost_max_depth': 4,
                'xgboost_learning_rate': 0.1,
                'xgboost_subsample': 0.8,
                'xgboost_colsample_bytree': 0.8,
                'xgboost_reg_alpha': 0.1,
                'xgboost_reg_lambda': 1.0,
                'xgboost_min_child_weight': 3,
                'xgboost_gamma': 0.1,
                'batch_size': 32,
                'epochs': 50,
                'learning_rate': 0.001
            }
        }
        
        # Run tests
        results = {}
        for config_name, params in test_configs.items():
            result = self.test_configuration(config_name, params)
            results[config_name] = result
        
        return results
    
    def analyze_results(self, results: Dict[str, Any]) -> None:
        """Analyze and display results."""
        logger.info("\n" + "="*80)
        logger.info("📊 HYPERPARAMETER TESTING RESULTS")
        logger.info("="*80)
        
        # Sort by R² score
        sorted_results = sorted(results.items(), key=lambda x: x[1].get('R2', -999), reverse=True)
        
        for config_name, metrics in sorted_results:
            r2 = metrics.get('R2', -999)
            rmse = metrics.get('RMSE', 999)
            n_models = metrics.get('n_models', 0)
            
            if r2 > -999:
                status = "✅" if r2 > 0 else "⚠️"
                logger.info(f"{status} {config_name:25} | R²: {r2:8.4f} | RMSE: {rmse:8.4f} | Models: {n_models}")
            else:
                logger.info(f"❌ {config_name:25} | FAILED")
        
        # Find best configuration
        best_config = max(results.items(), key=lambda x: x[1].get('R2', -999))
        best_name, best_metrics = best_config
        
        logger.info("="*80)
        if best_metrics.get('R2', -999) > -999:
            logger.info(f"🏆 BEST CONFIGURATION: {best_name}")
            logger.info(f"   R²: {best_metrics['R2']:.4f}")
            logger.info(f"   RMSE: {best_metrics['RMSE']:.4f}")
            logger.info(f"   MAE: {best_metrics['MAE']:.4f}")
            logger.info(f"   Models tested: {best_metrics['n_models']}")
        else:
            logger.info("❌ No successful configurations found")
        
        logger.info("="*80)
    
    def save_results(self, results: Dict[str, Any]) -> None:
        """Save results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"quick_hyperparameter_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"💾 Results saved to: {filename}")

def main():
    """Main function."""
    logger.info("🚀 Starting Quick Hyperparameter Testing")
    
    # Initialize tester
    tester = QuickHyperparameterTester()
    
    # Run tests
    results = tester.run_comprehensive_tests()
    
    # Analyze results
    tester.analyze_results(results)
    
    # Save results
    tester.save_results(results)
    
    logger.info("✅ Quick hyperparameter testing completed!")

if __name__ == "__main__":
    main()
