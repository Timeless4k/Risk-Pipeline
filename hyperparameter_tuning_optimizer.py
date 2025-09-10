#!/usr/bin/env python3
"""
Comprehensive Hyperparameter Tuning Script for RiskPipeline
Fixes negative R² scores and optimizes all models for better performance
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from risk_pipeline.core.pipeline import RiskPipeline
from risk_pipeline.models.lstm_model import LSTMModel
from risk_pipeline.models.stockmixer_model import StockMixerModel
from risk_pipeline.models.xgboost_model import XGBoostModel
from risk_pipeline.models.enhanced_arima_model import EnhancedARIMAModel
from risk_pipeline.models.arima_model import ARIMAModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    """Comprehensive hyperparameter optimization for all models."""
    
    def __init__(self, config_path: str = "configs/simple_test_config.json"):
        """Initialize the optimizer with a base configuration."""
        self.config_path = config_path
        self.results = {}
        self.best_configs = {}
        
        # Load base configuration
        with open(config_path, 'r') as f:
            self.base_config = json.load(f)
        
        # Define optimized hyperparameter search spaces for each model
        self.search_spaces = self._define_search_spaces()
        
        logger.info("🚀 Hyperparameter Optimizer initialized!")
        logger.info(f"Base config: {config_path}")
    
    def _define_search_spaces(self) -> Dict[str, Dict]:
        """Define comprehensive search spaces for each model."""
        return {
            'lstm': {
                'units': [[32, 16], [64, 32], [128, 64], [256, 128], [64, 32, 16]],
                'dropout': [0.1, 0.2, 0.3, 0.4],
                'learning_rate': [0.0001, 0.001, 0.01],
                'batch_size': [16, 32, 64, 128],
                'epochs': [50, 100, 200],
                'early_stopping_patience': [10, 20, 30],
                'reduce_lr_patience': [5, 10, 15]
            },
            'stockmixer': {
                'temporal_units': [32, 64, 128, 256],
                'indicator_units': [32, 64, 128, 256],
                'cross_stock_units': [32, 64, 128, 256],
                'fusion_units': [64, 128, 256, 512],
                'dropout': [0.1, 0.2, 0.3, 0.4],
                'learning_rate': [0.0001, 0.001, 0.01],
                'batch_size': [16, 32, 64, 128],
                'epochs': [50, 100, 200],
                'early_stopping_patience': [10, 20, 30]
            },
            'xgboost': {
                'n_estimators': [100, 200, 500, 1000],
                'max_depth': [3, 4, 5, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.7, 0.8, 0.9],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
                'reg_alpha': [0, 0.1, 0.5, 1.0],
                'reg_lambda': [0.1, 0.5, 1.0, 2.0],
                'min_child_weight': [1, 3, 5, 7],
                'gamma': [0, 0.1, 0.2, 0.5]
            },
            'enhanced_arima': {
                'order': [(1, 1, 1), (1, 1, 2), (2, 1, 1), (2, 1, 2), (1, 2, 1)],
                'top_k_features': [5, 10, 15, 20, 25],
                'use_log_vol_target': [True, False],
                'residual_model': ['xgb', 'none'],
                'auto_order': [True, False]
            },
            'arima': {
                'order': [(1, 1, 1), (1, 1, 2), (2, 1, 1), (2, 1, 2), (1, 2, 1)],
                'auto_order': [True, False]
            }
        }
    
    def create_test_config(self, model_type: str, params: Dict) -> Dict:
        """Create a test configuration with specific hyperparameters."""
        config = self.base_config.copy()
        
        # Update model-specific parameters
        if model_type == 'lstm':
            config['models']['lstm_units'] = params['units']
            config['models']['lstm_dropout'] = params['dropout']
            config['training']['learning_rate'] = params['learning_rate']
            config['training']['batch_size'] = params['batch_size']
            config['training']['epochs'] = params['epochs']
            config['training']['early_stopping_patience'] = params['early_stopping_patience']
            config['training']['reduce_lr_patience'] = params['reduce_lr_patience']
            
        elif model_type == 'stockmixer':
            config['models']['stockmixer_temporal_units'] = params['temporal_units']
            config['models']['stockmixer_indicator_units'] = params['indicator_units']
            config['models']['stockmixer_cross_stock_units'] = params['cross_stock_units']
            config['models']['stockmixer_fusion_units'] = params['fusion_units']
            config['models']['stockmixer_dropout'] = params['dropout']
            config['training']['learning_rate'] = params['learning_rate']
            config['training']['batch_size'] = params['batch_size']
            config['training']['epochs'] = params['epochs']
            config['training']['early_stopping_patience'] = params['early_stopping_patience']
            
        elif model_type == 'xgboost':
            config['models']['xgboost_n_estimators'] = params['n_estimators']
            config['models']['xgboost_max_depth'] = params['max_depth']
            config['models']['xgboost_learning_rate'] = params['learning_rate']
            config['models']['xgboost_subsample'] = params['subsample']
            config['models']['xgboost_colsample_bytree'] = params['colsample_bytree']
            config['models']['xgboost_reg_alpha'] = params['reg_alpha']
            config['models']['xgboost_reg_lambda'] = params['reg_lambda']
            config['models']['xgboost_min_child_weight'] = params['min_child_weight']
            config['models']['xgboost_gamma'] = params['gamma']
            
        elif model_type == 'enhanced_arima':
            config['models']['arima_order'] = params['order']
            config['models']['arima_top_k_features'] = params['top_k_features']
            config['models']['arima_use_log_vol_target'] = params['use_log_vol_target']
            config['models']['arima_residual_model'] = params['residual_model']
            config['models']['arima_auto_order'] = params['auto_order']
            
        elif model_type == 'arima':
            config['models']['arima_order'] = params['order']
            config['models']['arima_auto_order'] = params['auto_order']
        
        return config
    
    def test_single_model(self, model_type: str, params: Dict, asset: str = "AAPL") -> Dict[str, float]:
        """Test a single model configuration on a single asset."""
        try:
            # Create test configuration
            test_config = self.create_test_config(model_type, params)
            
            # Limit to single asset for faster testing
            test_config['data']['us_assets'] = [asset]
            test_config['data']['au_assets'] = []
            test_config['data']['start_date'] = "2020-01-01"  # Shorter period for faster testing
            test_config['data']['end_date'] = "2024-01-01"
            
            # Reduce training complexity for faster testing
            test_config['training']['walk_forward_splits'] = 3
            test_config['training']['test_size'] = 63  # ~3 months
            
            # Initialize pipeline
            pipeline = RiskPipeline(test_config)
            
            # Run pipeline
            results = pipeline.run()
            
            # Extract metrics for the specific model and asset
            if 'model_performance' in results:
                perf_df = results['model_performance']
                model_results = perf_df[
                    (perf_df['model_type'] == model_type) & 
                    (perf_df['asset'] == asset) & 
                    (perf_df['task'] == 'regression')
                ]
                
                if not model_results.empty:
                    return {
                        'R2': model_results['R2'].iloc[0],
                        'RMSE': model_results['RMSE'].iloc[0],
                        'MAE': model_results['MAE'].iloc[0],
                        'MSE': model_results['MSE'].iloc[0]
                    }
            
            return {'R2': -999, 'RMSE': 999, 'MAE': 999, 'MSE': 999}
            
        except Exception as e:
            logger.error(f"Error testing {model_type} with params {params}: {e}")
            return {'R2': -999, 'RMSE': 999, 'MAE': 999, 'MSE': 999}
    
    def grid_search_optimize(self, model_type: str, max_combinations: int = 20) -> Dict[str, Any]:
        """Perform grid search optimization for a specific model."""
        logger.info(f"🔍 Starting grid search optimization for {model_type}")
        
        search_space = self.search_spaces[model_type]
        best_score = -999
        best_params = None
        all_results = []
        
        # Generate parameter combinations
        import itertools
        
        # Get all parameter names and values
        param_names = list(search_space.keys())
        param_values = list(search_space.values())
        
        # Generate combinations (limit to max_combinations)
        combinations = list(itertools.product(*param_values))
        
        if len(combinations) > max_combinations:
            # Randomly sample combinations
            import random
            combinations = random.sample(combinations, max_combinations)
        
        logger.info(f"Testing {len(combinations)} parameter combinations for {model_type}")
        
        for i, combination in enumerate(combinations):
            params = dict(zip(param_names, combination))
            
            logger.info(f"Testing combination {i+1}/{len(combinations)}: {params}")
            
            # Test on multiple assets for robustness
            assets = ["AAPL", "MSFT"]
            scores = []
            
            for asset in assets:
                result = self.test_single_model(model_type, params, asset)
                scores.append(result['R2'])
            
            # Use average R² score across assets
            avg_r2 = np.mean(scores)
            
            result = {
                'params': params,
                'R2': avg_r2,
                'individual_scores': scores
            }
            all_results.append(result)
            
            logger.info(f"Combination {i+1} R²: {avg_r2:.4f}")
            
            if avg_r2 > best_score:
                best_score = avg_r2
                best_params = params
                logger.info(f"🎯 New best R²: {best_score:.4f}")
        
        return {
            'model_type': model_type,
            'best_params': best_params,
            'best_score': best_score,
            'all_results': all_results
        }
    
    def optimize_all_models(self, max_combinations_per_model: int = 15) -> Dict[str, Any]:
        """Optimize all models and return comprehensive results."""
        logger.info("🚀 Starting comprehensive hyperparameter optimization!")
        
        all_results = {}
        models_to_optimize = ['lstm', 'stockmixer', 'xgboost', 'enhanced_arima', 'arima']
        
        for model_type in models_to_optimize:
            logger.info(f"\n{'='*60}")
            logger.info(f"Optimizing {model_type.upper()}")
            logger.info(f"{'='*60}")
            
            try:
                result = self.grid_search_optimize(model_type, max_combinations_per_model)
                all_results[model_type] = result
                
                logger.info(f"✅ {model_type.upper()} optimization completed!")
                logger.info(f"Best R²: {result['best_score']:.4f}")
                logger.info(f"Best params: {result['best_params']}")
                
            except Exception as e:
                logger.error(f"❌ Failed to optimize {model_type}: {e}")
                all_results[model_type] = {
                    'model_type': model_type,
                    'best_params': None,
                    'best_score': -999,
                    'error': str(e)
                }
        
        # Save results
        self.save_results(all_results)
        
        return all_results
    
    def save_results(self, results: Dict[str, Any]) -> None:
        """Save optimization results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = f"hyperparameter_tuning_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create summary
        summary = self.create_summary(results)
        summary_file = f"hyperparameter_tuning_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create optimized configuration
        optimized_config = self.create_optimized_config(results)
        config_file = f"optimized_config_{timestamp}.json"
        with open(config_file, 'w') as f:
            json.dump(optimized_config, f, indent=2)
        
        logger.info(f"📊 Results saved:")
        logger.info(f"  - Detailed: {results_file}")
        logger.info(f"  - Summary: {summary_file}")
        logger.info(f"  - Config: {config_file}")
    
    def create_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of optimization results."""
        summary = {
            'optimization_timestamp': datetime.now().isoformat(),
            'models_optimized': len(results),
            'best_performers': [],
            'improvements': {}
        }
        
        for model_type, result in results.items():
            if 'best_score' in result and result['best_score'] > -999:
                summary['best_performers'].append({
                    'model': model_type,
                    'best_r2': result['best_score'],
                    'best_params': result['best_params']
                })
        
        # Sort by R² score
        summary['best_performers'].sort(key=lambda x: x['best_r2'], reverse=True)
        
        return summary
    
    def create_optimized_config(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create an optimized configuration file."""
        config = self.base_config.copy()
        
        # Update with best parameters for each model
        for model_type, result in results.items():
            if result.get('best_params') and result.get('best_score', -999) > -999:
                best_params = result['best_params']
                
                if model_type == 'lstm':
                    config['models']['lstm_units'] = best_params['units']
                    config['models']['lstm_dropout'] = best_params['dropout']
                    config['training']['learning_rate'] = best_params['learning_rate']
                    config['training']['batch_size'] = best_params['batch_size']
                    config['training']['epochs'] = best_params['epochs']
                    
                elif model_type == 'stockmixer':
                    config['models']['stockmixer_temporal_units'] = best_params['temporal_units']
                    config['models']['stockmixer_indicator_units'] = best_params['indicator_units']
                    config['models']['stockmixer_cross_stock_units'] = best_params['cross_stock_units']
                    config['models']['stockmixer_fusion_units'] = best_params['fusion_units']
                    config['models']['stockmixer_dropout'] = best_params['dropout']
                    
                elif model_type == 'xgboost':
                    config['models']['xgboost_n_estimators'] = best_params['n_estimators']
                    config['models']['xgboost_max_depth'] = best_params['max_depth']
                    config['models']['xgboost_learning_rate'] = best_params['learning_rate']
                    config['models']['xgboost_subsample'] = best_params['subsample']
                    config['models']['xgboost_colsample_bytree'] = best_params['colsample_bytree']
                    config['models']['xgboost_reg_alpha'] = best_params['reg_alpha']
                    config['models']['xgboost_reg_lambda'] = best_params['reg_lambda']
                    config['models']['xgboost_min_child_weight'] = best_params['min_child_weight']
                    config['models']['xgboost_gamma'] = best_params['gamma']
        
        return config
    
    def run_quick_test(self) -> None:
        """Run a quick test to verify the optimization process."""
        logger.info("🧪 Running quick test...")
        
        # Test XGBoost with conservative parameters
        xgb_params = {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_weight': 3,
            'gamma': 0.1
        }
        
        result = self.test_single_model('xgboost', xgb_params, 'AAPL')
        logger.info(f"Quick test result: {result}")
        
        if result['R2'] > -999:
            logger.info("✅ Quick test passed!")
        else:
            logger.warning("⚠️ Quick test failed - check configuration")

def main():
    """Main function to run hyperparameter optimization."""
    logger.info("🚀 Starting RiskPipeline Hyperparameter Optimization")
    
    # Initialize optimizer
    optimizer = HyperparameterOptimizer()
    
    # Run quick test first
    optimizer.run_quick_test()
    
    # Run full optimization
    results = optimizer.optimize_all_models(max_combinations_per_model=10)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("🎯 OPTIMIZATION SUMMARY")
    logger.info("="*80)
    
    for model_type, result in results.items():
        if result.get('best_score', -999) > -999:
            logger.info(f"{model_type.upper()}: R² = {result['best_score']:.4f}")
        else:
            logger.info(f"{model_type.upper()}: FAILED")
    
    logger.info("="*80)
    logger.info("✅ Optimization completed! Check the generated files for detailed results.")

if __name__ == "__main__":
    main()
