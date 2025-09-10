#!/usr/bin/env python3
"""
Simple Hyperparameter Test - Focus on R² improvement without SHAP analysis
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from risk_pipeline import RiskPipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleHyperparameterTester:
    """Simple hyperparameter tester focused on R² improvement."""
    
    def __init__(self):
        self.results = {}
        
    def create_test_config(self, name: str, **kwargs):
        """Create a test configuration."""
        base_config = {
            "data": {
                "all_assets": ["AAPL", "MSFT"],
                "start_date": "2020-01-01",
                "end_date": "2024-01-01",
                "cache_dir": "data_cache"
            },
            "training": {
                "walk_forward_splits": 3,
                "test_size": 63,
                "joblib_n_jobs": 22,
                "num_workers": 22
            },
            "output": {
                "results_dir": "experiments",
                "plots_dir": "visualizations",
                "log_dir": "logs",
                "shap_dir": "shap_plots"
            },
            "logging": {
                "level": "INFO"
            },
            "shap": {
                "background_samples": 50,
                "eval_samples": 100
            }
        }
        
        # Update with provided parameters
        base_config.update(kwargs)
        return base_config
    
    def run_test(self, config_name: str, config: dict):
        """Run a single test configuration."""
        logger.info(f"🧪 Testing configuration: {config_name}")
        
        try:
            # Create pipeline with the configuration
            pipeline = RiskPipeline(config=config)
            
            # Run pipeline without SHAP analysis to avoid the error
            results = pipeline.run_complete_pipeline(
                assets=config["data"]["all_assets"],
                models=['arima', 'xgboost'],  # Only working models
                save_models=True,
                run_shap=False,  # Disable SHAP to avoid the error
                description=f"Simple test: {config_name}"
            )
            
            # Extract R² scores
            r2_scores = {}
            for asset, asset_results in results.items():
                for task, task_results in asset_results.items():
                    for model, model_result in task_results.items():
                        if isinstance(model_result, dict) and 'metrics' in model_result:
                            metrics = model_result['metrics']
                            if 'R2' in metrics:
                                r2_scores[f"{asset}_{model}_{task}"] = metrics['R2']
            
            self.results[config_name] = {
                'success': True,
                'r2_scores': r2_scores,
                'results': results
            }
            
            logger.info(f"✅ {config_name} completed successfully")
            logger.info(f"📊 R² scores: {r2_scores}")
            
        except Exception as e:
            logger.error(f"❌ {config_name} failed: {str(e)}")
            self.results[config_name] = {
                'success': False,
                'error': str(e),
                'r2_scores': {}
            }
    
    def run_all_tests(self):
        """Run all test configurations."""
        logger.info("🚀 Starting Simple Hyperparameter Tests")
        
        # Test configurations focused on R² improvement
        test_configs = {
            "Basic_Optimized": self.create_test_config(
                "Basic_Optimized",
                models={
                    "arima": {
                        "order": (1, 1, 1),
                        "seasonal_order": (1, 1, 1, 12),
                        "n_jobs": 22
                    },
                    "xgboost": {
                        "n_estimators": 100,
                        "max_depth": 4,
                        "learning_rate": 0.1,
                        "subsample": 0.9,
                        "colsample_bytree": 0.9,
                        "reg_alpha": 0.0,
                        "reg_lambda": 0.0,
                        "random_state": 42,
                        "n_jobs": 22
                    }
                }
            ),
            
            "Conservative_XGBoost": self.create_test_config(
                "Conservative_XGBoost",
                models={
                    "arima": {
                        "order": (1, 1, 1),
                        "seasonal_order": (1, 1, 1, 12),
                        "n_jobs": 22
                    },
                    "xgboost": {
                        "n_estimators": 50,
                        "max_depth": 2,
                        "learning_rate": 0.05,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "reg_alpha": 0.1,
                        "reg_lambda": 1.0,
                        "random_state": 42,
                        "n_jobs": 22
                    }
                }
            ),
            
            "Enhanced_ARIMA_Focus": self.create_test_config(
                "Enhanced_ARIMA_Focus",
                models={
                    "arima": {
                        "order": (2, 1, 2),
                        "seasonal_order": (2, 1, 2, 12),
                        "n_jobs": 22
                    },
                    "xgboost": {
                        "n_estimators": 200,
                        "max_depth": 3,
                        "learning_rate": 0.05,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "reg_alpha": 0.1,
                        "reg_lambda": 1.0,
                        "random_state": 42,
                        "n_jobs": 22
                    }
                }
            ),
            
            "Minimal_Features": self.create_test_config(
                "Minimal_Features",
                models={
                    "arima": {
                        "order": (1, 1, 1),
                        "seasonal_order": (1, 1, 1, 12),
                        "n_jobs": 22
                    },
                    "xgboost": {
                        "n_estimators": 300,
                        "max_depth": 2,
                        "learning_rate": 0.03,
                        "subsample": 0.7,
                        "colsample_bytree": 0.7,
                        "reg_alpha": 0.2,
                        "reg_lambda": 2.0,
                        "random_state": 42,
                        "n_jobs": 22
                    }
                }
            ),
            
            "Aggressive_Regularization": self.create_test_config(
                "Aggressive_Regularization",
                models={
                    "arima": {
                        "order": (1, 1, 1),
                        "seasonal_order": (1, 1, 1, 12),
                        "n_jobs": 22
                    },
                    "xgboost": {
                        "n_estimators": 500,
                        "max_depth": 1,
                        "learning_rate": 0.01,
                        "subsample": 0.6,
                        "colsample_bytree": 0.6,
                        "reg_alpha": 0.5,
                        "reg_lambda": 5.0,
                        "random_state": 42,
                        "n_jobs": 22
                    }
                }
            )
        }
        
        # Run each test
        for config_name, config in test_configs.items():
            self.run_test(config_name, config)
        
        # Analyze results
        self.analyze_results()
        
        # Save results
        self.save_results()
    
    def analyze_results(self):
        """Analyze test results and provide recommendations."""
        logger.info("\n" + "="*80)
        logger.info("📊 SIMPLE HYPERPARAMETER TEST RESULTS")
        logger.info("="*80)
        
        successful_configs = []
        failed_configs = []
        
        for config_name, result in self.results.items():
            if result['success']:
                successful_configs.append((config_name, result['r2_scores']))
                logger.info(f"✅ {config_name:<25} | SUCCESS")
                for model_task, r2 in result['r2_scores'].items():
                    logger.info(f"   {model_task}: R² = {r2:.4f}")
            else:
                failed_configs.append(config_name)
                logger.error(f"❌ {config_name:<25} | FAILED: {result['error']}")
        
        logger.info("="*80)
        
        if successful_configs:
            # Find best configuration
            best_config = None
            best_avg_r2 = float('-inf')
            
            for config_name, r2_scores in successful_configs:
                if r2_scores:
                    avg_r2 = sum(r2_scores.values()) / len(r2_scores)
                    if avg_r2 > best_avg_r2:
                        best_avg_r2 = avg_r2
                        best_config = config_name
            
            logger.info(f"🏆 BEST CONFIGURATION: {best_config}")
            logger.info(f"📈 AVERAGE R²: {best_avg_r2:.4f}")
            
            # Provide recommendations
            logger.info("\n💡 RECOMMENDATIONS:")
            if best_avg_r2 > 0:
                logger.info("   ✅ R² is positive - models are performing better than baseline")
            elif best_avg_r2 > -1:
                logger.info("   ⚠️  R² is negative but close to 0 - models need improvement")
            else:
                logger.info("   ❌ R² is very negative - models are performing worse than baseline")
            
            logger.info(f"   🎯 Use configuration: {best_config}")
            logger.info("   🔧 Consider further tuning based on best performing model")
        else:
            logger.error("❌ No successful configurations found")
            logger.info("💡 RECOMMENDATIONS:")
            logger.info("   🔍 Check error messages above")
            logger.info("   🛠️  Fix underlying issues before tuning")
            logger.info("   📝 Consider simpler model configurations")
    
    def save_results(self):
        """Save results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simple_hyperparameter_results_{timestamp}.json"
        
        # Prepare results for JSON serialization
        serializable_results = {}
        for config_name, result in self.results.items():
            serializable_results[config_name] = {
                'success': result['success'],
                'r2_scores': result['r2_scores'],
                'error': result.get('error', None)
            }
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"💾 Results saved to: {filename}")

def main():
    """Main function."""
    tester = SimpleHyperparameterTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()
