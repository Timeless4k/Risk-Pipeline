#!/usr/bin/env python3
"""
Main execution script for RiskPipeline
Provides command-line interface and orchestrates the complete workflow
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import pipeline components
try:
    from risk_pipeline import RiskPipeline, AssetConfig
    from visualization import VolatilityVisualizer, create_publication_quality_plots
    from stockmixer_model import StockMixerExplainer
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all modules are in the same directory and dependencies are installed.")
    print("Run: python setup.py")
    sys.exit(1)


class PipelineRunner:
    """Orchestrates pipeline execution with configuration management"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.setup_logging()
        
    def _load_config(self, config_path: str = None) -> dict:
        """Load configuration from file or use defaults"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        default_config_path = Path('configs/pipeline_config.json')
        if default_config_path.exists():
            with open(default_config_path, 'r') as f:
                return json.load(f)
                
        # Fallback to hardcoded defaults
        return {
            "data": {
                "start_date": "2017-01-01",
                "end_date": "2024-03-31",
                "us_assets": ["AAPL", "MSFT", "^GSPC"],
                "au_assets": ["IOZ.AX", "CBA.AX", "BHP.AX"]
            },
            "training": {
                "walk_forward_splits": 5,
                "test_size": 252,
                "random_state": 42
            }
        }
    
    def setup_logging(self):
        """Configure logging with both file and console output"""
        # Clear any existing handlers to avoid conflicts
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Create logs directory
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'pipeline_run_{timestamp}.log'
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler - this will capture ALL logs
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Capture everything in file
        file_handler.setFormatter(formatter)
        
        # Console handler - less verbose for console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Configure root logger
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Create pipeline-specific logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        # Test logging
        self.logger.info(f"Logging initialized. Log file: {log_file}")
        self.logger.debug("Debug logging is working")
        self.logger.info("Info logging is working")
        self.logger.warning("Warning logging is working")
        
        self.log_file_path = log_file
        
    def run_quick_test(self):
        """Run pipeline on subset of data for testing"""
        print("\n" + "="*60)
        print("Running Quick Test Mode")
        print("="*60)
        
        # Override configuration for quick test
        test_config = AssetConfig()
        test_config.START_DATE = '2023-01-01'
        test_config.END_DATE = '2024-03-31'
        test_config.WALK_FORWARD_SPLITS = 2  # Reduced for quick test
        test_config.TEST_SIZE = 50  # Much smaller test size for limited data
        
        # Use only two assets (one from each market) + VIX
        test_assets = ['AAPL', 'CBA.AX']  # Removed ^VIX from assets, it will be downloaded separately
        
        # Initialize and run pipeline
        pipeline = RiskPipeline(config=test_config)
        
        try:
            pipeline.run_pipeline(assets=test_assets)
            
            # Only generate visualizations if we have results
            if pipeline.results and not self._is_results_empty(pipeline.results):
                visualizer = VolatilityVisualizer('visualizations/test')
                visualizer.plot_performance_comparison(pipeline.results, 'regression')
                print("✅ Visualizations generated")
            else:
                print("⚠️ No results generated - skipping visualization")
            
        except Exception as e:
            self.logger.error(f"Quick test failed: {e}", exc_info=True)
            print(f"❌ Quick test failed: {e}")
            return
        
        print("\n✅ Quick test completed!")
        print("Check results in 'results/' and 'visualizations/test/'")
        
    def _is_results_empty(self, results: dict) -> bool:
        """Check if results dictionary is effectively empty"""
        if not results:
            return True
        for asset_results in results.values():
            if not asset_results:
                continue
            for task_results in asset_results.values():
                if not task_results:
                    continue
                for model_results in task_results.values():
                    if model_results and any(k not in ['predictions', 'actuals'] and 
                                           v not in [float('inf'), -float('inf'), 0.0] 
                                           for k, v in model_results.items()):
                        return False
        return True
        
    def run_full_pipeline(self, assets: list = None):
        """Run complete pipeline with all features"""
        print("\n" + "="*60)
        print("Running Full Pipeline")
        print("="*60)
        
        # Initialize pipeline with configuration
        asset_config = AssetConfig()
        
        # Update from loaded configuration
        asset_config.START_DATE = self.config['data']['start_date']
        asset_config.END_DATE = self.config['data']['end_date']
        asset_config.WALK_FORWARD_SPLITS = self.config['training']['walk_forward_splits']
        asset_config.TEST_SIZE = self.config['training']['test_size']
        asset_config.RANDOM_STATE = self.config['training']['random_state']
        
        # Determine assets to process
        if assets is None:
            assets = self.config['data']['us_assets'] + self.config['data']['au_assets']
        
        print(f"\nProcessing assets: {', '.join(assets)}")
        print(f"Date range: {asset_config.START_DATE} to {asset_config.END_DATE}")
        print(f"Walk-forward splits: {asset_config.WALK_FORWARD_SPLITS}")
        
        # Initialize pipeline
        pipeline = RiskPipeline(config=asset_config)
        
        # Run pipeline
        try:
            pipeline.run_pipeline(assets=assets)
            
            # Generate comprehensive visualizations
            self._generate_visualizations(pipeline.results)
            
            # Generate summary report
            self._generate_summary_report(pipeline.results)
            
            print("\n✅ Pipeline completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            print(f"\n❌ Pipeline failed: {e}")
            print("Check logs for detailed error information.")
            sys.exit(1)
            
    def run_single_asset(self, asset: str):
        """Run pipeline for a single asset"""
        print(f"\n" + "="*60)
        print(f"Running Pipeline for {asset}")
        print("="*60)
        
        self.run_full_pipeline(assets=[asset])
        
    def _generate_visualizations(self, results: dict):
        """Generate all visualization outputs"""
        print("\nGenerating visualizations...")
        
        # Standard visualizations
        visualizer = VolatilityVisualizer()
        
        # Performance comparisons
        visualizer.plot_performance_comparison(results, 'regression')
        visualizer.plot_performance_comparison(results, 'classification')
        
        # Time series plots for best models
        for asset in results:
            # Find best regression model
            reg_results = results[asset].get('regression', {})
            if reg_results:
                best_model = max(reg_results.items(), 
                               key=lambda x: x[1].get('R2', -1) if 'R2' in x[1] else -1)
                if best_model[1].get('R2', -1) > 0:
                    visualizer.plot_time_series_predictions(
                        results, asset, best_model[0]
                    )
        
        # Publication quality plots
        create_publication_quality_plots(results)
        
        # Interactive dashboard
        visualizer.create_interactive_dashboard(results)
        
        print("✅ Visualizations generated")
        
    def _generate_summary_report(self, results: dict):
        """Generate comprehensive summary report"""
        print("\nGenerating summary report...")
        
        report_path = Path('results/summary_report.md')
        
        with open(report_path, 'w') as f:
            # Header
            f.write("# Volatility Forecasting Pipeline - Summary Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Configuration summary
            f.write("## Configuration\n\n")
            f.write(f"- Date Range: {self.config['data']['start_date']} to {self.config['data']['end_date']}\n")
            f.write(f"- Walk-forward Splits: {self.config['training']['walk_forward_splits']}\n")
            f.write(f"- Test Size: {self.config['training']['test_size']} days\n\n")
            
            # Results summary
            f.write("## Results Summary\n\n")
            
            # Best performing models
            f.write("### Best Performing Models\n\n")
            
            best_regression = self._find_best_model(results, 'regression', 'R2')
            best_classification = self._find_best_model(results, 'classification', 'F1')
            
            if best_regression:
                f.write(f"**Regression**: {best_regression['model']} on {best_regression['asset']} "
                       f"(R² = {best_regression['score']:.4f})\n\n")
                       
            if best_classification:
                f.write(f"**Classification**: {best_classification['model']} on {best_classification['asset']} "
                       f"(F1 = {best_classification['score']:.4f})\n\n")
            
            # Market comparison
            f.write("### Market Comparison\n\n")
            us_avg, au_avg = self._calculate_market_averages(results)
            f.write(f"- US Market Average R²: {us_avg:.4f}\n")
            f.write(f"- AU Market Average R²: {au_avg:.4f}\n\n")
            
            # Detailed results table
            f.write("### Detailed Results\n\n")
            f.write("See `model_performance.csv` for complete results.\n\n")
            
            # Key insights
            f.write("## Key Insights\n\n")
            insights = self._generate_insights(results)
            for insight in insights:
                f.write(f"- {insight}\n")
            
        print(f"✅ Summary report saved to {report_path}")
        
    def _find_best_model(self, results: dict, task: str, metric: str) -> dict:
        """Find best performing model across all assets"""
        best = {'score': -1 if metric == 'R2' else 0}
        
        for asset, asset_results in results.items():
            if task in asset_results:
                for model, metrics in asset_results[task].items():
                    if metric in metrics:
                        score = metrics[metric]
                        if (metric == 'R2' and score > best['score']) or \
                           (metric != 'R2' and score > best['score']):
                            best = {
                                'asset': asset,
                                'model': model,
                                'score': score
                            }
        
        return best if best['score'] != -1 else None
        
    def _calculate_market_averages(self, results: dict) -> tuple:
        """Calculate average R² scores by market"""
        us_scores = []
        au_scores = []
        
        us_assets = self.config['data']['us_assets']
        
        for asset, asset_results in results.items():
            if 'regression' in asset_results:
                for model, metrics in asset_results['regression'].items():
                    if 'R2' in metrics:
                        if asset in us_assets:
                            us_scores.append(metrics['R2'])
                        else:
                            au_scores.append(metrics['R2'])
        
        us_avg = sum(us_scores) / len(us_scores) if us_scores else 0
        au_avg = sum(au_scores) / len(au_scores) if au_scores else 0
        
        return us_avg, au_avg
        
    def _generate_insights(self, results: dict) -> list:
        """Generate key insights from results"""
        insights = []
        
        # Model comparison insights
        regression_scores = {}
        for asset, asset_results in results.items():
            if 'regression' in asset_results:
                for model, metrics in asset_results['regression'].items():
                    if 'R2' in metrics:
                        if model not in regression_scores:
                            regression_scores[model] = []
                        regression_scores[model].append(metrics['R2'])
        
        # Average scores by model
        avg_scores = {model: sum(scores)/len(scores) 
                     for model, scores in regression_scores.items() 
                     if scores}
        
        if avg_scores:
            best_model = max(avg_scores.items(), key=lambda x: x[1])
            worst_model = min(avg_scores.items(), key=lambda x: x[1])
            
            insights.append(
                f"{best_model[0]} achieved the highest average R² score of {best_model[1]:.3f}"
            )
            insights.append(
                f"{worst_model[0]} had the lowest average R² score of {worst_model[1]:.3f}"
            )
            
        # Market insights
        us_avg, au_avg = self._calculate_market_averages(results)
        if us_avg > au_avg:
            insights.append(
                f"US market models performed {((us_avg/au_avg - 1) * 100):.1f}% better on average"
            )
        else:
            insights.append(
                f"AU market models performed {((au_avg/us_avg - 1) * 100):.1f}% better on average"
            )
            
        # Deep learning vs traditional
        dl_models = ['LSTM', 'StockMixer']
        traditional = ['ARIMA', 'Naive_MA']
        
        dl_scores = [score for model, scores in regression_scores.items() 
                    if model in dl_models for score in scores]
        trad_scores = [score for model, scores in regression_scores.items() 
                      if model in traditional for score in scores]
        
        if dl_scores and trad_scores:
            dl_avg = sum(dl_scores) / len(dl_scores)
            trad_avg = sum(trad_scores) / len(trad_scores)
            
            if dl_avg > trad_avg:
                insights.append(
                    f"Deep learning models outperformed traditional models by "
                    f"{((dl_avg/trad_avg - 1) * 100):.1f}% on average"
                )
                
        return insights


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Run RiskPipeline for volatility forecasting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --quick           # Quick test with subset of data
  python run_pipeline.py --full            # Full pipeline on all assets
  python run_pipeline.py --asset AAPL      # Single asset analysis
  python run_pipeline.py --us-only         # US markets only
  python run_pipeline.py --au-only         # Australian markets only
        """
    )
    
    # Execution modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--quick', action='store_true',
                           help='Run quick test with subset of data')
    mode_group.add_argument('--full', action='store_true',
                           help='Run full pipeline on all assets')
    mode_group.add_argument('--asset', type=str,
                           help='Run pipeline for single asset (e.g., AAPL)')
    
    # Asset selection
    parser.add_argument('--us-only', action='store_true',
                       help='Process US assets only')
    parser.add_argument('--au-only', action='store_true',
                       help='Process Australian assets only')
    
    # Configuration
    parser.add_argument('--config', type=str,
                       help='Path to configuration file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Print header
    print("="*60)
    print("RiskPipeline - Volatility Forecasting Framework")
    print("Author: Gurudeep Singh Dhinjan")
    print("="*60)
    
    # Initialize runner
    runner = PipelineRunner(config_path=args.config)
    
    # Determine assets to process
    assets = None
    if args.us_only:
        assets = runner.config['data']['us_assets']
    elif args.au_only:
        assets = runner.config['data']['au_assets']
    
    # Execute based on mode
    if args.quick:
        runner.run_quick_test()
    elif args.full:
        runner.run_full_pipeline(assets=assets)
    elif args.asset:
        # Validate asset
        all_assets = runner.config['data']['us_assets'] + runner.config['data']['au_assets']
        if args.asset not in all_assets:
            print(f"❌ Unknown asset: {args.asset}")
            print(f"Available assets: {', '.join(all_assets)}")
            sys.exit(1)
        runner.run_single_asset(args.asset)
    
    print("\n" + "="*60)
    print("Pipeline execution completed!")
    print("="*60)


if __name__ == "__main__":
    main()
