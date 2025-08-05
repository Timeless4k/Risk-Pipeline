"""
Main entry point for RiskPipeline execution.
"""

import argparse
import logging
import json
from pathlib import Path
from risk_pipeline import RiskPipeline


def main():
    """Main execution function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='RiskPipeline - Volatility Forecasting')
    parser.add_argument('--config', type=str, default='configs/pipeline_config.json',
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['full', 'quick', 'train', 'evaluate'],
                       default='full', help='Execution mode')
    parser.add_argument('--assets', nargs='+', help='Override assets from config')
    parser.add_argument('--models', nargs='+', help='Override models to run')
    parser.add_argument('--skip-shap', action='store_true', help='Skip SHAP analysis')
    parser.add_argument('--experiment-name', type=str, help='Name for experiment tracking')
    parser.add_argument('--output-dir', type=str, help='Override output directory')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
        print(f"✅ Loaded configuration from {args.config}")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return
    
    # Override assets if specified
    if args.assets:
        config['data']['us_assets'] = [a for a in args.assets if not a.endswith('.AX')]
        config['data']['au_assets'] = [a for a in args.assets if a.endswith('.AX')]
        print(f"📊 Using assets: {args.assets}")
    
    # Override output directory if specified
    if args.output_dir:
        config['output']['results_dir'] = args.output_dir
        config['output']['plots_dir'] = f"{args.output_dir}/visualizations"
        config['output']['shap_dir'] = f"{args.output_dir}/shap_plots"
        config['output']['models_dir'] = f"{args.output_dir}/models"
        config['output']['log_dir'] = f"{args.output_dir}/logs"
        print(f"📁 Output directory: {args.output_dir}")
    
    # Initialize pipeline
    try:
        pipeline = RiskPipeline(config=config, experiment_name=args.experiment_name)
        print("✅ RiskPipeline initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        return
    
    # Execute based on mode
    try:
        if args.mode == 'quick':
            print("🚀 Running quick test...")
            results = pipeline.run_quick_test(assets=args.assets)
            
        elif args.mode == 'train':
            print("🚀 Training models only...")
            models_to_run = args.models or ['arima', 'lstm', 'xgboost', 'stockmixer']
            results = pipeline.train_models_only(
                assets=args.assets or config['data']['us_assets'] + config['data']['au_assets'],
                models=models_to_run
            )
            
        elif args.mode == 'evaluate':
            print("🚀 Evaluating saved models...")
            experiment_id = input("Enter experiment ID to evaluate: ")
            results = pipeline.analyze_saved_models(experiment_id)
            
        else:  # full mode
            print("🚀 Running complete pipeline...")
            results = pipeline.run_complete_pipeline(
                assets=args.assets,
                models=args.models,
                run_shap=not args.skip_shap
            )
        
        print("\n✅ Pipeline execution completed!")
        print(f"📊 Results saved to: {config['output']['results_dir']}")
        print(f"📈 Visualizations saved to: {config['output']['plots_dir']}")
        if not args.skip_shap:
            print(f"🔍 SHAP plots saved to: {config['output']['shap_dir']}")
        
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        logging.error(f"Pipeline execution failed: {e}", exc_info=True)
        return


if __name__ == "__main__":
    main() 