"""
Command-line interface for RiskPipeline with subcommands.
"""

import click
import json
import logging
from pathlib import Path
from risk_pipeline import RiskPipeline


@click.group()
def cli():
    """RiskPipeline CLI for volatility forecasting"""
    pass


@cli.command()
@click.option('--config', default='configs/pipeline_config.json', help='Config file path')
@click.option('--assets', multiple=True, help='Assets to process')
@click.option('--models', multiple=True, help='Models to run')
@click.option('--skip-shap', is_flag=True, help='Skip SHAP analysis')
@click.option('--experiment-name', help='Name for experiment tracking')
@click.option('--output-dir', help='Override output directory')
def run(config, assets, models, skip_shap, experiment_name, output_dir):
    """Run complete pipeline"""
    try:
        # Load configuration
        with open(config, 'r') as f:
            config_data = json.load(f)
        
        # Override assets if specified
        if assets:
            config_data['data']['us_assets'] = [a for a in assets if not a.endswith('.AX')]
            config_data['data']['au_assets'] = [a for a in assets if a.endswith('.AX')]
            click.echo(f"📊 Using assets: {list(assets)}")
        
        # Override output directory if specified
        if output_dir:
            config_data['output']['results_dir'] = output_dir
            config_data['output']['plots_dir'] = f"{output_dir}/visualizations"
            config_data['output']['shap_dir'] = f"{output_dir}/shap_plots"
            config_data['output']['models_dir'] = f"{output_dir}/models"
            config_data['output']['log_dir'] = f"{output_dir}/logs"
            click.echo(f"📁 Output directory: {output_dir}")
        
        # Initialize pipeline
        pipeline = RiskPipeline(config=config_data, experiment_name=experiment_name)
        
        # Run pipeline
        results = pipeline.run_complete_pipeline(
            assets=list(assets) if assets else None,
            models=list(models) if models else None,
            run_shap=not skip_shap
        )
        
        click.echo("✅ Pipeline completed successfully!")
        
    except Exception as e:
        click.echo(f"❌ Pipeline failed: {e}")
        raise


@cli.command()
@click.option('--config', default='configs/quick_test_config.json', help='Config file path')
@click.option('--assets', multiple=True, help='Assets to process')
@click.option('--experiment-name', help='Name for experiment tracking')
def test(config, assets, experiment_name):
    """Run quick test"""
    try:
        # Load configuration
        with open(config, 'r') as f:
            config_data = json.load(f)
        
        # Override assets if specified
        if assets:
            config_data['data']['us_assets'] = [a for a in assets if not a.endswith('.AX')]
            config_data['data']['au_assets'] = [a for a in assets if a.endswith('.AX')]
            click.echo(f"📊 Using assets: {list(assets)}")
        
        # Initialize pipeline
        pipeline = RiskPipeline(config=config_data, experiment_name=experiment_name)
        
        # Run quick test
        results = pipeline.run_quick_test(assets=list(assets) if assets else None)
        
        click.echo("✅ Quick test completed successfully!")
        
    except Exception as e:
        click.echo(f"❌ Quick test failed: {e}")
        raise


@cli.command()
@click.argument('experiment_id')
@click.option('--config', default='configs/pipeline_config.json', help='Config file path')
def evaluate(experiment_id, config):
    """Evaluate saved experiment"""
    try:
        # Load configuration
        with open(config, 'r') as f:
            config_data = json.load(f)
        
        # Initialize pipeline
        pipeline = RiskPipeline(config=config_data)
        
        # Evaluate saved models
        results = pipeline.analyze_saved_models(experiment_id)
        
        click.echo(f"✅ Evaluation completed for experiment: {experiment_id}")
        
    except Exception as e:
        click.echo(f"❌ Evaluation failed: {e}")
        raise


@cli.command()
@click.argument('experiment_ids', nargs=-1, required=True)
@click.option('--config', default='configs/pipeline_config.json', help='Config file path')
def compare(experiment_ids, config):
    """Compare multiple experiments"""
    try:
        # Load configuration
        with open(config, 'r') as f:
            config_data = json.load(f)
        
        # Initialize pipeline
        pipeline = RiskPipeline(config=config_data)
        
        # Compare experiments
        results = pipeline.compare_experiments(list(experiment_ids))
        
        click.echo(f"✅ Comparison completed for {len(experiment_ids)} experiments")
        
    except Exception as e:
        click.echo(f"❌ Comparison failed: {e}")
        raise


@cli.command()
@click.option('--config', default='configs/pipeline_config.json', help='Config file path')
@click.option('--assets', multiple=True, help='Assets to process')
@click.option('--models', multiple=True, help='Models to train')
@click.option('--experiment-name', help='Name for experiment tracking')
def train(config, assets, models, experiment_name):
    """Train models only (no evaluation)"""
    try:
        # Load configuration
        with open(config, 'r') as f:
            config_data = json.load(f)
        
        # Override assets if specified
        if assets:
            config_data['data']['us_assets'] = [a for a in assets if not a.endswith('.AX')]
            config_data['data']['au_assets'] = [a for a in assets if a.endswith('.AX')]
            click.echo(f"📊 Using assets: {list(assets)}")
        
        # Initialize pipeline
        pipeline = RiskPipeline(config=config_data, experiment_name=experiment_name)
        
        # Train models
        models_to_run = list(models) if models else ['arima', 'lstm', 'xgboost', 'stockmixer']
        results = pipeline.train_models_only(
            assets=list(assets) if assets else config_data['data']['us_assets'] + config_data['data']['au_assets'],
            models=models_to_run
        )
        
        click.echo("✅ Model training completed successfully!")
        
    except Exception as e:
        click.echo(f"❌ Model training failed: {e}")
        raise


@cli.command()
@click.option('--config', default='configs/pipeline_config.json', help='Config file path')
def info(config):
    """Show pipeline information and configuration"""
    try:
        # Load configuration
        with open(config, 'r') as f:
            config_data = json.load(f)
        
        click.echo("="*60)
        click.echo("RISKPIPELINE INFORMATION")
        click.echo("="*60)
        
        # Data configuration
        click.echo("\n📊 DATA CONFIGURATION:")
        click.echo(f"  Start Date: {config_data['data']['start_date']}")
        click.echo(f"  End Date: {config_data['data']['end_date']}")
        click.echo(f"  US Assets: {', '.join(config_data['data']['us_assets'])}")
        click.echo(f"  AU Assets: {', '.join(config_data['data']['au_assets'])}")
        click.echo(f"  Cache Directory: {config_data['data']['cache_dir']}")
        
        # Feature configuration
        click.echo("\n🔧 FEATURE CONFIGURATION:")
        click.echo(f"  Volatility Window: {config_data['features']['volatility_window']}")
        click.echo(f"  MA Short: {config_data['features']['ma_short']}")
        click.echo(f"  MA Long: {config_data['features']['ma_long']}")
        click.echo(f"  Correlation Window: {config_data['features']['correlation_window']}")
        click.echo(f"  Sequence Length: {config_data['features']['sequence_length']}")
        
        # Training configuration
        click.echo("\n🎯 TRAINING CONFIGURATION:")
        click.echo(f"  Walk Forward Splits: {config_data['training']['walk_forward_splits']}")
        click.echo(f"  Test Size: {config_data['training']['test_size']}")
        click.echo(f"  Batch Size: {config_data['training']['batch_size']}")
        click.echo(f"  Epochs: {config_data['training']['epochs']}")
        click.echo(f"  Random State: {config_data['training']['random_state']}")
        
        # Output configuration
        click.echo("\n📁 OUTPUT CONFIGURATION:")
        click.echo(f"  Results Directory: {config_data['output']['results_dir']}")
        click.echo(f"  Plots Directory: {config_data['output']['plots_dir']}")
        click.echo(f"  SHAP Directory: {config_data['output']['shap_dir']}")
        click.echo(f"  Models Directory: {config_data['output']['models_dir']}")
        click.echo(f"  Log Directory: {config_data['output']['log_dir']}")
        
        click.echo("\n" + "="*60)
        
    except Exception as e:
        click.echo(f"❌ Failed to load configuration: {e}")
        raise


@cli.command()
@click.option('--config', default='configs/pipeline_config.json', help='Config file path')
def validate(config):
    """Validate configuration and dependencies"""
    try:
        # Load configuration
        with open(config, 'r') as f:
            config_data = json.load(f)
        
        click.echo("🔍 Validating configuration and dependencies...")
        
        # Check required directories
        required_dirs = [
            config_data['output']['results_dir'],
            config_data['output']['plots_dir'],
            config_data['output']['shap_dir'],
            config_data['output']['models_dir'],
            config_data['output']['log_dir'],
            config_data['data']['cache_dir']
        ]
        
        for dir_path in required_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            click.echo(f"✅ Directory ready: {dir_path}")
        
        # Check configuration structure
        required_sections = ['data', 'features', 'models', 'training', 'output']
        for section in required_sections:
            if section in config_data:
                click.echo(f"✅ Configuration section: {section}")
            else:
                click.echo(f"❌ Missing configuration section: {section}")
        
        # Try to initialize pipeline
        try:
            pipeline = RiskPipeline(config=config_data)
            click.echo("✅ Pipeline initialization successful")
        except Exception as e:
            click.echo(f"❌ Pipeline initialization failed: {e}")
        
        click.echo("✅ Configuration validation completed!")
        
    except Exception as e:
        click.echo(f"❌ Validation failed: {e}")
        raise


if __name__ == '__main__':
    cli() 