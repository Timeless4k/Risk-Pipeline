#!/usr/bin/env python3
"""
Setup script for RiskPipeline environment
Handles dependency installation, directory creation, and initial configuration
"""

import os
import sys
import subprocess
import platform
import json
from pathlib import Path
import argparse


class PipelineSetup:
    """Automated setup for RiskPipeline environment"""
    
    def __init__(self):
        self.root_dir = Path.cwd()
        self.python_version = sys.version_info
        self.os_type = platform.system()
        
    def check_python_version(self):
        """Ensure Python version is compatible"""
        print(f"Checking Python version...")
        if self.python_version < (3, 8):
            print(f"❌ Python 3.8+ required. Current version: {self.python_version.major}.{self.python_version.minor}")
            sys.exit(1)
        print(f"✅ Python {self.python_version.major}.{self.python_version.minor} detected")
        
    def create_directory_structure(self):
        """Create required directory structure"""
        print("\nCreating directory structure...")
        
        directories = [
            'data_cache',
            'results',
            'shap_plots',
            'visualizations',
            'logs',
            'models',
            'configs',
            'tests',
            'notebooks',
            'publication_plots'
        ]
        
        for dir_name in directories:
            dir_path = self.root_dir / dir_name
            dir_path.mkdir(exist_ok=True)
            print(f"  ✅ Created {dir_name}/")
            
    def install_dependencies(self, use_gpu=False):
        """Install required Python packages"""
        print("\nInstalling dependencies...")
        
        # Check if in virtual environment
        if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            print("⚠️  WARNING: Not in a virtual environment. Consider creating one:")
            print("  python -m venv venv")
            print("  source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
            response = input("\nContinue anyway? (y/n): ")
            if response.lower() != 'y':
                sys.exit(0)
        
        # Upgrade pip
        print("Upgrading pip...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
        
        # Install requirements
        print("Installing requirements...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        
        # Install GPU-specific packages if requested
        if use_gpu:
            print("Installing GPU support...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorflow-gpu==2.13.0'])
            
    def create_config_file(self):
        """Create default configuration file"""
        print("\nCreating default configuration...")
        
        config = {
            "data": {
                "start_date": "2017-01-01",
                "end_date": "2024-03-31",
                "us_assets": ["AAPL", "MSFT", "^GSPC"],
                "au_assets": ["IOZ.AX", "CBA.AX", "BHP.AX"],
                "cache_dir": "data_cache"
            },
            "features": {
                "volatility_window": 5,
                "ma_short": 10,
                "ma_long": 50,
                "correlation_window": 30,
                "sequence_length": 20
            },
            "models": {
                "lstm_units": [50, 30],
                "lstm_dropout": 0.2,
                "stockmixer_temporal_units": 64,
                "stockmixer_indicator_units": 64,
                "stockmixer_cross_stock_units": 64,
                "stockmixer_fusion_units": 128,
                "xgboost_n_estimators": 100,
                "xgboost_max_depth": 5
            },
            "training": {
                "walk_forward_splits": 5,
                "test_size": 252,
                "batch_size": 32,
                "epochs": 50,
                "early_stopping_patience": 10,
                "reduce_lr_patience": 5,
                "random_state": 42
            },
            "output": {
                "results_dir": "results",
                "plots_dir": "visualizations",
                "shap_dir": "shap_plots",
                "log_dir": "logs"
            }
        }
        
        config_path = self.root_dir / 'configs' / 'pipeline_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"✅ Configuration saved to {config_path}")
        
    def download_sample_data(self):
        """Download a small sample of data for testing"""
        print("\nDownloading sample data for testing...")
        
        try:
            import yfinance as yf
            
            # Download 1 month of data for testing
            test_symbols = ['AAPL', '^VIX']
            for symbol in test_symbols:
                print(f"  Downloading {symbol}...")
                data = yf.download(
                    symbol, 
                    start='2024-01-01', 
                    end='2024-02-01',
                    progress=False
                )
                
                # Save to cache
                cache_file = self.root_dir / 'data_cache' / f'{symbol.replace("^", "")}_test.pkl'
                data.to_pickle(cache_file)
                
            print("✅ Sample data downloaded")
            
        except Exception as e:
            print(f"⚠️  Could not download sample data: {e}")
            
    def verify_installation(self):
        """Verify all components are properly installed"""
        print("\nVerifying installation...")
        
        # Check critical imports
        critical_packages = [
            'numpy',
            'pandas',
            'sklearn',
            'tensorflow',
            'xgboost',
            'statsmodels',
            'shap',
            'yfinance'
        ]
        
        failed = []
        for package in critical_packages:
            try:
                __import__(package)
                print(f"  ✅ {package}")
            except ImportError:
                failed.append(package)
                print(f"  ❌ {package}")
        
        if failed:
            print(f"\n❌ Failed to import: {', '.join(failed)}")
            print("Please check your installation.")
            sys.exit(1)
        
        # Check TensorFlow GPU support
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"\n✅ GPU support detected: {len(gpus)} GPU(s) available")
            else:
                print("\n⚠️  No GPU detected. Models will run on CPU.")
        except:
            pass
            
        print("\n✅ All components verified successfully!")
        
    def create_example_notebook(self):
        """Create a Jupyter notebook with usage examples"""
        print("\nCreating example notebook...")
        
        notebook_content = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "# RiskPipeline - Quick Start Guide\n",
                        "\n",
                        "This notebook demonstrates how to use the RiskPipeline for volatility forecasting.\n",
                        "\n",
                        "## 1. Basic Usage"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Import the pipeline\n",
                        "from risk_pipeline import RiskPipeline, AssetConfig\n",
                        "import pandas as pd\n",
                        "import matplotlib.pyplot as plt\n",
                        "\n",
                        "# Initialize pipeline\n",
                        "pipeline = RiskPipeline()\n",
                        "\n",
                        "print(\"Pipeline initialized successfully!\")"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## 2. Run Complete Pipeline\n",
                        "\n",
                        "Run the full pipeline on all assets:"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Run pipeline on subset of assets for quick test\n",
                        "test_assets = ['AAPL', 'IOZ.AX']\n",
                        "pipeline.run_pipeline(assets=test_assets)\n",
                        "\n",
                        "print(\"\\nPipeline execution completed!\")"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## 3. Visualize Results"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Load and display results\n",
                        "results_df = pd.read_csv('results/model_performance.csv')\n",
                        "print(\"Model Performance Summary:\")\n",
                        "print(results_df.head(10))\n",
                        "\n",
                        "# Plot performance comparison\n",
                        "from visualization import VolatilityVisualizer\n",
                        "visualizer = VolatilityVisualizer()\n",
                        "visualizer.plot_performance_comparison(pipeline.results, 'regression')"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## 4. Custom Analysis\n",
                        "\n",
                        "Example of running analysis on a single asset:"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Analyze specific asset\n",
                        "asset = 'AAPL'\n",
                        "if asset in pipeline.results:\n",
                        "    reg_results = pipeline.results[asset]['regression']\n",
                        "    \n",
                        "    # Compare models\n",
                        "    for model, metrics in reg_results.items():\n",
                        "        if 'R2' in metrics:\n",
                        "            print(f\"{model}: R² = {metrics['R2']:.4f}, RMSE = {metrics['RMSE']:.4f}\")"
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.8.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        notebook_path = self.root_dir / 'notebooks' / 'quick_start.ipynb'
        with open(notebook_path, 'w') as f:
            json.dump(notebook_content, f, indent=2)
            
        print(f"✅ Example notebook created at {notebook_path}")
        
    def run_setup(self, args):
        """Run complete setup process"""
        print("="*60)
        print("RiskPipeline Setup")
        print("="*60)
        
        # Run setup steps
        self.check_python_version()
        self.create_directory_structure()
        
        if not args.skip_install:
            self.install_dependencies(use_gpu=args.gpu)
            
        self.create_config_file()
        
        if args.download_sample:
            self.download_sample_data()
            
        self.verify_installation()
        
        if args.create_notebook:
            self.create_example_notebook()
        
        print("\n" + "="*60)
        print("✅ Setup completed successfully!")
        print("="*60)
        print("\nNext steps:")
        print("1. Review configuration in configs/pipeline_config.json")
        print("2. Run the pipeline with: python run_pipeline.py")
        print("3. Check results in the results/ directory")
        print("4. View visualizations in visualizations/ directory")
        
        if args.create_notebook:
            print("5. Open the example notebook: jupyter notebook notebooks/quick_start.ipynb")


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description='Setup RiskPipeline environment')
    parser.add_argument('--gpu', action='store_true', 
                       help='Install GPU support for TensorFlow')
    parser.add_argument('--skip-install', action='store_true',
                       help='Skip package installation')
    parser.add_argument('--download-sample', action='store_true',
                       help='Download sample data for testing')
    parser.add_argument('--create-notebook', action='store_true',
                       help='Create example Jupyter notebook')
    
    args = parser.parse_args()
    
    setup = PipelineSetup()
    setup.run_setup(args)


if __name__ == "__main__":
    main()
