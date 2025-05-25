# RiskPipeline: Interpretable Machine Learning for Volatility Forecasting

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.13.0-orange)
![License](https://img.shields.io/badge/license-MIT-green)

A comprehensive framework for volatility prediction across US and Australian equity markets using interpretable machine learning, implementing the methodology from:

**"Interpretable Machine Learning for Volatility Forecasting: A Regime-Aware Evaluation of ARIMA, LSTM, and StockMixer Across U.S. and Australian Equity Markets"**

Author: Gurudeep Singh Dhinjan  
Student ID: 24555981  
University of Technology Sydney (UTS)

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Models](#models)
- [Results](#results)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

## 🎯 Overview

RiskPipeline is a production-ready implementation of an interpretable volatility forecasting framework that:

- **Forecasts short-term (5-day) volatility** for equity markets
- **Compares traditional and modern ML approaches** including ARIMA, LSTM, and StockMixer
- **Provides SHAP-based interpretability** for all complex models
- **Uses walk-forward cross-validation** to simulate realistic deployment
- **Analyzes performance across market regimes** (Bull, Bear, Sideways)
- **Evaluates cross-market transferability** between US and Australian markets

### Key Contributions

1. **Dual-task framework**: Both regression (continuous volatility) and classification (volatility regimes)
2. **Novel feature engineering**: Inter-asset correlations and market-wide risk indicators
3. **Regime-aware evaluation**: Performance analysis across different market conditions
4. **Production-ready pipeline**: Modular, scalable, and fully documented

## ✨ Key Features

### Data & Markets
- **US Assets**: S&P 500 (^GSPC), Apple (AAPL), Microsoft (MSFT)
- **Australian Assets**: ASX 200 ETF (IOZ.AX), Commonwealth Bank (CBA.AX), BHP Group (BHP.AX)
- **Time Period**: January 2017 - March 2024
- **Data Sources**: Yahoo Finance, with automatic caching

### Models Implemented
- **Traditional**: ARIMA, Naive Moving Average
- **Deep Learning**: LSTM (custom architecture), StockMixer (MLP-based)
- **Ensemble**: XGBoost
- **Baselines**: Random Classifier, Naive predictor

### Features
- **Technical**: Lagged returns, moving averages, rolling volatility
- **Macroeconomic**: VIX index and changes
- **Inter-asset**: Rolling correlations between key asset pairs
- **Regime indicators**: Market state classification

### Evaluation
- **Walk-forward cross-validation**: 5 sequential folds
- **Comprehensive metrics**: RMSE, MAE, R² (regression); Accuracy, F1, Precision, Recall (classification)
- **SHAP interpretability**: Global and local feature importance
- **Regime analysis**: Performance breakdown by market conditions

## 🏗️ Architecture

```
RiskPipeline Architecture
├── Data Ingestion
│   ├── Yahoo Finance API
│   ├── Caching system
│   └── Data validation
├── Feature Engineering
│   ├── Technical indicators
│   ├── VIX integration
│   └── Correlation features
├── Model Training
│   ├── Traditional models
│   ├── Deep learning
│   └── Ensemble methods
├── Evaluation
│   ├── Walk-forward validation
│   ├── Performance metrics
│   └── Regime analysis
├── Interpretability
│   ├── SHAP analysis
│   ├── Feature importance
│   └── Pathway analysis (StockMixer)
└── Visualization & Reporting
    ├── Performance plots
    ├── Time series analysis
    └── Interactive dashboard
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- 16GB RAM recommended
- GPU optional but recommended for deep learning models

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/risk-pipeline.git
cd risk-pipeline
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Run Setup Script
```bash
# Basic setup
python setup.py

# With GPU support
python setup.py --gpu

# Full setup with sample data
python setup.py --gpu --download-sample --create-notebook
```

### Step 4: Verify Installation
```bash
python -c "from risk_pipeline import RiskPipeline; print('✅ Installation successful!')"
```

## 🏃 Quick Start

### 1. Quick Test (Subset of Data)
```bash
python run_pipeline.py --quick
```
This runs a fast test on 2 assets with reduced data to verify everything works.

### 2. Full Pipeline
```bash
python run_pipeline.py --full
```
Runs the complete pipeline on all 6 assets with full historical data.

### 3. Single Asset Analysis
```bash
python run_pipeline.py --asset AAPL
```

### 4. Market-Specific Analysis
```bash
# US markets only
python run_pipeline.py --full --us-only

# Australian markets only
python run_pipeline.py --full --au-only
```

## 📖 Usage

### Python API

```python
from risk_pipeline import RiskPipeline, AssetConfig
from visualization import VolatilityVisualizer

# Initialize pipeline
pipeline = RiskPipeline()

# Run on specific assets
assets = ['AAPL', 'CBA.AX']
pipeline.run_pipeline(assets=assets)

# Access results
results = pipeline.results
print(f"AAPL LSTM R²: {results['AAPL']['regression']['LSTM']['R2']:.4f}")

# Generate visualizations
visualizer = VolatilityVisualizer()
visualizer.plot_performance_comparison(results, 'regression')
```

### Jupyter Notebook

```bash
jupyter notebook notebooks/quick_start.ipynb
```

### Custom Configuration

```python
# Load custom configuration
import json

with open('configs/custom_config.json', 'r') as f:
    config = json.load(f)

# Modify parameters
config['training']['epochs'] = 100
config['features']['sequence_length'] = 30

# Run with custom config
runner = PipelineRunner(config_path='configs/custom_config.json')
runner.run_full_pipeline()
```

## 🤖 Models

### 1. ARIMA (Baseline)
- Traditional time series model
- Order: (1,1,1) with automatic parameter selection
- Provides interpretable baseline performance

### 2. LSTM (Deep Learning)
- Architecture: 2 LSTM layers (50, 30 units) with dropout
- Input: 20-day sequences
- Captures long-term dependencies

### 3. StockMixer (Novel Architecture)
- Three parallel pathways:
  - **Temporal**: Time-based pattern extraction
  - **Indicator**: Technical indicator processing
  - **Cross-stock**: Inter-asset relationship modeling
- Lightweight MLP-based design
- Interpretable pathway contributions

### 4. XGBoost (Ensemble)
- Gradient boosting for classification
- 100 estimators, max depth 5
- Feature importance via SHAP

## 📊 Results

### Expected Performance (from thesis)

| Model | Average R² (Regression) | Average F1 (Classification) |
|-------|------------------------|---------------------------|
| Naive MA | 0.45-0.55 | N/A |
| ARIMA | 0.55-0.65 | N/A |
| LSTM | 0.70-0.80 | 0.75-0.85 |
| StockMixer | 0.75-0.85 | 0.80-0.90 |
| XGBoost | N/A | 0.80-0.88 |

### Output Files

After running the pipeline, you'll find:

```
results/
├── model_performance.csv      # Detailed metrics for all models
├── detailed_results.pkl       # Full results with predictions
├── summary_report.md          # Executive summary
└── performance_summary.png    # Overview visualization

shap_plots/
├── AAPL_shap_summary.png     # SHAP analysis per asset
├── CBA.AX_shap_summary.png
└── ...

visualizations/
├── regression_performance_comparison.png
├── classification_performance_comparison.png
├── walk_forward_validation.png
├── cross_market_analysis.png
└── interactive_dashboard.html
```

## 📁 Project Structure

```
risk-pipeline/
├── risk_pipeline.py          # Main pipeline implementation
├── stockmixer_model.py       # StockMixer architecture
├── visualization.py          # Plotting and analysis
├── run_pipeline.py          # Command-line interface
├── setup.py                 # Installation script
├── test_pipeline.py         # Unit tests
├── requirements.txt         # Dependencies
├── README.md               # This file
│
├── configs/                # Configuration files
│   └── pipeline_config.json
├── data_cache/            # Downloaded data cache
├── results/               # Output results
├── shap_plots/           # SHAP visualizations
├── visualizations/       # Performance plots
├── logs/                 # Execution logs
├── models/               # Saved models
├── notebooks/            # Jupyter notebooks
└── tests/                # Additional tests
```

## ⚙️ Configuration

Default configuration (`configs/pipeline_config.json`):

```json
{
    "data": {
        "start_date": "2017-01-01",
        "end_date": "2024-03-31",
        "us_assets": ["AAPL", "MSFT", "^GSPC"],
        "au_assets": ["IOZ.AX", "CBA.AX", "BHP.AX"]
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
        "stockmixer_temporal_units": 64
    },
    "training": {
        "walk_forward_splits": 5,
        "test_size": 252,
        "epochs": 50,
        "batch_size": 32
    }
}
```

## 🧪 Testing

### Run All Tests
```bash
python test_pipeline.py
```

### Run Specific Test
```bash
python -m unittest test_pipeline.TestFeatureEngineer
```

### Test Coverage
```bash
pytest test_pipeline.py --cov=risk_pipeline --cov-report=html
```

## 🔧 Troubleshooting

### Common Issues

1. **Import Error**
   ```
   ImportError: No module named 'risk_pipeline'
   ```
   Solution: Ensure you're in the correct directory and virtual environment is activated.

2. **Memory Error**
   ```
   MemoryError: Unable to allocate array
   ```
   Solution: Reduce batch size or number of assets being processed simultaneously.

3. **Data Download Failed**
   ```
   Error downloading AAPL: HTTPError
   ```
   Solution: Check internet connection. Data is cached after first download.

4. **GPU Not Detected**
   ```
   No GPU detected. Models will run on CPU.
   ```
   Solution: Install CUDA and tensorflow-gpu. This is optional - CPU works fine for this scale.

### Debug Mode

```bash
# Run with verbose logging
python run_pipeline.py --full --debug

# Check logs
tail -f logs/pipeline_run_*.log
```

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@thesis{dhinjan2025interpretable,
  title={Interpretable Machine Learning for Volatility Forecasting: 
         A Regime-Aware Evaluation of ARIMA, LSTM, and StockMixer 
         Across U.S. and Australian Equity Markets},
  author={Dhinjan, Gurudeep Singh},
  year={2025},
  school={University of Technology Sydney},
  type={Honours Thesis}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 💬 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the test outputs
3. Check existing issues on GitHub
4. Create a new issue with detailed information

## 🙏 Acknowledgments

- Dr. Gnana Bharathy (Supervisor)
- University of Technology Sydney
- Authors of StockMixer, SHAP, and other referenced papers

---

**Note**: This implementation is for academic research purposes. Always perform your own due diligence before using any model for actual trading decisions.