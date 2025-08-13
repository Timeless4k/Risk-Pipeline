# 🚀 RiskPipeline CLI - Interactive Command Line Interface

A powerful, user-friendly command-line interface for running the RiskPipeline machine learning framework with various configurations and options.

## ✨ Features

- **🎯 Interactive Menu System** - Easy-to-use menu-driven interface
- **🚀 Max Performance Mode** - Default mode with optimal settings for best results
- **⚡ Quick Test Mode** - Fast validation for development and testing
- **🔧 Custom Configuration** - Fine-tune all pipeline parameters
- **📊 System Monitoring** - Real-time resource usage tracking
- **⚙️ Preset Management** - Save and manage custom configurations
- **🎨 Beautiful UI** - Colored output with emojis and clear formatting
- **🔄 Multiple Run Modes** - Interactive, command-line, and automation support

## 🚀 Quick Start

### 1. Interactive Mode (Recommended)
```bash
# Start the interactive CLI
python run_pipeline_cli.py
```

### 2. Direct Execution Modes
```bash
# Run with maximum performance settings
python run_pipeline_cli.py --max-performance

# Run quick test mode
python run_pipeline_cli.py --quick-test

# Run with custom configuration
python run_pipeline_cli.py --custom

# Non-interactive mode for automation
python run_pipeline_cli.py --non-interactive
```

### 3. Using the Scripts
```bash
# Windows
run_cli.bat

# Linux/Mac
./run_cli.sh
```

## 📋 Available Modes

### 🚀 Max Performance (Default)
- **Training Splits**: 10
- **Test Size**: 126 days
- **Epochs**: 200
- **Models**: All (ARIMA, LSTM, StockMixer, XGBoost)
- **SHAP Analysis**: Yes
- **Model Saving**: Yes
- **Assets**: All (US + AU markets)

### ⚡ Quick Test
- **Training Splits**: 3
- **Test Size**: 50 days
- **Epochs**: 50
- **Models**: XGBoost, LSTM
- **SHAP Analysis**: No
- **Model Saving**: No
- **Assets**: First 2 assets only

### 🏭 Production
- **Training Splits**: 7
- **Test Size**: 100 days
- **Epochs**: 150
- **Models**: All
- **SHAP Analysis**: Yes
- **Model Saving**: Yes
- **Assets**: All

### 🔬 Research
- **Training Splits**: 15
- **Test Size**: 200 days
- **Epochs**: 300
- **Models**: All
- **SHAP Analysis**: Yes
- **Model Saving**: Yes
- **Assets**: All

## 🎮 Interactive Menu Options

1. **🚀 Run Max Performance Pipeline** - Start with optimal settings
2. **⚡ Quick Test Mode** - Fast validation run
3. **🔧 Custom Configuration** - Create your own setup
4. **📊 View System Status** - Monitor resources
5. **⚙️ Manage Presets** - Save/load configurations
6. **📁 View Results** - Check pipeline outputs
7. **❓ Help & Information** - Get detailed help
8. **🚪 Exit** - Close the CLI

## 🔧 Custom Configuration Options

### Training Parameters
- Walk-forward splits (number of validation folds)
- Test size (days)
- Number of epochs
- Early stopping patience

### Model Selection
- **ARIMA**: Time series forecasting
- **LSTM**: Deep learning sequences
- **StockMixer**: Advanced multi-stock model
- **XGBoost**: Gradient boosting

### Pipeline Features
- SHAP interpretability analysis
- Model persistence
- Asset selection (US, AU, or custom)
- Feature engineering options

### Asset Selection
- **All**: Complete US + AU market coverage
- **US Only**: AAPL, MSFT, ^GSPC
- **AU Only**: IOZ.AX, CBA.AX, BHP.AX
- **Custom**: User-defined symbols

## 💻 System Requirements

- **Python**: 3.8+
- **Dependencies**: All packages from `requirements.txt`
- **Memory**: 8GB+ RAM recommended
- **Storage**: 10GB+ free space for data and models
- **Platform**: Windows, Linux, macOS

## 🚀 Performance Optimization

The CLI automatically detects your system capabilities:
- **CPU Cores**: Optimizes parallel processing
- **Memory**: Monitors usage during execution
- **Platform**: Adapts to OS-specific optimizations

## 📁 Output Structure

```
results/
├── experiments/
│   └── cli_run_[timestamp]/
│       ├── models/
│       ├── features/
│       ├── shap_plots/
│       ├── visualizations/
│       └── metadata.json
├── logs/
└── plots/
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure you're in the project root directory
   - Activate virtual environment: `source venv/bin/activate`
   - Install dependencies: `pip install -r requirements.txt`

2. **Memory Issues**
   - Use Quick Test mode for limited resources
   - Reduce training splits or test size
   - Monitor system resources during execution

3. **Model Availability**
   - Some models may not be available due to missing dependencies
   - Check the warning messages for specific issues

### Getting Help

- Use the built-in help system (Option 7 in main menu)
- Check logs in the `logs/` directory
- Monitor system status (Option 4 in main menu)

## 🎯 Best Practices

1. **Start with Max Performance** - Best default for most use cases
2. **Use Quick Test** - For development and validation
3. **Monitor Resources** - Check system status before long runs
4. **Save Presets** - Create custom configurations for repeated use
5. **Review Results** - Always check outputs and visualizations

## 🔄 Automation

For automated runs, use the non-interactive mode:

```bash
# Run max performance automatically
python run_pipeline_cli.py --non-interactive

# Use in scripts
python run_pipeline_cli.py --max-performance > pipeline.log 2>&1
```

## 📊 Example Workflow

1. **Start CLI**: `python run_pipeline_cli.py`
2. **Choose Mode**: Select "Max Performance" (Option 1)
3. **Confirm**: Accept the configuration
4. **Monitor**: Watch progress and system resources
5. **Review**: Check results and visualizations
6. **Save**: Models and results are automatically saved

## 🎉 What You Get

- **Trained Models**: All selected models for each asset
- **Performance Metrics**: Comprehensive evaluation results
- **SHAP Analysis**: Feature importance and interpretability
- **Visualizations**: Charts, plots, and analysis graphs
- **Experiment Tracking**: Organized results with metadata
- **Export Options**: CSV, JSON, and other formats

---

**Ready to run your risk analysis pipeline?** 🚀

Start with: `python run_pipeline_cli.py`
