# 🚀 RiskPipeline - Comprehensive Risk Analysis Pipeline

A comprehensive machine learning pipeline for risk analysis with **EVERYTHING ENABLED BY DEFAULT** for maximum performance and analysis depth.

## 🎯 **NEW: COMPREHENSIVE CLI MENU**

**`run_pipeline.py` is now your CENTRAL CONTROL SCRIPT with:**
- ✅ **ALL FEATURES ENABLED BY DEFAULT**
- ✅ **MAXIMUM PERFORMANCE SETTINGS (8 parallel workers)**
- ✅ **ALL MODELS: ARIMA, XGBoost, StockMixer, LSTM**
- ✅ **COMPREHENSIVE ANALYSIS: SHAP, visualizations, volatility**
- ✅ **VERBOSE LOGGING: Full detailed output**
- ✅ **ONE-CLICK "RUN EVERYTHING" OPTION**

## 📁 Project Structure

```
Risk-Pipeline/
├── 🚀 run_pipeline.py              # SINGLE COMPREHENSIVE PIPELINE SCRIPT
├── 📊 risk_pipeline/               # Core pipeline modules
│   ├── core/                       # Core pipeline components
│   ├── models/                     # ML model implementations
│   ├── visualization/              # Plotting and visualization
│   └── interpretability/           # SHAP and model interpretation
├── 📁 artifacts/                   # Output results and models
├── 📁 tests/                       # Test suite
└── 📚 docs/                        # Documentation
```

## 🚀 Quick Start (NEW SIMPLIFIED WORKFLOW)

### **Option 1: Run Everything (Recommended)**
```bash
python run_pipeline.py --run-all
```
**This runs everything with maximum settings - no configuration needed!**

### **Option 2: Interactive Menu**
```bash
python run_pipeline.py
```
**Choose from the comprehensive menu system**

### **Option 3: Direct Script Execution**
```bash
# Run main pipeline directly
python run_pipeline.py --run-all

# Use interactive menu directly
python run_pipeline.py
```

### **Option 4: Custom Configuration**
```bash
python run_pipeline.py --custom
```
**Tune individual settings as needed**

## 🎮 **NEW COMPREHENSIVE MENU SYSTEM**

The main menu now provides:

1. **🚀 RUN EVERYTHING** - All features, maximum performance (RECOMMENDED)
2. **⚡ QUICK RUN** - Skip settings, use defaults
3. **🔧 CUSTOM CONFIGURATION** - Tune individual settings
4. **📊 DATA CONFIGURATION** - Change data source
5. **🤖 MODEL SELECTION** - Choose which models to run
6. **⚙️ PERFORMANCE TUNING** - Adjust workers, CPU mode
7. **📁 OUTPUT SETTINGS** - Change directories, logging
8. **❓ HELP & INFORMATION** - Comprehensive guidance
9. **🚪 EXIT**

## 🎯 What You Get (Everything Enabled by Default)

✅ **ALL MODELS**: ARIMA, XGBoost, StockMixer, LSTM (if available)  
✅ **MAXIMUM PERFORMANCE**: 8 parallel workers by default  
✅ **SHAP ANALYSIS**: Full model interpretability and feature importance  
✅ **COMPREHENSIVE VISUALIZATIONS**: All plots, charts, and analysis  
✅ **VOLATILITY ANALYSIS**: Advanced financial analysis and plots  
✅ **MODEL PERSISTENCE**: Save all trained models for future use  
✅ **VERBOSE LOGGING**: Full detailed output for debugging  
✅ **COMPREHENSIVE REPORT**: Complete analysis summary  

## 🔧 **DEFAULT CONFIGURATION (Everything ON)**

| Setting | Default Value | Description |
|---------|---------------|-------------|
| **Models** | All Available | ARIMA, XGBoost, StockMixer, LSTM |
| **Parallel Workers** | 8 | Maximum performance processing |
| **SHAP Analysis** | ✅ Enabled | Full model interpretability |
| **Visualizations** | ✅ Enabled | All plots and charts |
| **Volatility Analysis** | ✅ Enabled | Financial analysis features |
| **Model Saving** | ✅ Enabled | Persist all trained models |
| **Verbose Logging** | ✅ Enabled | Full detailed output |
| **Comprehensive Report** | ✅ Enabled | Complete analysis summary |

## 🚀 **Usage Examples (Simplified)**

### **1. Run Everything (One Command)**
```bash
python run_pipeline.py --run-all
```
*This is all you need for comprehensive analysis!*

### **2. Interactive Menu**
```bash
python run_pipeline.py
# Choose option 1: "RUN EVERYTHING"
```

### **3. Custom Data Analysis**
```bash
python run_pipeline.py
# Navigate to Data Configuration → Custom CSV
# Then choose "RUN EVERYTHING"
```

### **4. Performance Tuning**
```bash
python run_pipeline.py
# Navigate to Performance Tuning
# Adjust workers, CPU mode, etc.
# Then choose "RUN EVERYTHING"
```

## 🔧 **Customization Options**

### **Performance Tuning**
- **Parallel Workers**: 1-16 (default: 8 maximum)
- **CPU Only Mode**: Force CPU processing
- **Memory Optimization**: Automatic handling

### **Model Selection**
- **Enable/Disable All**: One-click toggle
- **Individual Models**: Pick specific models
- **Automatic Detection**: LSTM availability checked

### **Feature Toggles**
- **SHAP Analysis**: Model interpretability
- **Visualizations**: All plots and charts
- **Volatility Analysis**: Financial analysis
- **Model Persistence**: Save trained models
- **Verbose Logging**: Detailed output
- **Comprehensive Reports**: Full analysis

## 📊 **Data Sources**

### **Demo Data (Default)**
Built-in sample data for immediate testing.

### **Custom CSV**
- Any time series CSV file
- Automatic date column detection
- Flexible data format handling

## 📈 **Output & Results**

When you run the pipeline, you'll get:

```
artifacts/comprehensive_run/
├── 📊 config.json              # Pipeline configuration
├── 🌍 env.json                 # Environment snapshot
├── 📋 splits.json              # Data split information
├── 📊 results.csv              # Model performance results
├── 📋 RESULTS.md               # Detailed results summary
├── 📈 visualizations/          # All generated plots
├── 🔍 shap_analysis/           # SHAP analysis results
├── 📊 volatility_analysis/     # Volatility analysis plots
├── 💾 models/                  # Saved trained models
└── 📝 pipeline.log             # Detailed execution log
```

## 🎯 **Key Benefits of New System**

1. **🚀 Everything Enabled by Default** - No configuration needed
2. **⚡ One-Click "Run Everything"** - Maximum performance instantly
3. **🔧 Full Customization** - Tune any setting as needed
4. **📊 Maximum Performance** - 8 parallel workers by default
5. **🎮 Interactive Menu** - Easy navigation and configuration
6. **📝 Verbose Logging** - Full debugging and monitoring
7. **💾 Complete Persistence** - Save all models and results

## 🚀 **Quick Start Guide**

### **For New Users:**
1. **Run**: `python run_pipeline.py --run-all`
2. **Done!** Everything runs with maximum settings

### **For Advanced Users:**
1. **Run**: `python run_pipeline.py`
2. **Navigate** through the comprehensive menu
3. **Customize** any setting as needed
4. **Execute** with your configuration

### **For Developers:**
1. **Direct execution**: `python risk_pipeline_main.py --run-all`
2. **Custom scripts**: Use the tools in `tools/` folder
3. **API integration**: Import and use pipeline modules directly

## 🐛 **Troubleshooting**

### **Common Issues**

1. **TensorFlow Not Available**
   - LSTM models automatically disabled
   - Other models work normally
   - Message displayed in CLI

2. **Memory Issues**
   - Use CPU-only mode in Performance Tuning
   - Reduce parallel workers
   - All handled in the menu system

3. **File Not Found Errors**
   - Ensure you're in Risk-Pipeline root directory
   - Check file paths in Data Configuration

### **Getting Help**

```bash
# Show help
python run_pipeline.py --help

# Interactive help
python run_pipeline.py
# Then select "Help & Information"
```

## 🚀 **Advanced Usage**

### **Batch Processing**
```bash
# Process multiple datasets
for dataset in data/*.csv; do
    python run_pipeline.py --run-all
    # Configure CSV in interactive menu
done
```

### **Custom Configurations**
```bash
# Start with custom settings
python run_pipeline.py --custom
# Navigate to specific configuration areas
```

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 **License**

This project is licensed under the MIT License - see the [license](license) file for details.

## 🆘 **Support**

- **Documentation**: Check this README and run `python run_pipeline.py --help`
- **Interactive Help**: Use the built-in help system in the CLI menu
- **Issues**: Report bugs and feature requests on GitHub

---

## 🎯 **The Bottom Line**

**You now have a RiskPipeline that:**
- 🚀 **Runs everything by default** with maximum performance
- ⚡ **Requires zero configuration** for comprehensive analysis
- 🔧 **Provides full customization** when you need it
- 🎮 **Offers intuitive navigation** through comprehensive menus
- 📊 **Delivers maximum results** with all features enabled

**🎯 Start using RiskPipeline now: `python run_pipeline.py --run-all`**

*This transforms your RiskPipeline into a comprehensive analysis platform that works out of the box with everything enabled, while maintaining full customization capabilities for advanced users.*
