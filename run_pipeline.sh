#!/bin/bash

echo ""
echo "========================================"
echo "    🚀 RiskPipeline - COMPREHENSIVE CLI"
echo "========================================"
echo ""
echo "Starting RiskPipeline with everything enabled..."
echo ""
echo "Options:"
echo "  - Press Enter for interactive menu"
echo "  - Or run: python run_pipeline.py --run-all"
echo ""

# Check if Python is available
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Error: Python not found. Please install Python 3.7+"
    exit 1
fi

# Run the pipeline launcher
$PYTHON_CMD run_pipeline.py

echo ""
echo "Pipeline completed. Press Enter to continue..."
read
