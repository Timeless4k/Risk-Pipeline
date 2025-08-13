#!/bin/bash

echo "Starting RiskPipeline..."
echo

# Check if virtual environment exists and activate it
if [ -f "venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Virtual environment not found. Please run 'python -m venv venv' first."
    exit 1
fi

# Run the simple pipeline
echo "Running RiskPipeline with maximum performance..."
python run_simple_pipeline.py

# Check exit status
if [ $? -ne 0 ]; then
    echo
    echo "An error occurred. Press Enter to exit..."
    read
fi
