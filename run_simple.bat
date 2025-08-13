@echo off
echo Starting RiskPipeline...
echo.

REM Check if virtual environment exists and activate it
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Virtual environment not found. Please run 'python -m venv venv' first.
    pause
    exit /b 1
)

REM Run the simple pipeline
echo Running RiskPipeline with maximum performance...
python run_simple_pipeline.py

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo An error occurred. Press any key to exit...
    pause
)
