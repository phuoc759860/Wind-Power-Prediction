@echo off
title AMG Wind Power Forecasting System
cd /d "%~dp0"

echo ========================================
echo AMG Wind Power Forecasting System
echo ========================================
echo.
echo [1/4] Checking Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python 3.10+.
    pause
    exit /b 1
)

echo [2/4] Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo WARNING: Some packages may not have installed correctly.
)

echo [3/4] Running main pipeline...
echo This will take approximately 10-15 minutes...
python main.py
if %errorlevel% neq 0 (
    echo ERROR: Pipeline failed. Check logs/wind_forecasting.log
    pause
    exit /b 1
)

echo [4/4] Starting API server...
echo Opening http://localhost:8000 in browser...
start http://localhost:8000
python -m uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

pause
