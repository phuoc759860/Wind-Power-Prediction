@echo off
title AMG Wind Power Forecasting System
cd /d "%~dp0"

rem Pick a Python that has the deps (default 'python' may be a dep-less stub).
set "PYTHON_CMD=py -3.13"
where py >nul 2>&1 || set "PYTHON_CMD=python"
if defined API_PYTHON set "PYTHON_CMD=%API_PYTHON%"

echo ========================================
echo AMG Wind Power Forecasting System
echo ========================================
echo.
echo [1/5] Checking Python...
%PYTHON_CMD% --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python 3.10+.
    pause
    exit /b 1
)

echo [2/5] Installing dependencies...
%PYTHON_CMD% -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo WARNING: Some packages may not have installed correctly.
)

echo [3/5] Running main pipeline...
echo This will take approximately 45-60 minutes...
echo NOTE: --no-wf-ml skips the optional ~30 min ML walk-forward validation.
echo       Remove it for the full validation set.
%PYTHON_CMD% main.py --no-wf-ml
if %errorlevel% neq 0 (
    echo ERROR: Pipeline failed. Check logs/wind_forecasting.log
    pause
    exit /b 1
)

echo [4/5] Running test suite and regenerating report...
%PYTHON_CMD% -m pytest tests/ -v > pytest_report.txt 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Test suite failed. See pytest_report.txt
    pause
    exit /b 1
)
%PYTHON_CMD% generate_report.py
if %errorlevel% neq 0 (
    echo ERROR: Report generation failed.
    pause
    exit /b 1
)

echo [5/5] Starting API server...
echo NOTE: no --reload in deployment (P2-01); set API_KEY before starting.
echo Opening http://localhost:8000 in browser...
start http://localhost:8000
%PYTHON_CMD% -m uvicorn src.api:app --host 0.0.0.0 --port 8000

pause
