@echo off
echo ============================================
echo   AMG Wind Farm Forecasting API
echo ============================================
echo.
echo Starting server on http://localhost:8000
echo API docs: http://localhost:8000/docs
echo Dashboard: http://localhost:8000
echo.
echo Press Ctrl+C to stop
echo.
python -m uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
pause
