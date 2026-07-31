@echo off
echo ============================================
echo   AMG Wind Farm Forecasting API
echo ============================================
echo.
echo Starting server on http://localhost:8000
echo API docs: http://localhost:8000/docs
echo Dashboard: http://localhost:8000
echo.
echo IMPORTANT (P2-01): set API_KEY before starting, e.g.
echo   set API_KEY=your-secret-key
echo Without API_KEY the server is FAIL-CLOSED (protected endpoints return 503).
echo.
echo Press Ctrl+C to stop
echo.
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
pause
