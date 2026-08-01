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
echo Optional tuning (P2-01):
echo   set PREWARM_MODELS=all     ^(default; pre-warms models in background at startup^)
echo   set PREWARM_MODELS=0       ^(disable pre-warm; lazy-load on first request^)
echo   set MODEL_LOAD_TIMEOUT=30  ^(seconds; slower loads return 503 instead of hanging^)
echo.
echo Press Ctrl+C to stop
echo.

rem Pick a Python that actually has the deps. The default 'python' on some
rem machines is a dependency-free stub, so prefer the py launcher (3.13)
rem or an explicit API_PYTHON override, e.g.
rem   set API_PYTHON=C:\Users\ASUS\...\python3.13.exe
set "PYTHON_CMD=py -3.13"
where py >nul 2>&1 || set "PYTHON_CMD=python"
if defined API_PYTHON set "PYTHON_CMD=%API_PYTHON%"
%PYTHON_CMD% -m uvicorn src.api:app --host 0.0.0.0 --port 8000
pause
