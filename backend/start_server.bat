@echo off
REM Vision Agent - Start server so http://localhost:8000 works
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Python not found. Install Python 3.10+ from https://www.python.org/downloads/
        echo Make sure to check "Add Python to PATH" during installation.
        pause
        exit /b 1
    )
    echo Installing dependencies...
    venv\Scripts\pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: pip install failed.
        pause
        exit /b 1
    )
)

echo.
echo Starting Vision Agent at http://localhost:8000
echo Keep this window open. Press Ctrl+C to stop.
echo.
venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
pause
