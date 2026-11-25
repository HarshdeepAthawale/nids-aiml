@echo off
title Network IDS — UNSW-NB15

echo.
echo  ========================================
echo   Network Intrusion Detection System
echo   UNSW-NB15 Project
echo  ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found!
    echo  Please install Python from https://python.org
    echo  Make sure to check "Add to PATH" during install.
    pause
    exit /b
)

REM Install dependencies if needed
echo  [1/3] Checking dependencies...
pip install -r requirements.txt -q
echo  Done.
echo.

REM Start backend in a new window
echo  [2/3] Starting backend API...
start "NIDS Backend (keep open)" cmd /k "python -m uvicorn backend.app:app --port 8000"
timeout /t 3 /nobreak >nul

REM Open frontend in browser
echo  [3/3] Opening frontend in browser...
start "" frontend\index.html

echo.
echo  ========================================
echo   App is running!
echo.
echo   Frontend : frontend\index.html (browser)
echo   API docs : http://localhost:8000/docs
echo.
echo   Close the "NIDS Backend" window to stop.
echo  ========================================
echo.
pause
