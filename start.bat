@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================
echo  DroneThreatTrainer - one-command setup/run
echo ============================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found on your PATH.
    echo Install Python 3.10+ from https://www.python.org/downloads/
    echo During install, check "Add python.exe to PATH".
    echo.
    pause
    exit /b 1
)

python -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
if errorlevel 1 (
    echo [WARNING] Your Python is older than 3.10. This project targets 3.10+.
    echo It may still work, but if you see errors, please upgrade Python.
    echo.
)

if not exist ".venv" (
    echo [1/3] Creating virtual environment .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create the virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [1/3] Virtual environment .venv already exists, reusing it.
)

echo [2/3] Installing/updating dependencies ...
".venv\Scripts\python.exe" -m pip install -q --upgrade pip
".venv\Scripts\python.exe" -m pip install -q -r requirements.txt
if errorlevel 1 (
    echo [ERROR] pip install failed. Check the error output above.
    pause
    exit /b 1
)

set PORT=8090
".venv\Scripts\python.exe" -c "import socket; s=socket.socket(); exit(0 if s.connect_ex(('127.0.0.1', %PORT%)) != 0 else 1)" >nul 2>nul
if errorlevel 1 (
    echo [WARNING] Port %PORT% is already in use. Trying 8099 instead.
    set PORT=8099
)

where npm >nul 2>nul
if not errorlevel 1 (
    if not exist "landing\dist\index.html" (
        echo [INFO] Building landing page ...
        pushd landing
        call npm install -q
        call npm run build
        popd
    )
    if not exist "app\dist\index.html" (
        echo [INFO] Building tactical app ...
        pushd app
        call npm install -q
        call npm run build
        popd
    )
)

echo [3/3] Starting the server on http://localhost:!PORT!/ ...
echo.
echo   Tactical App:     http://localhost:!PORT!/styleguide
echo   Desktop trainer:  http://localhost:!PORT!/trainer
echo   VR mode:          http://localhost:!PORT!/vr
echo   AAR dashboard:    http://localhost:!PORT!/aar
echo   Instructor:       http://localhost:!PORT!/instructor
echo   Team roles:       http://localhost:!PORT!/team
echo   Swarm simulation: http://localhost:!PORT!/swarm
echo.
echo Press CTRL+C in this window to stop the server.
echo ============================================
echo.

start "" "http://localhost:!PORT!/"
".venv\Scripts\python.exe" -m uvicorn server:app --port !PORT!

pause
