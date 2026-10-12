@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0\radar_classification"

echo ============================================
echo  DroneThreatTrainer - Radar Analytics (optional)
echo ============================================
echo.
echo This is a separate, heavier service (qiskit/catboost/xgboost/shap).
echo It is NOT required to run the main trainer - see README's
echo "Architecture note: two separate services, not one app".
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found on your PATH.
    pause
    exit /b 1
)

where node >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Node.js was not found on your PATH. Install it from https://nodejs.org/
    pause
    exit /b 1
)

if not exist ".venv_radar" (
    echo [1/4] Creating virtual environment .venv_radar ...
    python -m venv .venv_radar
    if errorlevel 1 (
        echo [ERROR] Failed to create the virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [1/4] Virtual environment .venv_radar already exists, reusing it.
)

echo [2/4] Installing/updating Python dependencies (this can take several minutes) ...
".venv_radar\Scripts\python.exe" -m pip install -q --upgrade pip
".venv_radar\Scripts\python.exe" -m pip install -q -r requirements.txt
if errorlevel 1 (
    echo [ERROR] pip install failed. Check the error output above.
    pause
    exit /b 1
)

if not exist "frontend\node_modules" (
    echo [3/4] Installing frontend dependencies ...
    pushd frontend
    call npm install
    popd
) else (
    echo [3/4] Frontend dependencies already installed, reusing them.
)

echo [4/4] Starting the API (port 8081) and frontend (port 3000) ...
echo.
echo   Radar Analytics dashboard:  http://localhost:3000/
echo   API health check:          http://localhost:8081/health
echo.
echo Two windows will open. Close both to stop.
echo ============================================
echo.

start "DroneThreatTrainer Radar Analytics API" cmd /k "cd /d "%~dp0radar_classification" && ".venv_radar\Scripts\python.exe" -m uvicorn api.app:app --port 8081"
timeout /t 3 /nobreak >nul
start "DroneThreatTrainer Radar Analytics Frontend" cmd /k "cd /d "%~dp0radar_classification\frontend" && npm run dev"
timeout /t 5 /nobreak >nul
start "" "http://localhost:3000/"

echo Both services are starting in their own windows.
pause
