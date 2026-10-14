#!/usr/bin/env bash
# DroneThreatTrainer - one-command setup/run (Mac/Linux)
set -e
cd "$(dirname "$0")"

echo "============================================"
echo " DroneThreatTrainer - one-command setup/run"
echo "============================================"
echo

if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] python3 was not found on your PATH."
    echo "Install Python 3.10+ from https://www.python.org/downloads/ (or via your package manager)."
    exit 1
fi

PYVER_OK=$(python3 -c "import sys; print(1 if sys.version_info >= (3,10) else 0)")
if [ "$PYVER_OK" != "1" ]; then
    echo "[WARNING] Your Python is older than 3.10. This project targets 3.10+."
    echo "It may still work, but if you see errors, please upgrade Python."
    echo
fi

if [ ! -d ".venv" ]; then
    echo "[1/3] Creating virtual environment .venv ..."
    python3 -m venv .venv
else
    echo "[1/3] Virtual environment .venv already exists, reusing it."
fi

echo "[2/3] Installing/updating dependencies ..."
./.venv/bin/python -m pip install -q --upgrade pip
./.venv/bin/python -m pip install -q -r requirements.txt

PORT=8090
if ./.venv/bin/python -c "import socket; s=socket.socket(); exit(0 if s.connect_ex(('127.0.0.1', $PORT)) != 0 else 1)"; then
    :
else
    echo "[WARNING] Port $PORT is already in use. Trying 8099 instead."
    PORT=8099
fi

if command -v npm >/dev/null 2>&1; then
    if [ ! -f "landing/dist/index.html" ]; then
        echo "[INFO] Building landing page ..."
        (cd landing && npm install -q && npm run build)
    fi
    if [ ! -f "app/dist/index.html" ]; then
        echo "[INFO] Building tactical app ..."
        (cd app && npm install -q && npm run build)
    fi
fi

echo "[3/3] Starting the server on http://localhost:$PORT/ ..."
echo
echo "  Tactical App:     http://localhost:$PORT/styleguide"
echo "  Desktop trainer:  http://localhost:$PORT/trainer"
echo "  VR mode:          http://localhost:$PORT/vr"
echo "  AAR dashboard:    http://localhost:$PORT/aar"
echo "  Instructor:       http://localhost:$PORT/instructor"
echo "  Team roles:       http://localhost:$PORT/team"
echo "  Swarm simulation: http://localhost:$PORT/swarm"
echo
echo "Press CTRL+C to stop the server."
echo "============================================"
echo

if command -v open >/dev/null 2>&1; then
    (sleep 1.5 && open "http://localhost:$PORT/") &
elif command -v xdg-open >/dev/null 2>&1; then
    (sleep 1.5 && xdg-open "http://localhost:$PORT/") &
fi

exec ./.venv/bin/python -m uvicorn server:app --port "$PORT"
