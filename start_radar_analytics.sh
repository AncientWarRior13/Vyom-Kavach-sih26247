#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/radar_classification"

echo "============================================"
echo " DroneThreatTrainer - Radar Analytics (optional)"
echo "============================================"
echo
echo "This is a separate, heavier service (qiskit/catboost/xgboost/shap)."
echo "It is NOT required to run the main trainer - see README's"
echo "\"Architecture note: two separate services, not one app\"."
echo

if ! command -v python3 >/dev/null 2>&1; then
  echo "[ERROR] python3 was not found on your PATH."
  exit 1
fi

if ! command -v node >/dev/null 2>&1; then
  echo "[ERROR] Node.js was not found on your PATH. Install it from https://nodejs.org/"
  exit 1
fi

if [ ! -d ".venv_radar" ]; then
  echo "[1/4] Creating virtual environment .venv_radar ..."
  python3 -m venv .venv_radar
else
  echo "[1/4] Virtual environment .venv_radar already exists, reusing it."
fi

echo "[2/4] Installing/updating Python dependencies (this can take several minutes) ..."
".venv_radar/bin/python" -m pip install -q --upgrade pip
".venv_radar/bin/python" -m pip install -q -r requirements.txt

if [ ! -d "frontend/node_modules" ]; then
  echo "[3/4] Installing frontend dependencies ..."
  (cd frontend && npm install)
else
  echo "[3/4] Frontend dependencies already installed, reusing them."
fi

echo "[4/4] Starting the API (port 8081) and frontend (port 3000) ..."
echo
echo "  Radar Analytics dashboard:  http://localhost:3000/"
echo "  API health check:          http://localhost:8081/health"
echo
echo "Press CTRL+C to stop both services."
echo "============================================"
echo

cleanup() {
  kill "$API_PID" "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup EXIT

".venv_radar/bin/python" -m uvicorn api.app:app --port 8081 &
API_PID=$!

(cd frontend && npm run dev) &
FRONTEND_PID=$!

sleep 5
if command -v open >/dev/null 2>&1; then
  open "http://localhost:3000/"
elif command -v xdg-open >/dev/null 2>&1; then
  xdg-open "http://localhost:3000/"
fi

wait "$API_PID" "$FRONTEND_PID"
