#!/usr/bin/env bash
set -o errexit

echo "=== Installing Python dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Checking frontend build assets ==="
if [ ! -f "app/dist/index.html" ] || [ ! -f "landing/dist/index.html" ]; then
  if command -v npm >/dev/null 2>&1; then
    echo "=== Building frontends via npm ==="
    (cd app && npm install && npm run build)
    (cd landing && npm install && npm run build)
  fi
fi

echo "=== Render build complete ==="
