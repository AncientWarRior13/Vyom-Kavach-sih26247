#!/usr/bin/env python3
"""ARES-X Simulation Backend Server launcher.
Launches the FastAPI + WebSocket backend server on port 8000, powering
the modern Next.js GCS Mission Control dashboard (frontend/).

Usage:
    python dashboard/serve.py
    # or
    python -m uvicorn ares.server:app --host 0.0.0.0 --port 8000
"""
import os
import sys
import uvicorn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

if __name__ == "__main__":
    port = int(os.environ.get("ARES_PORT", 8000))
    host = os.environ.get("ARES_HOST", "0.0.0.0")
    print(f"Starting ARES-X Live Autonomy Simulation Server on http://localhost:{port}")
    print(f"WebSocket endpoint: ws://localhost:{port}/ws")
    print(f"REST API: http://localhost:{port}/api/health")
    print(f"Frontend Next.js dashboard runs at http://localhost:3000\n")
    uvicorn.run("ares.server:app", host=host, port=port, reload=False, log_level="info")

