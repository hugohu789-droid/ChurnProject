#!/bin/bash
# -------------------------------------------------------------
# Uvicorn Deployment Script
# Description: Stop any existing instance and start a new one
# Author: Jenkins CI Automation
# -------------------------------------------------------------

APP_NAME="churn_api"
APP_PATH="/usr/local/test/backend/app/churn_api"
LOG_DIR="/usr/local/test/backend/app/logs"
LOG_PATH="${LOG_DIR}/uvicorn.log"
PYTHON_BIN="/usr/bin/python3"   # Change if using a virtual environment

echo "[INFO] Starting deployment for ${APP_NAME}..."

# 0️⃣ Ensure the log directory exists
if [ ! -d "$LOG_DIR" ]; then
  echo "[INFO] Log directory not found. Creating: $LOG_DIR"
  mkdir -p "$LOG_DIR"
  chmod 755 "$LOG_DIR"
else
  echo "[INFO] Log directory exists: $LOG_DIR"
fi

# 1️⃣ Stop existing process
echo "[INFO] Checking for existing Uvicorn process..."
PID=$(pgrep -f "uvicorn .*${APP_PATH}")
if [ -n "$PID" ]; then
  echo "[INFO] Stopping old process (PID: $PID)..."
  kill -9 $PID
else
  echo "[INFO] No existing Uvicorn process found."
fi

# 2️⃣ Start new process in background using the Python file
echo "[INFO] Launching new Uvicorn process from file..."
nohup $PYTHON_BIN -m uvicorn \
  "${APP_PATH%.*}:app" \
  --host 0.0.0.0 --port 8000 \
  --reload \
  > "$LOG_PATH" 2>&1 &

NEW_PID=$!
echo "[INFO] New process started (PID: $NEW_PID)"
echo "[INFO] Logs: $LOG_PATH"

# 3️⃣ Verify startup
sleep 2
if ps -p $NEW_PID > /dev/null; then
  echo "[SUCCESS] ${APP_NAME} is running on port 8000 (PID: $NEW_PID)"
else
  echo "[ERROR] Failed to start ${APP_NAME}. Check logs for details."
  exit 1
fi
