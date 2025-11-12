#!/bin/bash
# -------------------------------------------------------------
# Uvicorn Deployment Script
# Description: Stop any existing instance and start a new one
# Author: Jenkins CI Automation
# -------------------------------------------------------------

APP_NAME="churn_api"
APP_PATH="/usr/local/test/backend/app/churn_api.py"
LOG_PATH="/usr/local/test/backend/app/logs/uvicorn.log"
PYTHON_BIN="/usr/bin/python3"  

echo "[INFO] Starting deployment for ${APP_NAME}..."

# 1️⃣ Stop existing process
echo "[INFO] Checking for existing Uvicorn process..."
PID=$(pgrep -f "uvicorn .*${APP_PATH}")
if [ -n "$PID" ]; then
  echo "[INFO] Stopping old process (PID: $PID)..."
  kill -9 $PID
else
  echo "[INFO] No existing Uvicorn process found."
fi

# 2️⃣ Start new process in background
echo "[INFO] Launching new Uvicorn process..."
nohup $PYTHON_BIN -m uvicorn churn_api:app \
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

