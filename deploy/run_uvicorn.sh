#!/bin/bash
# -------------------------------------------------------------
# Uvicorn Deployment Script (Safe for repeated Jenkins runs)
# -------------------------------------------------------------

APP_NAME="churn_api"
APP_PATH="/usr/local/test/backend/app/churn_api.py"
APP_DIR=$(dirname "$APP_PATH")
APP_FILE=$(basename "$APP_PATH" .py)
LOG_DIR="/usr/local/test/backend/app/logs"
LOG_PATH="${LOG_DIR}/uvicorn.log"
PYTHON_BIN="/usr/bin/python3"
PORT=8000

echo "[INFO] Starting deployment for ${APP_NAME}..."

# 0️⃣ Ensure the log directory exists
if [ ! -d "$LOG_DIR" ]; then
  echo "[INFO] Log directory not found. Creating: $LOG_DIR"
  mkdir -p "$LOG_DIR"
  chmod 755 "$LOG_DIR"
else
  echo "[INFO] Log directory exists: $LOG_DIR"
fi

# 1️⃣ Stop any existing process using this port
echo "[INFO] Checking for existing process on port ${PORT}..."
EXISTING_PID=$(lsof -ti tcp:${PORT})
if [ -n "$EXISTING_PID" ]; then
  echo "[INFO] Port ${PORT} is currently used by PID ${EXISTING_PID}. Stopping it..."
  kill -9 $EXISTING_PID
  sleep 1
else
  echo "[INFO] No existing process found on port ${PORT}."
fi

# 2️⃣ Launch new Uvicorn process
echo "[INFO] Starting new Uvicorn process from ${APP_FILE}.py..."
nohup $PYTHON_BIN -m uvicorn \
  --app-dir "$APP_DIR" \
  "${APP_FILE}:app" \
  --host 0.0.0.0 --port ${PORT} \
  --reload \
  > "$LOG_PATH" 2>&1 &

NEW_PID=$!
echo "[INFO] New process started (PID: $NEW_PID)"
echo "[INFO] Logs: $LOG_PATH"

# 3️⃣ Verify startup
sleep 2
if ps -p $NEW_PID > /dev/null; then
  echo "[SUCCESS] ${APP_NAME} is running on port ${PORT} (PID: $NEW_PID)"
else
  echo "[ERROR] Failed to start ${APP_NAME}. Check logs for details."
  exit 1
fi
