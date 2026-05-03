#!/usr/bin/env bash
# ===================================================================
# stop-vllm.sh -- stop the running vLLM instance on a given port
# ===================================================================
set -euo pipefail

PORT="${VLLM_PORT:-8000}"
PID_FILE="/tmp/vllm-${PORT}.pid"

if [ ! -f "$PID_FILE" ]; then
  echo "No PID file at ${PID_FILE}. Is vLLM running on port ${PORT}?" >&2
  exit 1
fi

PID=$(cat "$PID_FILE")
if ! kill -0 "$PID" 2>/dev/null; then
  echo "PID ${PID} not running. Clean up stale pid file." >&2
  rm -f "$PID_FILE"
  exit 1
fi

echo "Stopping vLLM (pid ${PID}) ..."
kill "$PID"
for i in $(seq 1 30); do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "vLLM stopped."
    rm -f "$PID_FILE"
    exit 0
  fi
  sleep 1
done

echo "Graceful stop timed out — force killing." >&2
kill -9 "$PID" 2>/dev/null || true
rm -f "$PID_FILE"
echo "Force-killed."
