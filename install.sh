#!/usr/bin/env bash
# ===================================================================
# install.sh -- install Qwen3.6-27B server as a systemd service
#
# Usage: sudo bash install.sh [OPTIONS]
#
# Options:
#   --model PATH       Model directory (default: /home/michel/models/qwen3.6-27b-fp8)
#   --port NUM         HTTP port (default: 8000)
#   --user NAME        System user to run as (default: current user)
#   --dry-run          Show what would be done without making changes
# ===================================================================
set -euo pipefail

# ---- defaults -----------------------------------------------------
VLLM_PORT=8000
RUN_USER=""
DRY_RUN=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="qwen3.6-27b"
UNIT_PATH="/etc/systemd/system/${SERVICE_NAME}.service"
ENV_PATH="/etc/${SERVICE_NAME}.env"

# ---- parse args ---------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)  MODEL_PATH="$2"; shift 2 ;;
    --port)   VLLM_PORT="$2"; shift 2 ;;
    --user)   RUN_USER="$2";  shift 2 ;;
    --dry-run) DRY_RUN=1;     shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# ---- determine user -----------------------------------------------
if [ -z "$RUN_USER" ]; then
  if [ "$(id -u)" -eq 0 ]; then
    # Running as root — use SUDO_USER or first non-root user with nvidia group
    RUN_USER="${SUDO_USER:-$(find /home -maxdepth 1 -mindepth 1 -printf '%f\n' | head -1)}"
  else
    echo "ERROR: Run with sudo: sudo bash install.sh [--model PATH] [--port NUM]" >&2
    exit 1
  fi
fi

RUN_HOME=$(eval echo "~${RUN_USER}")

# Default model path uses the target user's home (set after user is known)
MODEL_PATH="${MODEL_PATH:-${RUN_HOME}/models/qwen3.6-27b-fp8}"

# ---- pre-flight checks --------------------------------------------
echo "=== Qwen3.6-27B Server Installer ==="
echo ""

# Check GPU
if ! command -v nvidia-smi &>/dev/null; then
  echo "ERROR: nvidia-smi not found — no GPU driver." >&2
  exit 1
fi
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "GPU: $GPU_COUNT GPU(s) detected"

# Check model
if [ ! -d "$MODEL_PATH" ]; then
  echo "ERROR: Model not found at $MODEL_PATH" >&2
  echo "Download the model first, then retry with --model /path/to/model" >&2
  exit 1
fi
echo "Model: $MODEL_PATH"

# Check repo
if [ ! -f "${SCRIPT_DIR}/daemon.sh" ]; then
  echo "ERROR: daemon.sh not found — run from the repo directory." >&2
  exit 1
fi

# Check venv
if [ ! -f "${SCRIPT_DIR}/.venv/bin/vllm" ]; then
  echo "ERROR: vLLM not found in ${SCRIPT_DIR}/.venv/" >&2
  echo "Create the virtual environment and install vLLM first." >&2
  exit 1
fi

echo "User: $RUN_USER ($RUN_HOME)"
echo "Port: $VLLM_PORT"
echo "Service: ${SERVICE_NAME}.service"
echo ""

if [ "$DRY_RUN" -eq 1 ]; then
  echo "--- DRY RUN ---"
  echo "Would create: $UNIT_PATH"
  echo "Would create: $ENV_PATH"
  echo "Would enable and start: ${SERVICE_NAME}.service"
  echo ""
  echo "--- Unit file ---"
  cat <<UNIT
[Unit]
Description=Qwen3.6-27B LLM Server (2x RTX 3090)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${RUN_USER}
Group=${RUN_USER}
WorkingDirectory=${SCRIPT_DIR}
EnvironmentFile=${ENV_PATH}
ExecStart=${SCRIPT_DIR}/daemon.sh
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
LogsDirectory=${SERVICE_NAME}
PIDFile=/run/${SERVICE_NAME}.pid

# GPU access
DevicePolicy=closed
ReadOnlyPaths=/
ReadWritePaths=/tmp /var/log/${SERVICE_NAME} ${SCRIPT_DIR}

[Install]
WantedBy=multi-user.target
UNIT
  echo ""
  echo "--- Environment file ---"
  cat <<ENV
MODEL_PATH=${MODEL_PATH}
VLLM_PORT=${VLLM_PORT}
VLLM_TP=2
VLLM_GPU_MEM=0.88
VLLM_MAX_LEN=131072
VLLM_MAX_SEQS=2
ENV
  exit 0
fi

# ---- create log directory -----------------------------------------
echo "Creating log directory /var/log/${SERVICE_NAME} ..."
mkdir -p /var/log/${SERVICE_NAME}
chown ${RUN_USER}:${RUN_USER} /var/log/${SERVICE_NAME}

# ---- write environment file ---------------------------------------
echo "Writing environment file: $ENV_PATH"
cat > "$ENV_PATH" <<EOF
MODEL_PATH=${MODEL_PATH}
VLLM_PORT=${VLLM_PORT}
VLLM_TP=2
VLLM_GPU_MEM=0.88
VLLM_MAX_LEN=131072
VLLM_MAX_SEQS=2
EOF
chmod 640 "$ENV_PATH"

# ---- write systemd unit -------------------------------------------
echo "Writing service unit: $UNIT_PATH"
cat > "$UNIT_PATH" <<EOF
[Unit]
Description=Qwen3.6-27B LLM Server (2x RTX 3090)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${RUN_USER}
Group=${RUN_USER}
WorkingDirectory=${SCRIPT_DIR}
EnvironmentFile=${ENV_PATH}
ExecStart=${SCRIPT_DIR}/daemon.sh
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
LogsDirectory=${SERVICE_NAME}
PIDFile=/run/${SERVICE_NAME}.pid

# Security
DevicePolicy=closed
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=/tmp /var/log/${SERVICE_NAME} ${SCRIPT_DIR}

[Install]
WantedBy=multi-user.target
EOF

# ---- enable and start ---------------------------------------------
echo "Reloading systemd ..."
systemctl daemon-reload

echo "Enabling service ..."
systemctl enable "${SERVICE_NAME}.service"

echo "Starting service ..."
systemctl start "${SERVICE_NAME}.service"

# ---- verify -------------------------------------------------------
echo ""
sleep 3
if systemctl is-active --quiet "${SERVICE_NAME}.service"; then
  echo "=== Service is running ==="
  echo ""
  echo "  systemd:  systemctl status ${SERVICE_NAME}"
  echo "  logs:     journalctl -u ${SERVICE_NAME} -f"
  echo "  stop:     systemctl stop ${SERVICE_NAME}"
  echo "  restart:  systemctl restart ${SERVICE_NAME}"
  echo "  uninstall: bash ${SCRIPT_DIR}/uninstall.sh"
  echo ""

  # Quick health check
  echo "  Waiting for health check ..."
  for i in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:${VLLM_PORT}/health" &>/dev/null; then
      echo "  Server is healthy on http://0.0.0.0:${VLLM_PORT}"
      break
    fi
    sleep 2
  done
else
  echo "ERROR: Service failed to start." >&2
  echo "Check logs: journalctl -u ${SERVICE_NAME} -n 50" >&2
  systemctl status "${SERVICE_NAME}.service" >&2 || true
  exit 1
fi
