#!/usr/bin/env bash
# ===================================================================
# install.sh -- install Qwen3.6-27B server as a systemd service
#
# Usage: sudo bash install.sh [OPTIONS]
#
# Options:
#   --model PATH       Model directory (default: ~/models/qwen3.6-27b-fp8)
#   --hf-repo REPO     Hugging Face repo (default: Qwen/Qwen3.6-27B-FP8)
#   --port NUM         HTTP port (default: 8000)
#   --user NAME        System user to run as (default: current user)
#   --skip-download    Skip model download (must already exist)
#   --dry-run          Show what would be done without making changes
#
# Hugging Face token:
#   Set HF_TOKEN environment variable, or the script will prompt you.
#   Get a token at: https://huggingface.co/settings/tokens
# ===================================================================
set -euo pipefail

# ---- defaults -----------------------------------------------------
VLLM_PORT=8000
HF_REPO="Qwen/Qwen3.6-27B-FP8"
RUN_USER=""
DRY_RUN=0
SKIP_DOWNLOAD=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="qwen3.6-27b"
UNIT_PATH="/etc/systemd/system/${SERVICE_NAME}.service"
ENV_PATH="/etc/${SERVICE_NAME}.env"

# ---- parse args ---------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)         MODEL_PATH="$2"; shift 2 ;;
    --hf-repo)       HF_REPO="$2";     shift 2 ;;
    --port)          VLLM_PORT="$2";   shift 2 ;;
    --user)          RUN_USER="$2";    shift 2 ;;
    --skip-download) SKIP_DOWNLOAD=1;  shift ;;
    --dry-run)       DRY_RUN=1;        shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# ---- determine user -----------------------------------------------
if [ -z "$RUN_USER" ]; then
  if [ "$(id -u)" -eq 0 ]; then
    RUN_USER="${SUDO_USER:-$(find /home -maxdepth 1 -mindepth 1 -printf '%f\n' | head -1)}"
  else
    echo "ERROR: Run with sudo: sudo bash install.sh" >&2
    exit 1
  fi
fi

RUN_HOME=$(eval echo "~${RUN_USER}")
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

# Check repo files
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

# ---- download model -----------------------------------------------
MODEL_EXISTS=0
if [ -f "${MODEL_PATH}/config.json" ]; then
  MODEL_EXISTS=1
fi

if [ "$MODEL_EXISTS" -eq 0 ]; then
  if [ "$SKIP_DOWNLOAD" -eq 1 ]; then
    echo "ERROR: Model not found at $MODEL_PATH and --skip-download is set." >&2
    exit 1
  fi

  echo ""
  echo "Model not found at $MODEL_PATH"
  echo "Downloading from Hugging Face: $HF_REPO"
  echo ""

  # Get HF token
  HF_TOKEN="${HF_TOKEN:-}"
  if [ -z "$HF_TOKEN" ]; then
    echo "No HF_TOKEN environment variable set."
    echo "Get a token at: https://huggingface.co/settings/tokens"
    echo ""
    read -rs -p "Hugging Face token: " HF_TOKEN
    echo ""
    if [ -z "$HF_TOKEN" ]; then
      echo "ERROR: Empty token." >&2
      exit 1
    fi
  fi

  echo "Downloading model (this may take several minutes) ..."
  echo ""

  export HF_TOKEN
  DOWNLOAD_OUTPUT=$(${SCRIPT_DIR}/.venv/bin/python3 ${SCRIPT_DIR}/download_model.py \
    "$HF_REPO" \
    --local-dir "$MODEL_PATH" \
    --token "$HF_TOKEN" 2>&1)
  DOWNLOAD_EXIT=$?

  # Print download progress line by line
  echo "$DOWNLOAD_OUTPUT"

  if [ "$DOWNLOAD_EXIT" -ne 0 ]; then
    echo "" >&2
    echo "ERROR: Model download failed." >&2
    echo "Check the output above for details." >&2
    echo "You can also download manually:" >&2
    echo "  HF_TOKEN=xxx ${SCRIPT_DIR}/.venv/bin/python3 ${SCRIPT_DIR}/download_model.py $HF_REPO --local-dir $MODEL_PATH" >&2
    exit 1
  fi

  # Fix ownership so the target user can read it
  chown -R "${RUN_USER}:${RUN_USER}" "$(dirname "$MODEL_PATH")" 2>/dev/null || true

  echo ""
  echo "Model downloaded successfully."
else
  echo "Model: $MODEL_PATH (already exists)"
fi

echo ""
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

# ---- fix triton cache ownership ------------------------------------
if [ -d "${RUN_HOME}/.triton" ]; then
  echo "Fixing Triton cache ownership ..."
  chown -R "${RUN_USER}:${RUN_USER}" "${RUN_HOME}/.triton"
fi

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
  for i in $(seq 1 60); do
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
