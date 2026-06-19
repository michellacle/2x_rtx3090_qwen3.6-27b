#!/usr/bin/env bash
# ===================================================================
# uninstall.sh -- remove the Qwen3.6-27B systemd service
#
# Usage: sudo bash uninstall.sh
# ===================================================================
set -euo pipefail

SERVICE_NAME="qwen3.6-27b"
UNIT_PATH="/etc/systemd/system/${SERVICE_NAME}.service"
ENV_PATH="/etc/${SERVICE_NAME}.env"

if [ "$(id -u)" -ne 0 ]; then
  echo "ERROR: Run with sudo: sudo bash uninstall.sh" >&2
  exit 1
fi

echo "=== Uninstalling Qwen3.6-27B Service ==="
echo ""

# Stop and disable
if systemctl list-unit-files | grep -q "${SERVICE_NAME}"; then
  echo "Stopping service ..."
  systemctl stop "${SERVICE_NAME}.service" 2>/dev/null || true
  echo "Disabling service ..."
  systemctl disable "${SERVICE_NAME}.service" 2>/dev/null || true
fi

# Remove files
echo "Removing unit file: $UNIT_PATH"
rm -f "$UNIT_PATH"

echo "Removing environment file: $ENV_PATH"
rm -f "$ENV_PATH"

echo "Removing log directory: /var/log/${SERVICE_NAME}"
rm -rf /var/log/${SERVICE_NAME}

# Reload
systemctl daemon-reload
systemctl reset-failed

echo ""
echo "Service removed. Repo files are untouched."
echo "To reinstall: sudo bash install.sh"
