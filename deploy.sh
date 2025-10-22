#!/usr/bin/env bash
set -euo pipefail

# Auto deploy systemd service for live paper trading
# - Creates /etc/systemd/system/auto-trading.service
# - Sets up a Python venv under repo and installs dependencies
# - Starts and enables the service via systemctl
#
# Defaults:
#   SYMBOL (default: BTCUSDT)
#   INTERVAL (default: 5m)
# Usage:
#   sudo ./deploy.sh
#   SYMBOL=ETHUSDT INTERVAL=5m sudo ./deploy.sh

SYMBOL="${SYMBOL:-BTCUSDT}"
INTERVAL="${INTERVAL:-5m}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="auto-trading"
UNIT_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
VENV_DIR="${REPO_DIR}/.venv"

# Use sudo if not root
SUDO_BIN=""
if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  echo "需要以 root 权限运行：请使用 sudo ./deploy.sh" >&2
  exit 1
fi
RUN_USER="${SUDO_USER:-$(id -un)}"

# 确认 systemctl 可用
if ! command -v systemctl >/dev/null 2>&1; then
  echo "未检测到 systemctl 或 systemd 未运行。请确认服务器是 systemd 系统。" >&2
  exit 1
fi

# Ensure Python tooling on Ubuntu
if command -v apt-get >/dev/null 2>&1; then
  apt-get update -y
  apt-get install -y python3 python3-venv python3-pip
fi

# Create venv
if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/pip" install --upgrade pip
"${VENV_DIR}/bin/pip" install websocket-client Flask pandas requests numpy

PYTHON_BIN="${VENV_DIR}/bin/python"

# Write systemd unit
read -r -d '' UNIT_CONTENT <<EOF
[Unit]
Description=Auto Trading Live Paper Trader (interval=${INTERVAL})
After=network.target

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${REPO_DIR}
ExecStart=${PYTHON_BIN} ${REPO_DIR}/live_paper_trading.py --symbol ${SYMBOL} --interval ${INTERVAL}
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1
SyslogIdentifier=${SERVICE_NAME}

[Install]
WantedBy=multi-user.target
EOF

echo "Creating unit: ${UNIT_FILE}"
echo "${UNIT_CONTENT}" > "${UNIT_FILE}"

# Reload and start service
systemctl daemon-reload
systemctl enable "${SERVICE_NAME}"
systemctl restart "${SERVICE_NAME}"

cat <<MSG
Service '${SERVICE_NAME}' deployed and started.
Check status:  sudo systemctl status ${SERVICE_NAME} --no-pager
Follow logs:   sudo journalctl -u ${SERVICE_NAME} -f
MSG