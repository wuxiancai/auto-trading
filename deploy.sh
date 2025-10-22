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
  SUDO_BIN="sudo"
fi
RUN_USER="${SUDO_USER:-$(id -un)}"

# Ensure Python tooling on Ubuntu
if command -v apt-get >/dev/null 2>&1; then
  ${SUDO_BIN} apt-get update -y
  ${SUDO_BIN} apt-get install -y python3 python3-venv python3-pip
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
echo "${UNIT_CONTENT}" | ${SUDO_BIN} tee "${UNIT_FILE}" >/dev/null

# Reload and start service
${SUDO_BIN} systemctl daemon-reload
${SUDO_BIN} systemctl enable "${SERVICE_NAME}"
${SUDO_BIN} systemctl restart "${SERVICE_NAME}"

cat <<MSG
Service '${SERVICE_NAME}' deployed and started.
Check status:  sudo systemctl status ${SERVICE_NAME} --no-pager
Follow logs:   sudo journalctl -u ${SERVICE_NAME} -f
MSG