#!/usr/bin/env bash
# ==============================================================================
# AI Trader - Raspberry Pi / Linux Systemd Services Installer
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Identify target user (the non-root user who owns the repository)
if [ -n "$SUDO_USER" ]; then
    TARGET_USER="$SUDO_USER"
else
    TARGET_USER="$USER"
fi

if [ "$EUID" -ne 0 ]; then
    echo "❌ Error: This script must be run with sudo privileges."
    echo "Usage: sudo bash systemd/install_services.sh"
    exit 1
fi

echo "============================================================"
echo " 🚀 Installing AI Trader Systemd Services on Raspberry Pi"
echo "============================================================"
echo " Target User:        ${TARGET_USER}"
echo " Project Directory:  ${PROJECT_DIR}"
echo " Virtualenv Python:  ${PROJECT_DIR}/venv/bin/python"
echo "============================================================"

# Verify venv exists
if [ ! -f "${PROJECT_DIR}/venv/bin/python" ]; then
    echo "❌ Virtual environment not found at ${PROJECT_DIR}/venv."
    echo "Please create it first: python3 -m venv venv && ./venv/bin/pip install -r requirements.txt"
    exit 1
fi

# Configure and install ai-trader-api.service
sed -e "s|{{USER}}|${TARGET_USER}|g" \
    -e "s|{{PROJECT_DIR}}|${PROJECT_DIR}|g" \
    "${SCRIPT_DIR}/ai-trader-api.service" > /etc/systemd/system/ai-trader-api.service

# Configure and install ai-trader-web.service
sed -e "s|{{USER}}|${TARGET_USER}|g" \
    -e "s|{{PROJECT_DIR}}|${PROJECT_DIR}|g" \
    "${SCRIPT_DIR}/ai-trader-web.service" > /etc/systemd/system/ai-trader-web.service

# Configure and install ai-trader-scanner.service & timer
sed -e "s|{{USER}}|${TARGET_USER}|g" \
    -e "s|{{PROJECT_DIR}}|${PROJECT_DIR}|g" \
    "${SCRIPT_DIR}/ai-trader-scanner.service" > /etc/systemd/system/ai-trader-scanner.service

cp "${SCRIPT_DIR}/ai-trader-scanner.timer" /etc/systemd/system/ai-trader-scanner.timer

# Reload systemd
echo "🔄 Reloading systemd daemon..."
systemctl daemon-reload

# Enable services to run on boot and start them now
echo "🟢 Enabling and starting ai-trader-api.service..."
systemctl enable --now ai-trader-api.service

echo "🟢 Enabling and starting ai-trader-web.service..."
systemctl enable --now ai-trader-web.service

echo "🟢 Enabling and starting ai-trader-scanner.timer (periodic scans)..."
systemctl enable --now ai-trader-scanner.timer

echo ""
echo "============================================================"
echo " ✅ Installation Complete!"
echo "============================================================"
echo " Services configured to automatically start on boot:"
echo "   1. ai-trader-api.service      (FastAPI REST backend on port 8000)"
echo "   2. ai-trader-web.service      (Streamlit Web Dashboard on port 8501)"
echo "   3. ai-trader-scanner.timer    (Periodic market scan runner)"
echo ""
echo " Useful management commands:"
echo "   sudo systemctl status ai-trader-api"
echo "   sudo systemctl status ai-trader-web"
echo "   journalctl -u ai-trader-api -f"
echo "   journalctl -u ai-trader-web -f"
echo "============================================================"
