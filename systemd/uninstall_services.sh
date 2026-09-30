#!/usr/bin/env bash
# ==============================================================================
# AI Trader - Raspberry Pi / Linux Systemd Services Uninstaller
# ==============================================================================
set -e

if [ "$EUID" -ne 0 ]; then
    echo "❌ Error: This script must be run with sudo privileges."
    echo "Usage: sudo bash systemd/uninstall_services.sh"
    exit 1
fi

echo "Stopping and disabling AI Trader services..."
systemctl stop ai-trader-scanner.timer 2>/dev/null || true
systemctl disable ai-trader-scanner.timer 2>/dev/null || true
systemctl stop ai-trader-web.service 2>/dev/null || true
systemctl disable ai-trader-web.service 2>/dev/null || true
systemctl stop ai-trader-api.service 2>/dev/null || true
systemctl disable ai-trader-api.service 2>/dev/null || true

echo "Removing unit files from /etc/systemd/system/..."
rm -f /etc/systemd/system/ai-trader-api.service
rm -f /etc/systemd/system/ai-trader-web.service
rm -f /etc/systemd/system/ai-trader-scanner.service
rm -f /etc/systemd/system/ai-trader-scanner.timer

systemctl daemon-reload
echo "✅ AI Trader services successfully uninstalled."
