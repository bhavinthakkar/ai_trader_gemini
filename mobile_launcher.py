#!/usr/bin/env python3
"""
AI Trader Mobile App Launcher
=============================
1. Verifies that the FastAPI backend (api.py) is active on port 8000.
2. Launches Expo Metro Bundler for the React Native Android application.
3. Displays connection QR code and LAN endpoints for your Android device.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

BASE_DIR = Path(__file__).resolve().parent
MOBILE_DIR = BASE_DIR / "mobile_app"
NODE_BIN = Path.home() / ".local" / "node" / "bin"


def ensure_api_server_running(port: int = 8000):
    url = f"http://127.0.0.1:{port}/api/health"
    try:
        with urlopen(url, timeout=1.0) as resp:
            if resp.status == 200:
                print(f"✅ FastAPI Backend is active at http://localhost:{port}")
                return True
    except Exception:
        pass

    print(f"⚡ Starting FastAPI Backend on port {port}...")
    api_starter = BASE_DIR / "api_server.py"
    subprocess.run([sys.executable, str(api_starter), "start"], check=True)
    time.sleep(1.0)
    return True


def main():
    ensure_api_server_running()

    # Prepend local node binary to PATH and disable Electron sandbox on Linux
    env = os.environ.copy()
    if NODE_BIN.exists():
        env["PATH"] = f"{NODE_BIN}:{env.get('PATH', '')}"
    env["ELECTRON_DISABLE_SANDBOX"] = "1"

    print("\n=======================================================")
    print("📱 Launching AI Trader Android Mobile App (Expo Metro)")
    print("=======================================================")
    print("1. Install 'Expo Go' from the Google Play Store on your Android phone.")
    print("2. Connect your phone to the same local WiFi as this computer.")
    print("3. Scan the QR code below using the Expo Go app or your camera.")
    print("=======================================================\n")

    cmd = ["npx", "expo", "start", "--lan"]
    try:
        subprocess.run(cmd, cwd=str(MOBILE_DIR), env=env)
    except KeyboardInterrupt:
        print("\n👋 Mobile server stopped.")


if __name__ == "__main__":
    main()
