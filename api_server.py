#!/usr/bin/env python3
"""
Managed FastAPI REST API Server Lifecycle Service
=================================================
Manages the background execution of the FastAPI mobile backend (`api.py`).
- Launches Uvicorn in a detached background daemon.
- Checks health endpoint (/api/health).
- Resolves local and network URLs for mobile device connection.
- CLI actions: start, status, stop, restart.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PORT = 8000
DEFAULT_HOST = "0.0.0.0"
DEFAULT_LOG_PATH = BASE_DIR / "api_server.log"
PID_FILE_PATH = BASE_DIR / ".api_server.pid"


def get_lan_ip() -> str:
    """Discovers the primary local network IP (e.g., 192.168.0.x)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def is_api_running(host: str = "127.0.0.1", port: int = DEFAULT_PORT, timeout: float = 1.0) -> bool:
    """Checks if the FastAPI health check answers on the port."""
    url = f"http://{host}:{port}/api/health"
    try:
        with urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def start_api_server(
    port: int = DEFAULT_PORT,
    host: str = DEFAULT_HOST,
    log_path: Path = DEFAULT_LOG_PATH,
    timeout: float = 15.0,
) -> Tuple[bool, str, str]:
    """Starts the Uvicorn FastAPI server in the background as a detached daemon."""
    lan_ip = get_lan_ip()
    local_url = f"http://localhost:{port}"
    network_url = f"http://{lan_ip}:{port}"

    if is_api_running(port=port):
        return True, local_url, network_url

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8")

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "api:app",
        f"--port={port}",
        f"--host={host}",
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        PID_FILE_PATH.write_text(str(proc.pid), encoding="utf-8")
    except Exception as e:
        print(f"[APIServer] Failed to start uvicorn: {e}")
        return False, "", ""

    # Wait for health endpoint
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_api_running(port=port):
            return True, local_url, network_url
        time.sleep(0.4)

    return False, local_url, network_url


def stop_api_server(port: int = DEFAULT_PORT) -> bool:
    """Stops the managed FastAPI background process."""
    stopped = False

    if PID_FILE_PATH.exists():
        try:
            pid = int(PID_FILE_PATH.read_text(encoding="utf-8").strip())
            os.kill(pid, signal.SIGTERM)
            for _ in range(20):
                if not is_api_running(port=port):
                    stopped = True
                    break
                time.sleep(0.2)
        except (ProcessLookupError, ValueError):
            pass
        except Exception as e:
            print(f"[APIServer] Error terminating PID: {e}")
        finally:
            if PID_FILE_PATH.exists():
                PID_FILE_PATH.unlink()

    # Fallback to fuser / lsof
    if is_api_running(port=port):
        try:
            subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True)
            time.sleep(0.5)
            stopped = not is_api_running(port=port)
        except Exception:
            pass

    return stopped or not is_api_running(port=port)


def main():
    parser = argparse.ArgumentParser(description="Managed FastAPI REST API Server")
    parser.add_argument(
        "action",
        nargs="?",
        choices=["start", "status", "stop", "restart"],
        default="status",
        help="Action to perform (default: status)",
    )
    parser.add_argument("--port", "-p", type=int, default=DEFAULT_PORT, help="Port to bind/check (default: 8000)")
    args = parser.parse_args()

    port = args.port
    lan_ip = get_lan_ip()

    if args.action == "status":
        if is_api_running(port=port):
            print(f"🟢 AI Trader API is RUNNING on port {port}:")
            print(f"   • Local Docs:   http://localhost:{port}/docs")
            print(f"   • Network Docs: http://{lan_ip}:{port}/docs")
            print(f"   • Health:       http://localhost:{port}/api/health")
        else:
            print(f"🔴 AI Trader API is NOT running on port {port}.")

    elif args.action == "start":
        if is_api_running(port=port):
            print(f"✅ AI Trader API is already running on port {port}.")
        else:
            ok, loc, net = start_api_server(port=port)
            if ok:
                print(f"🚀 AI Trader API started successfully!")
                print(f"   • Local Docs:   {loc}/docs")
                print(f"   • Network Docs: {net}/docs")
            else:
                print(f"❌ Failed to start AI Trader API. Check log at {DEFAULT_LOG_PATH}")

    elif args.action == "stop":
        if stop_api_server(port=port):
            print(f"🛑 AI Trader API stopped successfully.")
        else:
            print(f"⚠️ Could not stop server or server was not running.")

    elif args.action == "restart":
        stop_api_server(port=port)
        time.sleep(0.5)
        ok, loc, net = start_api_server(port=port)
        if ok:
            print(f"✅ AI Trader API restarted successfully!")
            print(f"   • Local Docs:   {loc}/docs")
            print(f"   • Network Docs: {net}/docs")
        else:
            print(f"❌ Failed to restart AI Trader API.")


if __name__ == "__main__":
    main()
