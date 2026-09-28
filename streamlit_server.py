#!/usr/bin/env python3
"""
Managed Streamlit Server Lifecycle Service
==========================================
Manages the background execution of the Streamlit dashboard (`app.py`).
- Detects if Streamlit is already running on the configured port.
- Launches Streamlit in a detached background daemon if not active.
- Waits for health endpoint check (/_stcore/health) to confirm readiness.
- Resolves local loopback and LAN/WiFi IP addresses for multi-device access.
- Provides start, status, and stop CLI actions.
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
APP_SCRIPT = BASE_DIR / "app.py"
DEFAULT_PORT = 8501
DEFAULT_HOST = "0.0.0.0"
DEFAULT_LOG_PATH = BASE_DIR / "streamlit.log"
PID_FILE_PATH = BASE_DIR / ".streamlit_server.pid"


def get_lan_ip() -> str:
    """Discovers the machine's primary local network IP (e.g., 192.168.0.x)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Connecting to a public routing target without sending data determines the outgoing interface IP
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def is_streamlit_running(host: str = "127.0.0.1", port: int = DEFAULT_PORT, timeout: float = 1.0) -> bool:
    """
    Checks if a healthy Streamlit server is answering requests on the target port.
    Uses the official Streamlit health check endpoint '/_stcore/health'.
    """
    url = f"http://{host}:{port}/_stcore/health"
    try:
        with urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except (HTTPError, URLError, socket.timeout, ConnectionRefusedError, OSError):
        # Fallback probe to root URL '/'
        try:
            root_url = f"http://{host}:{port}/"
            with urlopen(root_url, timeout=timeout) as resp:
                return resp.status in (200, 304)
        except Exception:
            return False


def start_streamlit_server(
    port: int = DEFAULT_PORT,
    host: str = DEFAULT_HOST,
    log_path: Path = DEFAULT_LOG_PATH,
    timeout: float = 15.0,
    headless: bool = True,
) -> Tuple[bool, str, str]:
    """
    Starts Streamlit in the background as a detached daemon process.
    Returns (success, local_url, network_url).
    """
    lan_ip = get_lan_ip()
    local_url = f"http://localhost:{port}"
    network_url = f"http://{lan_ip}:{port}"

    if is_streamlit_running(port=port):
        return True, local_url, network_url

    if not APP_SCRIPT.exists():
        print(f"[StreamlitServer] Error: Target application script '{APP_SCRIPT}' not found.")
        return False, "", ""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8")

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_SCRIPT),
        f"--server.port={port}",
        f"--server.address={host}",
    ]
    if headless:
        cmd.append("--server.headless=true")

    try:
        # Launch detached background daemon process
        process = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        PID_FILE_PATH.write_text(str(process.pid), encoding="utf-8")
    except Exception as e:
        print(f"[StreamlitServer] Failed to launch process: {e}")
        log_file.close()
        return False, "", ""

    # Poll until ready or timeout
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_streamlit_running(port=port):
            return True, local_url, network_url
        if process.poll() is not None:
            # Process exited prematurely
            print(f"[StreamlitServer] Server exited prematurely with code {process.returncode}.")
            log_file.close()
            return False, "", ""
        time.sleep(0.5)

    print(f"[StreamlitServer] Timed out waiting for server to answer on port {port}.")
    return False, local_url, network_url


def ensure_streamlit_running(
    port: int = DEFAULT_PORT,
    host: str = DEFAULT_HOST,
    timeout: float = 15.0,
    verbose: bool = True,
) -> Tuple[bool, str, str]:
    """
    Guarantees that Streamlit is running. If already running, adopts it immediately.
    If not running, starts it in the background.
    """
    lan_ip = get_lan_ip()
    local_url = f"http://localhost:{port}"
    network_url = f"http://{lan_ip}:{port}"

    if is_streamlit_running(port=port):
        if verbose:
            print(f"[Streamlit] Server is active and accessible:")
            print(f"  • Local:   {local_url}")
            print(f"  • Network: {network_url} (accessible on WiFi)")
        return True, local_url, network_url

    if verbose:
        print(f"[Streamlit] Starting dashboard server on port {port}...")

    ok, local_url, network_url = start_streamlit_server(
        port=port, host=host, timeout=timeout
    )

    if ok:
        if verbose:
            print(f"[Streamlit] Dashboard server launched successfully!")
            print(f"  • Local:   {local_url}")
            print(f"  • Network: {network_url} (accessible on WiFi)")
    else:
        if verbose:
            print(f"[Streamlit] Warning: Failed to automatically start Streamlit dashboard.")

    return ok, local_url, network_url


def stop_streamlit_server(port: int = DEFAULT_PORT) -> bool:
    """Stops the managed Streamlit server if running."""
    stopped = False

    # Try PID file first
    if PID_FILE_PATH.exists():
        try:
            pid = int(PID_FILE_PATH.read_text(encoding="utf-8").strip())
            os.kill(pid, signal.SIGTERM)
            stopped = True
            print(f"[StreamlitServer] Sent SIGTERM to PID {pid}.")
        except ProcessLookupError:
            pass
        except Exception as e:
            print(f"[StreamlitServer] Error terminating PID: {e}")
        finally:
            try:
                PID_FILE_PATH.unlink(missing_ok=True)
            except Exception:
                pass

    # Ensure port is released
    if is_streamlit_running(port=port):
        time.sleep(1.0)
        if not is_streamlit_running(port=port):
            stopped = True

    return stopped


def main() -> None:
    parser = argparse.ArgumentParser(description="Managed Streamlit Dashboard Server")
    parser.add_argument(
        "action",
        nargs="?",
        choices=["start", "status", "stop", "restart"],
        default="status",
        help="Action to perform: start, status, stop, restart (default: status)",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to bind/check (default: {DEFAULT_PORT})",
    )
    args = parser.parse_args()

    if args.action == "status":
        running = is_streamlit_running(port=args.port)
        lan_ip = get_lan_ip()
        if running:
            print(f"✅ Streamlit server is RUNNING on port {args.port}.")
            print(f"   • Local URL:   http://localhost:{args.port}")
            print(f"   • Network URL: http://{lan_ip}:{args.port}")
            if PID_FILE_PATH.exists():
                print(f"   • Recorded PID: {PID_FILE_PATH.read_text().strip()}")
        else:
            print(f"❌ Streamlit server is NOT running on port {args.port}.")
            sys.exit(1)

    elif args.action == "start":
        ok, loc, net = ensure_streamlit_running(port=args.port, verbose=True)
        if not ok:
            sys.exit(1)

    elif args.action == "stop":
        if is_streamlit_running(port=args.port) or PID_FILE_PATH.exists():
            stop_streamlit_server(port=args.port)
            print(f"🛑 Streamlit server stopped.")
        else:
            print(f"Streamlit server is not running on port {args.port}.")

    elif args.action == "restart":
        stop_streamlit_server(port=args.port)
        time.sleep(1.0)
        ok, loc, net = start_streamlit_server(port=args.port)
        if ok:
            print(f"✅ Streamlit server restarted successfully.")
            print(f"   • Local URL:   {loc}")
            print(f"   • Network URL: {net}")
        else:
            print(f"❌ Failed to restart Streamlit server.")
            sys.exit(1)


if __name__ == "__main__":
    main()
