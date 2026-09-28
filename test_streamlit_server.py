#!/usr/bin/env python3
"""
Unit tests for streamlit_server.py lifecycle manager.
"""

import os
import signal
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import streamlit_server


class TestStreamlitServer(unittest.TestCase):
    """Tests for Streamlit server detection, startup, and shutdown."""

    def test_get_lan_ip(self):
        ip = streamlit_server.get_lan_ip()
        self.assertIsInstance(ip, str)
        # Should be a valid IPv4 format
        parts = ip.split(".")
        self.assertEqual(len(parts), 4)

    @patch("streamlit_server.urlopen")
    def test_is_streamlit_running_true(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        self.assertTrue(streamlit_server.is_streamlit_running(port=8501))

    @patch("streamlit_server.urlopen")
    def test_is_streamlit_running_false(self, mock_urlopen):
        mock_urlopen.side_effect = ConnectionRefusedError("Connection refused")
        self.assertFalse(streamlit_server.is_streamlit_running(port=8501))

    @patch("streamlit_server.is_streamlit_running")
    def test_ensure_streamlit_running_adopts_existing(self, mock_is_running):
        mock_is_running.return_value = True
        with patch("streamlit_server.start_streamlit_server") as mock_start:
            ok, local_url, net_url = streamlit_server.ensure_streamlit_running(
                port=8501, verbose=False
            )
            self.assertTrue(ok)
            self.assertIn("8501", local_url)
            self.assertIn("8501", net_url)
            mock_start.assert_not_called()

    @patch("streamlit_server.subprocess.Popen")
    @patch("streamlit_server.is_streamlit_running")
    def test_start_streamlit_server_success(self, mock_is_running, mock_popen):
        # First call False (not running), then True (started)
        mock_is_running.side_effect = [False, True]
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        with tempfile.TemporaryDirectory() as tmpdir:
            test_pid_file = Path(tmpdir) / ".test.pid"
            test_log = Path(tmpdir) / "test.log"

            with patch("streamlit_server.PID_FILE_PATH", test_pid_file):
                ok, local_url, net_url = streamlit_server.start_streamlit_server(
                    port=8501,
                    log_path=test_log,
                    timeout=2.0,
                )
                self.assertTrue(ok)
                mock_popen.assert_called_once()
                self.assertTrue(test_pid_file.exists())
                self.assertEqual(test_pid_file.read_text().strip(), "12345")

    @patch("os.kill")
    def test_stop_streamlit_server(self, mock_kill):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_pid_file = Path(tmpdir) / ".test.pid"
            test_pid_file.write_text("99999", encoding="utf-8")

            with patch("streamlit_server.PID_FILE_PATH", test_pid_file):
                with patch("streamlit_server.is_streamlit_running", return_value=False):
                    stopped = streamlit_server.stop_streamlit_server(port=8501)
                    self.assertTrue(stopped)
                    mock_kill.assert_called_once_with(99999, signal.SIGTERM)
                    self.assertFalse(test_pid_file.exists())


if __name__ == "__main__":
    unittest.main()
