#!/usr/bin/env python3
"""
Unit and integration tests for master.py orchestration engine.
"""

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, call, patch

import master


class TestMasterExtraction(unittest.TestCase):
    """Tests for extracting active stocks and database signals."""

    def test_extract_most_traded_stocks_empty(self):
        self.assertEqual(master.extract_most_traded_stocks(None), [])
        self.assertEqual(master.extract_most_traded_stocks({}), [])
        self.assertEqual(master.extract_most_traded_stocks({"stocks": "not_a_list"}), [])

    def test_extract_most_traded_stocks_limit(self):
        sample = {
            "stocks": [
                {"symbol": "NVDA", "price": 120.0, "volume": 50000000},
                {"symbol": "TSLA", "price": 250.0, "volume": 40000000},
                {"symbol": "AAPL", "price": 220.0, "volume": 30000000},
                {"symbol": "AMD", "price": 160.0, "volume": 20000000},
            ]
        }
        res_2 = master.extract_most_traded_stocks(sample, limit=2)
        self.assertEqual(len(res_2), 2)
        self.assertEqual(res_2[0]["symbol"], "NVDA")
        self.assertEqual(res_2[1]["symbol"], "TSLA")

        res_10 = master.extract_most_traded_stocks(sample, limit=10)
        self.assertEqual(len(res_10), 4)

    def test_get_latest_signal(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            db_path = tf.name

        try:
            conn = sqlite3.connect(db_path)
            with conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        decision TEXT NOT NULL,
                        confidence REAL,
                        buy_score REAL,
                        hold_score REAL,
                        sell_score REAL,
                        quant_score REAL,
                        horizon_days INTEGER,
                        timestamp TEXT,
                        model_used TEXT
                    )
                """)
                cursor.execute("""
                    INSERT INTO signals (symbol, decision, confidence, quant_score, model_used)
                    VALUES ('NVDA', 'BUY', 0.88, 72.5, 'free')
                """)
            conn.close()

            sig = master.get_latest_signal("NVDA", db_path=db_path)
            self.assertIsNotNone(sig)
            self.assertEqual(sig["symbol"], "NVDA")
            self.assertEqual(sig["decision"], "BUY")
            self.assertEqual(sig["confidence"], 0.88)
            self.assertEqual(sig["quant_score"], 72.5)

            # Test missing symbol
            missing = master.get_latest_signal("NONEXISTENT", db_path=db_path)
            self.assertIsNone(missing)
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)


class TestMasterExecutionSubprocess(unittest.TestCase):
    """Tests for subprocess invocation and parameter pass-through."""

    @patch("subprocess.run")
    def test_run_dashboard_script(self, mock_subproc):
        mock_subproc.return_value = MagicMock(returncode=0)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            tf.write(json.dumps({"stocks": [{"symbol": "NVDA"}]}).encode("utf-8"))
            json_path = tf.name

        try:
            code, data = master.run_dashboard_script(
                limit=10,
                tickers="NVDA,TSLA",
                html_path="test_dash.html",
                json_path=json_path,
            )
            self.assertEqual(code, 0)
            self.assertIsNotNone(data)
            self.assertEqual(data["stocks"][0]["symbol"], "NVDA")

            # Verify subprocess call args
            mock_subproc.assert_called_once()
            args, kwargs = mock_subproc.call_args
            cmd = args[0]
            self.assertIn(master.DASHBOARD_SCRIPT, cmd)
            self.assertIn("--limit", cmd)
            self.assertIn("10", cmd)
            self.assertIn("--tickers", cmd)
            self.assertIn("NVDA,TSLA", cmd)
            self.assertIn("--html", cmd)
            self.assertIn("test_dash.html", cmd)
            self.assertIn("--json", cmd)
            self.assertIn(json_path, cmd)
        finally:
            if os.path.exists(json_path):
                os.remove(json_path)

    @patch("subprocess.run")
    def test_run_main_analysis(self, mock_subproc):
        mock_subproc.return_value = MagicMock(returncode=0)

        code, elapsed = master.run_main_analysis(
            symbol="AAPL",
            model="gemini",
            temperature=0.3,
            reasoning_budget=8000,
            reasoning_effort="high",
        )
        self.assertEqual(code, 0)
        self.assertGreaterEqual(elapsed, 0.0)

        mock_subproc.assert_called_once()
        args, kwargs = mock_subproc.call_args
        cmd = args[0]
        self.assertIn(master.MAIN_SCRIPT, cmd)
        self.assertIn("gemini", cmd)
        self.assertIn("AAPL", cmd)
        self.assertIn("--temperature", cmd)
        self.assertIn("0.3", cmd)
        self.assertIn("--reasoning-budget", cmd)
        self.assertIn("8000", cmd)
        self.assertIn("--reasoning-effort", cmd)
        self.assertIn("high", cmd)

    @patch("subprocess.run")
    def test_run_dashboard_script_eu(self, mock_subproc):
        mock_subproc.return_value = MagicMock(returncode=0)
        code, _ = master.run_dashboard_script(
            limit=5,
            market="EU",
            prefer_exchange="DE",
            no_html=True,
        )
        self.assertEqual(code, 0)
        mock_subproc.assert_called_once()
        cmd = mock_subproc.call_args[0][0]
        self.assertIn("--market", cmd)
        self.assertIn("EU", cmd)
        self.assertIn("--prefer-exchange", cmd)
        self.assertIn("DE", cmd)

    @patch("subprocess.run")
    def test_run_main_analysis_eu(self, mock_subproc):
        mock_subproc.return_value = MagicMock(returncode=0)
        code, _ = master.run_main_analysis(
            symbol="SAP.DE",
            model="free",
            market="EU",
            prefer_exchange="DE",
        )
        self.assertEqual(code, 0)
        mock_subproc.assert_called_once()
        cmd = mock_subproc.call_args[0][0]
        self.assertIn("SAP.DE", cmd)
        self.assertIn("--market", cmd)
        self.assertIn("EU", cmd)
        self.assertIn("--prefer-exchange", cmd)
        self.assertIn("DE", cmd)


class TestMasterOrchestration(unittest.TestCase):
    """Tests full orchestration workflows and sequential guarantees."""

    def setUp(self):
        self.patcher = patch("master.ensure_streamlit_running", return_value=(True, "http://localhost:8501", "http://192.168.0.241:8501"))
        self.mock_streamlit = self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    @patch("master.run_main_analysis")
    @patch("master.run_dashboard_script")
    def test_sequential_execution_order(self, mock_dash, mock_main):
        fake_stocks = [
            {"symbol": "SYM1", "price": 10.0, "volume": 1000000, "catalyst_type": "NEWS"},
            {"symbol": "SYM2", "price": 20.0, "volume": 2000000, "catalyst_type": "EARNINGS"},
            {"symbol": "SYM3", "price": 30.0, "volume": 3000000, "catalyst_type": "M&A"},
        ]
        mock_dash.return_value = (0, {"stocks": fake_stocks, "fear_greed": "Neutral"})
        mock_main.return_value = (0, 1.5)

        executed_symbols = []

        def side_effect(symbol, **kwargs):
            executed_symbols.append(symbol)
            return (0, 0.1)

        mock_main.side_effect = side_effect

        report = master.run_master(
            model="free",
            limit=3,
            delay=0.0,
            dry_run=False,
        )

        self.assertEqual(mock_dash.call_count, 1)
        self.assertEqual(mock_main.call_count, 3)
        self.assertEqual(executed_symbols, ["SYM1", "SYM2", "SYM3"])
        self.assertEqual(report["stocks_analyzed"], 3)
        self.assertEqual(report["model"], "free")

    @patch("master.run_main_analysis")
    @patch("master.run_dashboard_script")
    def test_stop_on_error_behavior(self, mock_dash, mock_main):
        fake_stocks = [
            {"symbol": "FAIL1", "price": 10.0, "volume": 1000000},
            {"symbol": "PASS2", "price": 20.0, "volume": 2000000},
        ]
        mock_dash.return_value = (0, {"stocks": fake_stocks})
        mock_main.return_value = (1, 0.5)  # Return failure code 1

        report = master.run_master(
            model="super",
            limit=2,
            delay=0.0,
            stop_on_error=True,
            dry_run=False,
        )

        # Should halt after the first failed symbol
        self.assertEqual(mock_main.call_count, 1)
        self.assertEqual(len(report["results"]), 1)
        self.assertEqual(report["results"][0]["symbol"], "FAIL1")
        self.assertEqual(report["results"][0]["returncode"], 1)

    @patch("master.run_dashboard_script")
    def test_dry_run_does_not_call_main(self, mock_dash):
        fake_stocks = [
            {"symbol": "NVDA", "price": 125.0, "volume": 80000000},
            {"symbol": "TSLA", "price": 240.0, "volume": 60000000},
        ]
        mock_dash.return_value = (0, {"stocks": fake_stocks})

        with patch("master.run_main_analysis") as mock_main:
            report = master.run_master(
                model="free",
                limit=2,
                dry_run=True,
            )
            mock_main.assert_not_called()
            self.assertEqual(len(report["results"]), 2)
            self.assertTrue(report["results"][0]["dry_run"])

    @patch("master.run_main_analysis")
    def test_skip_dashboard_with_json(self, mock_main):
        mock_main.return_value = (0, 0.2)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            tf.write(json.dumps({"stocks": [{"symbol": "INTC", "price": 30.0, "volume": 10000000}]}).encode("utf-8"))
            json_file = tf.name

        try:
            with patch("master.run_dashboard_script") as mock_dash:
                report = master.run_master(
                    model="gemini",
                    limit=1,
                    skip_dashboard=True,
                    dashboard_json=json_file,
                    delay=0.0,
                )
                mock_dash.assert_not_called()
                mock_main.assert_called_once()
                self.assertEqual(report["stocks_analyzed"], 1)
                self.assertEqual(report["results"][0]["symbol"], "INTC")
        finally:
            if os.path.exists(json_file):
                os.remove(json_file)

    @patch("master.run_main_analysis")
    @patch("master.run_dashboard_script")
    def test_export_json_output(self, mock_dash, mock_main):
        fake_stocks = [{"symbol": "MSFT", "price": 400.0, "volume": 20000000}]
        mock_dash.return_value = (0, {"stocks": fake_stocks, "fear_greed": "Greed (65)"})
        mock_main.return_value = (0, 0.3)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            out_json = tf.name

        try:
            report = master.run_master(
                model="free",
                limit=1,
                output_json=out_json,
                delay=0.0,
            )
            self.assertTrue(os.path.exists(out_json))
            with open(out_json, "r", encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual(saved["model"], "free")
            self.assertEqual(len(saved["results"]), 1)
            self.assertEqual(saved["results"][0]["symbol"], "MSFT")
        finally:
            if os.path.exists(out_json):
                os.remove(out_json)


class TestMasterCLI(unittest.TestCase):
    """Tests CLI argument parsing and error handling."""

    def test_valid_models(self):
        for m in ["free", "gemini", "nemotron", "super", "ultra", "kimi", "qwen", "llamacpp"]:
            self.assertIn(m, master.VALID_MODELS)


if __name__ == "__main__":
    unittest.main()
