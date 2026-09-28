import os
import unittest
import tempfile
import sqlite3
from datetime import datetime, timedelta, timezone

from db import init_db, get_latest_signals, get_summary_stats, delete_signals_older_than, get_connection


class TestDBRecency(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp_dir.name, "test_trader.db")
        init_db(self.db_path)

        # Seed test data with different timestamps
        now = datetime.now(timezone.utc)
        t_2d = (now - timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S")
        t_5d = (now - timedelta(days=5)).strftime("%Y-%m-%d %H:%M:%S")
        t_10d = (now - timedelta(days=10)).strftime("%Y-%m-%d %H:%M:%S")
        t_30d = (now - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")

        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            # Stock A: recent signal (2 days ago)
            cursor.execute("""
                INSERT INTO signals (timestamp, symbol, decision, confidence, reason, model_used)
                VALUES (?, 'STOCK_A', 'BUY', 0.85, 'Recent setup', 'Gemini')
            """, (t_2d,))

            # Stock B: older signal (5 days ago) AND an even older signal (10 days ago)
            cursor.execute("""
                INSERT INTO signals (timestamp, symbol, decision, confidence, reason, model_used)
                VALUES (?, 'STOCK_B', 'HOLD', 0.50, 'Old setup', 'Gemini')
            """, (t_10d,))
            cursor.execute("""
                INSERT INTO signals (timestamp, symbol, decision, confidence, reason, model_used)
                VALUES (?, 'STOCK_B', 'SELL', 0.75, 'Newer setup', 'Gemini')
            """, (t_5d,))

            # Stock C: very old signal (30 days ago)
            cursor.execute("""
                INSERT INTO signals (timestamp, symbol, decision, confidence, reason, model_used)
                VALUES (?, 'STOCK_C', 'BUY', 0.90, 'Ancient setup', 'Gemini')
            """, (t_30d,))
            conn.commit()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_get_latest_signals_default_7_days(self):
        # By default (max_age_days=7), STOCK_C (30d ago) should be excluded
        # STOCK_A (2d) and STOCK_B (5d) should be included.
        signals = get_latest_signals(db_path=self.db_path)
        symbols = [s["symbol"] for s in signals]
        self.assertIn("STOCK_A", symbols)
        self.assertIn("STOCK_B", symbols)
        self.assertNotIn("STOCK_C", symbols)
        self.assertEqual(len(signals), 2)

        # STOCK_B should have its latest signal (SELL from 5d ago, not HOLD from 10d ago)
        b_sig = next(s for s in signals if s["symbol"] == "STOCK_B")
        self.assertEqual(b_sig["decision"], "SELL")

    def test_get_latest_signals_all_time(self):
        # max_age_days=None returns all stocks with latest record
        signals = get_latest_signals(db_path=self.db_path, max_age_days=None)
        symbols = [s["symbol"] for s in signals]
        self.assertIn("STOCK_A", symbols)
        self.assertIn("STOCK_B", symbols)
        self.assertIn("STOCK_C", symbols)
        self.assertEqual(len(signals), 3)

    def test_get_latest_signals_custom_days(self):
        # max_age_days=3 should only return STOCK_A (2d ago)
        signals = get_latest_signals(db_path=self.db_path, max_age_days=3)
        symbols = [s["symbol"] for s in signals]
        self.assertEqual(symbols, ["STOCK_A"])

    def test_get_summary_stats_recency(self):
        # 7-day stats: STOCK_A (BUY) and STOCK_B (SELL). STOCK_C (BUY) is excluded.
        stats = get_summary_stats(db_path=self.db_path, max_age_days=7)
        self.assertEqual(stats["total_tracked"], 2)
        self.assertEqual(stats["buy_count"], 1)
        self.assertEqual(stats["sell_count"], 1)
        self.assertEqual(stats["hold_count"], 0)

        # All-time stats: STOCK_A (BUY), STOCK_B (SELL), STOCK_C (BUY)
        stats_all = get_summary_stats(db_path=self.db_path, max_age_days=None)
        self.assertEqual(stats_all["total_tracked"], 3)
        self.assertEqual(stats_all["buy_count"], 2)
        self.assertEqual(stats_all["sell_count"], 1)

    def test_delete_signals_older_than(self):
        # Delete signals older than 7 days (should delete STOCK_B's 10d signal and STOCK_C's 30d signal)
        deleted = delete_signals_older_than(days=7, db_path=self.db_path)
        self.assertEqual(deleted, 2)

        # After deletion, all-time query should only have STOCK_A and STOCK_B
        all_signals = get_latest_signals(db_path=self.db_path, max_age_days=None)
        symbols = [s["symbol"] for s in all_signals]
        self.assertIn("STOCK_A", symbols)
        self.assertIn("STOCK_B", symbols)
        self.assertNotIn("STOCK_C", symbols)


if __name__ == "__main__":
    unittest.main()
