"""
Unit tests for Forward Workflow Accuracy Testing:
- Strictly binary BUY/SELL recommendation from LLM
- Database persistence of testing analysis in signals
- Outcome evaluation comparing BUY/SELL decisions to price action at horizon completion
- Retrieval of pending forward signals for UI tracking
"""

import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from db import init_db, save_results, get_latest_signals, update_signal_outcomes, get_pending_signals
from signal_schema import validate_signal_json
from main import normalize_master_trader_json


class TestForwardAccuracyWorkflow(unittest.TestCase):
    def setUp(self):
        self.symbol = "NVDA"
        self.base_m_data = {
            "current_price": 100.0,
            "ema20": 98.5,
            "ema50": 95.0,
            "ema200": 88.0,
            "change_5d_pct": 2.5,
            "atr": 2.5,
            "suggested_stop_loss": 95.0,
            "suggested_target_price": 115.0,
            "rsi14": 55.0,
            "rvol_20d": 1.2,
            "high_20d": 105.0,
            "low_20d": 92.0,
            "avg_dollar_vol_20d": 50_000_000.0,
            "days_to_earnings": 30,
            "deterministic_data_completeness": 0.95,
            "deterministic_5pillar_scores": {
                "composite_quantitative_score": 75.0,
                "trend_score": 75.0,
                "sector_relative_score": 70.0,
                "market_alpha_score": 74.0,
                "valuation_history_score": 68.0,
                "peer_valuation_score": 73.0,
            },
        }

    def test_schema_binary_decision_enforcement(self):
        """Tests that validate_signal_json in binary mode normalizes or enforces strictly BUY or SELL."""
        # BUY output is valid
        buy_obj = {
            "stock": self.symbol,
            "decision": "BUY",
            "primary_driver": "QUANT_STRUCTURE",
            "confidence": 0.80,
            "buy_score": 0.70,
            "hold_score": 0.0,
            "sell_score": 0.30,
            "horizon_days": 20,
            "horizon_sessions": 20,
        }
        norm_buy, err_buy = validate_signal_json(buy_obj, require_binary_decision=True)
        self.assertEqual(err_buy, [])
        self.assertEqual(norm_buy["decision"], "BUY")

        # SELL output is valid
        sell_obj = {
            "stock": self.symbol,
            "decision": "SELL",
            "primary_driver": "QUANT_STRUCTURE",
            "confidence": 0.80,
            "buy_score": 0.20,
            "hold_score": 0.0,
            "sell_score": 0.80,
            "horizon_days": 20,
            "horizon_sessions": 20,
        }
        norm_sell, err_sell = validate_signal_json(sell_obj, require_binary_decision=True)
        self.assertEqual(err_sell, [])
        self.assertEqual(norm_sell["decision"], "SELL")

        # HOLD output gets resolved directionally in binary mode
        hold_obj = {
            "stock": self.symbol,
            "decision": "HOLD",
            "primary_driver": "QUANT_STRUCTURE",
            "confidence": 0.60,
            "buy_score": 0.55,
            "hold_score": 0.0,
            "sell_score": 0.45,
            "horizon_days": 20,
            "horizon_sessions": 20,
        }
        norm_hold, err_hold = validate_signal_json(hold_obj, require_binary_decision=True)
        self.assertEqual(err_hold, [])
        self.assertEqual(norm_hold["decision"], "BUY")

    def test_database_persistence_and_pending_signals(self):
        """Tests that analysis is saved to SQLite with test_decision and retrieved via get_pending_signals."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            db_path = tf.name

        try:
            init_db(db_path)

            raw_output = {
                "stock": self.symbol,
                "decision": "BUY",
                "primary_driver": "QUANT_STRUCTURE",
                "confidence": 0.85,
                "buy_score": 0.80,
                "hold_score": 0.0,
                "sell_score": 0.20,
                "data_completeness": 0.95,
            }
            res = normalize_master_trader_json(raw_output, self.symbol, self.base_m_data, {}, {})
            self.assertEqual(res["model_decision"], "BUY")
            self.assertEqual(res["test_decision"], "BUY")

            save_results([res], db_path=db_path)

            latest = get_latest_signals(db_path=db_path)
            self.assertEqual(len(latest), 1)
            self.assertEqual(latest[0]["symbol"], self.symbol)
            self.assertEqual(latest[0]["model_decision"], "BUY")

            # Check that pending signal is retrieved for outcome tracking
            pending = get_pending_signals(db_path=db_path)
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0]["symbol"], self.symbol)
            self.assertEqual(pending[0]["model_decision"], "BUY")
            self.assertEqual(pending[0]["entry_price"], 100.0)

        finally:
            import os
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_outcome_evaluator_waits_for_horizon_completion(self):
        """Tests that update_signal_outcomes does not prematurely evaluate fresh signals before horizon completion."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            db_path = tf.name

        try:
            init_db(db_path)

            # Insert a signal created just 2 days ago (less than 20 trading sessions / 28 calendar days)
            two_days_ago = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S")
            res = {
                "stock": "AAPL",
                "decision": "BUY",
                "model_decision": "BUY",
                "test_decision": "BUY",
                "entry_price": 150.0,
                "target_price": 170.0,
                "stop_loss_price": 140.0,
                "horizon_days": 20,
                "timestamp": two_days_ago,
            }
            save_results([res], db_path=db_path)

            # Outcome evaluator without force should skip fresh 20-session signals
            evaluated = update_signal_outcomes(db_path=db_path, min_bars=20, force=False)
            self.assertEqual(evaluated, 0)

            # Signal remains in pending list
            pending = get_pending_signals(db_path=db_path)
            self.assertEqual(len(pending), 1)

        finally:
            import os
            if os.path.exists(db_path):
                os.remove(db_path)


if __name__ == "__main__":
    unittest.main()
