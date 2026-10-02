#!/usr/bin/env python3
"""
Unit tests for SELL criteria symmetry with BUY, and formal distinction between
short-term bearish directional trade (SHORT) and reducing/exiting an existing holding (LONG_EXIT).
"""

import os
import tempfile
import unittest
from unittest.mock import patch

from db import init_db, save_results, get_latest_signals
from main import normalize_master_trader_json
from quantitative_scoring_service import QuantitativeScoringService


class TestSellCriteriaSymmetry(unittest.TestCase):
    def setUp(self):
        self.symbol = "TEST"
        # Standard bearish snapshot clearing short trade geometry
        # price=100, atr=2.0, high_20d=102, low_20d=90
        # structural_short_stop = max(100 + 3.0, 102 + 1.0) = 103.0 (risk = 3.0)
        # structural_short_target = max(100 - 5.0, 90 - 1.0) = 95.0 (reward = 5.0)
        # short_rr = 5.0 / 3.0 = 1.67 >= 1.5
        self.base_m_data = {
            "current_price": 100.0,
            "atr": 2.0,
            "rsi14": 42.0,
            "rvol_20d": 1.2,
            "avg_dollar_vol_20d": 15_000_000.0,
            "high_20d": 102.0,
            "low_20d": 90.0,
            "suggested_short_stop_loss": 103.0,
            "suggested_short_target_price": 95.0,
            "suggested_stop_loss": 97.0,
            "suggested_target_price": 105.0,
            "deterministic_data_completeness": 0.90,
            "deterministic_5pillar_scores": {
                "composite_quantitative_score": 35.0,  # <= 40 structural breakdown
                "trend_score": 30.0,
                "sector_relative_score": 40.0,
                "market_alpha_score": 35.0,
                "valuation_history_score": 40.0,
                "peer_valuation_score": 35.0,
            },
        }

        self.base_sell_data = {
            "stock": "TEST",
            "decision": "SELL",
            "primary_driver": "QUANT_STRUCTURE",
            "confidence": 0.75,
            "buy_score": 0.10,
            "hold_score": 0.20,
            "sell_score": 0.70,
            "data_completeness": 0.90,
            "horizon_days": 10,
            "quant_score": 35,
            "pillar_scores": {
                "trend": 30.0,
                "sector": 40.0,
                "alpha": 35.0,
                "valuation_history": 40.0,
                "peer_valuation": 35.0,
            },
            "bull_case": ["oversold support near 90"],
            "bear_case": ["breakdown below 20d moving average", "negative momentum"],
            "key_risks": [],
            "missing_information": [],
        }

    def test_valid_short_directional_trade_passes(self):
        """A valid short with composite <= 40, RR >= 1.5, ample liquidity, and calm vol passes as SELL."""
        out = normalize_master_trader_json(
            self.base_sell_data, self.symbol, self.base_m_data, {}, {}
        )
        self.assertEqual(out["decision"], "SELL")
        self.assertEqual(out["sell_type"], "SHORT")
        self.assertIsNotNone(out["short_reward_risk_ratio"])
        self.assertGreaterEqual(out["short_reward_risk_ratio"], 1.5)
        self.assertEqual(out["structural_short_stop_price"], 103.0)
        self.assertEqual(out["structural_short_target_price"], 95.0)

    def test_short_vetoed_when_composite_above_ceiling(self):
        """A short is vetoed if composite > 40 and no extension certificate exists."""
        m_data = dict(self.base_m_data)
        m_data["deterministic_5pillar_scores"] = {
            "composite_quantitative_score": 55.0,  # Neutral/bullish, not a breakdown
            "trend_score": 55.0,
            "sector_relative_score": 55.0,
            "market_alpha_score": 55.0,
            "valuation_history_score": 55.0,
            "peer_valuation_score": 55.0,
        }
        out = normalize_master_trader_json(
            self.base_sell_data, self.symbol, m_data, {}, {}
        )
        self.assertEqual(out["decision"], "HOLD")
        self.assertEqual(out["no_trade_reason"], "INSUFFICIENT_EVIDENCE")
        self.assertTrue(any("deterministic composite" in r and "> 40" in r for r in out["key_risks"]))

    def test_short_vetoed_when_reward_risk_too_low(self):
        """A short is vetoed if short RR < 1.5 (e.g. overhead stop is too distant relative to support)."""
        m_data = dict(self.base_m_data)
        # Price at 100, 20d high at 115 (massive overhead resistance stop = 116, risk = 16)
        # 20d low at 96 (target = 95.5, reward = 4.5) -> RR = 4.5 / 16 = 0.28 < 1.5
        m_data["high_20d"] = 115.0
        m_data["low_20d"] = 96.0
        m_data["suggested_short_stop_loss"] = 116.0
        m_data["suggested_short_target_price"] = 96.0

        out = normalize_master_trader_json(
            self.base_sell_data, self.symbol, m_data, {}, {}
        )
        self.assertEqual(out["decision"], "HOLD")
        self.assertEqual(out["no_trade_reason"], "RR_TOO_LOW")
        self.assertTrue(any("short reward:risk ratio" in r and "< 1.5" in r for r in out["key_risks"]))

    def test_short_vetoed_during_earnings_blackout(self):
        """A short is vetoed when earnings report is within 3 days due to infinite upside gap risk."""
        m_data = dict(self.base_m_data)
        m_data["days_to_earnings"] = 2

        out = normalize_master_trader_json(
            self.base_sell_data, self.symbol, m_data, {}, {}
        )
        self.assertEqual(out["decision"], "HOLD")
        self.assertEqual(out["no_trade_reason"], "EARNINGS_BLACKOUT")
        self.assertTrue(any("earnings report in 2 day(s)" in r for r in out["key_risks"]))

    def test_short_vetoed_near_ex_dividend_date(self):
        """A short is vetoed when ex-dividend is within 2 days due to dividend payout liability and borrow recall."""
        m_data = dict(self.base_m_data)
        m_data["days_to_ex_dividend"] = 1
        m_data["ex_dividend_date"] = "2026-10-03"

        out = normalize_master_trader_json(
            self.base_sell_data, self.symbol, m_data, {}, {}
        )
        self.assertEqual(out["decision"], "HOLD")
        self.assertEqual(out["no_trade_reason"], "EVENT_RISK")
        self.assertTrue(any("ex-dividend date in 1 day(s)" in r for r in out["key_risks"]))

    def test_short_vetoed_when_expected_move_exceeds_overhead_stop(self):
        """A short is vetoed if option expected move exceeds the distance to overhead stop."""
        m_data = dict(self.base_m_data)
        # Overhead stop is at 103.0 (stop distance = $3.00 / 3%)
        # Expected move is $4.50 (4.5%), which is greater than $3.00 stop distance
        gloomberb = {
            "options": {
                "implied_volatility_pct": 55.0,
                "expected_move": 4.50,
                "expected_move_pct": 4.5,
                "days_to_expiration": 10,
            }
        }
        out = normalize_master_trader_json(
            self.base_sell_data, self.symbol, m_data, {}, gloomberb
        )
        self.assertEqual(out["decision"], "HOLD")
        self.assertEqual(out["no_trade_reason"], "EXPECTED_MOVE_EXCEEDS_STOP")
        self.assertTrue(any("expected move ($4.50" in r and "exceeds short overhead stop distance ($3.00" in r for r in out["key_risks"]))

    def test_short_vetoed_on_low_liquidity(self):
        """A short is vetoed if turnover is too low (RVOL < 0.5 or dollar volume < $1M)."""
        m_data = dict(self.base_m_data)
        m_data["rvol_20d"] = 0.35
        m_data["avg_dollar_vol_20d"] = 400_000.0

        out = normalize_master_trader_json(
            self.base_sell_data, self.symbol, m_data, {}, {}
        )
        self.assertEqual(out["decision"], "HOLD")
        self.assertEqual(out["no_trade_reason"], "LOW_LIQUIDITY")
        self.assertTrue(any("insufficient liquidity" in r for r in out["key_risks"]))

    def test_short_vetoed_under_extreme_volatility(self):
        """A short is vetoed under extreme market volatility regime (IV >= 70% crisis ceiling)."""
        m_data = dict(self.base_m_data)
        gloomberb = {
            "options": {
                "implied_volatility_pct": 75.0,  # Extreme crisis IV
            }
        }

        out = normalize_master_trader_json(
            self.base_sell_data, self.symbol, m_data, {}, gloomberb
        )
        self.assertEqual(out["decision"], "HOLD")
        self.assertEqual(out["no_trade_reason"], "VOLATILITY_REGIME")
        self.assertTrue(any("extreme volatility" in r for r in out["key_risks"]))

    # --------------------------------------------------------------------------
    # LONG HOLDING EXIT (LONG_EXIT) TESTS
    # --------------------------------------------------------------------------

    def test_long_exit_holding_thesis_intact_vetoed_to_hold(self):
        """An exit on a long holding whose composite is >= 70 and stop/target not breached is gated to HOLD."""
        m_data = dict(self.base_m_data)
        m_data["is_holding"] = True
        m_data["current_price"] = 100.0
        m_data["suggested_stop_loss"] = 92.0
        m_data["suggested_target_price"] = 115.0
        m_data["deterministic_5pillar_scores"] = {
            "composite_quantitative_score": 75.0,  # Strong bull thesis
            "trend_score": 80.0,
            "sector_relative_score": 70.0,
            "market_alpha_score": 75.0,
            "valuation_history_score": 70.0,
            "peer_valuation_score": 70.0,
        }

        data = dict(self.base_sell_data)
        data["action_type"] = "EXIT"

        out = normalize_master_trader_json(data, self.symbol, m_data, {}, {})
        self.assertEqual(out["decision"], "HOLD")
        self.assertEqual(out["sell_type"], "LONG_EXIT")
        self.assertEqual(out["no_trade_reason"], "HOLDING_THESIS_INTACT")
        self.assertTrue(any("holding thesis is intact" in r for r in out["key_risks"]))

    def test_long_exit_allowed_when_stop_loss_breached(self):
        """A long holding exit is allowed when current price breaches the stop loss."""
        m_data = dict(self.base_m_data)
        m_data["is_holding"] = True
        m_data["current_price"] = 91.0  # Breached stop of 92.0
        m_data["suggested_stop_loss"] = 92.0
        m_data["suggested_target_price"] = 115.0
        m_data["deterministic_5pillar_scores"] = {
            "composite_quantitative_score": 72.0,
            "trend_score": 70.0,
        }

        data = dict(self.base_sell_data)
        data["action_type"] = "EXIT"

        out = normalize_master_trader_json(data, self.symbol, m_data, {}, {})
        self.assertEqual(out["decision"], "SELL")
        self.assertEqual(out["sell_type"], "LONG_EXIT")

    def test_long_exit_allowed_when_target_reached(self):
        """A long holding exit is allowed when current price reaches or exceeds profit target."""
        m_data = dict(self.base_m_data)
        m_data["is_holding"] = True
        m_data["current_price"] = 116.0  # Exceeds target of 115.0
        m_data["suggested_stop_loss"] = 92.0
        m_data["suggested_target_price"] = 115.0
        m_data["deterministic_5pillar_scores"] = {
            "composite_quantitative_score": 72.0,
            "trend_score": 75.0,
        }

        data = dict(self.base_sell_data)
        data["sell_type"] = "LONG_EXIT"

        out = normalize_master_trader_json(data, self.symbol, m_data, {}, {})
        self.assertEqual(out["decision"], "SELL")
        self.assertEqual(out["sell_type"], "LONG_EXIT")

    def test_long_exit_allowed_on_thesis_breakdown(self):
        """A long holding exit is allowed when quantitative composite deteriorates below 50."""
        m_data = dict(self.base_m_data)
        m_data["is_holding"] = True
        m_data["current_price"] = 100.0
        m_data["suggested_stop_loss"] = 90.0
        m_data["suggested_target_price"] = 120.0
        m_data["deterministic_5pillar_scores"] = {
            "composite_quantitative_score": 46.0,  # < 50 breakdown
            "trend_score": 38.0,
        }

        data = dict(self.base_sell_data)
        data["action_type"] = "EXIT"

        out = normalize_master_trader_json(data, self.symbol, m_data, {}, {})
        self.assertEqual(out["decision"], "SELL")
        self.assertEqual(out["sell_type"], "LONG_EXIT")

    def test_long_exit_allowed_for_earnings_derisking(self):
        """A long holding exit is allowed for event de-risking before earnings."""
        m_data = dict(self.base_m_data)
        m_data["is_holding"] = True
        m_data["days_to_earnings"] = 2
        m_data["current_price"] = 100.0
        m_data["suggested_stop_loss"] = 90.0
        m_data["suggested_target_price"] = 120.0
        m_data["deterministic_5pillar_scores"] = {
            "composite_quantitative_score": 72.0,
        }

        data = dict(self.base_sell_data)
        data["sell_type"] = "LONG_EXIT"

        out = normalize_master_trader_json(data, self.symbol, m_data, {}, {})
        self.assertEqual(out["decision"], "SELL")
        self.assertEqual(out["sell_type"], "LONG_EXIT")

    def test_short_vetoed_on_low_data_coverage(self):
        """A short is vetoed if deterministic data coverage is below 80%."""
        m_data = dict(self.base_m_data)
        m_data["deterministic_data_completeness"] = 0.65

        data = dict(self.base_sell_data)
        data["data_completeness"] = 0.65

        out = normalize_master_trader_json(data, self.symbol, m_data, {}, {})
        self.assertEqual(out["decision"], "HOLD")
        self.assertEqual(out["no_trade_reason"], "INSUFFICIENT_EVIDENCE")
        self.assertTrue(any("data coverage 65% < 80% bar" in r for r in out["key_risks"]))

    def test_short_vetoed_on_scheduled_company_event(self):
        """A short is vetoed if a scheduled corporate action or event is imminent."""
        m_data = dict(self.base_m_data)
        gloomberb = {
            "events": {
                "days_to_next_event": 1,
                "next_event_type": "SHAREHOLDER_MEETING",
            }
        }
        out = normalize_master_trader_json(
            self.base_sell_data, self.symbol, m_data, {}, gloomberb
        )
        self.assertEqual(out["decision"], "HOLD")
        self.assertEqual(out["no_trade_reason"], "EVENT_RISK")
        self.assertTrue(any("scheduled company event" in r.lower() for r in out["key_risks"]))

    def test_short_allowed_with_certified_extension_sell(self):
        """A short is allowed with composite up to 65 if unlocked by certified EXTENSION_SELL."""
        import datetime
        today_date = datetime.date.today()
        fresh_news = [{
            "title": "TEST hits new high as valuation stretches",
            "summary": "TEST stock trades at multi-year highs",
            "published_at": today_date.isoformat(),
            "category": "News",
        }]

        m_data = dict(self.base_m_data)
        m_data.update({
            "current_price": 110.0,
            "change_5d_pct": 12.0,
            "rsi14": 78.0,
            "rvol_20d": 1.8,
            "high_20d": 111.0,
            "low_20d": 80.0,
            "atr": 3.0,
            "suggested_short_stop_loss": 114.5,
            "suggested_short_target_price": 102.5,
            "suggested_stop_loss": None,
            "suggested_target_price": None,
            "deterministic_5pillar_scores": {
                "composite_quantitative_score": 62.0,  # <= 65 EXTENSION_MAX_COMPOSITE
                "trend_score": 80.0,
                "valuation_history_score": 35.0,  # Rich
                "peer_valuation_score": 30.0,     # Rich
                "sector_relative_score": 50.0,
                "market_alpha_score": 65.0,
            },
        })

        data = dict(self.base_sell_data)
        data.update({
            "primary_driver": "NEWS_CATALYST",
            "sell_type": "SHORT",
        })

        out = normalize_master_trader_json(
            data, self.symbol, m_data, {}, {"news": fresh_news}
        )
        self.assertEqual(out["decision"], "SELL")
        self.assertEqual(out["sell_type"], "SHORT")
        self.assertGreaterEqual(out["short_reward_risk_ratio"], 1.5)

    # --------------------------------------------------------------------------
    # DATABASE PERSISTENCE TESTS
    # --------------------------------------------------------------------------

    def test_database_persistence_of_short_metrics(self):
        """Verify db.py correctly persists sell_type, short RR, and short geometry columns."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            db_path = tf.name

        try:
            init_db(db_path)
            res = normalize_master_trader_json(
                self.base_sell_data, self.symbol, self.base_m_data, {}, {}
            )
            self.assertEqual(res["decision"], "SELL")
            save_results([res], model_used="TestModel", db_path=db_path)

            signals = get_latest_signals(db_path=db_path, max_age_days=None)
            self.assertEqual(len(signals), 1)
            row = signals[0]
            self.assertEqual(row["symbol"], "TEST")
            self.assertEqual(row["decision"], "SELL")
            self.assertEqual(row["sell_type"], "SHORT")
            self.assertEqual(row["short_reward_risk_ratio"], res["short_reward_risk_ratio"])
            self.assertEqual(row["structural_short_stop_price"], 103.0)
            self.assertEqual(row["structural_short_target_price"], 95.0)
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)


if __name__ == "__main__":
    unittest.main()
