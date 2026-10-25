#!/usr/bin/env python3
"""
Unit tests for Step 2 Authoritative Decision Contract.

Verifies that:
1. Each output has one authoritative final action (final_decision == decision).
2. The model's suggestion is preserved in `model_decision` without controlling the final action.
3. Explicit candidate qualification produces candidate_action, candidate_type, setup_evidence.
4. Gate results and rejection_reasons are deterministically structured.
5. Model's HOLD does NOT override a qualified deterministic setup (e.g. PULLBACK_BUY).
6. Model's BUY does NOT override deterministic gate failures (e.g. BUY_FLOOR).
7. Full persistence into SQLite database retains all contract fields for the dashboard.
"""

import json
import tempfile
import unittest

from db import init_db, save_results, get_latest_signals
from main import normalize_master_trader_json
from quant.policy_config import get_default_policy


class TestAuthoritativeDecisionContract(unittest.TestCase):
    def setUp(self):
        self.policy = get_default_policy()
        self.symbol = "NVDA"

        # Baseline healthy market snapshot that clears all technical & liquidity gates
        self.base_m_data = {
            "current_price": 100.0,
            "ema20": 98.5,
            "ema50": 95.0,
            "ema200": 88.0,
            "change_5d_pct": -3.5,
            "atr": 2.5,
            "suggested_stop_loss": 95.0,        # Risk = 5.0
            "suggested_target_price": 112.5,     # Reward = 12.5 -> RR = 2.5 >= 1.2
            "rsi14": 48.0,                      # Pullback in uptrend (40 <= RSI <= 55)
            "rvol_20d": 1.3,
            "high_20d": 104.0,
            "low_20d": 94.0,
            "avg_dollar_vol_20d": 50_000_000.0,
            "days_to_earnings": 30,
            "deterministic_data_completeness": 0.95,
            "deterministic_5pillar_scores": {
                "composite_quantitative_score": 72.0,   # >= 60.0 buy floor
                "trend_score": 75.0,
                "sector_relative_score": 70.0,
                "market_alpha_score": 74.0,
                "valuation_history_score": 68.0,
                "peer_valuation_score": 73.0,
            },
        }

    def test_model_hold_does_not_override_qualified_pullback_buy(self):
        """
        When the model proposes HOLD, but quantitative criteria qualify a PULLBACK_BUY
        and all compliance gates pass, final_decision must be BUY.
        The model's HOLD remains visible in model_decision without overriding the trade.
        """
        raw_output = {
            "stock": self.symbol,
            "decision": "HOLD",                  # LLM suggests HOLD
            "primary_driver": "QUANT_STRUCTURE",
            "confidence": 0.50,
            "buy_score": 0.20,
            "hold_score": 0.70,
            "sell_score": 0.10,
            "data_completeness": 0.95,
            "bull_case": ["Strong technical support near 95"],
            "bear_case": ["Macro rate uncertainty"],
        }

        res = normalize_master_trader_json(raw_output, self.symbol, self.base_m_data, {}, {})

        # Contract assertion: Authoritative final action returned to caller
        self.assertEqual(res["decision"], "BUY")
        self.assertEqual(res["final_decision"], "BUY")
        self.assertEqual(res["position_action"], "OPEN_LONG")
        self.assertEqual(res["recommendation"], "BUY")
        self.assertIsNone(res["staged_entry_plan"])

        # Contract assertion: Model's raw suggestion preserved separately
        self.assertEqual(res["model_decision"], "HOLD")

        # Contract assertion: Candidate setup qualification
        self.assertEqual(res["candidate_action"], "BUY")
        self.assertEqual(res["candidate_type"], "PULLBACK_BUY")
        self.assertIsInstance(res["setup_evidence"], dict)
        self.assertEqual(res["setup_evidence"].get("trigger"), "pullback_in_uptrend")
        self.assertEqual(res["setup_evidence"].get("composite_score"), 72.0)

        # Contract assertion: Gates and reasons
        self.assertEqual(res["rejection_reasons"], [])
        gate_res = res["gate_results"]
        self.assertIn("BUY_FLOOR", gate_res["passed_gates"])
        self.assertIn("REWARD_TO_RISK", gate_res["passed_gates"])
        self.assertIn("CONFIDENCE_HURDLE", gate_res["passed_gates"])
        self.assertEqual(gate_res["failed_gates"], [])

    def test_model_buy_does_not_override_deterministic_gate_failure(self):
        """
        When the model proposes BUY, but the quantitative composite fails the buy floor,
        final_decision must be HOLD.
        The model's BUY remains visible in model_decision without forcing an invalid trade.
        """
        m_data = dict(self.base_m_data)
        m_data["deterministic_5pillar_scores"] = {
            "composite_quantitative_score": 54.0,   # < 60.0 buy floor
            "trend_score": 55.0,
            "sector_relative_score": 55.0,
            "market_alpha_score": 52.0,
            "valuation_history_score": 54.0,
            "peer_valuation_score": 54.0,
        }

        raw_output = {
            "stock": self.symbol,
            "decision": "BUY",                   # LLM hallucinating / proposing BUY
            "primary_driver": "QUANT_STRUCTURE",
            "confidence": 0.85,
            "buy_score": 0.80,
            "hold_score": 0.15,
            "sell_score": 0.05,
            "data_completeness": 0.95,
        }

        res = normalize_master_trader_json(raw_output, self.symbol, m_data, {}, {})

        # Contract assertion: Final decision is gated to HOLD
        self.assertEqual(res["decision"], "HOLD")
        self.assertEqual(res["final_decision"], "HOLD")
        self.assertEqual(res["position_action"], "NO_ACTION")

        # Contract assertion: Model's proposed action preserved
        self.assertEqual(res["model_decision"], "BUY")

        # Contract assertion: Rejection reasons document why the trade was held
        self.assertIn("BUY_FLOOR", res["gate_results"]["failed_gates"])
        self.assertIn("SCORE_BELOW_BUY_THRESHOLD", res["rejection_reasons"])

    def test_catalyst_dip_buy_taxonomy_and_contract(self):
        """
        Verifies that certified catalyst dip buys retain the correct contract taxonomy.
        """
        fresh_news = [{
            "title": "NVDA slips on export restriction rumor",
            "summary": "Short-term headline dip for NVDA guidance",
            "published_at": "2026-10-25T10:00:00Z",
            "category": "News",
        }]

        m_data = dict(self.base_m_data)
        m_data.update({
            "change_5d_pct": -7.5,
            "rsi14": 38.0,
            "deterministic_5pillar_scores": {
                "composite_quantitative_score": 54.0,   # >= 52.0 dip floor
                "trend_score": 45.0,
                "sector_relative_score": 60.0,
                "market_alpha_score": 50.0,
                "valuation_history_score": 72.0,       # Valuation intact >= 60.0
                "peer_valuation_score": 65.0,          # Valuation intact >= 55.0
            },
        })

        # Subcase A: Model proposed BUY
        raw_buy = {
            "stock": self.symbol,
            "decision": "BUY",
            "primary_driver": "NEWS_CATALYST",
            "confidence": 0.70,
            "buy_score": 0.65,
            "hold_score": 0.25,
            "sell_score": 0.10,
            "data_completeness": 0.95,
        }
        res_buy = normalize_master_trader_json(raw_buy, self.symbol, m_data, {}, {"news": fresh_news})
        self.assertEqual(res_buy["decision"], "BUY")
        self.assertEqual(res_buy["final_decision"], "BUY")
        self.assertEqual(res_buy["model_decision"], "BUY")
        self.assertEqual(res_buy["candidate_action"], "BUY")
        self.assertEqual(res_buy["candidate_type"], "CATALYST_DIP_BUY")
        self.assertEqual(res_buy["position_action"], "OPEN_LONG")

        # Subcase B: Model proposed DEEP BUY
        raw_deep = dict(raw_buy)
        raw_deep["decision"] = "DEEP BUY"
        res_deep = normalize_master_trader_json(raw_deep, self.symbol, m_data, {}, {"news": fresh_news})
        self.assertEqual(res_deep["decision"], "DEEP BUY")
        self.assertEqual(res_deep["final_decision"], "DEEP BUY")
        self.assertEqual(res_deep["model_decision"], "DEEP BUY")
        self.assertEqual(res_deep["candidate_action"], "DEEP BUY")
        self.assertEqual(res_deep["candidate_type"], "CATALYST_DIP_BUY")
        self.assertEqual(res_deep["position_action"], "OPEN_LONG")

    def test_database_persistence_of_authoritative_contract_fields(self):
        """
        Verifies that SQLite signals table persists all 7 new authoritative contract fields
        and that get_latest_signals retrieves them properly for dashboard inspection.
        """
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            db_path = tf.name

        try:
            init_db(db_path)

            raw_output = {
                "stock": self.symbol,
                "decision": "HOLD",
                "primary_driver": "QUANT_STRUCTURE",
                "confidence": 0.50,
                "buy_score": 0.20,
                "hold_score": 0.70,
                "sell_score": 0.10,
                "data_completeness": 0.95,
            }
            res = normalize_master_trader_json(raw_output, self.symbol, self.base_m_data, {}, {})
            save_results([res], db_path=db_path)

            latest = get_latest_signals(db_path=db_path)
            self.assertEqual(len(latest), 1)
            sig = latest[0]

            # Contract checks on persisted row
            self.assertEqual(sig["symbol"], "NVDA")
            self.assertEqual(sig["decision"], "BUY")
            self.assertEqual(sig["final_decision"], "BUY")
            self.assertEqual(sig["model_decision"], "HOLD")
            self.assertEqual(sig["candidate_action"], "BUY")
            self.assertEqual(sig["candidate_type"], "PULLBACK_BUY")
            self.assertEqual(sig["position_action"], "OPEN_LONG")

            # JSON-serialized fields
            setup_ev = json.loads(sig["setup_evidence"])
            self.assertEqual(setup_ev.get("trigger"), "pullback_in_uptrend")

            gate_res = json.loads(sig["gate_results"])
            self.assertIn("BUY_FLOOR", gate_res.get("passed_gates", []))

            # Since the trade passed with no rejections, rejection_reasons is empty
            self.assertEqual(sig["rejection_reasons"], "")

        finally:
            import os
            if os.path.exists(db_path):
                os.remove(db_path)


if __name__ == "__main__":
    unittest.main()
