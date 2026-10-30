"""Unit Tests for Centralized Trading Policy Configuration (Step 5.1).

Validates:
1. Version-controlled YAML loading and structure verification.
2. Contradiction resolution and threshold consistency validation.
3. Separate Dip-Buy branch specification vs. standard BUY floor precedence.
4. Holding-thesis guard conflict resolution (preventing premature exits on noise).
5. Integration of active policy in normalize_master_trader_json.
6. Environmental variable override mechanics.
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

# Ensure repo root is on sys.path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from policy_config import (
    DipBuyConfig,
    ScoresConfig,
    TradeQualityConfig,
    TradingPolicy,
    get_default_policy,
    load_policy_config,
)

try:
    from main import normalize_master_trader_json
except ImportError:
    normalize_master_trader_json = None


class TestTradingPolicyConfig(unittest.TestCase):
    """Test suite for Step 5.1 Centralized Trading Policy Configuration."""

    def setUp(self):
        self.policy = load_policy_config()

    # -------------------------------------------------------------------------
    # 1. Loading & Schema Completeness
    # -------------------------------------------------------------------------
    def test_policy_yaml_loading_completeness(self):
        """Verifies that config/trading_policy.yaml loads with exact expected sections."""
        self.assertEqual(self.policy.version, "1.0.0")
        self.assertIn("Compliance Policy", self.policy.policy_name)

        # Scores
        self.assertEqual(self.policy.scores.buy_minimum, 60.0)
        self.assertEqual(self.policy.scores.dip_buy_minimum, 48.0)
        self.assertEqual(self.policy.scores.short_maximum, 45.0)
        self.assertEqual(self.policy.scores.holding_thesis_minimum, 70.0)

        # Dip Buy
        self.assertTrue(self.policy.dip_buy.enabled)
        self.assertEqual(self.policy.dip_buy.score_minimum, 48.0)
        self.assertEqual(self.policy.dip_buy.min_drop_pct_5d, 5.0)
        self.assertEqual(self.policy.dip_buy.max_drop_pct_5d, 30.0)
        self.assertEqual(self.policy.dip_buy.min_rsi, 20.0)
        self.assertEqual(self.policy.dip_buy.max_rsi, 45.0)
        self.assertEqual(self.policy.dip_buy.min_valuation_history, 60.0)
        self.assertEqual(self.policy.dip_buy.min_peer_valuation, 55.0)

        # Trade Quality
        self.assertEqual(self.policy.trade_quality.reward_risk_minimum, 1.20)
        self.assertEqual(self.policy.trade_quality.min_data_completeness, 0.80)
        self.assertEqual(self.policy.trade_quality.min_rvol_20d, 0.50)
        self.assertEqual(self.policy.trade_quality.min_avg_dollar_volume, 1_000_000.0)

        # Events
        self.assertEqual(self.policy.events.earnings_blackout_sessions, 3)
        self.assertTrue(self.policy.events.event_risk_blackout)
        self.assertEqual(self.policy.events.ex_dividend_short_blackout_days, 2)

        # Volatility
        self.assertEqual(self.policy.volatility.min_vol_factor, 0.80)
        self.assertEqual(self.policy.volatility.max_stop_expected_move_ratio, 0.85)

        # Data Quality
        self.assertTrue(self.policy.data_quality.require_fresh_market_data)
        self.assertTrue(self.policy.data_quality.require_fresh_ml_data_for_in_scope_tickers)

        # Governance
        self.assertFalse(self.policy.policy.llm_has_execution_authority)
        self.assertTrue(self.policy.policy.fail_closed_for_new_entries)

    # -------------------------------------------------------------------------
    # 2. Contradiction Resolution & Schema Validation
    # -------------------------------------------------------------------------
    def test_rejection_of_contradictory_dip_buy_floor(self):
        """Rejects policies where dip-buy minimum exceeds standard buy minimum."""
        bad_scores = ScoresConfig(buy_minimum=60.0, dip_buy_minimum=65.0)
        with self.assertRaises(ValueError) as ctx:
            bad_scores.validate()
        self.assertIn("cannot be greater than standard buy_minimum", str(ctx.exception))

    def test_rejection_of_overlapping_long_short_thresholds(self):
        """Rejects policies where short ceiling overlaps with buy floor."""
        bad_scores = ScoresConfig(buy_minimum=50.0, dip_buy_minimum=40.0, short_maximum=55.0)
        with self.assertRaises(ValueError) as ctx:
            bad_scores.validate()
        self.assertIn("must be strictly less than buy_minimum", str(ctx.exception))

    def test_rejection_of_invalid_bounds(self):
        """Rejects negative reward:risk and inverted RSI/drop bounds."""
        bad_tq = TradeQualityConfig(reward_risk_minimum=-1.0)
        with self.assertRaises(ValueError):
            bad_tq.validate()

        bad_dip = DipBuyConfig(min_rsi=55.0, max_rsi=35.0)
        with self.assertRaises(ValueError):
            bad_dip.validate()

    # -------------------------------------------------------------------------
    # 3. Dip-Buy Branch Resolution Precedence
    # -------------------------------------------------------------------------
    def test_resolve_buy_score_floor_precedence(self):
        """Verifies precedence resolution between standard BUY and certified Dip-Buy."""
        # Standard BUY candidate (no certified dip)
        floor, rule = self.policy.resolve_buy_score_floor(is_certified_dip_buy=False)
        self.assertEqual(floor, 60.0)
        self.assertEqual(rule, "RULE_STANDARD_BUY_FLOOR")

        # Certified Dip-Buy candidate
        dip_floor, dip_rule = self.policy.resolve_buy_score_floor(is_certified_dip_buy=True)
        self.assertEqual(dip_floor, 48.0)
        self.assertEqual(dip_rule, "RULE_DIP_BUY_DISCOUNT_FLOOR")

        # If dip-buy is disabled, standard floor is universally enforced
        disabled_policy = load_policy_config()
        disabled_policy.dip_buy.enabled = False
        dis_floor, dis_rule = disabled_policy.resolve_buy_score_floor(is_certified_dip_buy=True)
        self.assertEqual(dis_floor, 60.0)
        self.assertEqual(dis_rule, "RULE_STANDARD_BUY_FLOOR")

    # -------------------------------------------------------------------------
    # 4. Holding-Thesis Guard
    # -------------------------------------------------------------------------
    def test_holding_thesis_guard_prevents_premature_exit(self):
        """Verifies that an active holding with intact thesis (score >= 70) blocks premature SELL."""
        # Active holding with strong score -> Thesis intact
        self.assertTrue(self.policy.is_holding_thesis_intact(is_holding=True, composite_score=72.0))

        # Active holding with weak score -> Thesis broken, allow exit
        self.assertFalse(self.policy.is_holding_thesis_intact(is_holding=True, composite_score=65.0))

        # Not a holding (new candidate short) -> Does not apply
        self.assertFalse(self.policy.is_holding_thesis_intact(is_holding=False, composite_score=85.0))

        # Critical evidence risk flag overrides thesis guard
        self.assertFalse(
            self.policy.is_holding_thesis_intact(is_holding=True, composite_score=75.0, has_critical_risk=True)
        )

    # -------------------------------------------------------------------------
    # 5. Integration with normalize_master_trader_json
    # -------------------------------------------------------------------------
    @unittest.skipIf(normalize_master_trader_json is None, "normalize_master_trader_json required")
    def test_normalize_enforces_policy_thresholds(self):
        """Verifies that normalize_master_trader_json enforces the active policy thresholds."""
        # Construct mock market data
        m_data = {
            "current_price": 100.0,
            "atr": 2.5,
            "suggested_stop_loss": 96.0,
            "suggested_target_price": 108.0,  # RR = (108-100)/(100-96) = 2.0 >= 1.2
            "rsi14": 55.0,
            "rvol_20d": 1.5,
            "high_20d": 109.0,
            "low_20d": 95.0,
            "avg_dollar_vol_20d": 10_000_000.0,
            "days_to_earnings": 30,
            "deterministic_data_completeness": 0.88,
            "deterministic_5pillar_scores": {
                "composite_quantitative_score": 58.0,  # Below 60.0 standard buy floor
                "trend_score": 60.0,
                "sector_relative_score": 60.0,
                "market_alpha_score": 60.0,
                "valuation_history_score": 55.0,
                "peer_valuation_score": 55.0,
            },
        }

        # Candidate BUY with composite 58.0 < 60.0 must be downgraded to HOLD
        raw_output = {
            "decision": "BUY",
            "buy_score": 0.70,
            "hold_score": 0.20,
            "sell_score": 0.10,
            "data_completeness": 0.88,
        }
        res = normalize_master_trader_json(raw_output, "TEST", m_data=m_data, policy=self.policy)
        self.assertEqual(res["decision"], "HOLD")
        self.assertEqual(res["no_trade_reason"], "INSUFFICIENT_EVIDENCE")
        self.assertTrue(any("58.0/100 < 60" in r for r in res.get("key_risks", [])))

        # Candidate BUY with composite 62.0 >= 60.0 must be CERTIFIED
        m_data["deterministic_5pillar_scores"]["composite_quantitative_score"] = 62.0
        res_pass = normalize_master_trader_json(raw_output, "TEST", m_data=m_data, policy=self.policy)
        self.assertEqual(res_pass["decision"], "BUY")
        self.assertIsNone(res_pass["no_trade_reason"])

    # -------------------------------------------------------------------------
    # 6. Environmental Overrides
    # -------------------------------------------------------------------------
    def test_environment_variable_overrides(self):
        """Verifies that environment variables override configuration defaults when set."""
        with patch.dict(os.environ, {"BUY_COMPOSITE_FLOOR": "65.0", "MIN_REWARD_RISK_RATIO": "1.50"}):
            override_policy = load_policy_config()
            self.assertEqual(override_policy.scores.buy_minimum, 65.0)
            self.assertEqual(override_policy.trade_quality.reward_risk_minimum, 1.50)


if __name__ == "__main__":
    unittest.main()
