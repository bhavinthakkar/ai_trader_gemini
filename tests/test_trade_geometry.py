"""
Comprehensive Unit Test Suite for Trade Geometry & Reward-to-Risk Engine (Step 5.2).

Validates:
1. Mathematical correctness of Long RR = (Target - Entry) / (Entry - Stop)
2. Mathematical correctness of Short RR = (Entry - Target) / (Stop - Entry)
3. Rejection of invalid price ordering (non-positive prices, non-monotonic sequences)
4. Rejection of zero or negative risk denominators
5. Execution assumptions & transaction cost accounting (slippage, fees, borrow costs, cost drag)
6. Deterministic functions for entry, stop, target & rejection of unvalidated LLM prices
7. Realized trade outcome justification: Break-even win rate before & after costs, statistical edge & positive expectancy
"""

import math
import unittest
from quant.trade_geometry import (
    TradeDirection,
    PriceOrderingStatus,
    ExecutionCostModel,
    RewardRiskResult,
    validate_price_ordering,
    compute_breakeven_win_rate,
    estimate_empirical_win_probability,
    compute_expected_value_r,
    compute_validated_reward_risk,
    derive_deterministic_channel_geometry,
)
from quantitative_scoring_service import QuantitativeScoringService


class TestTradeGeometryEngine(unittest.TestCase):
    """Exhaustive test suite for Step 5.2 reward-to-risk and price validation."""

    def setUp(self):
        self.standard_costs = ExecutionCostModel(
            slippage_bps=5.0,
            commission_bps=1.0,
            fixed_fee_per_share=0.0,
        )

    # -------------------------------------------------------------------------
    # 1. Long Entry Formula: RR_long = (Target - Entry) / (Entry - Stop)
    # -------------------------------------------------------------------------
    def test_long_reward_risk_calculation(self):
        """Verifies exact formula computation for long entry."""
        entry = 100.0
        stop = 95.0   # risk = 5.0
        target = 110.0 # reward = 10.0
        # Zero cost model to verify gross formula
        zero_costs = ExecutionCostModel(slippage_bps=0.0, commission_bps=0.0)

        res = compute_validated_reward_risk(
            direction=TradeDirection.LONG,
            entry_price=entry,
            stop_loss=stop,
            target_price=target,
            execution_costs=zero_costs,
        )

        self.assertTrue(res.is_valid)
        self.assertEqual(res.status, PriceOrderingStatus.VALID)
        self.assertEqual(res.gross_reward, 10.0)
        self.assertEqual(res.gross_risk, 5.0)
        self.assertEqual(res.gross_reward_risk_ratio, 2.0)
        # Break-even win rate for 2.0 RR: 1 / (1 + 2.0) = 0.3333
        self.assertAlmostEqual(res.breakeven_win_rate_gross, 1.0 / 3.0, places=3)

    # -------------------------------------------------------------------------
    # 2. Short Entry Formula: RR_short = (Entry - Target) / (Stop - Entry)
    # -------------------------------------------------------------------------
    def test_short_reward_risk_calculation(self):
        """Verifies exact formula computation for short entry."""
        entry = 100.0
        stop = 104.0  # risk = 4.0
        target = 92.0  # reward = 8.0
        zero_costs = ExecutionCostModel(slippage_bps=0.0, commission_bps=0.0)

        res = compute_validated_reward_risk(
            direction=TradeDirection.SHORT,
            entry_price=entry,
            stop_loss=stop,
            target_price=target,
            execution_costs=zero_costs,
        )

        self.assertTrue(res.is_valid)
        self.assertEqual(res.status, PriceOrderingStatus.VALID)
        self.assertEqual(res.gross_reward, 8.0)
        self.assertEqual(res.gross_risk, 4.0)
        self.assertEqual(res.gross_reward_risk_ratio, 2.0)
        self.assertAlmostEqual(res.breakeven_win_rate_gross, 1.0 / 3.0, places=3)

    # -------------------------------------------------------------------------
    # 3. Price Ordering & Rejection of Invalid Geometry
    # -------------------------------------------------------------------------
    def test_reject_non_positive_prices(self):
        """Rejects zero or negative prices across all fields."""
        invalid_cases = [
            (0.0, 95.0, 105.0),
            (-100.0, 95.0, 105.0),
            (100.0, 0.0, 105.0),
            (100.0, -95.0, 105.0),
            (100.0, 95.0, 0.0),
            (100.0, 95.0, -105.0),
        ]
        for entry, stop, target in invalid_cases:
            is_valid, status, reason = validate_price_ordering(TradeDirection.LONG, entry, stop, target)
            self.assertFalse(is_valid)
            self.assertEqual(status, PriceOrderingStatus.NON_POSITIVE_PRICE)
            self.assertIn("strictly positive", reason)

    def test_reject_invalid_long_ordering(self):
        """Rejects inverted price ordering for long entries (target <= entry or entry <= stop)."""
        # Case A: target below entry (bearish target on long)
        is_valid, status, reason = validate_price_ordering(TradeDirection.LONG, entry=100.0, stop=95.0, target=98.0)
        self.assertFalse(is_valid)
        self.assertEqual(status, PriceOrderingStatus.ZERO_OR_NEGATIVE_REWARD)
        self.assertIn("target must be strictly above entry", reason)

        # Case B: stop above entry (stop placed in profit zone)
        is_valid, status, reason = validate_price_ordering(TradeDirection.LONG, entry=100.0, stop=105.0, target=110.0)
        self.assertFalse(is_valid)
        self.assertEqual(status, PriceOrderingStatus.ZERO_OR_NEGATIVE_RISK)
        self.assertIn("risk denominator (entry - stop) must be > 0", reason)

        # Case C: target equal to entry (zero reward)
        is_valid, status, reason = validate_price_ordering(TradeDirection.LONG, entry=100.0, stop=95.0, target=100.0)
        self.assertFalse(is_valid)
        self.assertEqual(status, PriceOrderingStatus.ZERO_OR_NEGATIVE_REWARD)

    def test_reject_invalid_short_ordering(self):
        """Rejects inverted price ordering for short entries (target >= entry or entry >= stop)."""
        # Case A: target above entry (bullish target on short)
        is_valid, status, reason = validate_price_ordering(TradeDirection.SHORT, entry=100.0, stop=105.0, target=102.0)
        self.assertFalse(is_valid)
        self.assertEqual(status, PriceOrderingStatus.ZERO_OR_NEGATIVE_REWARD)
        self.assertIn("target must be strictly below entry", reason)

        # Case B: stop below entry (stop placed in short profit zone)
        is_valid, status, reason = validate_price_ordering(TradeDirection.SHORT, entry=100.0, stop=95.0, target=90.0)
        self.assertFalse(is_valid)
        self.assertEqual(status, PriceOrderingStatus.ZERO_OR_NEGATIVE_RISK)
        self.assertIn("risk denominator (stop - entry) must be > 0", reason)

        # Case C: target equal to entry
        is_valid, status, reason = validate_price_ordering(TradeDirection.SHORT, entry=100.0, stop=105.0, target=100.0)
        self.assertFalse(is_valid)
        self.assertEqual(status, PriceOrderingStatus.ZERO_OR_NEGATIVE_REWARD)

    # -------------------------------------------------------------------------
    # 4. Zero or Negative Risk Denominators
    # -------------------------------------------------------------------------
    def test_reject_zero_risk_denominator(self):
        """Rejects setups where stop == entry (infinite / undefined ratio)."""
        # Long: entry == stop
        is_valid, status, reason = validate_price_ordering(TradeDirection.LONG, entry=100.0, stop=100.0, target=110.0)
        self.assertFalse(is_valid)
        self.assertEqual(status, PriceOrderingStatus.ZERO_OR_NEGATIVE_RISK)

        # Short: entry == stop
        is_valid, status, reason = validate_price_ordering(TradeDirection.SHORT, entry=100.0, stop=100.0, target=90.0)
        self.assertFalse(is_valid)
        self.assertEqual(status, PriceOrderingStatus.ZERO_OR_NEGATIVE_RISK)

    # -------------------------------------------------------------------------
    # 5. Transaction Costs & Execution Assumptions
    # -------------------------------------------------------------------------
    def test_execution_cost_accounting_reduces_rr(self):
        """
        Verifies that slippage, commission, and execution drag:
        1. Increase effective entry for long (paying ask + fees)
        2. Decrease effective target on take-profit (selling bid - fees)
        3. Decrease effective stop on loss (adverse liquidation slip + fees)
        4. Lower net reward, expand net risk, and strictly decrease net RR
        """
        entry = 100.0
        stop = 95.0   # gross risk = 5.0
        target = 106.0 # gross reward = 6.0 (gross RR = 1.20)

        res = compute_validated_reward_risk(
            direction=TradeDirection.LONG,
            entry_price=entry,
            stop_loss=stop,
            target_price=target,
            execution_costs=self.standard_costs,  # 5 bps slip + 1 bps comm per leg
        )

        self.assertTrue(res.is_valid)
        self.assertEqual(res.gross_reward_risk_ratio, 1.20)
        self.assertAlmostEqual(res.breakeven_win_rate_gross, 1.0 / 2.20, places=4)

        # Post-cost: effective entry is slightly higher (100 + 0.05 = 100.05)
        self.assertGreater(res.effective_entry, entry)
        # Effective target is slightly lower (106 - 0.053 = 105.95)
        self.assertLess(res.effective_target, target)
        # Effective stop is slightly lower (95 - 0.0475 = 94.95)
        self.assertLess(res.effective_stop, stop)

        # Net reward < Gross reward
        self.assertLess(res.net_reward, res.gross_reward)
        # Net risk > Gross risk
        self.assertGreater(res.net_risk, res.gross_risk)
        # Net RR < Gross RR
        self.assertLess(res.net_reward_risk_ratio, res.gross_reward_risk_ratio)
        # Net break-even win rate > Gross break-even win rate
        self.assertGreater(res.breakeven_win_rate_net, res.breakeven_win_rate_gross)
        # Cost drag is positive
        self.assertGreater(res.cost_drag_bps, 0.0)

    def test_short_borrow_fee_impact(self):
        """Verifies that short borrow fees expand downside risk and lower net short RR."""
        entry = 100.0
        stop = 103.0  # gross risk = 3.0
        target = 95.0  # gross reward = 5.0 (gross RR = 1.67)

        costs_with_borrow = ExecutionCostModel(
            slippage_bps=5.0,
            commission_bps=1.0,
            borrow_rate_annual_pct=3.0,  # 3% annual borrow cost
            holding_days_assumption=10.0,
        )

        res = compute_validated_reward_risk(
            direction=TradeDirection.SHORT,
            entry_price=entry,
            stop_loss=stop,
            target_price=target,
            execution_costs=costs_with_borrow,
        )

        self.assertTrue(res.is_valid)
        self.assertEqual(res.gross_reward_risk_ratio, 1.67)
        self.assertLess(res.net_reward_risk_ratio, 1.67)
        self.assertGreater(res.breakeven_win_rate_net, res.breakeven_win_rate_gross)

    # -------------------------------------------------------------------------
    # 6. Realized Trade Outcomes & Break-Even Win Rate Justification
    # -------------------------------------------------------------------------
    def test_breakeven_win_rate_for_1_20_ratio(self):
        """
        Verifies prompt assertion:
        Under fixed reward/loss sizes, a 1.20-to-1 ratio has a break-even win rate
        of about 45.5% before costs (1 / 2.20 = 0.4545).
        """
        be_gross = compute_breakeven_win_rate(1.20)
        self.assertAlmostEqual(be_gross, 0.4545, places=4)
        self.assertAlmostEqual(round(be_gross * 100.0, 1), 45.5, places=1)

    def test_trade_rejected_when_realized_outcomes_lack_statistical_edge(self):
        """
        A trade clearing 1.20 RR alone does NOT justify entering if estimated
        realized win probability does not clear net break-even rate with sufficient buffer.
        """
        entry = 100.0
        stop = 95.0
        target = 106.0  # Gross RR = 1.20 -> BE gross = 45.45% -> BE net ~46.5%

        # Case A: Low conviction composite score (50.0 = 50.0% win rate)
        # With BE net = 46.5% and min_edge_pct = 5.0%, required win rate is 51.5% -> edge fails
        res_no_edge = compute_validated_reward_risk(
            direction=TradeDirection.LONG,
            entry_price=entry,
            stop_loss=stop,
            target_price=target,
            composite_score=50.0,  # p_win = 50.0%
            min_edge_pct=0.05,     # Requires 5% probability buffer
        )
        self.assertTrue(res_no_edge.is_valid)
        self.assertTrue(res_no_edge.meets_minimum_rr)
        self.assertFalse(res_no_edge.has_statistical_edge)
        self.assertFalse(res_no_edge.trade_justified)
        self.assertIn("Insufficient statistical edge", res_no_edge.rejection_reason)

        # Case B: High conviction composite score (65.0 -> p_win = 55.25%)
        res_with_edge = compute_validated_reward_risk(
            direction=TradeDirection.LONG,
            entry_price=entry,
            stop_loss=stop,
            target_price=target,
            composite_score=65.0,
            min_edge_pct=0.05,
        )
        self.assertTrue(res_with_edge.is_valid)
        self.assertTrue(res_with_edge.meets_minimum_rr)
        self.assertTrue(res_with_edge.has_statistical_edge)
        self.assertTrue(res_with_edge.trade_justified)
        self.assertGreater(res_with_edge.expected_value_r, 0.0)

    # -------------------------------------------------------------------------
    # 7. Deterministic Functions & LLM Price Validation Guard
    # -------------------------------------------------------------------------
    def test_llm_cannot_bypass_ordering_or_set_unvalidated_risk(self):
        """
        Verifies that LLM candidate suggestions:
        1. Cannot bypass price ordering (e.g. LLM hallucinates stop above entry)
        2. Cannot set unvalidated excessive risk (e.g. stop distance > 30%)
        3. Are sanitized and snapped back to deterministic structural channel levels
        """
        price = 100.0
        atr = 2.0
        low_20d = 96.0
        high_20d = 108.0

        # LLM attempts to suggest inverted stop (stop = 105 > entry 100 on LONG)
        entry, stop, target, meta = derive_deterministic_channel_geometry(
            direction=TradeDirection.LONG,
            current_price=price,
            atr=atr,
            high_20d=high_20d,
            low_20d=low_20d,
            suggested_stop_loss=105.0,  # Inverted!
            suggested_target_price=110.0,
        )
        self.assertEqual(entry, 100.0)
        self.assertTrue(meta["candidate_override_attempted"])
        self.assertFalse(meta["candidate_accepted"])
        # Snapped to deterministic structural stop: min(96 - 1.0, 100 - 3.0) = 95.0
        self.assertEqual(stop, 95.0)
        self.assertTrue(any("Candidate prices rejected" in n for n in meta["sanitization_notes"]))

    def test_llm_excessive_risk_rejected(self):
        """LLM suggesting catastrophic 40% stop loss is rejected as excessive unvalidated risk."""
        price = 100.0
        atr = 2.0
        low_20d = 96.0
        high_20d = 108.0

        entry, stop, target, meta = derive_deterministic_channel_geometry(
            direction=TradeDirection.LONG,
            current_price=price,
            atr=atr,
            high_20d=high_20d,
            low_20d=low_20d,
            suggested_stop_loss=60.0,  # 40% loss!
            suggested_target_price=120.0,
            max_stop_distance_pct=0.25,
        )
        self.assertFalse(meta["candidate_accepted"])
        # Snapped to conservative structural stop
        self.assertEqual(stop, 95.0)

    def test_quantitative_service_backward_compatibility(self):
        """
        Verifies QuantitativeScoringService methods maintain exact backwards compatibility
        with legacy dictionary keys while exposing net metrics and audit fields.
        """
        res_long = QuantitativeScoringService.compute_reward_risk(100.0, 95.0, 110.0)
        self.assertIn("reward_risk_ratio", res_long)
        self.assertIn("breakeven_win_rate", res_long)
        self.assertIn("net_reward_risk_ratio", res_long)
        self.assertIn("effective_entry", res_long)
        self.assertIn("cost_drag_bps", res_long)
        self.assertEqual(res_long["reward_risk_ratio"], 2.0)

        res_short = QuantitativeScoringService.compute_short_reward_risk(100.0, 104.0, 92.0)
        self.assertIn("reward_risk_ratio", res_short)
        self.assertIn("net_reward_risk_ratio", res_short)
        self.assertEqual(res_short["reward_risk_ratio"], 2.0)

        # Invalid geometry returns None for reward_risk_ratio
        res_bad = QuantitativeScoringService.compute_reward_risk(100.0, 105.0, 95.0)
        self.assertIsNone(res_bad["reward_risk_ratio"])
        self.assertIsNone(res_bad["breakeven_win_rate"])
        self.assertFalse(res_bad["is_valid"])


if __name__ == "__main__":
    unittest.main()
