"""Tests for Risk-Based Position Sizing and Portfolio Risk Management (Step 7.1).

Validates:
1. Mathematical Position Sizing Formula:
   N_risk = floor( (E * r) / (|P - S| + C + G) )
2. Inviolable Invariant:
   A score of 90 does NOT get a larger risk budget than a score of 70
   unless explicitly validated by the formal investment mandate.
3. Constraint Hierarchy and Binding Caps:
   N_allowed = min(N_risk, N_position_cap, N_liquidity_cap, N_portfolio_cap)
   - Binding constraint correctly identified: RISK_BUDGET, POSITION_CAP, LIQUIDITY_CAP, PORTFOLIO_CAP, ZERO_ALLOCATION.
4. Friction and Gap Reserve:
   - Round-trip costs C per share (slippage + commissions on entry and exit).
   - Explicit gap risk reserve G in price units (ATR multiple or price fallback).
5. Safe Fail-Closed Behavior:
   - Non-positive equity, price, stop loss, or risk fraction.
   - Inverted stops (stop >= entry for long, stop <= entry for short).
   - Zero denominators and extreme float values (NaN, Inf).
6. Integration with StructuredDecision and TradingPolicy.
"""

import math
import unittest

from quant.policy_config import TradingPolicy, get_default_policy, load_policy_config
from quant.position_sizing import (
    BindingConstraint,
    InvestmentMandate,
    PositionSizingResult,
    calculate_risk_based_position_size,
    compute_gap_risk_reserve,
    compute_round_trip_cost_per_share,
)
from quant.structured_decision import StructuredDecisionEngine


class TestPositionSizingFormulas(unittest.TestCase):
    """Verifies core mathematical equations for N_risk, C, and G."""

    def setUp(self):
        self.mandate = InvestmentMandate(
            portfolio_equity=100_000.0,
            max_risk_fraction=0.010,  # 1.0% = $1,000 risk budget
            max_position_pct=0.20,   # 20% cap to allow risk budget to bind
            max_adv_pct=0.05,
            max_gross_exposure_pct=1.00,
            current_gross_exposure=0.0,
            slippage_bps=5.0,
            commission_bps=1.0,
            gap_risk_atr_multiple=0.50,
            default_gap_risk_pct=0.01,
        )

    def test_round_trip_cost_computation(self):
        """C = P * (slippage + commissions) * 2 / 10,000."""
        # P = 100, slippage = 5 bps, commission = 1 bps -> total 12 bps = 0.0012
        # C = 100 * 0.0012 = 0.12
        c = compute_round_trip_cost_per_share(100.0, slippage_bps=5.0, commission_bps=1.0)
        self.assertEqual(c, 0.12)

        # Non-positive price returns 0.0
        self.assertEqual(compute_round_trip_cost_per_share(0.0), 0.0)
        self.assertEqual(compute_round_trip_cost_per_share(-50.0), 0.0)

    def test_gap_risk_reserve_computation(self):
        """G = gap_risk_atr_multiple * ATR or entry_price * default_gap_pct."""
        # ATR provided: 0.50 * 2.0 = 1.00
        g = compute_gap_risk_reserve(100.0, atr=2.0, gap_risk_atr_multiple=0.50)
        self.assertEqual(g, 1.00)

        # ATR missing: 100.0 * 0.01 = 1.00
        g_fallback = compute_gap_risk_reserve(100.0, atr=None, default_gap_risk_pct=0.01)
        self.assertEqual(g_fallback, 1.00)

        # Invalid ATR (<= 0, NaN, Inf) falls back cleanly
        self.assertEqual(compute_gap_risk_reserve(100.0, atr=0.0, default_gap_risk_pct=0.015), 1.50)
        self.assertEqual(compute_gap_risk_reserve(100.0, atr=float("nan"), default_gap_risk_pct=0.02), 2.00)

    def test_n_risk_preliminary_size_long(self):
        """
        Calculates N_risk for a long position:
        E = 100,000, r = 0.01 -> Risk Budget = $1,000.
        P = 100.0, S = 95.0 -> stop_distance = 5.0.
        C = 0.12 (12 bps), G = 1.00 (0.5 * ATR 2.0).
        effective_per_share_risk = 5.0 + 0.12 + 1.0 = 6.12.
        N_risk = floor(1,000 / 6.12) = floor(163.398) = 163.
        """
        res = calculate_risk_based_position_size(
            entry_price=100.0,
            stop_loss=95.0,
            direction="LONG",
            atr=2.0,
            mandate=self.mandate,
        )
        self.assertEqual(res.entry_price, 100.0)
        self.assertEqual(res.stop_loss, 95.0)
        self.assertEqual(res.stop_distance, 5.0)
        self.assertEqual(res.cost_per_share, 0.12)
        self.assertEqual(res.gap_reserve_per_share, 1.0)
        self.assertEqual(res.effective_per_share_risk, 6.12)
        self.assertEqual(res.equity_at_risk_dollars, 1000.0)
        self.assertEqual(res.n_risk, 163)
        self.assertEqual(res.shares_allowed, 163)
        self.assertEqual(res.binding_constraint, BindingConstraint.RISK_BUDGET.value)
        self.assertTrue(res.is_tradeable)

    def test_n_risk_preliminary_size_short(self):
        """
        Calculates N_risk for a short position:
        P = 100.0, S = 105.0 -> stop_distance = 5.0.
        With identical C = 0.12, G = 1.0 -> effective risk = 6.12.
        N_risk = floor(1,000 / 6.12) = 163.
        """
        res = calculate_risk_based_position_size(
            entry_price=100.0,
            stop_loss=105.0,
            direction="SHORT",
            atr=2.0,
            mandate=self.mandate,
        )
        self.assertEqual(res.stop_distance, 5.0)
        self.assertEqual(res.effective_per_share_risk, 6.12)
        self.assertEqual(res.n_risk, 163)
        self.assertEqual(res.shares_allowed, 163)
        self.assertEqual(res.binding_constraint, BindingConstraint.RISK_BUDGET.value)


class TestPositionSizingInvariants(unittest.TestCase):
    """Verifies mandatory risk-based invariants from Step 7.1."""

    def test_invariant_score_budget_equality(self):
        """
        INVARIANT: Do not automatically give a score of 90 a larger risk budget
        than a score of 70 unless the sizing rule has been validated.
        """
        mandate = InvestmentMandate(portfolio_equity=100_000.0, max_risk_fraction=0.010)

        # Stock A has high score 90.0
        res_90 = calculate_risk_based_position_size(
            entry_price=120.0,
            stop_loss=114.0,
            atr=2.5,
            composite_score=90.0,
            mandate=mandate,
        )

        # Stock B has lower score 70.0
        res_70 = calculate_risk_based_position_size(
            entry_price=120.0,
            stop_loss=114.0,
            atr=2.5,
            composite_score=70.0,
            mandate=mandate,
        )

        # Stock C has minimum buy score 60.0
        res_60 = calculate_risk_based_position_size(
            entry_price=120.0,
            stop_loss=114.0,
            atr=2.5,
            composite_score=60.0,
            mandate=mandate,
        )

        # Both receive the exact identical equity at risk and share count
        self.assertEqual(res_90.equity_at_risk_dollars, res_70.equity_at_risk_dollars)
        self.assertEqual(res_70.equity_at_risk_dollars, res_60.equity_at_risk_dollars)
        self.assertEqual(res_90.n_risk, res_70.n_risk)
        self.assertEqual(res_70.n_risk, res_60.n_risk)
        self.assertEqual(res_90.shares_allowed, res_70.shares_allowed)
        self.assertEqual(res_70.shares_allowed, res_60.shares_allowed)

    def test_validated_score_budget_scaling_when_enabled(self):
        """When validated scaling is explicitly permitted in mandate, budget scales monotonically."""
        mandate_scaled = InvestmentMandate(
            portfolio_equity=100_000.0,
            max_risk_fraction=0.010,
            max_position_pct=0.25,  # Allows risk budget to bind
            allow_score_budget_scaling=True,  # Explicitly enabled
        )
        res_90 = calculate_risk_based_position_size(
            entry_price=100.0,
            stop_loss=95.0,
            atr=2.0,
            composite_score=90.0,
            mandate=mandate_scaled,
        )
        res_70 = calculate_risk_based_position_size(
            entry_price=100.0,
            stop_loss=95.0,
            atr=2.0,
            composite_score=70.0,
            mandate=mandate_scaled,
        )
        # Score 90 gets slightly higher conviction allocation than score 70
        self.assertGreater(res_90.equity_at_risk_dollars, res_70.equity_at_risk_dollars)
        self.assertGreater(res_90.shares_allowed, res_70.shares_allowed)


class TestPositionSizingCapsAndHierarchy(unittest.TestCase):
    """
    Verifies that the permissible share count is strictly:
    N_allowed = min(N_risk, N_position_cap, N_liquidity_cap, N_portfolio_cap)
    and correctly identifies which constraint is binding.
    """

    def setUp(self):
        self.mandate = InvestmentMandate(
            portfolio_equity=100_000.0,
            max_risk_fraction=0.010,  # $1,000 risk budget
            max_position_pct=0.10,   # Max $10,000 position
            max_adv_pct=0.01,        # Max 1% of ADV
            max_gross_exposure_pct=1.00,
            current_gross_exposure=0.0,
            slippage_bps=5.0,
            commission_bps=1.0,
        )

    def test_binding_constraint_position_cap(self):
        """Tight stop loss makes N_risk large, but Position Cap (10% of E) binds."""
        # Entry = 100, Stop = 99.50 -> stop distance = 0.50
        # C = 0.12, G = 0.10 -> effective risk = 0.72
        # N_risk = floor(1,000 / 0.72) = 1,388 shares
        # N_position_cap = floor(10,000 / 100) = 100 shares
        res = calculate_risk_based_position_size(
            entry_price=100.0,
            stop_loss=99.50,
            atr=0.20,
            adv_shares_20d=1_000_000.0,  # Liquidity cap = 10,000 shares
            mandate=self.mandate,
        )
        self.assertEqual(res.n_position_cap, 100)
        self.assertGreater(res.n_risk, 100)
        self.assertEqual(res.shares_allowed, 100)
        self.assertEqual(res.binding_constraint, BindingConstraint.POSITION_CAP.value)
        self.assertEqual(res.position_market_value, 10_000.0)
        self.assertEqual(res.position_equity_pct, 0.10)

    def test_binding_constraint_liquidity_cap(self):
        """Illiquid ticker (low ADV) limits size even when risk and position caps permit more."""
        # 20d ADV is 4,000 shares -> 1% ADV is 40 shares
        # Risk allows ~163 shares, Position cap allows 100 shares
        # Liquidity cap of 40 shares binds!
        res = calculate_risk_based_position_size(
            entry_price=100.0,
            stop_loss=95.0,
            atr=2.0,
            adv_shares_20d=4_000.0,
            mandate=self.mandate,
        )
        self.assertEqual(res.n_liquidity_cap, 40)
        self.assertEqual(res.n_position_cap, 100)
        self.assertEqual(res.n_risk, 163)
        self.assertEqual(res.shares_allowed, 40)
        self.assertEqual(res.binding_constraint, BindingConstraint.LIQUIDITY_CAP.value)

    def test_binding_constraint_portfolio_cap(self):
        """Exhausted portfolio gross capacity limits sizing."""
        # Mandate has $100,000 max gross, current exposure is $96,000
        # Remaining capacity = $4,000 -> at $100/share = 40 shares
        mandate_tight = InvestmentMandate(
            portfolio_equity=100_000.0,
            max_risk_fraction=0.010,
            max_position_pct=0.20,
            max_gross_exposure_pct=1.00,
            current_gross_exposure=96_000.0,  # Only $4k left
        )
        res = calculate_risk_based_position_size(
            entry_price=100.0,
            stop_loss=95.0,
            atr=2.0,
            mandate=mandate_tight,
        )
        self.assertEqual(res.n_portfolio_cap, 40)
        self.assertEqual(res.shares_allowed, 40)
        self.assertEqual(res.binding_constraint, BindingConstraint.PORTFOLIO_CAP.value)

    def test_binding_constraint_zero_allocation_when_portfolio_full(self):
        """When portfolio is 100% full (0 capacity), N_allowed = 0."""
        mandate_full = InvestmentMandate(
            portfolio_equity=100_000.0,
            max_gross_exposure_pct=1.00,
            current_gross_exposure=100_000.0,  # 100% utilized
        )
        res = calculate_risk_based_position_size(
            entry_price=100.0,
            stop_loss=95.0,
            atr=2.0,
            mandate=mandate_full,
        )
        self.assertEqual(res.shares_allowed, 0)
        self.assertEqual(res.binding_constraint, BindingConstraint.ZERO_ALLOCATION.value)
        self.assertFalse(res.is_tradeable)
        self.assertIn("0 shares", res.rejection_reason or "")


class TestPositionSizingFailSafeBoundaries(unittest.TestCase):
    """Verifies fail-closed execution on invalid, missing, or malformed inputs."""

    def setUp(self):
        self.mandate = InvestmentMandate()

    def test_fail_closed_on_zero_or_negative_equity(self):
        """E <= 0 returns 0 shares and INVALID_INPUTS."""
        res_zero = calculate_risk_based_position_size(entry_price=100.0, stop_loss=95.0, portfolio_equity=0.0)
        self.assertEqual(res_zero.shares_allowed, 0)
        self.assertEqual(res_zero.binding_constraint, BindingConstraint.INVALID_INPUTS.value)

        res_neg = calculate_risk_based_position_size(entry_price=100.0, stop_loss=95.0, portfolio_equity=-50_000.0)
        self.assertEqual(res_neg.shares_allowed, 0)

    def test_fail_closed_on_non_positive_price(self):
        """P <= 0 returns 0 shares."""
        res_zero = calculate_risk_based_position_size(entry_price=0.0, stop_loss=95.0)
        self.assertEqual(res_zero.shares_allowed, 0)
        self.assertEqual(res_zero.binding_constraint, BindingConstraint.INVALID_INPUTS.value)

        res_neg = calculate_risk_based_position_size(entry_price=-100.0, stop_loss=95.0)
        self.assertEqual(res_neg.shares_allowed, 0)

    def test_fail_closed_on_non_positive_stop(self):
        """S <= 0 returns 0 shares."""
        res = calculate_risk_based_position_size(entry_price=100.0, stop_loss=0.0)
        self.assertEqual(res.shares_allowed, 0)
        self.assertEqual(res.binding_constraint, BindingConstraint.INVALID_INPUTS.value)

    def test_fail_closed_on_stop_equals_entry(self):
        """P == S returns 0 shares (zero risk denominator)."""
        res = calculate_risk_based_position_size(entry_price=100.0, stop_loss=100.0)
        self.assertEqual(res.shares_allowed, 0)
        self.assertIn("cannot equal", res.rejection_reason or "")

    def test_fail_closed_on_inverted_long_stop(self):
        """Long stop S >= P is inverted geometry -> returns 0 shares."""
        res = calculate_risk_based_position_size(entry_price=100.0, stop_loss=105.0, direction="LONG")
        self.assertEqual(res.shares_allowed, 0)
        self.assertIn("must be strictly below", res.rejection_reason or "")

    def test_fail_closed_on_inverted_short_stop(self):
        """Short stop S <= P is inverted geometry -> returns 0 shares."""
        res = calculate_risk_based_position_size(entry_price=100.0, stop_loss=95.0, direction="SHORT")
        self.assertEqual(res.shares_allowed, 0)
        self.assertIn("must be strictly above", res.rejection_reason or "")

    def test_fail_closed_on_nan_and_inf(self):
        """Float NaN and Inf inputs fail closed safely without unhandled exceptions."""
        res_nan = calculate_risk_based_position_size(entry_price=float("nan"), stop_loss=95.0)
        self.assertEqual(res_nan.shares_allowed, 0)

        res_inf = calculate_risk_based_position_size(entry_price=100.0, stop_loss=float("inf"))
        self.assertEqual(res_inf.shares_allowed, 0)


class TestPositionSizingIntegration(unittest.TestCase):
    """Verifies integration with StructuredDecision and PolicyConfig."""

    def test_mandate_from_policy_config(self):
        """InvestmentMandate correctly loads institutional defaults from TradingPolicy."""
        policy = get_default_policy()
        mandate = InvestmentMandate.from_policy(policy)
        self.assertEqual(mandate.portfolio_equity, 100_000.0)
        self.assertEqual(mandate.max_risk_fraction, 0.010)
        self.assertEqual(mandate.max_position_pct, 0.10)
        self.assertEqual(mandate.max_adv_pct, 0.01)
        self.assertEqual(mandate.gap_risk_atr_multiple, 0.50)
        self.assertFalse(mandate.allow_score_budget_scaling)

    def test_yaml_config_loading_portfolio_risk(self):
        """trading_policy.yaml loads portfolio_risk section properly."""
        policy = load_policy_config()
        self.assertEqual(policy.portfolio_risk.default_portfolio_equity, 100_000.0)
        self.assertEqual(policy.portfolio_risk.max_risk_fraction_per_trade, 0.010)
        self.assertEqual(policy.portfolio_risk.max_position_equity_pct, 0.10)

    def test_certified_buy_includes_position_sizing_metrics(self):
        """A certified BUY decision automatically calculates and embeds risk-based sizing."""
        engine = StructuredDecisionEngine(get_default_policy())
        # Setup that passes all gates for BUY:
        # Score 75.0, entry 100, stop 92 (distance 8 > 0.85 * sqrt(20) * 1.0 = 3.80), target 120 (RR = 20 / 8 = 2.50)
        dec = engine.evaluate(
            ticker="NVDA",
            candidate_decision="BUY",
            composite_score=75.0,
            current_price=100.0,
            stop_loss=92.0,
            target_price=120.0,
            atr=1.0,
            days_to_earnings=45,
            rvol_20d=1.5,
            avg_dollar_volume_20d=50_000_000.0,
            pillar_scores={"trend": 75.0, "sector": 75.0, "alpha": 75.0, "valuation_history": 75.0, "peer_valuation": 75.0},
        )
        self.assertEqual(dec.decision, "BUY")
        self.assertEqual(dec.position_action, "OPEN_LONG")
        self.assertIn("position_sizing", dec.metrics)
        sizing = dec.metrics["position_sizing"]
        self.assertGreater(sizing["shares_allowed"], 0)
        self.assertIn(sizing["binding_constraint"], ["RISK_BUDGET", "POSITION_CAP", "LIQUIDITY_CAP", "PORTFOLIO_CAP"])
        self.assertEqual(sizing["ticker"], "NVDA")
        self.assertEqual(sizing["direction"], "LONG")
        self.assertEqual(sizing["entry_price"], 100.0)
        self.assertEqual(sizing["stop_loss"], 92.0)

    def test_pre_earnings_haircut_reduces_shares(self):
        """Verifies that pre-earnings mode applies a 50% haircut to position sizing."""
        policy = get_default_policy()
        policy.events.allow_pre_earnings_trading = True
        policy.events.earnings_risk_haircut = 0.50

        # Normal sizing (earnings 30 days away)
        res_normal = calculate_risk_based_position_size(
            entry_price=100.0,
            stop_loss=90.0,
            portfolio_equity=100_000.0,
            risk_fraction=0.01,
            days_to_earnings=30,
            policy=policy,
        )

        # Pre-earnings sizing (earnings 2 days away)
        res_event = calculate_risk_based_position_size(
            entry_price=100.0,
            stop_loss=90.0,
            portfolio_equity=100_000.0,
            risk_fraction=0.01,
            days_to_earnings=2,
            policy=policy,
        )

        self.assertTrue(res_event.metrics.get("pre_earnings_event_mode"))
        self.assertEqual(res_event.metrics.get("earnings_risk_haircut"), 0.50)
        self.assertAlmostEqual(res_event.shares_allowed, res_normal.shares_allowed // 2, delta=2)


if __name__ == "__main__":
    unittest.main()
