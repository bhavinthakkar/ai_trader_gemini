"""
Comprehensive Unit Test Suite for Step 5.3 Deterministic Compliance Gates.

Validates the explicit semantics of all five compliance gates:
1. BUY floor: Ordinary branch vs separate Dip-Buy branch; LLM override prohibition.
2. Reward-to-risk: Valid stop/target geometry, threshold clearance, cost accounting, statistical edge.
3. Volatility, events & liquidity: Options 1-sigma expected move, T=20 horizon, ATR fallback, stop outside noise,
   earnings blackout, universal liquidity controls (ADV participation limit, square-root market impact).
4. SELL and exit symmetry: Long exits vs short entries; holding-thesis guard forced exit invariants (stop breach,
   critical evidence veto, earnings blackout, catastrophic collapse).
5. Gated confidence: Exact mathematical formula, 5-pillar consensus dispersion, data completeness scaling,
   missing-data policy (< 3 pillars), and empirical 0.65 threshold.
"""

import math
import unittest
from quant.policy_config import TradingPolicy, get_default_policy
from quant.trade_geometry import TradeDirection, ExecutionCostModel
from quant.compliance_gates import (
    GateStatus,
    GateResult,
    ComplianceReport,
    BuyFloorGate,
    RewardToRiskGate,
    VolatilityEventLiquidityGate,
    SellExitSymmetryGate,
    GatedConfidenceGate,
    CompliancePolicyEngine,
)


class TestComplianceGatesSemantics(unittest.TestCase):
    """Exhaustive tests for Step 5.3 compliance gate semantics."""

    def setUp(self):
        self.policy = get_default_policy()
        self.engine = CompliancePolicyEngine(self.policy)

    # -------------------------------------------------------------------------
    # 1. Gate 1: BUY Floor Semantics
    # -------------------------------------------------------------------------
    def test_ordinary_buy_branch_semantics(self):
        """Standard BUY requires composite >= 60.0; composite < 60.0 is strictly rejected."""
        # Score 62.0 >= 60.0 clears
        res_pass = BuyFloorGate.evaluate(
            candidate_decision="BUY",
            composite_score=62.0,
            policy=self.policy,
        )
        self.assertTrue(res_pass.passed)
        self.assertEqual(res_pass.metrics["rule_name"], "RULE_STANDARD_BUY_FLOOR")

        # Score 58.0 < 60.0 fails
        res_fail = BuyFloorGate.evaluate(
            candidate_decision="BUY",
            composite_score=58.0,
            policy=self.policy,
        )
        self.assertFalse(res_fail.passed)
        self.assertEqual(res_fail.reason_code, "COMPOSITE_BELOW_FLOOR")
        self.assertIn("58.0/100 < required floor 60.0", res_fail.message)

    def test_dip_buy_separate_branch_semantics(self):
        """
        Certified dip-buy lowers floor to 52.0 when all dislocation conditions hold.
        Reverts immediately to 60.0 if any condition fails.
        """
        # Certified dip-buy: RSI=38 (in [30,45]), 5d drop=8% (in [5,15]), val_hist=65 (>= 60), no value trap
        res_dip_pass = BuyFloorGate.evaluate(
            candidate_decision="BUY",
            composite_score=54.0,  # 54.0 clears 52.0 dip floor
            catalyst_path="DIP_BUY",
            rsi14=38.0,
            drawdown_5d_pct=8.0,
            valuation_history_score=65.0,
            is_value_trap=False,
            policy=self.policy,
        )
        self.assertTrue(res_dip_pass.passed)
        self.assertEqual(res_dip_pass.metrics["required_floor"], 52.0)
        self.assertEqual(res_dip_pass.metrics["rule_name"], "RULE_DIP_BUY_DISCOUNT_FLOOR")

        # Disqualified dip-buy: RSI=25 (too low / structural collapse)
        # Floor reverts to 60.0 -> score 54.0 fails
        res_dip_bad_rsi = BuyFloorGate.evaluate(
            candidate_decision="BUY",
            composite_score=54.0,
            catalyst_path="DIP_BUY",
            rsi14=25.0,  # Fails RSI
            drawdown_5d_pct=8.0,
            valuation_history_score=65.0,
            is_value_trap=False,
            policy=self.policy,
        )
        self.assertFalse(res_dip_bad_rsi.passed)
        self.assertEqual(res_dip_bad_rsi.metrics["required_floor"], 60.0)
        self.assertEqual(res_dip_bad_rsi.metrics["rule_name"], "RULE_STANDARD_BUY_FLOOR")

        # Disqualified dip-buy: Confirmed value trap
        res_dip_trap = BuyFloorGate.evaluate(
            candidate_decision="BUY",
            composite_score=54.0,
            catalyst_path="DIP_BUY",
            rsi14=38.0,
            drawdown_5d_pct=8.0,
            valuation_history_score=65.0,
            is_value_trap=True,  # Value trap!
            policy=self.policy,
        )
        self.assertFalse(res_dip_trap.passed)
        self.assertEqual(res_dip_trap.metrics["required_floor"], 60.0)

    def test_llm_cannot_override_failed_score_requirement(self):
        """Inviolable Invariant: An LLM candidate BUY cannot override a failed score floor."""
        # Missing composite
        res_none = BuyFloorGate.evaluate(
            candidate_decision="BUY",
            composite_score=None,
            policy=self.policy,
        )
        self.assertFalse(res_none.passed)
        self.assertEqual(res_none.reason_code, "COMPOSITE_UNAVAILABLE")

        # Sub-floor composite (59.9 < 60.0)
        res_sub = BuyFloorGate.evaluate(
            candidate_decision="BUY",
            composite_score=59.9,
            policy=self.policy,
        )
        self.assertFalse(res_sub.passed)
        self.assertEqual(res_sub.reason_code, "COMPOSITE_BELOW_FLOOR")

    # -------------------------------------------------------------------------
    # 2. Gate 2: Reward-to-Risk Semantics
    # -------------------------------------------------------------------------
    def test_reward_to_risk_gate_validation(self):
        """Requires valid geometry, threshold clearance, and statistical edge."""
        # Valid setup: entry 100, stop 95, target 110 (gross RR = 2.0, clears 1.20)
        res_pass = RewardToRiskGate.evaluate(
            direction="LONG",
            entry_price=100.0,
            stop_loss=95.0,
            target_price=110.0,
            composite_score=65.0,
            policy=self.policy,
        )
        self.assertTrue(res_pass.passed)
        self.assertGreaterEqual(res_pass.metrics["gross_reward_risk_ratio"], 1.20)

        # Invalid price ordering: stop >= entry
        res_inverted = RewardToRiskGate.evaluate(
            direction="LONG",
            entry_price=100.0,
            stop_loss=105.0,
            target_price=110.0,
            composite_score=65.0,
            policy=self.policy,
        )
        self.assertFalse(res_inverted.passed)
        self.assertEqual(res_inverted.reason_code, "ZERO_OR_NEGATIVE_RISK")

        # Ratio too low: entry 100, stop 95, target 104 (RR = 4/5 = 0.80 < 1.20)
        res_low_rr = RewardToRiskGate.evaluate(
            direction="LONG",
            entry_price=100.0,
            stop_loss=95.0,
            target_price=104.0,
            composite_score=65.0,
            policy=self.policy,
        )
        self.assertFalse(res_low_rr.passed)
        self.assertEqual(res_low_rr.reason_code, "RR_TOO_LOW")

    # -------------------------------------------------------------------------
    # 3. Gate 3: Volatility, Events & Universal Liquidity Semantics
    # -------------------------------------------------------------------------
    def test_earnings_and_event_blackout_boundaries(self):
        """Earnings within 3 sessions or corporate event risk forces mandatory HOLD."""
        # Earnings in 2 sessions triggers blackout
        res_earn = VolatilityEventLiquidityGate.evaluate(
            direction="LONG",
            price=100.0,
            stop_loss=92.0,
            atr=2.5,
            days_to_earnings=2,
            policy=self.policy,
        )
        self.assertFalse(res_earn.passed)
        self.assertEqual(res_earn.reason_code, "EARNINGS_BLACKOUT")

        # Scheduled corporate event risk triggers blackout
        res_event = VolatilityEventLiquidityGate.evaluate(
            direction="LONG",
            price=100.0,
            stop_loss=92.0,
            atr=2.5,
            days_to_earnings=20,
            is_event_risk_blackout=True,
            policy=self.policy,
        )
        self.assertFalse(res_event.passed)
        self.assertEqual(res_event.reason_code, "EVENT_RISK")

    def test_options_expected_move_and_noise_stop_semantics(self):
        """
        Verifies 1-sigma options expected move calculation over T=20 horizon,
        ATR fallback when options are missing, and rejection of stops inside noise zone.
        """
        price = 100.0
        atr = 2.5
        iv = 35.0  # 35% IV

        # Expected move from IV: 100 * 0.35 * sqrt(20/252) = 35 * 0.2817 = $9.86
        move_iv, source_iv = VolatilityEventLiquidityGate.calculate_expected_move(
            price=price, iv_pct=iv, atr=atr, horizon_days=20
        )
        self.assertEqual(source_iv, "OPTIONS_ATM_IMPLIED_VOLATILITY")
        self.assertAlmostEqual(move_iv, 9.86, places=1)

        # Fallback to ATR when IV missing: ATR * sqrt(20) = 2.5 * 4.472 = $11.18
        move_fb, source_fb = VolatilityEventLiquidityGate.calculate_expected_move(
            price=price, iv_pct=None, atr=atr, horizon_days=20
        )
        self.assertEqual(source_fb, "REALIZED_ATR_FALLBACK")
        self.assertAlmostEqual(move_fb, 11.18, places=1)

        # Stop inside noise zone: Stop at 95 (stop distance $5.00 < 0.85 * 9.86 = $8.38)
        res_noise = VolatilityEventLiquidityGate.evaluate(
            direction="LONG",
            price=price,
            stop_loss=95.0,  # $5 distance
            atr=atr,
            iv_pct=iv,
            days_to_earnings=30,
            policy=self.policy,
        )
        self.assertFalse(res_noise.passed)
        self.assertEqual(res_noise.reason_code, "STOP_INSIDE_NOISE_ZONE")
        self.assertIn("inside the 1-sigma noise zone", res_noise.message)

        # Stop outside noise zone: Stop at 91 (stop distance $9.00 >= $8.38)
        res_clean = VolatilityEventLiquidityGate.evaluate(
            direction="LONG",
            price=price,
            stop_loss=91.0,  # $9 distance
            atr=atr,
            iv_pct=iv,
            days_to_earnings=30,
            avg_dollar_volume_20d=50_000_000.0,
            intended_order_value=25_000.0,
            policy=self.policy,
        )
        self.assertTrue(res_clean.passed)

    def test_universal_liquidity_and_market_impact_controls(self):
        """
        Universal liquidity controls: Intended order must not exceed 1.0% ADV
        and square-root market impact must not exceed 15 bps.
        """
        price = 100.0
        atr = 2.5
        adv = 2_000_000.0  # $2M ADV -> 1.0% cap is $20,000

        # Case A: Order value $50,000 > 1.0% ADV ($20,000) -> Rejected
        res_large_order = VolatilityEventLiquidityGate.evaluate(
            direction="LONG",
            price=price,
            stop_loss=90.0,
            atr=atr,
            days_to_earnings=30,
            avg_dollar_volume_20d=adv,
            intended_order_value=50_000.0,
            policy=self.policy,
        )
        self.assertFalse(res_large_order.passed)
        self.assertEqual(res_large_order.reason_code, "LIQUIDITY_INSUFFICIENT")
        self.assertIn("exceeds 1.0% ADV cap", res_large_order.message)

        # Case B: Order value $10,000 <= 1.0% ADV ($20,000) with low impact -> Clears
        res_ok_order = VolatilityEventLiquidityGate.evaluate(
            direction="LONG",
            price=price,
            stop_loss=90.0,
            atr=atr,
            days_to_earnings=30,
            avg_dollar_volume_20d=adv,
            intended_order_value=10_000.0,
            policy=self.policy,
        )
        self.assertTrue(res_ok_order.passed)
        self.assertLessEqual(res_ok_order.metrics["estimated_market_impact_bps"], 15.0)

    # -------------------------------------------------------------------------
    # 4. Gate 4: SELL and Exit Symmetry Semantics
    # -------------------------------------------------------------------------
    def test_holding_thesis_guard_interception(self):
        """Intact active holding (composite >= 70, price > stop) intercepts premature SELL -> HOLD."""
        res = SellExitSymmetryGate.evaluate(
            candidate_decision="SELL",
            is_holding=True,
            current_price=105.0,
            stop_loss=95.0,
            target_price=115.0,
            composite_score=75.0,  # Intact thesis >= 70
            has_critical_evidence_veto=False,
            policy=self.policy,
        )
        self.assertFalse(res.passed)
        self.assertEqual(res.reason_code, "HOLDING_THESIS_INTACT")
        self.assertIn("Holding-Thesis Guard", res.message)

    def test_holding_guard_never_blocks_forced_exits(self):
        """
        Invariants: The holding guard must NEVER block:
        1. Stop loss breach
        2. Critical evidence veto
        3. Earnings blackout
        4. Catastrophic composite collapse (< 50.0)
        """
        # Invariant 1: Stop breach (price 94 <= stop 95)
        res_stop = SellExitSymmetryGate.evaluate(
            candidate_decision="SELL",
            is_holding=True,
            current_price=94.0,  # Breached!
            stop_loss=95.0,
            target_price=115.0,
            composite_score=80.0,  # Even with high composite!
            policy=self.policy,
        )
        self.assertTrue(res_stop.passed)
        self.assertEqual(res_stop.reason_code, "STOP_LOSS_BREACH")

        # Invariant 2: Critical evidence veto
        res_veto = SellExitSymmetryGate.evaluate(
            candidate_decision="SELL",
            is_holding=True,
            current_price=105.0,
            stop_loss=95.0,
            target_price=115.0,
            composite_score=80.0,
            has_critical_evidence_veto=True,  # Critical fraud/investigation!
            policy=self.policy,
        )
        self.assertTrue(res_veto.passed)
        self.assertEqual(res_veto.reason_code, "EVIDENCE_VETO_FORCED_EXIT")

        # Invariant 3: Earnings blackout exit
        res_earn_exit = SellExitSymmetryGate.evaluate(
            candidate_decision="SELL",
            is_holding=True,
            current_price=105.0,
            stop_loss=95.0,
            target_price=115.0,
            composite_score=80.0,
            is_earnings_blackout=True,
            policy=self.policy,
        )
        self.assertTrue(res_earn_exit.passed)
        self.assertEqual(res_earn_exit.reason_code, "EARNINGS_BLACKOUT_EXIT")

        # Invariant 4: Catastrophic composite collapse (composite 45.0 < 50.0)
        res_collapse = SellExitSymmetryGate.evaluate(
            candidate_decision="SELL",
            is_holding=True,
            current_price=105.0,
            stop_loss=95.0,
            target_price=115.0,
            composite_score=45.0,  # Thesis collapsed!
            policy=self.policy,
        )
        self.assertTrue(res_collapse.passed)
        self.assertEqual(res_collapse.reason_code, "THESIS_COLLAPSED")

    def test_short_entry_ceiling_enforcement(self):
        """Directional short entry requires composite <= short_maximum (45.0)."""
        # Composite 35.0 <= 45.0 clears
        res_short_ok = SellExitSymmetryGate.evaluate(
            candidate_decision="SELL",
            is_holding=False,  # Fresh entry!
            current_price=100.0,
            stop_loss=104.0,
            target_price=92.0,
            composite_score=35.0,
            policy=self.policy,
        )
        self.assertTrue(res_short_ok.passed)

        # Composite 50.0 > 45.0 fails
        res_short_fail = SellExitSymmetryGate.evaluate(
            candidate_decision="SELL",
            is_holding=False,
            current_price=100.0,
            stop_loss=104.0,
            target_price=92.0,
            composite_score=50.0,
            policy=self.policy,
        )
        self.assertFalse(res_short_fail.passed)
        self.assertEqual(res_short_fail.reason_code, "SHORT_CEILING_EXCEEDED")

    # -------------------------------------------------------------------------
    # 5. Gate 5: Gated Confidence Semantics
    # -------------------------------------------------------------------------
    def test_gated_confidence_formula_and_dispersion(self):
        """
        Verifies exact formula computation:
        Distance = min(1.0, |S - 50| / 50)
        Dispersion & Agreement from 5 pillars
        Completeness scaling
        """
        # Perfect alignment: All pillars 80.0, composite 80.0, completeness 1.0
        # Distance = |80 - 50| / 50 = 0.60
        # Mean = 80, Dispersion = 0, Agreement = 1.0
        # Edge Strength = 0.70 * 0.60 + 0.30 * 0.50 = 0.42 + 0.15 = 0.57
        # Base = 0.50 + 0.50 * 0.57 = 0.785
        # Conf = 0.785 * 1.0 = 0.785 -> rounds to 0.78
        pillars = {
            "trend": 80.0,
            "sector": 80.0,
            "alpha": 80.0,
            "valuation_history": 80.0,
            "peer_valuation": 80.0,
        }
        conf, meta = GatedConfidenceGate.calculate_confidence(
            composite_score=80.0,
            pillar_scores=pillars,
            data_completeness=1.0,
            decision="BUY",
        )
        self.assertEqual(conf, 0.78)
        self.assertEqual(meta["valid_pillars_count"], 5)

        # Neutral HOLD is capped at 0.60
        conf_hold, _ = GatedConfidenceGate.calculate_confidence(
            composite_score=80.0,
            pillar_scores=pillars,
            data_completeness=1.0,
            decision="HOLD",
        )
        self.assertEqual(conf_hold, 0.60)

    def test_missing_data_policy_and_empirical_threshold(self):
        """
        Missing-data policy: < 3 available pillars rejects directional setup.
        Empirical threshold: Confidence must be >= 0.65 to authorize directional entry.
        """
        # Case A: Only 2 pillars available -> Rejected
        sparse_pillars = {"trend": 80.0, "sector": 80.0}
        res_sparse = GatedConfidenceGate.evaluate(
            candidate_decision="BUY",
            composite_score=80.0,
            pillar_scores=sparse_pillars,
            data_completeness=0.50,
            policy=self.policy,
        )
        self.assertFalse(res_sparse.passed)
        self.assertEqual(res_sparse.reason_code, "DATA_COVERAGE_INADEQUATE")

        # Case B: 5 pillars, but weak edge (composite 53.0 -> distance 0.06 -> confidence ~0.53 < 0.65)
        neutral_pillars = {
            "trend": 53.0,
            "sector": 53.0,
            "alpha": 53.0,
            "valuation_history": 53.0,
            "peer_valuation": 53.0,
        }
        res_weak = GatedConfidenceGate.evaluate(
            candidate_decision="BUY",
            composite_score=53.0,
            pillar_scores=neutral_pillars,
            data_completeness=1.0,
            policy=self.policy,
        )
        self.assertFalse(res_weak.passed)
        self.assertEqual(res_weak.reason_code, "CONFIDENCE_TOO_LOW")

    # -------------------------------------------------------------------------
    # 6. Unified Compliance Engine Orchestration
    # -------------------------------------------------------------------------
    def test_unified_compliance_engine_end_to_end(self):
        """End-to-end evaluation of all 5 gates in unified engine."""
        # Strong healthy setup passing all 5 gates
        report_pass = self.engine.evaluate_setup(
            candidate_decision="BUY",
            is_holding=False,
            current_price=100.0,
            stop_loss=90.0,  # Outside noise
            target_price=115.0,  # RR = 15/10 = 1.50 >= 1.20
            atr=2.5,
            composite_score=68.0,
            pillar_scores={
                "trend": 70.0,
                "sector": 68.0,
                "alpha": 68.0,
                "valuation_history": 65.0,
                "peer_valuation": 65.0,
            },
            data_completeness=1.0,
            days_to_earnings=45,
            avg_dollar_volume_20d=50_000_000.0,
            intended_order_value=25_000.0,
        )
        self.assertTrue(report_pass.all_passed)
        self.assertEqual(report_pass.final_decision, "BUY")
        self.assertIsNone(report_pass.no_trade_reason)

        # Setup failing Gate 1 (composite 55.0 < 60.0) without dip-buy catalyst
        report_fail_g1 = self.engine.evaluate_setup(
            candidate_decision="BUY",
            is_holding=False,
            current_price=100.0,
            stop_loss=90.0,
            target_price=115.0,
            atr=2.5,
            composite_score=55.0,  # Sub-floor for ordinary BUY!
            pillar_scores={
                "trend": 55.0,
                "sector": 55.0,
                "alpha": 55.0,
                "valuation_history": 55.0,
                "peer_valuation": 55.0,
            },
            data_completeness=1.0,
            days_to_earnings=45,
            avg_dollar_volume_20d=50_000_000.0,
        )
        self.assertFalse(report_fail_g1.all_passed)
        self.assertEqual(report_fail_g1.final_decision, "HOLD")
        self.assertEqual(report_fail_g1.no_trade_reason, "COMPOSITE_BELOW_FLOOR")
        self.assertEqual(report_fail_g1.veto_gate, "1. BUY Floor")

        # Certified Dip-Buy setup with composite 55.0 (< 60.0) passes all gates (no universal HOLD downgrade)
        report_pass_dip = self.engine.evaluate_setup(
            candidate_decision="BUY",
            is_holding=False,
            current_price=100.0,
            stop_loss=90.0,
            target_price=120.0,
            atr=2.5,
            composite_score=55.0,
            catalyst_path="DIP_BUY",
            rsi14=36.0,
            drawdown_5d_pct=7.5,
            valuation_history_score=65.0,
            is_value_trap=False,
            pillar_scores={
                "trend": 52.0,
                "sector": 52.0,
                "alpha": 52.0,
                "valuation_history": 65.0,
                "peer_valuation": 55.0,
            },
            data_completeness=1.0,
            days_to_earnings=45,
            avg_dollar_volume_20d=50_000_000.0,
        )
        self.assertTrue(report_pass_dip.all_passed)
        self.assertEqual(report_pass_dip.final_decision, "BUY")
        self.assertIsNone(report_pass_dip.no_trade_reason)


if __name__ == "__main__":
    unittest.main()
