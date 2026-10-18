"""
Unit Test Suite for Deterministic Decision Precedence Engine (Step 5.4).

Verifies the strict 6-stage decision hierarchy:
Stage 1: Validate data and instrument eligibility.
         - Fail-closed on data outage for new entries (no neutral score substitution).
         - Open-position fail-safe surveillance during outage (stop breach, evidence veto, outage timeout).
Stage 2: Apply mandatory existing-position risk controls.
         - Valid stop breach, forced liquidation, or critical evidence veto cannot be blocked by high composite score.
         - Binary earnings blackout and catastrophic thesis collapse forced liquidations.
Stage 3: Apply event, liquidity, and compliance restrictions for new entries.
         - Blackouts, expected move exceeding stop, or low liquidity block fresh entries.
Stage 4: Evaluate existing-position exits and reductions.
         - Target profit-taking exit.
         - Holding-thesis guard protects intact winning positions against premature discretionary exit.
Stage 5: Evaluate new long or short entries.
         - Fully certified long generates BUY.
         - Fully certified short generates OPEN_SHORT.
         - Gate failures degrade to HOLD.
Stage 6: Fallback to HOLD.

Also tests:
- Five distinct internal action types (BUY, HOLD, SELL_TO_CLOSE, OPEN_SHORT, COVER_SHORT).
- External UI mapping compatibility (decision, sell_type).
- PrecedenceEvaluationResult structure and stage audit logging.
"""

import unittest
from quant.policy_config import TradingPolicy, get_default_policy
from quant.decision_precedence import (
    InternalActionType,
    DecisionStage,
    PositionState,
    PrecedenceEvaluationResult,
    DecisionPrecedenceEngine,
    evaluate_decision_precedence,
)


class TestDecisionPrecedence(unittest.TestCase):
    """Exhaustive test suite for 6-stage decision precedence engine."""

    def setUp(self):
        self.policy = get_default_policy()
        self.engine = DecisionPrecedenceEngine(self.policy)

    # -------------------------------------------------------------------------
    # InternalActionType and UI Mapping
    # -------------------------------------------------------------------------
    def test_internal_action_types_and_ui_mapping(self):
        """Validates all 5 action types and their backward-compatible external UI representation."""
        # BUY
        dec, sell_type = InternalActionType.BUY.to_external_ui()
        self.assertEqual(dec, "BUY")
        self.assertIsNone(sell_type)

        # HOLD
        dec, sell_type = InternalActionType.HOLD.to_external_ui()
        self.assertEqual(dec, "HOLD")
        self.assertIsNone(sell_type)

        # SELL_TO_CLOSE
        dec, sell_type = InternalActionType.SELL_TO_CLOSE.to_external_ui()
        self.assertEqual(dec, "SELL")
        self.assertEqual(sell_type, "LONG_EXIT")

        # OPEN_SHORT
        dec, sell_type = InternalActionType.OPEN_SHORT.to_external_ui()
        self.assertEqual(dec, "SELL")
        self.assertEqual(sell_type, "SHORT")

        # COVER_SHORT
        dec, sell_type = InternalActionType.COVER_SHORT.to_external_ui()
        self.assertEqual(dec, "BUY")
        self.assertEqual(sell_type, "COVER_SHORT")

    # -------------------------------------------------------------------------
    # STAGE 1: Data and Instrument Eligibility
    # -------------------------------------------------------------------------
    def test_stage1_data_outage_blocks_new_entry(self):
        """Data outage disables new entries rather than substituting neutral scores (50.0)."""
        res = self.engine.evaluate(
            candidate_decision="BUY",
            position_state=PositionState.FLAT,
            current_price=100.0,
            stop_loss=90.0,
            target_price=120.0,
            atr=3.0,
            composite_score=75.0,
            market_data_fresh=False,  # Outage!
        )
        self.assertEqual(res.resolved_stage, DecisionStage.STAGE_1_DATA_AND_INSTRUMENT_ELIGIBILITY)
        self.assertEqual(res.internal_action, InternalActionType.HOLD)
        self.assertEqual(res.external_decision, "HOLD")
        self.assertEqual(res.reason_code, "DATA_OUTAGE_NEW_ENTRIES_BLOCKED")
        self.assertIn("Market or ML data is missing, stale, or experiencing an outage", res.message)

    def test_stage1_missing_price_blocks_new_entry(self):
        """Missing or non-positive price triggers Stage 1 block for new entries."""
        res = self.engine.evaluate(
            candidate_decision="BUY",
            position_state=PositionState.FLAT,
            current_price=0.0,  # Invalid price
            stop_loss=90.0,
            target_price=120.0,
            atr=3.0,
            composite_score=75.0,
            market_data_fresh=True,
        )
        self.assertEqual(res.resolved_stage, DecisionStage.STAGE_1_DATA_AND_INSTRUMENT_ELIGIBILITY)
        self.assertEqual(res.internal_action, InternalActionType.HOLD)
        self.assertEqual(res.reason_code, "DATA_OUTAGE_NEW_ENTRIES_BLOCKED")

    def test_stage1_open_position_failsafe_stop_breach_during_outage(self):
        """
        Open position during data outage does NOT ignore risk.
        If last known quote breaches stop, mandatory exit triggers in Stage 1 fail-safe.
        """
        # Long position with price <= stop
        res_long = self.engine.evaluate(
            candidate_decision="HOLD",
            position_state=PositionState.LONG,
            current_price=88.0,
            stop_loss=90.0,
            target_price=120.0,
            atr=3.0,
            composite_score=75.0,
            market_data_fresh=False,  # Outage!
        )
        self.assertEqual(res_long.resolved_stage, DecisionStage.STAGE_1_DATA_AND_INSTRUMENT_ELIGIBILITY)
        self.assertEqual(res_long.internal_action, InternalActionType.SELL_TO_CLOSE)
        self.assertEqual(res_long.external_decision, "SELL")
        self.assertEqual(res_long.external_sell_type, "LONG_EXIT")
        self.assertEqual(res_long.reason_code, "DATA_OUTAGE_FAILSAFE_STOP_BREACH")

        # Short position with price >= stop
        res_short = self.engine.evaluate(
            candidate_decision="HOLD",
            position_state=PositionState.SHORT,
            current_price=105.0,
            stop_loss=102.0,
            target_price=85.0,
            atr=3.0,
            composite_score=40.0,
            market_data_fresh=False,  # Outage!
        )
        self.assertEqual(res_short.resolved_stage, DecisionStage.STAGE_1_DATA_AND_INSTRUMENT_ELIGIBILITY)
        self.assertEqual(res_short.internal_action, InternalActionType.COVER_SHORT)
        self.assertEqual(res_short.external_decision, "BUY")
        self.assertEqual(res_short.external_sell_type, "COVER_SHORT")
        self.assertEqual(res_short.reason_code, "DATA_OUTAGE_FAILSAFE_SHORT_STOP_BREACH")

    def test_stage1_open_position_failsafe_evidence_veto_during_outage(self):
        """Critical qualitative evidence veto forces exit during data outage."""
        res = self.engine.evaluate(
            candidate_decision="HOLD",
            position_state=PositionState.LONG,
            current_price=100.0,
            stop_loss=90.0,
            target_price=120.0,
            atr=3.0,
            composite_score=75.0,
            market_data_fresh=False,
            has_critical_evidence_veto=True,
        )
        self.assertEqual(res.resolved_stage, DecisionStage.STAGE_1_DATA_AND_INSTRUMENT_ELIGIBILITY)
        self.assertEqual(res.internal_action, InternalActionType.SELL_TO_CLOSE)
        self.assertEqual(res.reason_code, "DATA_OUTAGE_FAILSAFE_EVIDENCE_VETO")

    def test_stage1_open_position_failsafe_outage_timeout(self):
        """Outage duration exceeding max_outage_tolerance_hours (2.0h) triggers protective liquidation."""
        res = self.engine.evaluate(
            candidate_decision="HOLD",
            position_state=PositionState.LONG,
            current_price=100.0,
            stop_loss=90.0,
            target_price=120.0,
            atr=3.0,
            composite_score=75.0,
            market_data_fresh=False,
            outage_duration_hours=2.5,  # Exceeds default 2.0 hours
        )
        self.assertEqual(res.resolved_stage, DecisionStage.STAGE_1_DATA_AND_INSTRUMENT_ELIGIBILITY)
        self.assertEqual(res.internal_action, InternalActionType.SELL_TO_CLOSE)
        self.assertEqual(res.reason_code, "DATA_OUTAGE_PROTECTIVE_LIQUIDATION")
        self.assertIn("exceeded maximum tolerance 2.0h", res.message)

    # -------------------------------------------------------------------------
    # STAGE 2: Mandatory Existing-Position Risk Controls
    # -------------------------------------------------------------------------
    def test_stage2_high_score_cannot_block_stop_breach(self):
        """
        Core Invariant: A high score (e.g. 85.0) can NEVER block a stop-loss breach.
        """
        # Long position: composite=85.0 (huge conviction!), price 89.0 <= stop 90.0
        res_long = self.engine.evaluate(
            candidate_decision="BUY",  # Model even wants to buy more
            position_state=PositionState.LONG,
            current_price=89.0,
            stop_loss=90.0,
            target_price=120.0,
            atr=3.0,
            composite_score=85.0,
        )
        self.assertEqual(res_long.resolved_stage, DecisionStage.STAGE_2_MANDATORY_EXISTING_POSITION_RISK)
        self.assertEqual(res_long.internal_action, InternalActionType.SELL_TO_CLOSE)
        self.assertEqual(res_long.reason_code, "STOP_LOSS_BREACH")

        # Short position: composite=25.0 (ultra bearish!), price 106.0 >= short stop 105.0
        res_short = self.engine.evaluate(
            candidate_decision="SELL",
            position_state=PositionState.SHORT,
            current_price=106.0,
            stop_loss=105.0,
            target_price=80.0,
            atr=3.0,
            composite_score=25.0,
        )
        self.assertEqual(res_short.resolved_stage, DecisionStage.STAGE_2_MANDATORY_EXISTING_POSITION_RISK)
        self.assertEqual(res_short.internal_action, InternalActionType.COVER_SHORT)
        self.assertEqual(res_short.reason_code, "SHORT_STOP_LOSS_BREACH")

    def test_stage2_high_score_cannot_block_critical_evidence_veto(self):
        """Critical qualitative evidence veto forces liquidation even with composite=88.0."""
        res = self.engine.evaluate(
            candidate_decision="HOLD",
            position_state=PositionState.LONG,
            current_price=100.0,
            stop_loss=90.0,
            target_price=120.0,
            atr=3.0,
            composite_score=88.0,
            has_critical_evidence_veto=True,
        )
        self.assertEqual(res.resolved_stage, DecisionStage.STAGE_2_MANDATORY_EXISTING_POSITION_RISK)
        self.assertEqual(res.internal_action, InternalActionType.SELL_TO_CLOSE)
        self.assertEqual(res.reason_code, "EVIDENCE_VETO_FORCED_EXIT")

    def test_stage2_earnings_blackout_forced_liquidation(self):
        """Binary earnings blackout triggers mandatory liquidation for open positions."""
        res = self.engine.evaluate(
            candidate_decision="HOLD",
            position_state=PositionState.LONG,
            current_price=100.0,
            stop_loss=90.0,
            target_price=120.0,
            atr=3.0,
            composite_score=75.0,
            days_to_earnings=2,  # <= 3 sessions
        )
        self.assertEqual(res.resolved_stage, DecisionStage.STAGE_2_MANDATORY_EXISTING_POSITION_RISK)
        self.assertEqual(res.internal_action, InternalActionType.SELL_TO_CLOSE)
        self.assertEqual(res.reason_code, "EARNINGS_BLACKOUT_FORCED_EXIT")

    def test_stage2_catastrophic_thesis_collapse(self):
        """Composite falling below catastrophic floor (< 50.0 for long, > 60.0 for short) forces exit."""
        res_long = self.engine.evaluate(
            candidate_decision="HOLD",
            position_state=PositionState.LONG,
            current_price=100.0,
            stop_loss=90.0,
            target_price=120.0,
            atr=3.0,
            composite_score=48.0,  # Below 50.0
        )
        self.assertEqual(res_long.resolved_stage, DecisionStage.STAGE_2_MANDATORY_EXISTING_POSITION_RISK)
        self.assertEqual(res_long.internal_action, InternalActionType.SELL_TO_CLOSE)
        self.assertEqual(res_long.reason_code, "THESIS_COLLAPSED")

        res_short = self.engine.evaluate(
            candidate_decision="HOLD",
            position_state=PositionState.SHORT,
            current_price=90.0,
            stop_loss=100.0,
            target_price=80.0,
            atr=3.0,
            composite_score=62.0,  # Recovered above 60.0
        )
        self.assertEqual(res_short.resolved_stage, DecisionStage.STAGE_2_MANDATORY_EXISTING_POSITION_RISK)
        self.assertEqual(res_short.internal_action, InternalActionType.COVER_SHORT)
        self.assertEqual(res_short.reason_code, "SHORT_THESIS_COLLAPSED")

    # -------------------------------------------------------------------------
    # STAGE 3: Event, Liquidity, and Compliance Restrictions
    # -------------------------------------------------------------------------
    def test_stage3_earnings_blackout_blocks_fresh_entry(self):
        """Earnings blackout blocks candidate BUY on FLAT position in Stage 3."""
        res = self.engine.evaluate(
            candidate_decision="BUY",
            position_state=PositionState.FLAT,
            current_price=100.0,
            stop_loss=92.0,
            target_price=115.0,
            atr=3.0,
            composite_score=75.0,
            days_to_earnings=2,
        )
        self.assertEqual(res.resolved_stage, DecisionStage.STAGE_3_EVENT_LIQUIDITY_COMPLIANCE_RESTRICTIONS)
        self.assertEqual(res.internal_action, InternalActionType.HOLD)
        self.assertEqual(res.reason_code, "EARNINGS_BLACKOUT")

    def test_stage3_expected_move_exceeds_stop_blocks_fresh_entry(self):
        """Options expected move exceeding stop distance blocks candidate BUY in Stage 3."""
        res = self.engine.evaluate(
            candidate_decision="BUY",
            position_state=PositionState.FLAT,
            current_price=100.0,
            stop_loss=98.0,  # Stop distance = $2.00
            target_price=115.0,
            atr=2.0,
            composite_score=75.0,
            iv_pct=50.0,  # 1-sigma move ≈ $100 * 0.50 * sqrt(20/252) ≈ $14.08 >> $2.00
        )
        self.assertEqual(res.resolved_stage, DecisionStage.STAGE_3_EVENT_LIQUIDITY_COMPLIANCE_RESTRICTIONS)
        self.assertEqual(res.internal_action, InternalActionType.HOLD)
        self.assertEqual(res.reason_code, "STOP_INSIDE_NOISE_ZONE")

    def test_stage3_insufficient_liquidity_blocks_fresh_entry(self):
        """Insufficient ADV or RVOL blocks candidate BUY in Stage 3."""
        res = self.engine.evaluate(
            candidate_decision="BUY",
            position_state=PositionState.FLAT,
            current_price=100.0,
            stop_loss=88.0,  # Stop distance = 12.0 >= 0.85 * (2.0 * sqrt(20)) = 7.60 (outside noise!)
            target_price=120.0,
            atr=2.0,
            composite_score=75.0,
            rvol_20d=0.3,  # < 0.5
            avg_dollar_volume_20d=500_000.0,  # < $1.0M
        )
        self.assertEqual(res.resolved_stage, DecisionStage.STAGE_3_EVENT_LIQUIDITY_COMPLIANCE_RESTRICTIONS)
        self.assertEqual(res.internal_action, InternalActionType.HOLD)
        self.assertEqual(res.reason_code, "LOW_LIQUIDITY")

    # -------------------------------------------------------------------------
    # STAGE 4: Existing-Position Exits and Reductions
    # -------------------------------------------------------------------------
    def test_stage4_target_price_achieved(self):
        """Reaching profit target triggers SELL_TO_CLOSE for long, COVER_SHORT for short."""
        # Long position reaches target
        res_long = self.engine.evaluate(
            candidate_decision="HOLD",
            position_state=PositionState.LONG,
            current_price=122.0,
            stop_loss=90.0,
            target_price=120.0,
            atr=3.0,
            composite_score=75.0,
        )
        self.assertEqual(res_long.resolved_stage, DecisionStage.STAGE_4_EXISTING_POSITION_EXITS_AND_REDUCTIONS)
        self.assertEqual(res_long.internal_action, InternalActionType.SELL_TO_CLOSE)
        self.assertEqual(res_long.reason_code, "TARGET_ACHIEVED")

        # Short position reaches target
        res_short = self.engine.evaluate(
            candidate_decision="HOLD",
            position_state=PositionState.SHORT,
            current_price=79.0,
            stop_loss=105.0,
            target_price=80.0,
            atr=3.0,
            composite_score=35.0,
        )
        self.assertEqual(res_short.resolved_stage, DecisionStage.STAGE_4_EXISTING_POSITION_EXITS_AND_REDUCTIONS)
        self.assertEqual(res_short.internal_action, InternalActionType.COVER_SHORT)
        self.assertEqual(res_short.reason_code, "SHORT_TARGET_ACHIEVED")

    def test_stage4_holding_thesis_guard_prevents_premature_exit(self):
        """Holding-thesis guard protects intact winning position (composite=75.0, stop not breached) from LLM SELL."""
        res = self.engine.evaluate(
            candidate_decision="SELL",
            position_state=PositionState.LONG,
            current_price=105.0,
            stop_loss=95.0,
            target_price=120.0,
            atr=3.0,
            composite_score=75.0,  # Thesis intact!
        )
        self.assertEqual(res.resolved_stage, DecisionStage.STAGE_4_EXISTING_POSITION_EXITS_AND_REDUCTIONS)
        self.assertEqual(res.internal_action, InternalActionType.HOLD)
        self.assertEqual(res.reason_code, "HOLDING_THESIS_INTACT")

    def test_stage4_holding_thesis_broken_allows_discretionary_exit(self):
        """If holding thesis is broken (e.g. composite=55.0 < 70.0), discretionary SELL is permitted."""
        res = self.engine.evaluate(
            candidate_decision="SELL",
            position_state=PositionState.LONG,
            current_price=102.0,
            stop_loss=95.0,
            target_price=120.0,
            atr=3.0,
            composite_score=55.0,  # Below holding minimum of 70.0
        )
        self.assertEqual(res.resolved_stage, DecisionStage.STAGE_4_EXISTING_POSITION_EXITS_AND_REDUCTIONS)
        self.assertEqual(res.internal_action, InternalActionType.SELL_TO_CLOSE)
        self.assertEqual(res.external_decision, "SELL")
        self.assertEqual(res.external_sell_type, "LONG_EXIT")

    # -------------------------------------------------------------------------
    # STAGE 5: Evaluate New Entries
    # -------------------------------------------------------------------------
    def test_stage5_certified_long_produces_buy(self):
        """A candidate BUY on FLAT position clearing all gates produces InternalActionType.BUY."""
        res = self.engine.evaluate(
            candidate_decision="BUY",
            position_state=PositionState.FLAT,
            current_price=100.0,
            stop_loss=90.0,  # Risk = 10
            target_price=120.0,  # Reward = 20 -> RR = 2.0 > 1.2
            atr=2.0,  # 0.85 * (2.0 * sqrt(20)) = 7.60 <= 10.0
            composite_score=75.0,  # >= 60.0
            pillar_scores={"trend": 75.0, "sector": 70.0, "alpha": 72.0, "valuation_history": 68.0, "peer_valuation": 74.0},
            data_completeness=1.0,
            rvol_20d=1.5,
            avg_dollar_volume_20d=50_000_000.0,
        )
        self.assertEqual(res.resolved_stage, DecisionStage.STAGE_5_NEW_ENTRIES)
        self.assertEqual(res.internal_action, InternalActionType.BUY)
        self.assertEqual(res.external_decision, "BUY")
        self.assertIsNone(res.external_sell_type)
        self.assertEqual(res.reason_code, "BUY_ORDER_CERTIFIED")

    def test_stage5_certified_short_produces_open_short(self):
        """A candidate SELL on FLAT position clearing all short gates produces InternalActionType.OPEN_SHORT."""
        res = self.engine.evaluate(
            candidate_decision="SELL",
            position_state=PositionState.FLAT,
            current_price=100.0,
            stop_loss=110.0,  # Short risk = 10 (stop > entry)
            target_price=80.0,  # Short reward = 20 (target < entry) -> RR = 2.0 > 1.2
            atr=2.0,
            composite_score=35.0,  # <= 45.0
            pillar_scores={"trend": 30.0, "sector": 35.0, "alpha": 32.0, "valuation_history": 40.0, "peer_valuation": 38.0},
            data_completeness=1.0,
            rvol_20d=1.5,
            avg_dollar_volume_20d=50_000_000.0,
        )
        self.assertEqual(res.resolved_stage, DecisionStage.STAGE_5_NEW_ENTRIES)
        self.assertEqual(res.internal_action, InternalActionType.OPEN_SHORT)
        self.assertEqual(res.external_decision, "SELL")
        self.assertEqual(res.external_sell_type, "SHORT")
        self.assertEqual(res.reason_code, "SHORT_ORDER_CERTIFIED")

    def test_stage5_gate_failure_degrades_to_hold(self):
        """Low reward-to-risk in Stage 5 degrades candidate BUY to HOLD."""
        res = self.engine.evaluate(
            candidate_decision="BUY",
            position_state=PositionState.FLAT,
            current_price=100.0,
            stop_loss=90.0,  # Risk = 10
            target_price=108.0,  # Reward = 8 -> Gross RR = 0.8 < 1.2
            atr=2.0,
            composite_score=75.0,
            pillar_scores={"trend": 75.0, "sector": 70.0, "alpha": 72.0, "valuation_history": 68.0, "peer_valuation": 74.0},
            data_completeness=1.0,
            rvol_20d=1.5,
            avg_dollar_volume_20d=50_000_000.0,
        )
        self.assertEqual(res.resolved_stage, DecisionStage.STAGE_5_NEW_ENTRIES)
        self.assertEqual(res.internal_action, InternalActionType.HOLD)
        self.assertEqual(res.external_decision, "HOLD")
        self.assertEqual(res.reason_code, "RR_TOO_LOW")

    # -------------------------------------------------------------------------
    # STAGE 6: Fallback
    # -------------------------------------------------------------------------
    def test_stage6_no_trade_fallback(self):
        """Candidate decision HOLD on FLAT position defaults to Stage 6 NO_TRADE fallback."""
        res = self.engine.evaluate(
            candidate_decision="HOLD",
            position_state=PositionState.FLAT,
            current_price=100.0,
            stop_loss=90.0,
            target_price=120.0,
            atr=3.0,
            composite_score=55.0,
        )
        self.assertEqual(res.resolved_stage, DecisionStage.STAGE_6_NO_TRADE_FALLBACK)
        self.assertEqual(res.internal_action, InternalActionType.HOLD)
        self.assertEqual(res.reason_code, "NO_TRADE_DEFAULT")

    # -------------------------------------------------------------------------
    # Convenience Wrapper and Audit Logging
    # -------------------------------------------------------------------------
    def test_convenience_wrapper_and_audit_logging(self):
        """Validates evaluate_decision_precedence and presence of stage audit logs."""
        res = evaluate_decision_precedence(
            candidate_decision="BUY",
            position_state="FLAT",
            current_price=100.0,
            stop_loss=90.0,
            target_price=120.0,
            atr=2.0,
            composite_score=75.0,
            pillar_scores={"trend": 75.0, "sector": 70.0, "alpha": 72.0, "valuation_history": 68.0, "peer_valuation": 74.0},
            data_completeness=1.0,
            rvol_20d=1.5,
            avg_dollar_volume_20d=50_000_000.0,
        )
        self.assertIsInstance(res, PrecedenceEvaluationResult)
        self.assertEqual(res.internal_action, InternalActionType.BUY)
        self.assertTrue(len(res.stage_audit_log) >= 2)
        dict_rep = res.to_dict()
        self.assertEqual(dict_rep["internal_action"], "BUY")
        self.assertEqual(dict_rep["resolved_stage"], "STAGE_5_NEW_ENTRIES")


if __name__ == "__main__":
    unittest.main()
