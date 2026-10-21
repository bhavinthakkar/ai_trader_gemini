"""
Step 6.2: Automated Regression Tests for Mandatory Decision Scenarios.

Implements automated verification for the 10 mandatory decision scenarios and
an end-to-end integration test spanning data ingestion to SQLite persistence:

1. Score is 59.99, ordinary BUY threshold at 60 -> Do not BUY.
2. Score is 52.00, fails separately defined dip-buy criteria -> Do not BUY.
3. LLM proposes BUY but reward-to-risk fails -> Do not BUY.
4. Composite score is high but earnings blackout is active -> Block a new entry.
5. Existing long hits valid stop while score remains high -> Execute exit policy; thesis guard cannot override.
6. ML data is stale -> Use only approved fallback, otherwise block entry.
7. Ticker is outside ML universe -> Ensure ML weight is zero and quant-only fallback applies.
8. LLM returns invalid JSON or unsupported action -> Reject proposal; never issue order.
9. Short signal passes score threshold but borrow is unavailable -> Block the short entry.
10. Market data is stale or contradictory -> Block new entries and activate operational fail-safe.
11. Full End-to-End Integration: Data ingestion -> Decision JSON -> Database persistence.
"""

import json
import os
import sqlite3
import tempfile
import unittest
from typing import Any, Dict

from quant.policy_config import TradingPolicy, get_default_policy
from quant.trade_geometry import TradeDirection, ExecutionCostModel
from quant.compliance_gates import (
    GateStatus,
    BuyFloorGate,
    RewardToRiskGate,
    VolatilityEventLiquidityGate,
    SellExitSymmetryGate,
    GatedConfidenceGate,
    CompliancePolicyEngine,
)
from quant.decision_precedence import (
    InternalActionType,
    DecisionStage,
    PositionState,
    PrecedenceEvaluationResult,
    DecisionPrecedenceEngine,
    evaluate_decision_precedence,
)
from quant.composite_score import (
    blend_quant_and_ml_scores,
    compute_deterministic_composite,
)
from signal_schema import validate_signal_json
from main import normalize_master_trader_json
import db


class TestMandatoryDecisionScenarios(unittest.TestCase):
    """Automated regression suite for the 10 mandatory decision scenarios."""

    def setUp(self):
        self.policy = get_default_policy()
        self.engine = DecisionPrecedenceEngine(self.policy)

    # -------------------------------------------------------------------------
    # Scenario 1: Score is 59.99, ordinary BUY threshold at 60 -> Do not BUY
    # -------------------------------------------------------------------------
    def test_scenario_1_buy_score_5999_rejected(self):
        """Ordinary BUY with score 59.99 must be rejected and downgraded to HOLD."""
        # 1. Gate 1 Evaluation
        g1_res = BuyFloorGate.evaluate(
            candidate_decision="BUY",
            composite_score=59.99,
            policy=self.policy,
        )
        self.assertFalse(g1_res.passed)
        self.assertEqual(g1_res.status, GateStatus.FAIL)
        self.assertEqual(g1_res.reason_code, "COMPOSITE_BELOW_FLOOR")

        # 2. Decision Precedence Orchestration
        prec_res = self.engine.evaluate(
            candidate_decision="BUY",
            position_state=PositionState.FLAT,
            current_price=100.0,
            stop_loss=90.0,
            target_price=120.0,
            atr=2.0,
            composite_score=59.99,
            avg_dollar_volume_20d=10_000_000.0,
            rvol_20d=1.0,
        )
        self.assertEqual(prec_res.internal_action, InternalActionType.HOLD)
        self.assertEqual(prec_res.external_decision, "HOLD")
        self.assertEqual(prec_res.reason_code, "COMPOSITE_BELOW_FLOOR")

    def test_scenario_1_buy_score_6000_clears_floor(self):
        """Ordinary BUY with score 60.00 clears Gate 1 floor."""
        g1_res = BuyFloorGate.evaluate(
            candidate_decision="BUY",
            composite_score=60.00,
            policy=self.policy,
        )
        self.assertTrue(g1_res.passed)
        self.assertEqual(g1_res.status, GateStatus.PASS)

    # -------------------------------------------------------------------------
    # Scenario 2: Score is 52.00, fails dip-buy criteria -> Do not BUY
    # -------------------------------------------------------------------------
    def test_scenario_2_dip_buy_score_52_fails_criteria(self):
        """Score 52.00 is below standard 60 floor; fails dip-buy if any condition unmet."""
        # Case A: Catalyst path is standard momentum, not DIP_BUY
        g1_not_dip = BuyFloorGate.evaluate(
            candidate_decision="BUY",
            composite_score=52.00,
            catalyst_path="MOMENTUM_BREAKOUT",
            policy=self.policy,
        )
        self.assertFalse(g1_not_dip.passed)
        self.assertEqual(g1_not_dip.reason_code, "COMPOSITE_BELOW_FLOOR")

        # Case B: Catalyst is DIP_BUY, but RSI is 48.0 (> 45.0 upper limit)
        g1_high_rsi = BuyFloorGate.evaluate(
            candidate_decision="BUY",
            composite_score=52.00,
            catalyst_path="DIP_BUY",
            rsi14=48.0,
            drawdown_5d_pct=8.0,
            valuation_history_score=70.0,
            is_value_trap=False,
            policy=self.policy,
        )
        self.assertFalse(g1_high_rsi.passed)
        self.assertEqual(g1_high_rsi.reason_code, "COMPOSITE_BELOW_FLOOR")
        self.assertFalse(g1_high_rsi.metrics["dip_certified"])

        # Case C: Catalyst is DIP_BUY, but drawdown is only 2.0% (< 5.0% min pullback)
        g1_low_dd = BuyFloorGate.evaluate(
            candidate_decision="BUY",
            composite_score=52.00,
            catalyst_path="DIP_BUY",
            rsi14=35.0,
            drawdown_5d_pct=2.0,
            valuation_history_score=70.0,
            is_value_trap=False,
            policy=self.policy,
        )
        self.assertFalse(g1_low_dd.passed)
        self.assertEqual(g1_low_dd.reason_code, "COMPOSITE_BELOW_FLOOR")
        self.assertFalse(g1_low_dd.metrics["dip_certified"])

        # Case D: Catalyst is DIP_BUY, but certified as a Value Trap
        g1_value_trap = BuyFloorGate.evaluate(
            candidate_decision="BUY",
            composite_score=52.00,
            catalyst_path="DIP_BUY",
            rsi14=35.0,
            drawdown_5d_pct=8.0,
            valuation_history_score=70.0,
            is_value_trap=True,
            policy=self.policy,
        )
        self.assertFalse(g1_value_trap.passed)
        self.assertEqual(g1_value_trap.reason_code, "COMPOSITE_BELOW_FLOOR")
        self.assertFalse(g1_value_trap.metrics["dip_certified"])

        # Decision engine confirms rejection for failed dip-buy
        prec_res = self.engine.evaluate(
            candidate_decision="BUY",
            position_state=PositionState.FLAT,
            current_price=100.0,
            stop_loss=90.0,
            target_price=120.0,
            atr=2.0,
            composite_score=52.00,
            catalyst_path="DIP_BUY",
            rsi14=50.0,  # Fails RSI
            avg_dollar_volume_20d=10_000_000.0,
            rvol_20d=1.0,
        )
        self.assertEqual(prec_res.internal_action, InternalActionType.HOLD)
        self.assertEqual(prec_res.reason_code, "COMPOSITE_BELOW_FLOOR")

    def test_scenario_2_dip_buy_score_52_passes_when_all_criteria_met(self):
        """Score 52.00 passes Gate 1 when all dip-buy conditions are met and certifies BUY without HOLD conversion."""
        g1_pass = BuyFloorGate.evaluate(
            candidate_decision="BUY",
            composite_score=52.00,
            catalyst_path="DIP_BUY",
            rsi14=36.0,
            drawdown_5d_pct=7.5,
            valuation_history_score=65.0,
            is_value_trap=False,
            policy=self.policy,
        )
        self.assertTrue(g1_pass.passed)
        self.assertEqual(g1_pass.status, GateStatus.PASS)

        # Full engine verification: No subsequent universal rule converts score 52.00 to HOLD
        prec_res = self.engine.evaluate(
            candidate_decision="BUY",
            position_state=PositionState.FLAT,
            current_price=100.0,
            stop_loss=90.0,
            target_price=120.0,
            atr=2.0,
            composite_score=52.00,
            catalyst_path="DIP_BUY",
            rsi14=36.0,
            drawdown_5d_pct=7.5,
            valuation_history_score=65.0,
            is_value_trap=False,
            avg_dollar_volume_20d=10_000_000.0,
            rvol_20d=1.5,
            pillar_scores={"trend": 52.0, "sector": 52.0, "alpha": 52.0, "valuation_history": 65.0, "peer_valuation": 55.0},
            data_completeness=1.0,
        )
        self.assertEqual(prec_res.internal_action, InternalActionType.BUY)
        self.assertEqual(prec_res.external_decision, "BUY")
        self.assertEqual(prec_res.reason_code, "BUY_ORDER_CERTIFIED")

    # -------------------------------------------------------------------------
    # Scenario 3: LLM proposes BUY but reward-to-risk fails -> Do not BUY
    # -------------------------------------------------------------------------
    def test_scenario_3_llm_proposes_buy_rr_fails(self):
        """LLM candidate BUY with RR < 1.20 is rejected and downgraded to HOLD."""
        # Entry 100, Stop 90 (Risk 10), Target 110 (Reward 10) -> RR = 1.00 < 1.20
        g2_res = RewardToRiskGate.evaluate(
            direction=TradeDirection.LONG,
            entry_price=100.0,
            stop_loss=90.0,
            target_price=110.0,
            composite_score=75.0,  # High score cannot bypass RR gate
            policy=self.policy,
        )
        self.assertFalse(g2_res.passed)
        self.assertEqual(g2_res.reason_code, "RR_TOO_LOW")

        # Full precedence evaluation verifies downgrade to HOLD
        prec_res = self.engine.evaluate(
            candidate_decision="BUY",
            position_state=PositionState.FLAT,
            current_price=100.0,
            stop_loss=90.0,
            target_price=110.0,
            atr=2.0,
            composite_score=75.0,
            avg_dollar_volume_20d=10_000_000.0,
            rvol_20d=1.0,
        )
        self.assertEqual(prec_res.internal_action, InternalActionType.HOLD)
        self.assertEqual(prec_res.external_decision, "HOLD")
        self.assertEqual(prec_res.reason_code, "RR_TOO_LOW")

    def test_scenario_3_llm_proposes_buy_invalid_price_ordering(self):
        """LLM candidate BUY with Target <= Entry or Stop >= Entry is rejected."""
        prec_res = self.engine.evaluate(
            candidate_decision="BUY",
            position_state=PositionState.FLAT,
            current_price=100.0,
            stop_loss=90.0,
            target_price=95.0,  # Target below entry!
            atr=2.0,
            composite_score=80.0,
            avg_dollar_volume_20d=10_000_000.0,
            rvol_20d=1.0,
        )
        self.assertEqual(prec_res.internal_action, InternalActionType.HOLD)
        self.assertEqual(prec_res.reason_code, "ZERO_OR_NEGATIVE_REWARD")

    # -------------------------------------------------------------------------
    # Scenario 4: Composite score is high but earnings blackout is active -> Block entry
    # -------------------------------------------------------------------------
    def test_scenario_4_high_score_earnings_blackout_blocks_entry(self):
        """Composite 85.0 with earnings report in 2 sessions must be blocked."""
        # Gate 3 evaluation
        g3_res = VolatilityEventLiquidityGate.evaluate(
            direction=TradeDirection.LONG,
            price=100.0,
            stop_loss=90.0,
            atr=2.0,
            days_to_earnings=2,  # Blackout <= 3 sessions
            policy=self.policy,
        )
        self.assertFalse(g3_res.passed)
        self.assertEqual(g3_res.reason_code, "EARNINGS_BLACKOUT")

        # Decision Precedence evaluates Stage 3 before geometry or entries
        prec_res = self.engine.evaluate(
            candidate_decision="BUY",
            position_state=PositionState.FLAT,
            current_price=100.0,
            stop_loss=90.0,
            target_price=125.0,
            atr=2.0,
            composite_score=85.0,
            days_to_earnings=2,
        )
        self.assertEqual(prec_res.internal_action, InternalActionType.HOLD)
        self.assertEqual(prec_res.resolved_stage, DecisionStage.STAGE_3_EVENT_LIQUIDITY_COMPLIANCE_RESTRICTIONS)
        self.assertEqual(prec_res.reason_code, "EARNINGS_BLACKOUT")

    # -------------------------------------------------------------------------
    # Scenario 5: Existing long hits valid stop while score remains high -> Execute exit
    # -------------------------------------------------------------------------
    def test_scenario_5_existing_long_stop_loss_breach_bypasses_thesis_guard(self):
        """
        INVIOLABLE MANDATORY CONTROL:
        When an active position hits its stop loss, the exit MUST execute.
        A high composite score (e.g. 85.0) CANNOT intercept or override it via thesis guard.
        """
        # Long position: Entry 100, Stop 90, Current Price 88.0 (Stop Breached), Score 85.0
        prec_res = self.engine.evaluate(
            candidate_decision="SELL",
            position_state=PositionState.LONG,
            current_price=88.0,
            stop_loss=90.0,
            target_price=120.0,
            atr=2.0,
            composite_score=85.0,  # High score!
        )
        # Stage 2 mandatory risk controls must fire, producing SELL_TO_CLOSE
        self.assertEqual(prec_res.internal_action, InternalActionType.SELL_TO_CLOSE)
        self.assertEqual(prec_res.external_decision, "SELL")
        self.assertEqual(prec_res.external_sell_type, "LONG_EXIT")
        self.assertEqual(prec_res.resolved_stage, DecisionStage.STAGE_2_MANDATORY_EXISTING_POSITION_RISK)
        self.assertEqual(prec_res.reason_code, "STOP_LOSS_BREACH")

    # -------------------------------------------------------------------------
    # Scenario 6: ML data is stale -> Approved fallback or block entry
    # -------------------------------------------------------------------------
    def test_scenario_6_stale_ml_data_approved_fallback(self):
        """Stale ML data receives strictly 0.0 weight and falls back to pure quant."""
        stale_ctx = {
            "model_status": "STALE_SCORE",
            "in_scope": False,
            "ml_score": 85.0,
            "ml_weight": 0.20,
        }
        comp, w, s = blend_quant_and_ml_scores(65.0, stale_ctx)
        self.assertEqual(comp, 65.0)
        self.assertEqual(w, 0.0)
        self.assertIsNone(s)

    def test_scenario_6_stale_ml_data_blocks_entry_if_policy_requires(self):
        """When ml_data_fresh is False and policy fail-closed is active, entry is blocked in Stage 1."""
        prec_res = self.engine.evaluate(
            candidate_decision="BUY",
            position_state=PositionState.FLAT,
            current_price=100.0,
            stop_loss=90.0,
            target_price=120.0,
            atr=2.0,
            composite_score=70.0,
            ml_data_fresh=False,  # Stale!
        )
        self.assertEqual(prec_res.internal_action, InternalActionType.HOLD)
        self.assertEqual(prec_res.resolved_stage, DecisionStage.STAGE_1_DATA_AND_INSTRUMENT_ELIGIBILITY)
        self.assertEqual(prec_res.reason_code, "DATA_OUTAGE_NEW_ENTRIES_BLOCKED")

    # -------------------------------------------------------------------------
    # Scenario 7: Ticker is outside ML universe -> ML weight is 0.0, quant-only fallback
    # -------------------------------------------------------------------------
    def test_scenario_7_ticker_outside_ml_universe_quant_fallback(self):
        """Unmodeled/out-of-scope ticker receives 0.0 weight, 100% quant fallback."""
        out_of_scope_ctx = {
            "model_status": "OUT_OF_SCOPE",
            "in_scope": False,
            "ml_score": None,
            "ml_weight": 0.0,
        }
        comp, w, s = blend_quant_and_ml_scores(74.5, out_of_scope_ctx)
        self.assertEqual(comp, 74.5)
        self.assertEqual(w, 0.0)
        self.assertIsNone(s)

    # -------------------------------------------------------------------------
    # Scenario 8: LLM returns invalid JSON or unsupported action -> Reject proposal
    # -------------------------------------------------------------------------
    def test_scenario_8_llm_invalid_json_or_unsupported_action(self):
        """
        Invalid syntax, forbidden broker commands, or unsupported actions
        must be rejected by schema validation, never issuing an order.
        """
        # Case A: Non-dict payload
        _, errs_non_dict = validate_signal_json("not a json object")
        self.assertGreater(len(errs_non_dict), 0)

        # Case B: Prohibited broker directives
        broker_payload = {
            "decision": "BUY",
            "order_type": "LIMIT",
            "shares": 100,
            "execute_now": True,
        }
        _, errs_broker = validate_signal_json(broker_payload)
        self.assertGreater(len(errs_broker), 0)
        self.assertTrue(any("prohibited broker command" in e for e in errs_broker))

        # Case C: Unsupported decision action
        unsupported_action_payload = {
            "decision": "YOLO_CALLS",
            "confidence": 0.90,
        }
        _, errs_unsupported = validate_signal_json(unsupported_action_payload)
        self.assertGreater(len(errs_unsupported), 0)
        self.assertTrue(any("invalid decision" in e for e in errs_unsupported))

    # -------------------------------------------------------------------------
    # Scenario 9: Short signal passes score threshold but borrow is unavailable -> Block
    # -------------------------------------------------------------------------
    def test_scenario_9_short_passes_threshold_borrow_unavailable(self):
        """Bearish breakdown score (<=45.0) rejected for short entry if borrow unavailable."""
        # Gate 4 check
        g4_short = SellExitSymmetryGate.evaluate(
            candidate_decision="SELL",
            is_holding=False,
            current_price=100.0,
            stop_loss=110.0,
            target_price=80.0,
            composite_score=20.0,  # Strong breakdown
            borrow_available=False,  # Unborrowable!
            policy=self.policy,
        )
        self.assertFalse(g4_short.passed)
        self.assertEqual(g4_short.reason_code, "BORROW_UNAVAILABLE")

        # Decision Precedence orchestrates fallback to HOLD
        prec_res = self.engine.evaluate(
            candidate_decision="SELL",
            position_state=PositionState.FLAT,
            current_price=100.0,
            stop_loss=110.0,
            target_price=80.0,
            atr=2.0,
            composite_score=20.0,
            borrow_available=False,
            avg_dollar_volume_20d=10_000_000.0,
            rvol_20d=1.0,
        )
        self.assertEqual(prec_res.internal_action, InternalActionType.HOLD)
        self.assertEqual(prec_res.reason_code, "BORROW_UNAVAILABLE")

    def test_scenario_9_short_passes_threshold_borrow_available(self):
        """Bearish breakdown score with borrow available certifies short order."""
        prec_res = self.engine.evaluate(
            candidate_decision="SELL",
            position_state=PositionState.FLAT,
            current_price=100.0,
            stop_loss=108.0,
            target_price=80.0,
            atr=2.0,
            composite_score=20.0,
            pillar_scores={"trend": 20.0, "sector": 20.0, "alpha": 20.0, "valuation_history": 20.0, "peer_valuation": 20.0},
            borrow_available=True,
            avg_dollar_volume_20d=10_000_000.0,
            rvol_20d=1.0,
        )
        self.assertEqual(prec_res.internal_action, InternalActionType.OPEN_SHORT)
        self.assertEqual(prec_res.external_decision, "SELL")
        self.assertEqual(prec_res.external_sell_type, "SHORT")
        self.assertEqual(prec_res.reason_code, "SHORT_ORDER_CERTIFIED")

    # -------------------------------------------------------------------------
    # Scenario 10: Market data is stale or contradictory -> Fail-safe
    # -------------------------------------------------------------------------
    def test_scenario_10_market_data_stale_new_entries_blocked(self):
        """Stale market data activates Stage 1 fail-safe, blocking new entries."""
        prec_res = self.engine.evaluate(
            candidate_decision="BUY",
            position_state=PositionState.FLAT,
            current_price=100.0,
            stop_loss=90.0,
            target_price=120.0,
            atr=2.0,
            composite_score=75.0,
            market_data_fresh=False,  # Stale!
        )
        self.assertEqual(prec_res.internal_action, InternalActionType.HOLD)
        self.assertEqual(prec_res.resolved_stage, DecisionStage.STAGE_1_DATA_AND_INSTRUMENT_ELIGIBILITY)
        self.assertEqual(prec_res.reason_code, "DATA_OUTAGE_NEW_ENTRIES_BLOCKED")

    def test_scenario_10_outage_tolerance_exceeded_protective_liquidation(self):
        """Outage duration exceeding max tolerance triggers protective liquidation."""
        prec_res = self.engine.evaluate(
            candidate_decision="HOLD",
            position_state=PositionState.LONG,
            current_price=100.0,
            stop_loss=90.0,
            target_price=120.0,
            atr=2.0,
            composite_score=70.0,
            market_data_fresh=False,
            outage_duration_hours=3.0,  # Exceeds 2.0h tolerance
        )
        self.assertEqual(prec_res.internal_action, InternalActionType.SELL_TO_CLOSE)
        self.assertEqual(prec_res.reason_code, "DATA_OUTAGE_PROTECTIVE_LIQUIDATION")


# =============================================================================
# 11. END-TO-END PIPELINE INTEGRATION TEST
# =============================================================================
class TestEndToEndPipelineIntegration(unittest.TestCase):
    """
    Integration tests covering the entire execution path from raw data ingestion
    to deterministic normalization, decision JSON, and SQLite persistence.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_trader.db")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_end_to_end_data_to_decision_and_db_persistence(self):
        """
        Executes end-to-end:
        1. Ingest market & fundamental data
        2. Compute deterministic quantitative scoring & ML blending
        3. Evaluate compliance gates & decision precedence
        4. Normalize master trader JSON
        5. Persist to SQLite signals table and rag_evidence_records table
        6. Verify database persistence and schema fidelity
        """
        symbol = "MSFT"

        # 1. Raw Market Data Ingestion
        m_data = {
            "current_price": 400.0,
            "ema20": 395.0,
            "ema50": 390.0,
            "ema200": 370.0,
            "rsi14": 58.0,
            "atr": 6.0,
            "suggested_stop_loss": 392.0,       # Risk = $9 (from min(391, 392))
            "suggested_target_price": 435.0,    # Target capped at 415 -> Reward = $15 -> RR = 1.67
            "change_20d_pct": 5.0,
            "relative_alpha_20d": 3.0,
            "beta_60d": 1.1,
            "rvol_20d": 1.25,
            "avg_dollar_vol_20d": 50_000_000.0,
            "days_to_earnings": 45,             # Outside blackout
        }

        # 2. Fundamentals & Sector Data
        gloomberb_payload = {
            "financials": {
                "key_ratios": {
                    "forward_pe": 28.0,
                    "trailing_pe": 32.0,
                    "revenue_growth": 0.16,
                },
                "financial_quality": {
                    "fcf_margin": 32.0,
                    "operating_margin": 44.0,
                    "net_debt": -15000.0,  # Fortress net cash
                    "interest_coverage": 25.0,
                },
            },
            "peer_valuation": {
                "ev_to_ebitda": 20.0,
                "direct_peer_benchmarks": [
                    {"forward_pe": 35.0, "ev_to_ebitda": 24.0},
                    {"forward_pe": 30.0, "ev_to_ebitda": 22.0},
                ],
            },
        }
        sector_bench = {
            "sector_change_20d_pct": 2.0,
            "sector_etf": "XLK",
        }

        # 3. Deterministic Quantitative Score
        quant_res = compute_deterministic_composite(
            technical_data=m_data,
            gloomberb_payload=gloomberb_payload,
            sector_bench=sector_bench,
        )
        self.assertIsNotNone(quant_res.composite_quantitative_score)
        self.assertGreaterEqual(quant_res.composite_quantitative_score, 60.0)

        # Attach computed deterministic scores to m_data
        m_data["deterministic_5pillar_scores"] = quant_res.to_dict()
        m_data["deterministic_data_completeness"] = (
            quant_res.available_pillars_count / quant_res.total_pillars_count
        )

        # 4. In-Scope ML Score Blending (Experiment D: 30% Quant + 70% ML)
        ml_ctx = {
            "model_status": "IN_SCOPE",
            "in_scope": True,
            "ml_score": 85.0,
            "ml_weight": 0.70,
        }
        decision_comp, ml_w, ml_s = blend_quant_and_ml_scores(
            quant_res.composite_quantitative_score, ml_ctx
        )
        self.assertGreaterEqual(decision_comp, 60.0)
        m_data["ml_context"] = {
            "model_status": "IN_SCOPE",
            "in_scope": True,
            "score_0_100": 85.0,
            "ml_weight": 0.70,
            "predicted_20d_excess_return": 0.045,
            "rank": 1,
            "universe_size": 100,
        }

        # 5. Auditable RAG Evidence Records
        evidence_records = [
            {
                "document_id": "doc_sec_10q_01",
                "ticker": symbol,
                "source_type": "SEC_EDGAR_FILINGS",
                "source_url": "https://www.sec.gov/edgar/data/msft/10q.htm",
                "publication_timestamp": "2026-10-15T12:00:00Z",
                "available_timestamp": "2026-10-15T12:05:00Z",
                "retrieval_timestamp": "2026-10-18T10:00:00Z",
                "passage_or_excerpt": "Cloud revenue increased 29% YoY driven by enterprise Azure demand.",
                "event_type": "EARNINGS_SURPRISE",
                "event_horizon": "INTERMEDIATE_TERM",
            }
        ]

        # 6. LLM Advisory Output Structure (Compliant with Step 4.2 LLMEvidenceAnalysis contract)
        llm_raw_obj = {
            "ticker": symbol,
            "as_of": "2026-10-18T12:00:00Z",
            "analysis_status": "COMPLETE",
            "advisory_action": "CANDIDATE_BUY",
            "advisory_confidence": 0.72,
            "catalysts": [
                {
                    "catalyst_id": "cat_01",
                    "description": "Azure Cloud Growth Acceleration",
                    "category": "EARNINGS_SURPRISE",
                    "polarity": "BULLISH",
                    "event_horizon": "MEDIUM_TERM_20D",
                    "timestamp": "2026-10-15T12:00:00Z",
                    "supporting_document_ids": ["doc_sec_10q_01"],
                    "relevant_passage": "Cloud revenue increased 29% YoY driven by enterprise Azure demand.",
                    "is_priced_in": False,
                }
            ],
            "risk_flags": [],
            "supporting_document_ids": ["doc_sec_10q_01"],
            "contradictions": [],
            "missing_information": [],
            "summary": "Robust enterprise cloud demand and fortress balance sheet support long setup.",
            "forecast_20d": {
                "relative_outlook": "OUTPERFORM",
                "confidence": 0.70,
                "rationale": "High operating margins and accelerating cloud ARR.",
            },
        }

        # 7. Validate LLM Signal Schema
        valid_obj, val_errs = validate_signal_json(
            llm_raw_obj,
            expected_ticker=symbol,
            valid_document_ids={"doc_sec_10q_01"},
            require_ml_assessment=False,
            require_forecast_20d=True,
        )
        self.assertEqual(len(val_errs), 0, f"Validation errors: {val_errs}")

        # 8. Deterministic Normalization & Precedence Gate Execution
        macro_data = {"us_10y_yield": 4.15, "yield_curve_spread_10y2y": 0.22, "fear_greed_score": 55.0}
        normalized_signal = normalize_master_trader_json(
            valid_obj,
            symbol,
            m_data=m_data,
            macro_data=macro_data,
            gloomberb_payload=gloomberb_payload,
        )
        normalized_signal["evidence_records"] = evidence_records

        # Assert certified execution decision
        self.assertEqual(normalized_signal["decision"], "BUY")
        self.assertGreaterEqual(normalized_signal["quant_score"], 60.0)
        self.assertGreaterEqual(normalized_signal["reward_risk_ratio"], 1.20)

        # 9. Database Persistence
        db.save_results([normalized_signal], model_used="Gemini 3.8 Flash", db_path=self.db_path)

        # 10. Verify SQLite Persistence
        with db.get_connection(self.db_path) as conn:
            cursor = conn.cursor()

            # Verify signals row
            cursor.execute("SELECT * FROM signals WHERE symbol = ?;", (symbol,))
            row = cursor.fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["symbol"], symbol)
            self.assertEqual(row["decision"], "BUY")
            self.assertEqual(row["model_used"], "Gemini 3.8 Flash")
            self.assertAlmostEqual(row["entry_price"], 400.0)
            self.assertAlmostEqual(row["stop_loss_price"], 392.0)
            self.assertAlmostEqual(row["target_price"], 435.0)
            self.assertGreaterEqual(row["quant_score"], 60.0)

            signal_id = row["id"]

            # Verify linked rag_evidence_records
            cursor.execute("SELECT * FROM rag_evidence_records WHERE signal_id = ?;", (signal_id,))
            ev_rows = cursor.fetchall()
            self.assertEqual(len(ev_rows), 1)
            ev_row = ev_rows[0]
            self.assertEqual(ev_row["document_id"], "doc_sec_10q_01")
            self.assertEqual(ev_row["ticker"], symbol)
            self.assertEqual(ev_row["source_type"], "SEC_EDGAR_FILINGS")
            self.assertEqual(ev_row["event_type"], "EARNINGS_SURPRISE")


if __name__ == "__main__":
    unittest.main()
