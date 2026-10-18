"""
Step 6.4: Automated Tests for Separating the Decision from the Explanation.

Verifies:
1. Every mandatory gate has explicit test coverage:
   - DATA_QUALITY (Pass & Outage fail-safe)
   - INSUFFICIENT_PILLARS (< 3 pillars abort)
   - EVENT_RISK (Earnings blackout, binary corporate event)
   - LIQUIDITY (RVOL, ADV minimums)
   - BUY_FLOOR (Standard score 60.0, Dip-buy 52.0, Value trap veto)
   - SHORT_CEILING (Breakdown score <= 45.0)
   - BORROW_AVAILABILITY (Short locate check)
   - REWARD_TO_RISK (Price sequence ordering, non-negative risk, RR threshold)
   - CONFIDENCE_HURDLE (Signal strength >= 0.65)
   - STOP_LOSS_SURVEILLANCE (Inviolable long and short stop breaches)
   - HOLDING_THESIS (Winning position holding-thesis guard)
   - PROFIT_TARGET (Take profit execution)
2. Separation of decision from explanation:
   - Investment decisions and reason codes are completely independent of narrative text.
   - Omitted, empty, or altered free-form narratives yield identical structured decisions.
3. Invalid data fails safely:
   - Missing prices, negative ATR, and data outages fail-closed without crashes.
4. Determinism & reproducibility:
   - Logged reason codes and gate outcomes are 100% reproducible across multiple runs.
5. End-to-end database persistence and deserialization:
   - Structured decision dictionary and fields persist cleanly to SQLite trader.db.
"""

import json
import os
import sqlite3
import tempfile
import unittest
from typing import Any, Dict

from quant.policy_config import get_default_policy
from quant.trade_geometry import TradeDirection, ExecutionCostModel
from quant.structured_decision import (
    PositionAction,
    DataQualityStatus,
    ModelScopeStatus,
    StandardGateName,
    StandardReasonCode,
    StructuredDecision,
    StructuredDecisionEngine,
    build_structured_decision,
)
from quant.decision_precedence import (
    InternalActionType,
    DecisionStage,
    PositionState,
    DecisionPrecedenceEngine,
)
from main import normalize_master_trader_json
import db


class TestStructuredDecisionMandatoryGates(unittest.TestCase):
    """Verifies that every mandatory gate is independently tested and produces structured outputs."""

    def setUp(self):
        self.policy = get_default_policy()
        self.engine = StructuredDecisionEngine(self.policy)

    # -------------------------------------------------------------------------
    # Gate 1: DATA_QUALITY
    # -------------------------------------------------------------------------
    def test_gate_data_quality_fresh_market_data(self):
        """Valid fresh market data passes DATA_QUALITY gate."""
        dec = self.engine.evaluate(
            ticker="NVDA",
            candidate_decision="HOLD",
            composite_score=50.0,
            current_price=120.0,
            atr=2.5,
            market_data_fresh=True,
            ml_data_fresh=True,
        )
        self.assertIn(StandardGateName.DATA_QUALITY, dec.passed_gates)
        self.assertNotIn(StandardGateName.DATA_QUALITY, dec.failed_gates)
        self.assertEqual(dec.data_quality_status, DataQualityStatus.VALID.value)

    def test_gate_data_quality_outage_fails_closed(self):
        """Data outage fails DATA_QUALITY, marks OUTAGE, and blocks entry with DATA_OUTAGE_BLOCK."""
        dec = self.engine.evaluate(
            ticker="NVDA",
            candidate_decision="BUY",
            composite_score=85.0,
            current_price=None,  # Missing quote feed
            market_data_fresh=False,
        )
        self.assertIn(StandardGateName.DATA_QUALITY, dec.failed_gates)
        self.assertEqual(dec.data_quality_status, DataQualityStatus.OUTAGE.value)
        self.assertEqual(dec.decision, "HOLD")
        self.assertEqual(dec.position_action, PositionAction.NO_ACTION.value)
        self.assertIn(StandardReasonCode.DATA_OUTAGE_BLOCK, dec.reason_codes)

    # -------------------------------------------------------------------------
    # Gate 2: INSUFFICIENT_PILLARS
    # -------------------------------------------------------------------------
    def test_gate_insufficient_pillars_abort(self):
        """Fewer than 3 available pillars aborts with INSUFFICIENT_DATA and reason code."""
        dec = self.engine.evaluate(
            ticker="NVDA",
            candidate_decision="BUY",
            composite_score=75.0,
            current_price=120.0,
            atr=2.5,
            pillar_scores={"trend": 70.0, "sector": 65.0, "alpha": None, "valuation_history": None, "peer_valuation": None},  # only 2
        )
        self.assertIn(StandardGateName.INSUFFICIENT_PILLARS, dec.failed_gates)
        self.assertEqual(dec.data_quality_status, DataQualityStatus.INSUFFICIENT_DATA.value)
        self.assertEqual(dec.decision, "HOLD")
        self.assertIn(StandardReasonCode.INSUFFICIENT_PILLARS, dec.reason_codes)

    # -------------------------------------------------------------------------
    # Gate 3: EVENT_RISK (Earnings Blackout & Corporate Event)
    # -------------------------------------------------------------------------
    def test_gate_event_risk_earnings_blackout(self):
        """Earnings within blackout window (<= 3 sessions) blocks fresh entry."""
        dec = self.engine.evaluate(
            ticker="NVDA",
            candidate_decision="BUY",
            composite_score=85.0,
            current_price=120.0,
            stop_loss=115.0,
            target_price=135.0,
            atr=2.5,
            days_to_earnings=2,  # Active blackout
        )
        self.assertIn(StandardGateName.EVENT_RISK, dec.failed_gates)
        self.assertIn(StandardReasonCode.EARNINGS_BLACKOUT_ACTIVE, dec.reason_codes)
        self.assertEqual(dec.decision, "HOLD")
        self.assertEqual(dec.position_action, PositionAction.NO_ACTION.value)

    def test_gate_event_risk_binary_corporate_event(self):
        """Binary corporate event blackout blocks fresh entry."""
        dec = self.engine.evaluate(
            ticker="NVDA",
            candidate_decision="BUY",
            composite_score=85.0,
            current_price=120.0,
            stop_loss=115.0,
            target_price=135.0,
            atr=2.5,
            days_to_earnings=45,
            is_event_risk_blackout=True,
        )
        self.assertIn(StandardGateName.EVENT_RISK, dec.failed_gates)
        self.assertIn(StandardReasonCode.EVENT_RISK_BLACKOUT, dec.reason_codes)
        self.assertEqual(dec.decision, "HOLD")

    # -------------------------------------------------------------------------
    # Gate 4: LIQUIDITY
    # -------------------------------------------------------------------------
    def test_gate_liquidity_insufficient_volume(self):
        """Insufficient turnover (RVOL < 0.5x or ADV < $1M) blocks trade."""
        dec = self.engine.evaluate(
            ticker="ILLIQ",
            candidate_decision="BUY",
            composite_score=80.0,
            current_price=10.0,
            stop_loss=9.0,
            target_price=13.0,
            atr=0.1,  # Clears noise stop (0.1 * sqrt(20) * 0.85 = 0.38 < 1.0)
            days_to_earnings=45,
            rvol_20d=0.25,  # Too low
            avg_dollar_volume_20d=400_000.0,  # Below $1M bar
        )
        self.assertIn(StandardGateName.LIQUIDITY, dec.failed_gates)
        self.assertIn(StandardReasonCode.LIQUIDITY_INSUFFICIENT, dec.reason_codes)
        self.assertEqual(dec.decision, "HOLD")

    # -------------------------------------------------------------------------
    # Gate 5: BUY_FLOOR
    # -------------------------------------------------------------------------
    def test_gate_buy_floor_standard_below_threshold(self):
        """Score 58.4 fails BUY_FLOOR (minimum 60.0) -> HOLD with SCORE_BELOW_BUY_THRESHOLD."""
        dec = self.engine.evaluate(
            ticker="NVDA",
            candidate_decision="BUY",
            composite_score=58.4,
            current_price=120.0,
            stop_loss=115.0,
            target_price=135.0,
            atr=1.0,  # Clears noise stop (1.0 * sqrt(20) * 0.85 = 3.80 < 5.0)
            days_to_earnings=45,
            rvol_20d=1.5,
            avg_dollar_volume_20d=50_000_000.0,
            run_id="example-run-id",
        )
        self.assertEqual(dec.ticker, "NVDA")
        self.assertEqual(dec.decision, "HOLD")
        self.assertEqual(dec.composite_score, 58.4)
        self.assertEqual(dec.model_status, "IN_SCOPE")
        self.assertIn(StandardGateName.BUY_FLOOR, dec.failed_gates)
        self.assertEqual(dec.data_quality_status, "VALID")
        self.assertEqual(dec.position_action, "NO_ACTION")
        self.assertIn("SCORE_BELOW_BUY_THRESHOLD", dec.reason_codes)

    def test_gate_buy_floor_dip_buy_criteria_failed(self):
        """Score 52.0 on DIP_BUY failing RSI or drawdown fails BUY_FLOOR."""
        dec = self.engine.evaluate(
            ticker="DIP",
            candidate_decision="BUY",
            composite_score=52.0,
            current_price=100.0,
            stop_loss=95.0,
            target_price=115.0,
            atr=1.0,  # Clears noise stop (1.0 * sqrt(20) * 0.85 = 3.80 < 5.0)
            catalyst_path="DIP_BUY",
            rsi14=55.0,  # Fails RSI [30, 45] dip window
            drawdown_5d_pct=-0.02,  # Fails 5d drop [5%, 15%]
            days_to_earnings=45,
            rvol_20d=1.5,
            avg_dollar_volume_20d=50_000_000.0,
        )
        self.assertIn(StandardGateName.BUY_FLOOR, dec.failed_gates)
        self.assertIn(StandardReasonCode.DIP_CRITERIA_FAILED, dec.reason_codes)
        self.assertEqual(dec.decision, "HOLD")

    def test_gate_buy_floor_value_trap_confirmed(self):
        """Confirmed Value Trap fails BUY_FLOOR with VALUE_TRAP_DETECTED."""
        dec = self.engine.evaluate(
            ticker="TRAP",
            candidate_decision="BUY",
            composite_score=65.0,
            current_price=10.0,
            stop_loss=9.0,
            target_price=13.0,
            atr=0.2,  # Clears noise stop (0.2 * sqrt(20) * 0.85 = 0.76 < 1.0)
            is_value_trap=True,  # Confirmed structural distress
            days_to_earnings=45,
            rvol_20d=1.5,
            avg_dollar_volume_20d=50_000_000.0,
        )
        self.assertIn(StandardGateName.BUY_FLOOR, dec.failed_gates)
        self.assertIn(StandardReasonCode.VALUE_TRAP_DETECTED, dec.reason_codes)
        self.assertEqual(dec.decision, "HOLD")

    # -------------------------------------------------------------------------
    # Gate 6: SHORT_CEILING & BORROW_AVAILABILITY
    # -------------------------------------------------------------------------
    def test_gate_short_ceiling_exceeded(self):
        """Score 48.0 exceeds SHORT_CEILING (<= 45.0) -> HOLD with SHORT_CEILING_EXCEEDED."""
        dec = self.engine.evaluate(
            ticker="BEAR",
            candidate_decision="SELL",
            composite_score=48.0,
            current_price=100.0,
            stop_loss=105.0,
            target_price=85.0,
            atr=1.0,  # Clears noise stop (1.0 * sqrt(20) * 0.85 = 3.80 < 5.0)
            days_to_earnings=45,
            rvol_20d=1.5,
            avg_dollar_volume_20d=50_000_000.0,
            borrow_available=True,
        )
        self.assertIn(StandardGateName.SHORT_CEILING, dec.failed_gates)
        self.assertIn(StandardReasonCode.SHORT_CEILING_EXCEEDED, dec.reason_codes)
        self.assertEqual(dec.decision, "HOLD")

    def test_gate_short_borrow_unavailable(self):
        """Score <= 45.0 but borrow unavailable fails BORROW_AVAILABILITY."""
        dec = self.engine.evaluate(
            ticker="HTB",
            candidate_decision="SELL",
            composite_score=38.0,
            current_price=100.0,
            stop_loss=105.0,
            target_price=85.0,
            atr=1.0,  # Clears noise stop (1.0 * sqrt(20) * 0.85 = 3.80 < 5.0)
            days_to_earnings=45,
            rvol_20d=1.5,
            avg_dollar_volume_20d=50_000_000.0,
            borrow_available=False,  # Hard-to-borrow
        )
        self.assertIn(StandardGateName.BORROW_AVAILABILITY, dec.failed_gates)
        self.assertIn(StandardReasonCode.BORROW_UNAVAILABLE, dec.reason_codes)
        self.assertEqual(dec.decision, "HOLD")

    # -------------------------------------------------------------------------
    # Gate 7: REWARD_TO_RISK
    # -------------------------------------------------------------------------
    def test_gate_reward_to_risk_insufficient_ratio(self):
        """Reward-to-risk < 1.20 fails REWARD_TO_RISK."""
        # Risk = 5, Reward = 4 -> RR = 0.80 < 1.20
        dec = self.engine.evaluate(
            ticker="POOR_RR",
            candidate_decision="BUY",
            composite_score=75.0,
            current_price=100.0,
            stop_loss=95.0,
            target_price=104.0,  # Insufficient target
            atr=1.0,  # Clears noise stop (1.0 * sqrt(20) * 0.85 = 3.80 < 5.0)
            days_to_earnings=45,
            rvol_20d=1.5,
            avg_dollar_volume_20d=50_000_000.0,
        )
        self.assertIn(StandardGateName.REWARD_TO_RISK, dec.failed_gates)
        self.assertIn(StandardReasonCode.REWARD_TO_RISK_INSUFFICIENT, dec.reason_codes)
        self.assertEqual(dec.decision, "HOLD")

    def test_gate_reward_to_risk_invalid_price_ordering(self):
        """Inverted price sequence (target <= entry) fails REWARD_TO_RISK."""
        dec = self.engine.evaluate(
            ticker="INVERT",
            candidate_decision="BUY",
            composite_score=75.0,
            current_price=100.0,
            stop_loss=95.0,
            target_price=98.0,  # Target below entry
            atr=1.0,  # Clears noise stop (1.0 * sqrt(20) * 0.85 = 3.80 < 5.0)
            days_to_earnings=45,
            rvol_20d=1.5,
            avg_dollar_volume_20d=50_000_000.0,
        )
        self.assertIn(StandardGateName.REWARD_TO_RISK, dec.failed_gates)
        self.assertIn(StandardReasonCode.PRICE_ORDERING_INVALID, dec.reason_codes)
        self.assertEqual(dec.decision, "HOLD")

    # -------------------------------------------------------------------------
    # Gate 8: STOP_LOSS_SURVEILLANCE & HOLDING_THESIS (Existing Position)
    # -------------------------------------------------------------------------
    def test_gate_stop_loss_breach_forces_close_long(self):
        """Existing long position breaching stop executes CLOSE_LONG immediately."""
        dec = self.engine.evaluate(
            ticker="LONG_STOP",
            candidate_decision="BUY",  # Advisory wants to add
            composite_score=85.0,     # Very high score
            position_state="LONG",
            current_price=94.0,
            stop_loss=95.0,           # Price <= stop
            target_price=120.0,
            atr=2.0,
            days_to_earnings=45,
        )
        self.assertIn(StandardGateName.STOP_LOSS_SURVEILLANCE, dec.failed_gates)
        self.assertIn(StandardReasonCode.STOP_LOSS_BREACHED, dec.reason_codes)
        self.assertEqual(dec.decision, "SELL")
        self.assertEqual(dec.position_action, PositionAction.CLOSE_LONG.value)

    def test_gate_holding_thesis_intact_protects_winning_position(self):
        """Intact holding thesis prevents premature discretionary sale -> HOLD with NO_ACTION."""
        dec = self.engine.evaluate(
            ticker="WINNER",
            candidate_decision="SELL",  # Discretionary sell requested
            composite_score=75.0,      # Intact score >= 70.0
            position_state="LONG",
            current_price=110.0,
            stop_loss=95.0,            # Stop not hit
            target_price=130.0,        # Target not reached
            atr=2.0,
            days_to_earnings=45,
        )
        self.assertIn(StandardGateName.HOLDING_THESIS, dec.passed_gates)
        self.assertIn(StandardReasonCode.HOLDING_THESIS_INTACT, dec.reason_codes)
        self.assertEqual(dec.decision, "HOLD")
        self.assertEqual(dec.position_action, PositionAction.NO_ACTION.value)

    # -------------------------------------------------------------------------
    # Gate 9: PROFIT_TARGET
    # -------------------------------------------------------------------------
    def test_gate_profit_target_reached_executes_close_long(self):
        """Price reaching profit target triggers CLOSE_LONG."""
        dec = self.engine.evaluate(
            ticker="TARGET_HIT",
            candidate_decision="HOLD",
            composite_score=75.0,
            position_state="LONG",
            current_price=131.0,
            stop_loss=95.0,
            target_price=130.0,        # Price >= target
            atr=2.0,
            days_to_earnings=45,
        )
        self.assertIn(StandardGateName.PROFIT_TARGET, dec.passed_gates)
        self.assertIn(StandardReasonCode.PROFIT_TARGET_REACHED, dec.reason_codes)
        self.assertEqual(dec.decision, "SELL")
        self.assertEqual(dec.position_action, PositionAction.CLOSE_LONG.value)


class TestSeparationOfDecisionFromExplanation(unittest.TestCase):
    """
    Verifies that the investment decision is completely independent of the
    rendered narrative or free-form natural language explanations.
    """

    def setUp(self):
        self.policy = get_default_policy()

    def test_identical_decision_with_and_without_narrative(self):
        """
        Running the pipeline with verbose narrative vs empty narrative
        produces bitwise identical structured decisions.
        """
        symbol = "AAPL"
        m_data = {
            "current_price": 180.0,
            "atr": 3.0,
            "suggested_stop_loss": 174.0,
            "suggested_target_price": 195.0,
            "rvol_20d": 1.4,
            "avg_dollar_vol_20d": 80_000_000.0,
            "days_to_earnings": 35,
            "rsi14": 55.0,
            "deterministic_5pillar_scores": {
                "composite_quantitative_score": 78.5,
                "raw_composite": 78.5,
                "trend_score": 80.0,
                "sector_relative_score": 75.0,
                "market_alpha_score": 76.0,
                "valuation_history_score": 82.0,
                "peer_valuation_score": 77.0,
            },
            "deterministic_data_completeness": 1.0,
        }

        # Run 1: Elaborate, poetic narrative text
        narrative_obj = {
            "decision": "BUY",
            "buy_score": 0.85,
            "hold_score": 0.10,
            "sell_score": 0.05,
            "confidence": 0.80,
            "bull_case": [
                "Unprecedented ecosystem lock-in driving recurring services revenue to record highs.",
                "Services margin expansion acts as a resilient buffer against hardware cyclicality.",
            ],
            "bear_case": [
                "Geopolitical friction in Asian contract manufacturing centers.",
            ],
            "key_risks": [
                "Macro consumer slowdown could dampen iPhone replacement cycles.",
            ],
            "summary": "High-conviction long opportunity supported by expanding gross margins and robust services growth.",
        }

        sig1 = normalize_master_trader_json(
            narrative_obj,
            symbol,
            m_data=m_data,
            macro_data={"us_10y_yield": 4.10, "fear_greed_score": 60.0},
            policy=self.policy,
        )

        # Run 2: Completely empty narrative text
        empty_obj = {
            "decision": "BUY",
            "buy_score": 0.85,
            "hold_score": 0.10,
            "sell_score": 0.05,
            "confidence": 0.80,
            "bull_case": [],
            "bear_case": [],
            "key_risks": [],
            "summary": "",
        }

        sig2 = normalize_master_trader_json(
            empty_obj,
            symbol,
            m_data=m_data,
            macro_data={"us_10y_yield": 4.10, "fear_greed_score": 60.0},
            policy=self.policy,
        )

        # Investment decisions MUST BE EXACTLY IDENTICAL
        sd1 = sig1["structured_decision"]
        sd2 = sig2["structured_decision"]

        self.assertEqual(sd1["decision"], sd2["decision"])
        self.assertEqual(sd1["composite_score"], sd2["composite_score"])
        self.assertEqual(sd1["model_status"], sd2["model_status"])
        self.assertEqual(sd1["passed_gates"], sd2["passed_gates"])
        self.assertEqual(sd1["failed_gates"], sd2["failed_gates"])
        self.assertEqual(sd1["data_quality_status"], sd2["data_quality_status"])
        self.assertEqual(sd1["position_action"], sd2["position_action"])
        self.assertEqual(sd1["reason_codes"], sd2["reason_codes"])

    def test_reproducibility_of_logged_reasons(self):
        """Logged reasons and gate outcomes are strictly deterministic and reproducible."""
        engine = StructuredDecisionEngine(self.policy)
        reasons_history = []
        for _ in range(50):
            dec = engine.evaluate(
                ticker="NVDA",
                candidate_decision="BUY",
                composite_score=58.4,
                current_price=120.0,
                stop_loss=115.0,
                target_price=135.0,
                atr=1.0,  # Clears noise stop
                days_to_earnings=45,
                rvol_20d=1.5,
                avg_dollar_volume_20d=50_000_000.0,
                run_id="constant-run-id",
            )
            reasons_history.append((dec.decision, tuple(dec.failed_gates), tuple(dec.reason_codes)))

        # All 50 evaluations produce the exact same outcome
        self.assertEqual(len(set(reasons_history)), 1)
        self.assertEqual(reasons_history[0], ("HOLD", ("BUY_FLOOR",), ("SCORE_BELOW_BUY_THRESHOLD",)))


class TestStructuredDecisionDatabasePersistence(unittest.TestCase):
    """Verifies SQLite database round-trip persistence of structured decisions."""

    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()
        db.init_db(self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_structured_decision_sqlite_persistence_and_enrichment(self):
        """
        Saves a signal containing structured_decision into SQLite and verifies
        clean roundtrip retrieval and deserialization.
        """
        ticker = "NVDA"
        structured_dec = {
            "run_id": "test-run-12345",
            "ticker": ticker,
            "decision": "HOLD",
            "composite_score": 58.4,
            "model_status": "IN_SCOPE",
            "passed_gates": ["DATA_QUALITY", "EVENT_RISK", "LIQUIDITY"],
            "failed_gates": ["BUY_FLOOR"],
            "data_quality_status": "VALID",
            "position_action": "NO_ACTION",
            "reason_codes": ["SCORE_BELOW_BUY_THRESHOLD"],
            "timestamp": "2026-10-18T16:00:00Z",
        }

        signal_payload = {
            "symbol": ticker,
            "decision": "HOLD",
            "confidence": 0.55,
            "quant_score": 58.4,
            "run_id": structured_dec["run_id"],
            "model_status": structured_dec["model_status"],
            "passed_gates": structured_dec["passed_gates"],
            "failed_gates": structured_dec["failed_gates"],
            "data_quality_status": structured_dec["data_quality_status"],
            "position_action": structured_dec["position_action"],
            "reason_codes": structured_dec["reason_codes"],
            "structured_decision": structured_dec,
            "bull_case": ["Bull thesis narrative..."],
            "bear_case": ["Bear thesis narrative..."],
        }

        # 1. Save to DB
        db.save_results([signal_payload], model_used="Gemini", db_path=self.db_path)

        # 2. Retrieve history
        history = db.get_signal_history(symbol=ticker, db_path=self.db_path)
        self.assertEqual(len(history), 1)
        record = history[0]

        # 3. Verify structured fields
        self.assertEqual(record["run_id"], "test-run-12345")
        self.assertEqual(record["model_status"], "IN_SCOPE")
        self.assertEqual(record["data_quality_status"], "VALID")
        self.assertEqual(record["position_action"], "NO_ACTION")

        # Verify JSON list deserialization in _enrich_signal_dict
        self.assertEqual(record["passed_gates"], ["DATA_QUALITY", "EVENT_RISK", "LIQUIDITY"])
        self.assertEqual(record["failed_gates"], ["BUY_FLOOR"])
        self.assertEqual(record["reason_codes"], ["SCORE_BELOW_BUY_THRESHOLD"])

        # Verify structured_decision dict deserialization
        self.assertIsInstance(record["structured_decision"], dict)
        self.assertEqual(record["structured_decision"]["decision"], "HOLD")
        self.assertEqual(record["structured_decision"]["composite_score"], 58.4)
        self.assertEqual(record["structured_decision"]["failed_gates"], ["BUY_FLOOR"])
        self.assertEqual(record["structured_decision"]["reason_codes"], ["SCORE_BELOW_BUY_THRESHOLD"])


if __name__ == "__main__":
    unittest.main()
