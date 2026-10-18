"""
Step 6.3: Property-Based Testing Suite with Invariant Verification.

Uses Hypothesis with fixed random seeds (@seed(42)) and explicit regression
examples (@example) to verify core system invariants across a wide range of
valid and invalid inputs:

Invariant 1: Quantitative and hybrid scores always remain within permitted bounds [0.0, 100.0].
Invariant 2: A missing required input cannot create a stronger signal merely because it is missing.
Invariant 3: No BUY is certified when a mandatory BUY gate has failed.
Invariant 4: No short is opened without passing short-specific eligibility checks.
Invariant 5: No malformed LLM output can authorize an order.
Invariant 6: The same inputs, configuration and model versions produce the same deterministic decision.
"""

import math
import unittest
from typing import Any, Dict, Optional

from hypothesis import given, settings, seed, example, strategies as st

from quant.policy_config import TradingPolicy, get_default_policy
from quant.normalization import clamp, logistic_scale, normal_cdf_scale
from quant.volatility import compute_vol_factor, apply_volatility_dampener
from quant.composite_score import compute_deterministic_composite, blend_quant_and_ml_scores
from quant.trade_geometry import (
    TradeDirection,
    PriceOrderingStatus,
    compute_validated_reward_risk,
    derive_deterministic_channel_geometry,
    validate_price_ordering,
)
from quant.signal_strength import (
    calculate_signal_strength,
    compute_score_decisiveness,
    compute_pillar_agreement,
)
from quant.compliance_gates import (
    GateStatus,
    BuyFloorGate,
    RewardToRiskGate,
    VolatilityEventLiquidityGate,
    SellExitSymmetryGate,
    GatedConfidenceGate,
)
from quant.decision_precedence import (
    DecisionPrecedenceEngine,
    PositionState,
    InternalActionType,
    DecisionStage,
)
from signal_schema import validate_signal_json
from main import normalize_master_trader_json


# =============================================================================
# INVARIANT 1: Quantitative and hybrid scores always remain within permitted bounds
# =============================================================================
class TestInvariant1ScoreBounds(unittest.TestCase):
    """Verifies that all scoring, normalization, and dampening functions stay bounded."""

    @given(
        val=st.floats(allow_nan=True, allow_infinity=True),
        low=st.floats(min_value=-1000.0, max_value=500.0),
        high=st.floats(min_value=500.0, max_value=2000.0),
    )
    @settings(max_examples=100, deadline=None)
    @seed(42)
    @example(val=float("nan"), low=0.0, high=100.0)
    @example(val=float("inf"), low=0.0, high=100.0)
    @example(val=float("-inf"), low=0.0, high=100.0)
    @example(val=-99999.0, low=0.0, high=100.0)
    @example(val=99999.0, low=0.0, high=100.0)
    def test_clamp_always_within_bounds(self, val: float, low: float, high: float):
        """clamp(val, low, high) must strictly lie within [low, high] for all numeric inputs."""
        res = clamp(val, min_val=low, max_val=high)
        self.assertTrue(math.isfinite(res))
        self.assertGreaterEqual(res, low)
        self.assertLessEqual(res, high)

    @given(val=st.floats(min_value=-1e6, max_value=1e6))
    @settings(max_examples=100, deadline=None)
    @seed(42)
    @example(val=-1000.0)
    @example(val=0.0)
    @example(val=1000.0)
    def test_logistic_scale_strictly_in_0_to_100(self, val: float):
        """logistic_scale(val) must strictly be bounded in [0.0, 100.0]."""
        res = logistic_scale(val)
        self.assertTrue(math.isfinite(res))
        self.assertGreaterEqual(res, 0.0)
        self.assertLessEqual(res, 100.0)

    @given(val=st.floats(min_value=-1e6, max_value=1e6))
    @settings(max_examples=100, deadline=None)
    @seed(42)
    @example(val=-100.0)
    @example(val=0.0)
    @example(val=100.0)
    def test_normal_cdf_scale_strictly_in_0_to_100(self, val: float):
        """normal_cdf_scale(val) must strictly be bounded in [0.0, 100.0]."""
        res = normal_cdf_scale(val)
        self.assertTrue(math.isfinite(res))
        self.assertGreaterEqual(res, 0.0)
        self.assertLessEqual(res, 100.0)

    @given(
        atr=st.floats(min_value=-100.0, max_value=1000.0),
        price=st.floats(min_value=-100.0, max_value=5000.0),
    )
    @settings(max_examples=100, deadline=None)
    @seed(42)
    @example(atr=0.0, price=100.0)
    @example(atr=1.0, price=100.0)  # Calm (1.0%) -> 1.00
    @example(atr=2.0, price=100.0)  # Normal (2.0%) -> 0.95
    @example(atr=3.5, price=100.0)  # Elevated (3.5%) -> 0.90
    @example(atr=5.0, price=100.0)  # Extreme (5.0%) -> 0.85
    @example(atr=-5.0, price=100.0)
    @example(atr=5.0, price=-100.0)
    def test_compute_vol_factor_strictly_in_bounds(self, atr: float, price: float):
        """compute_vol_factor must return f_vol in [0.85, 1.00] for any inputs."""
        f_vol = compute_vol_factor(atr, price)
        self.assertTrue(math.isfinite(f_vol))
        self.assertGreaterEqual(f_vol, 0.85)
        self.assertLessEqual(f_vol, 1.00)

    @given(
        score=st.floats(min_value=0.0, max_value=100.0),
        vol_factor=st.floats(min_value=0.85, max_value=1.00),
    )
    @settings(max_examples=100, deadline=None)
    @seed(42)
    @example(score=50.0, vol_factor=0.85)  # Neutral invariance
    @example(score=50.0, vol_factor=0.90)  # Neutral invariance
    @example(score=50.0, vol_factor=0.95)  # Neutral invariance
    @example(score=50.0, vol_factor=1.00)  # Neutral invariance
    @example(score=100.0, vol_factor=0.85)
    @example(score=0.0, vol_factor=0.85)
    def test_apply_volatility_dampener_invariance_and_bounds(self, score: float, vol_factor: float):
        """apply_volatility_dampener must remain in [0, 100] and preserve neutral invariance."""
        damped = apply_volatility_dampener(score, vol_factor)
        self.assertTrue(math.isfinite(damped))
        self.assertGreaterEqual(damped, 0.0)
        self.assertLessEqual(damped, 100.0)
        if abs(score - 50.0) < 1e-9:
            self.assertAlmostEqual(damped, 50.0, places=4)

    @given(
        quant_score=st.one_of(st.none(), st.floats(min_value=0.0, max_value=100.0)),
        ml_score=st.one_of(st.none(), st.floats(min_value=0.0, max_value=100.0)),
        ml_weight=st.floats(min_value=0.0, max_value=1.0),
        in_scope=st.booleans(),
    )
    @settings(max_examples=100, deadline=None)
    @seed(42)
    @example(quant_score=75.0, ml_score=85.0, ml_weight=0.70, in_scope=True)
    @example(quant_score=75.0, ml_score=85.0, ml_weight=0.70, in_scope=False)
    @example(quant_score=59.99, ml_score=None, ml_weight=0.0, in_scope=False)
    def test_blend_quant_and_ml_scores_bounds(
        self, quant_score: Optional[float], ml_score: Optional[float], ml_weight: float, in_scope: bool
    ):
        """Blended score must either be None or strictly bounded in [0.0, 100.0]."""
        ctx = {
            "model_status": "IN_SCOPE" if in_scope else "OUT_OF_SCOPE",
            "in_scope": in_scope,
            "ml_score": ml_score,
            "ml_weight": ml_weight,
        }
        res, eff_w, eff_s = blend_quant_and_ml_scores(quant_score, ctx)
        if res is not None:
            self.assertTrue(math.isfinite(res))
            self.assertGreaterEqual(res, 0.0)
            self.assertLessEqual(res, 100.0)
            self.assertGreaterEqual(eff_w, 0.0)
            self.assertLessEqual(eff_w, 1.0)


# =============================================================================
# INVARIANT 2: Missing inputs cannot create a stronger signal
# =============================================================================
class TestInvariant2MissingInputsCannotStrengthenSignal(unittest.TestCase):
    """Verifies that missing data strictly penalizes or maintains confidence, never inflating it."""

    @given(
        score=st.floats(min_value=0.0, max_value=100.0),
        c_low=st.floats(min_value=0.0, max_value=0.50),
        c_high=st.floats(min_value=0.50, max_value=1.00),
        decision=st.sampled_from(["BUY", "SELL", "HOLD"]),
    )
    @settings(max_examples=100, deadline=None)
    @seed(42)
    @example(score=85.0, c_low=0.40, c_high=0.90, decision="BUY")
    @example(score=20.0, c_low=0.30, c_high=1.00, decision="SELL")
    @example(score=50.0, c_low=0.20, c_high=0.80, decision="HOLD")
    def test_confidence_monotonically_non_decreasing_with_completeness(
        self, score: float, c_low: float, c_high: float, decision: str
    ):
        """For any fixed setup, lower data completeness cannot yield higher confidence."""
        pillars = {"trend": score, "sector": score, "alpha": score, "valuation_history": score}
        res_low = calculate_signal_strength(
            composite_score=score,
            pillar_scores=pillars,
            data_completeness=c_low,
            decision=decision,
        )
        res_high = calculate_signal_strength(
            composite_score=score,
            pillar_scores=pillars,
            data_completeness=c_high,
            decision=decision,
        )
        # Completeness monotonic property: Confidence(low) <= Confidence(high)
        self.assertLessEqual(
            res_low.signal_strength,
            res_high.signal_strength + 1e-9,
            f"Confidence increased when completeness dropped: {res_low.signal_strength} > {res_high.signal_strength}",
        )

    @given(num_available_pillars=st.integers(min_value=0, max_value=2))
    @settings(max_examples=50, deadline=None)
    @seed(42)
    @example(num_available_pillars=0)
    @example(num_available_pillars=1)
    @example(num_available_pillars=2)
    def test_insufficient_pillars_gate_always_aborts(self, num_available_pillars: int):
        """If fewer than 3 pillars are available, composite MUST be None and is_tradeable False."""
        all_pillars = ["trend", "sector", "alpha", "valuation_history", "peer_valuation"]
        active_set = set(all_pillars[:num_available_pillars])

        m_data = {
            "current_price": 100.0,
            "ema20": 98.0 if "trend" in active_set else None,
            "ema50": 95.0 if "trend" in active_set else None,
            "rsi14": 55.0 if "trend" in active_set else None,
            "change_20d_pct": 3.0 if "sector" in active_set else None,
            "relative_alpha_20d": 2.0 if "alpha" in active_set else None,
            "beta_60d": 1.0,
            "atr": 2.0,
        }
        gloom = {
            "financials": {"key_ratios": {"trailing_pe": 20.0 if "valuation_history" in active_set else None}},
            "peer_valuation": {"ev_to_ebitda": 15.0 if "peer_valuation" in active_set else None},
        }
        res = compute_deterministic_composite(technical_data=m_data, gloomberb_payload=gloom)
        self.assertIsNone(res.composite_quantitative_score)
        self.assertFalse(res.is_tradeable)
        self.assertEqual(res.no_trade_reason, "INSUFFICIENT_EVIDENCE")


# =============================================================================
# INVARIANT 3: No BUY is certified when a mandatory BUY gate has failed
# =============================================================================
class TestInvariant3NoBuyWhenMandatoryBuyGateFailed(unittest.TestCase):
    """Verifies that no candidate BUY can slip through when any mandatory gate fails."""

    def setUp(self):
        self.policy = get_default_policy()
        self.engine = DecisionPrecedenceEngine(self.policy)

    @given(sub_floor_score=st.floats(min_value=0.0, max_value=59.999))
    @settings(max_examples=100, deadline=None)
    @seed(42)
    @example(sub_floor_score=59.99)
    @example(sub_floor_score=59.00)
    @example(sub_floor_score=52.00)
    @example(sub_floor_score=0.0)
    def test_ordinary_buy_below_floor_never_certifies_buy(self, sub_floor_score: float):
        """Any ordinary BUY candidate with score < 60.00 is strictly rejected and downgraded."""
        prec_res = self.engine.evaluate(
            candidate_decision="BUY",
            position_state=PositionState.FLAT,
            current_price=100.0,
            stop_loss=90.0,
            target_price=120.0,
            atr=2.0,
            composite_score=sub_floor_score,
            catalyst_path=None,
            avg_dollar_volume_20d=10_000_000.0,
            rvol_20d=1.0,
        )
        self.assertNotEqual(prec_res.internal_action, InternalActionType.BUY)
        self.assertEqual(prec_res.external_decision, "HOLD")
        self.assertEqual(prec_res.reason_code, "COMPOSITE_BELOW_FLOOR")

    @given(
        target_price=st.floats(min_value=10.0, max_value=111.99),
        composite_score=st.floats(min_value=60.0, max_value=100.0),
    )
    @settings(max_examples=100, deadline=None)
    @seed(42)
    @example(target_price=111.99, composite_score=95.0)  # RR = 1.199 < 1.20
    @example(target_price=100.0, composite_score=95.0)   # Target == Entry
    @example(target_price=95.0, composite_score=95.0)    # Target < Entry
    def test_sub_threshold_or_inverted_rr_never_certifies_buy(self, target_price: float, composite_score: float):
        """High composite score CANNOT bypass reward-to-risk (< 1.20) or invalid target ordering."""
        # Entry 100, Stop 90 (Risk 10). Required Reward for 1.20 is >= 12.0 (Target >= 112.0)
        prec_res = self.engine.evaluate(
            candidate_decision="BUY",
            position_state=PositionState.FLAT,
            current_price=100.0,
            stop_loss=90.0,
            target_price=target_price,
            atr=2.0,
            composite_score=composite_score,
            data_completeness=1.0,
            avg_dollar_volume_20d=10_000_000.0,
            rvol_20d=1.0,
        )
        self.assertNotEqual(prec_res.internal_action, InternalActionType.BUY)
        self.assertEqual(prec_res.external_decision, "HOLD")
        self.assertIn(prec_res.reason_code, ("RR_TOO_LOW", "ZERO_OR_NEGATIVE_REWARD", "INVALID_PRICE_ORDERING", "DATA_COVERAGE_INADEQUATE"))

    @given(
        days_to_earnings=st.integers(min_value=0, max_value=3),
        composite_score=st.floats(min_value=60.0, max_value=100.0),
    )
    @settings(max_examples=50, deadline=None)
    @seed(42)
    @example(days_to_earnings=0, composite_score=99.0)
    @example(days_to_earnings=1, composite_score=90.0)
    @example(days_to_earnings=2, composite_score=85.0)
    @example(days_to_earnings=3, composite_score=80.0)
    def test_earnings_blackout_never_certifies_buy(self, days_to_earnings: int, composite_score: float):
        """Active earnings blackout (<= 3 sessions) unconditionally blocks fresh BUY."""
        prec_res = self.engine.evaluate(
            candidate_decision="BUY",
            position_state=PositionState.FLAT,
            current_price=100.0,
            stop_loss=90.0,
            target_price=125.0,
            atr=2.0,
            composite_score=composite_score,
            days_to_earnings=days_to_earnings,
            avg_dollar_volume_20d=10_000_000.0,
            rvol_20d=1.0,
        )
        self.assertNotEqual(prec_res.internal_action, InternalActionType.BUY)
        self.assertEqual(prec_res.external_decision, "HOLD")
        self.assertEqual(prec_res.reason_code, "EARNINGS_BLACKOUT")

    @given(composite_score=st.floats(min_value=60.0, max_value=100.0))
    @settings(max_examples=50, deadline=None)
    @seed(42)
    @example(composite_score=65.0)
    @example(composite_score=85.0)
    def test_stale_market_data_never_certifies_buy(self, composite_score: float):
        """Stale market data unconditionally blocks fresh entries in Stage 1."""
        prec_res = self.engine.evaluate(
            candidate_decision="BUY",
            position_state=PositionState.FLAT,
            current_price=100.0,
            stop_loss=90.0,
            target_price=125.0,
            atr=2.0,
            composite_score=composite_score,
            market_data_fresh=False,  # Data outage!
            avg_dollar_volume_20d=10_000_000.0,
            rvol_20d=1.0,
        )
        self.assertNotEqual(prec_res.internal_action, InternalActionType.BUY)
        self.assertEqual(prec_res.external_decision, "HOLD")
        self.assertEqual(prec_res.reason_code, "DATA_OUTAGE_NEW_ENTRIES_BLOCKED")


# =============================================================================
# INVARIANT 4: No short is opened without passing short-specific checks
# =============================================================================
class TestInvariant4NoShortWithoutPassingShortEligibilityChecks(unittest.TestCase):
    """Verifies that short trades are never opened unless all short criteria pass."""

    def setUp(self):
        self.policy = get_default_policy()
        self.engine = DecisionPrecedenceEngine(self.policy)

    @given(composite_score=st.floats(min_value=0.0, max_value=45.0))
    @settings(max_examples=50, deadline=None)
    @seed(42)
    @example(composite_score=20.0)
    @example(composite_score=35.0)
    @example(composite_score=44.99)
    def test_short_without_borrow_never_opens_short(self, composite_score: float):
        """Short trade without borrow available is strictly rejected and converted to HOLD."""
        prec_res = self.engine.evaluate(
            candidate_decision="SELL",
            position_state=PositionState.FLAT,
            current_price=100.0,
            stop_loss=108.0,
            target_price=80.0,
            atr=2.0,
            composite_score=composite_score,
            borrow_available=False,  # Unborrowable!
            avg_dollar_volume_20d=10_000_000.0,
            rvol_20d=1.0,
        )
        self.assertNotEqual(prec_res.internal_action, InternalActionType.OPEN_SHORT)
        self.assertEqual(prec_res.external_decision, "HOLD")
        self.assertEqual(prec_res.reason_code, "BORROW_UNAVAILABLE")

    @given(above_ceiling_score=st.floats(min_value=45.01, max_value=100.0))
    @settings(max_examples=50, deadline=None)
    @seed(42)
    @example(above_ceiling_score=45.01)
    @example(above_ceiling_score=50.00)
    @example(above_ceiling_score=80.00)
    def test_short_above_ceiling_score_never_opens_short(self, above_ceiling_score: float):
        """Score above 45.0 ceiling fails short breakdown criteria and cannot open short."""
        prec_res = self.engine.evaluate(
            candidate_decision="SELL",
            position_state=PositionState.FLAT,
            current_price=100.0,
            stop_loss=108.0,
            target_price=80.0,
            atr=2.0,
            composite_score=above_ceiling_score,
            borrow_available=True,
            avg_dollar_volume_20d=10_000_000.0,
            rvol_20d=1.0,
        )
        self.assertNotEqual(prec_res.internal_action, InternalActionType.OPEN_SHORT)
        self.assertEqual(prec_res.external_decision, "HOLD")

    @given(days_to_ex_div=st.integers(min_value=0, max_value=2))
    @settings(max_examples=30, deadline=None)
    @seed(42)
    @example(days_to_ex_div=0)
    @example(days_to_ex_div=1)
    @example(days_to_ex_div=2)
    def test_ex_dividend_blackout_never_opens_short(self, days_to_ex_div: int):
        """Shorting within 2 sessions of ex-dividend is strictly prohibited."""
        prec_res = self.engine.evaluate(
            candidate_decision="SELL",
            position_state=PositionState.FLAT,
            current_price=100.0,
            stop_loss=108.0,
            target_price=80.0,
            atr=2.0,
            composite_score=20.0,
            borrow_available=True,
            days_to_ex_dividend=days_to_ex_div,
            avg_dollar_volume_20d=10_000_000.0,
            rvol_20d=1.0,
        )
        self.assertNotEqual(prec_res.internal_action, InternalActionType.OPEN_SHORT)
        self.assertEqual(prec_res.external_decision, "HOLD")
        self.assertEqual(prec_res.reason_code, "EX_DIVIDEND_BLACKOUT")

    @given(composite_score=st.floats(min_value=0.0, max_value=40.0))
    @settings(max_examples=50, deadline=None)
    @seed(42)
    @example(composite_score=20.0)
    def test_existing_long_position_never_opens_short_on_sell(self, composite_score: float):
        """When an existing long position receives SELL, it must close the long, never open a short."""
        prec_res = self.engine.evaluate(
            candidate_decision="SELL",
            position_state=PositionState.LONG,
            current_price=100.0,
            stop_loss=90.0,
            target_price=120.0,
            atr=2.0,
            composite_score=composite_score,
            borrow_available=True,
            avg_dollar_volume_20d=10_000_000.0,
            rvol_20d=1.0,
        )
        self.assertNotEqual(prec_res.internal_action, InternalActionType.OPEN_SHORT)
        self.assertIn(prec_res.internal_action, (InternalActionType.SELL_TO_CLOSE, InternalActionType.HOLD))


# =============================================================================
# INVARIANT 5: No malformed LLM output can authorize an order
# =============================================================================
class TestInvariant5MalformedLlmOutputCannotAuthorizeOrder(unittest.TestCase):
    """Verifies that arbitrary, malformed, or hostile LLM outputs cannot issue orders."""

    @given(
        forbidden_key=st.sampled_from([
            "order_type", "shares", "broker_action", "execute_now", "limit_price",
            "time_in_force", "account_id", "route", "portfolio_pct",
        ]),
        decision=st.sampled_from(["BUY", "SELL"]),
    )
    @settings(max_examples=50, deadline=None)
    @seed(42)
    def test_prohibited_broker_commands_always_rejected(self, forbidden_key: str, decision: str):
        """Any payload attempting to dictate broker execution commands is rejected at schema gate."""
        payload = {
            "ticker": "AAPL",
            "decision": decision,
            "confidence": 0.90,
            forbidden_key: "MALICIOUS_BROKER_INSTRUCTION",
        }
        _, errs = validate_signal_json(payload)
        self.assertGreater(len(errs), 0)
        self.assertTrue(any("prohibited broker command" in e for e in errs))

    @given(
        malformed_decision=st.text(min_size=1, max_size=20).filter(
            lambda s: s.strip().upper() not in ("BUY", "SELL", "HOLD")
        )
    )
    @settings(max_examples=50, deadline=None)
    @seed(42)
    @example(malformed_decision="YOLO_CALLS")
    @example(malformed_decision="ALL_IN")
    @example(malformed_decision="STRONG_BUY")
    @example(malformed_decision="PUMP_AND_DUMP")
    def test_unsupported_decision_actions_always_rejected(self, malformed_decision: str):
        """Non-standard decision strings are rejected by schema validation."""
        payload = {
            "ticker": "AAPL",
            "decision": malformed_decision,
            "confidence": 0.80,
            "buy_score": 0.8,
            "hold_score": 0.1,
            "sell_score": 0.1,
        }
        _, errs = validate_signal_json(payload)
        self.assertGreater(len(errs), 0)


# =============================================================================
# INVARIANT 6: Same inputs produce identical deterministic decisions
# =============================================================================
class TestInvariant6DeterminismAndIdempotence(unittest.TestCase):
    """Verifies strict mathematical determinism and idempotence across evaluations."""

    def setUp(self):
        self.policy = get_default_policy()
        self.engine = DecisionPrecedenceEngine(self.policy)

    @given(
        score=st.floats(min_value=0.0, max_value=100.0),
        c_val=st.floats(min_value=0.0, max_value=1.0),
        decision=st.sampled_from(["BUY", "SELL", "HOLD"]),
    )
    @settings(max_examples=50, deadline=None)
    @seed(42)
    def test_signal_strength_purely_deterministic(self, score: float, c_val: float, decision: str):
        """calculate_signal_strength evaluated twice on identical inputs produces identical output."""
        pillars = {"trend": score, "sector": score * 0.9, "alpha": score * 1.1}
        res1 = calculate_signal_strength(composite_score=score, pillar_scores=pillars, data_completeness=c_val, decision=decision)
        res2 = calculate_signal_strength(composite_score=score, pillar_scores=pillars, data_completeness=c_val, decision=decision)
        self.assertEqual(res1.signal_strength, res2.signal_strength)
        self.assertEqual(res1.score_decisiveness, res2.score_decisiveness)
        self.assertEqual(res1.pillar_agreement, res2.pillar_agreement)

    @given(
        current_price=st.floats(min_value=10.0, max_value=500.0),
        stop_loss=st.floats(min_value=5.0, max_value=450.0),
        target_price=st.floats(min_value=15.0, max_value=600.0),
        composite_score=st.floats(min_value=0.0, max_value=100.0),
    )
    @settings(max_examples=50, deadline=None)
    @seed(42)
    def test_precedence_engine_purely_deterministic(
        self, current_price: float, stop_loss: float, target_price: float, composite_score: float
    ):
        """DecisionPrecedenceEngine evaluated twice on identical inputs produces identical results."""
        res1 = self.engine.evaluate(
            candidate_decision="BUY",
            position_state=PositionState.FLAT,
            current_price=current_price,
            stop_loss=stop_loss,
            target_price=target_price,
            atr=current_price * 0.02,
            composite_score=composite_score,
            avg_dollar_volume_20d=10_000_000.0,
            rvol_20d=1.0,
        )
        res2 = self.engine.evaluate(
            candidate_decision="BUY",
            position_state=PositionState.FLAT,
            current_price=current_price,
            stop_loss=stop_loss,
            target_price=target_price,
            atr=current_price * 0.02,
            composite_score=composite_score,
            avg_dollar_volume_20d=10_000_000.0,
            rvol_20d=1.0,
        )
        self.assertEqual(res1.internal_action, res2.internal_action)
        self.assertEqual(res1.external_decision, res2.external_decision)
        self.assertEqual(res1.resolved_stage, res2.resolved_stage)
        self.assertEqual(res1.reason_code, res2.reason_code)


if __name__ == "__main__":
    unittest.main()
