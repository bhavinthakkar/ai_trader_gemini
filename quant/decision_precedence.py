"""
Deterministic Decision Precedence Engine (Step 5.4).

Implements the strict 6-stage decision hierarchy before order generation:
1. Validate data and instrument eligibility (Fail-closed on outage, no silent neutral substitution).
2. Apply mandatory existing-position risk controls (Stops, critical vetos, collapse; NEVER blocked by high score).
3. Apply event, liquidity, and compliance restrictions (Determines fresh entry eligibility).
4. Evaluate existing-position exits and reductions (Target profit-taking, holding-thesis guard).
5. Evaluate new long or short entries (BUY floor, short ceiling, RR, confidence).
6. Return HOLD or NO_TRADE fallback.

Defines five distinct internal action types:
- BUY: Opening or adding to a long position
- HOLD: Maintaining existing state or no-trade
- SELL_TO_CLOSE: Closing / liquidating an existing long position
- OPEN_SHORT: Opening a fresh directional short position
- COVER_SHORT: Buying to close / covering an existing short position
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple, Union

from quant.policy_config import TradingPolicy, get_default_policy, load_policy_config
from quant.trade_geometry import TradeDirection, ExecutionCostModel
from quant.compliance_gates import (
    GateStatus,
    GateResult,
    BuyFloorGate,
    RewardToRiskGate,
    VolatilityEventLiquidityGate,
    SellExitSymmetryGate,
    GatedConfidenceGate,
)


class InternalActionType(str, Enum):
    """Five distinct internal trading action types (Step 5.4)."""
    BUY = "BUY"
    HOLD = "HOLD"
    SELL_TO_CLOSE = "SELL_TO_CLOSE"
    OPEN_SHORT = "OPEN_SHORT"
    COVER_SHORT = "COVER_SHORT"

    def to_external_ui(self) -> Tuple[str, Optional[str]]:
        """
        Maps internal 5-action type to backward-compatible UI decision and sell_type.
        Returns:
            (decision, sell_type)
        """
        if self == InternalActionType.BUY:
            return "BUY", None
        elif self == InternalActionType.HOLD:
            return "HOLD", None
        elif self == InternalActionType.SELL_TO_CLOSE:
            return "SELL", "LONG_EXIT"
        elif self == InternalActionType.OPEN_SHORT:
            return "SELL", "SHORT"
        elif self == InternalActionType.COVER_SHORT:
            return "BUY", "COVER_SHORT"
        return "HOLD", None


class DecisionStage(str, Enum):
    """The 6 sequential stages of decision precedence."""
    STAGE_1_DATA_AND_INSTRUMENT_ELIGIBILITY = "STAGE_1_DATA_AND_INSTRUMENT_ELIGIBILITY"
    STAGE_2_MANDATORY_EXISTING_POSITION_RISK = "STAGE_2_MANDATORY_EXISTING_POSITION_RISK"
    STAGE_3_EVENT_LIQUIDITY_COMPLIANCE_RESTRICTIONS = "STAGE_3_EVENT_LIQUIDITY_COMPLIANCE_RESTRICTIONS"
    STAGE_4_EXISTING_POSITION_EXITS_AND_REDUCTIONS = "STAGE_4_EXISTING_POSITION_EXITS_AND_REDUCTIONS"
    STAGE_5_NEW_ENTRIES = "STAGE_5_NEW_ENTRIES"
    STAGE_6_NO_TRADE_FALLBACK = "STAGE_6_NO_TRADE_FALLBACK"


class PositionState(str, Enum):
    """Position classification of the instrument."""
    FLAT = "FLAT"
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class PrecedenceEvaluationResult:
    """Detailed audit result of the 6-stage decision precedence evaluation."""
    internal_action: InternalActionType
    external_decision: str
    external_sell_type: Optional[str]
    resolved_stage: DecisionStage
    reason_code: str
    message: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    stage_audit_log: List[Dict[str, Any]] = field(default_factory=list)
    passed_gates: List[str] = field(default_factory=list)
    failed_gates: List[str] = field(default_factory=list)
    reason_codes: List[str] = field(default_factory=list)
    data_quality_status: str = "VALID"
    position_action: str = "NO_ACTION"
    model_status: str = "IN_SCOPE"

    def __post_init__(self):
        if not self.position_action or self.position_action == "NO_ACTION":
            action_map = {
                InternalActionType.BUY: "OPEN_LONG",
                InternalActionType.HOLD: "NO_ACTION",
                InternalActionType.SELL_TO_CLOSE: "CLOSE_LONG",
                InternalActionType.OPEN_SHORT: "OPEN_SHORT",
                InternalActionType.COVER_SHORT: "CLOSE_SHORT",
            }
            self.position_action = action_map.get(self.internal_action, "NO_ACTION")

        if not self.reason_codes and self.reason_code:
            code_map = {
                "COMPOSITE_BELOW_FLOOR": "SCORE_BELOW_BUY_THRESHOLD",
                "DIP_BUY_CRITERIA_FAILED": "DIP_CRITERIA_FAILED",
                "VALUE_TRAP_CONFIRMED": "VALUE_TRAP_DETECTED",
                "EARNINGS_BLACKOUT": "EARNINGS_BLACKOUT_ACTIVE",
                "EARNINGS_BLACKOUT_FORCED_EXIT": "EARNINGS_BLACKOUT_ACTIVE",
                "DATA_OUTAGE_NEW_ENTRIES_BLOCKED": "DATA_OUTAGE_BLOCK",
                "DATA_OUTAGE_FAILSAFE_STOP_BREACH": "DATA_OUTAGE_FAILSAFE_EXIT",
                "DATA_OUTAGE_FAILSAFE_SHORT_STOP_BREACH": "DATA_OUTAGE_FAILSAFE_EXIT",
                "DATA_OUTAGE_FAILSAFE_EVIDENCE_VETO": "DATA_OUTAGE_FAILSAFE_EXIT",
                "DATA_OUTAGE_PROTECTIVE_LIQUIDATION": "DATA_OUTAGE_FAILSAFE_EXIT",
                "STOP_LOSS_BREACH": "STOP_LOSS_BREACHED",
                "SHORT_STOP_LOSS_BREACH": "SHORT_STOP_BREACHED",
                "TARGET_ACHIEVED": "PROFIT_TARGET_REACHED",
                "SHORT_TARGET_ACHIEVED": "PROFIT_TARGET_REACHED",
                "HOLDING_THESIS_INTACT": "HOLDING_THESIS_INTACT",
                "THESIS_COLLAPSED": "THESIS_COLLAPSED",
                "SHORT_THESIS_COLLAPSED": "THESIS_COLLAPSED",
                "EVIDENCE_VETO_FORCED_EXIT": "CRITICAL_EVIDENCE_VETO",
                "BORROW_UNAVAILABLE": "BORROW_UNAVAILABLE",
                "SHORT_CEILING_EXCEEDED": "SHORT_CEILING_EXCEEDED",
                "RR_TOO_LOW": "REWARD_TO_RISK_INSUFFICIENT",
                "ZERO_OR_NEGATIVE_REWARD": "REWARD_TO_RISK_INSUFFICIENT",
                "INVALID_PRICE_ORDERING": "PRICE_ORDERING_INVALID",
                "CONFIDENCE_TOO_LOW": "CONFIDENCE_BELOW_HURDLE",
                "BUY_ORDER_CERTIFIED": "BUY_ORDER_CERTIFIED",
                "SHORT_ORDER_CERTIFIED": "SHORT_ORDER_CERTIFIED",
                "NO_TRADE_DEFAULT": "NO_TRADE_DEFAULT",
            }
            mapped_rc = code_map.get(self.reason_code, self.reason_code)
            self.reason_codes = [mapped_rc]

        if not self.passed_gates and not self.failed_gates:
            if "DATA_OUTAGE" in self.reason_code:
                self.failed_gates.append("DATA_QUALITY")
                self.data_quality_status = "OUTAGE"
            else:
                self.passed_gates.append("DATA_QUALITY")

            if self.reason_code in ("COMPOSITE_BELOW_FLOOR", "DIP_BUY_CRITERIA_FAILED", "VALUE_TRAP_CONFIRMED"):
                self.failed_gates.append("BUY_FLOOR")
            elif self.reason_code in ("SHORT_CEILING_EXCEEDED",):
                self.failed_gates.append("SHORT_CEILING")
            elif self.reason_code in ("BORROW_UNAVAILABLE",):
                self.failed_gates.append("BORROW_AVAILABILITY")
            elif self.reason_code in ("RR_TOO_LOW", "ZERO_OR_NEGATIVE_REWARD", "INVALID_PRICE_ORDERING", "NO_STATISTICAL_EDGE"):
                self.failed_gates.append("REWARD_TO_RISK")
            elif self.reason_code in ("CONFIDENCE_TOO_LOW",):
                self.failed_gates.append("CONFIDENCE_HURDLE")
            elif self.reason_code in ("EARNINGS_BLACKOUT", "EVENT_RISK_BLACKOUT", "EX_DIVIDEND_BLACKOUT", "EARNINGS_BLACKOUT_FORCED_EXIT"):
                self.failed_gates.append("EVENT_RISK")
            elif self.reason_code in ("STOP_LOSS_BREACH", "SHORT_STOP_LOSS_BREACH"):
                self.failed_gates.append("STOP_LOSS_SURVEILLANCE")
            elif self.reason_code in ("TARGET_ACHIEVED", "SHORT_TARGET_ACHIEVED"):
                self.passed_gates.append("PROFIT_TARGET")
            elif self.reason_code in ("HOLDING_THESIS_INTACT",):
                self.passed_gates.append("HOLDING_THESIS")
            elif self.reason_code in ("BUY_ORDER_CERTIFIED",):
                self.passed_gates = ["DATA_QUALITY", "EVENT_RISK", "LIQUIDITY", "BUY_FLOOR", "REWARD_TO_RISK", "CONFIDENCE_HURDLE"]
            elif self.reason_code in ("SHORT_ORDER_CERTIFIED",):
                self.passed_gates = ["DATA_QUALITY", "EVENT_RISK", "LIQUIDITY", "SHORT_CEILING", "BORROW_AVAILABILITY", "REWARD_TO_RISK", "CONFIDENCE_HURDLE"]

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["internal_action"] = self.internal_action.value
        res["resolved_stage"] = self.resolved_stage.value
        return res

    def to_structured_decision(
        self,
        ticker: str,
        run_id: Optional[str] = None,
        composite_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Converts precedence evaluation to independent structured decision format (Step 6.4)."""
        reasons = list(self.reason_codes) if self.reason_codes else ([self.reason_code] if self.reason_code else [])
        from quant.structured_decision import StructuredDecision
        dec = StructuredDecision(
            run_id=run_id or "run-default",
            ticker=ticker,
            decision=self.external_decision,
            composite_score=composite_score,
            model_status=self.model_status,
            passed_gates=list(self.passed_gates),
            failed_gates=list(self.failed_gates),
            data_quality_status=self.data_quality_status,
            position_action=self.position_action,
            reason_codes=reasons,
            metrics=self.metrics,
        )
        return dec.to_dict()



class DecisionPrecedenceEngine:
    """
    Deterministic Decision Precedence Orchestrator.
    Executes the 6-stage decision pipeline in strict order.
    """

    def __init__(self, policy: Optional[TradingPolicy] = None):
        self.policy = policy or load_policy_config()

    def evaluate(
        self,
        candidate_decision: str,
        position_state: Union[PositionState, str],
        current_price: Optional[float],
        stop_loss: Optional[float],
        target_price: Optional[float],
        atr: Optional[float],
        composite_score: Optional[float],
        pillar_scores: Optional[Dict[str, Optional[float]]] = None,
        data_completeness: float = 1.0,
        market_data_fresh: bool = True,
        ml_data_fresh: bool = True,
        outage_duration_hours: float = 0.0,
        catalyst_path: Optional[str] = None,
        rsi14: Optional[float] = None,
        drawdown_5d_pct: Optional[float] = None,
        valuation_history_score: Optional[float] = None,
        is_value_trap: bool = False,
        days_to_earnings: Optional[int] = None,
        is_event_risk_blackout: bool = False,
        days_to_ex_dividend: Optional[int] = None,
        iv_pct: Optional[float] = None,
        rvol_20d: Optional[float] = None,
        avg_dollar_volume_20d: Optional[float] = None,
        intended_order_value: float = 25_000.0,
        has_critical_evidence_veto: bool = False,
        execution_costs: Optional[ExecutionCostModel] = None,
        borrow_available: bool = True,
        borrow_status: str = "ETB",
        borrow_fee_bps: float = 0.0,
        available_borrow_shares: Optional[int] = None,
        ema_20: Optional[float] = None,
        ema_50: Optional[float] = None,
        trend_score: Optional[float] = None,
        financial_quality_score: Optional[float] = None,
        ticker: str = "TICKER",
    ) -> PrecedenceEvaluationResult:
        """
        Evaluates trading decision following the strict 6-stage precedence order.
        """
        if isinstance(position_state, str):
            pos_str = position_state.upper()
            if pos_str in ("LONG", "HOLDING", "TRUE", "1"):
                position_state = PositionState.LONG
            elif pos_str in ("SHORT", "BORROWED"):
                position_state = PositionState.SHORT
            else:
                position_state = PositionState.FLAT

        audit_log: List[Dict[str, Any]] = []
        is_open_position = position_state in (PositionState.LONG, PositionState.SHORT)
        price_val = float(current_price) if (current_price is not None and current_price > 0) else None

        # =========================================================================
        # STAGE 1: Validate data and instrument eligibility
        # =========================================================================
        data_outage = (
            not market_data_fresh
            or not ml_data_fresh
            or price_val is None
            or (atr is not None and atr <= 0.0)
        )

        stage1_meta = {
            "stage": DecisionStage.STAGE_1_DATA_AND_INSTRUMENT_ELIGIBILITY.value,
            "data_outage": data_outage,
            "market_data_fresh": market_data_fresh,
            "ml_data_fresh": ml_data_fresh,
            "price_available": price_val is not None,
            "is_open_position": is_open_position,
        }
        audit_log.append(stage1_meta)

        if data_outage:
            # Policy invariant: Never silently substitute neutral scores (50.0)
            if self.policy.data_quality.block_new_entries_on_outage and not is_open_position:
                return PrecedenceEvaluationResult(
                    internal_action=InternalActionType.HOLD,
                    external_decision="HOLD",
                    external_sell_type=None,
                    resolved_stage=DecisionStage.STAGE_1_DATA_AND_INSTRUMENT_ELIGIBILITY,
                    reason_code="DATA_OUTAGE_NEW_ENTRIES_BLOCKED",
                    message="New entries blocked: Market or ML data is missing, stale, or experiencing an outage.",
                    metrics=stage1_meta,
                    stage_audit_log=audit_log,
                )

            # Separate fail-safe policy for open positions during data outage
            if is_open_position and self.policy.data_quality.enable_open_position_failsafe:
                # If emergency price or last known quote indicates stop breach:
                if price_val is not None and stop_loss is not None:
                    if position_state == PositionState.LONG and price_val <= stop_loss:
                        return PrecedenceEvaluationResult(
                            internal_action=InternalActionType.SELL_TO_CLOSE,
                            external_decision="SELL",
                            external_sell_type="LONG_EXIT",
                            resolved_stage=DecisionStage.STAGE_1_DATA_AND_INSTRUMENT_ELIGIBILITY,
                            reason_code="DATA_OUTAGE_FAILSAFE_STOP_BREACH",
                            message=f"Fail-safe exit during data outage: Price ${price_val:.2f} <= stop ${stop_loss:.2f}.",
                            metrics=stage1_meta,
                            stage_audit_log=audit_log,
                        )
                    elif position_state == PositionState.SHORT and price_val >= stop_loss:
                        return PrecedenceEvaluationResult(
                            internal_action=InternalActionType.COVER_SHORT,
                            external_decision="BUY",
                            external_sell_type="COVER_SHORT",
                            resolved_stage=DecisionStage.STAGE_1_DATA_AND_INSTRUMENT_ELIGIBILITY,
                            reason_code="DATA_OUTAGE_FAILSAFE_SHORT_STOP_BREACH",
                            message=f"Fail-safe short cover during data outage: Price ${price_val:.2f} >= stop ${stop_loss:.2f}.",
                            metrics=stage1_meta,
                            stage_audit_log=audit_log,
                        )

                # Critical evidence veto during outage forces liquidation regardless of quote feed
                if has_critical_evidence_veto:
                    action = (
                        InternalActionType.SELL_TO_CLOSE
                        if position_state == PositionState.LONG
                        else InternalActionType.COVER_SHORT
                    )
                    ext_dec, ext_sell = action.to_external_ui()
                    return PrecedenceEvaluationResult(
                        internal_action=action,
                        external_decision=ext_dec,
                        external_sell_type=ext_sell,
                        resolved_stage=DecisionStage.STAGE_1_DATA_AND_INSTRUMENT_ELIGIBILITY,
                        reason_code="DATA_OUTAGE_FAILSAFE_EVIDENCE_VETO",
                        message="Fail-safe exit during data outage: Critical evidence veto detected.",
                        metrics=stage1_meta,
                        stage_audit_log=audit_log,
                    )

                # Outage duration exceeds tolerance threshold during trading sessions
                if outage_duration_hours >= self.policy.data_quality.max_outage_tolerance_hours:
                    action = (
                        InternalActionType.SELL_TO_CLOSE
                        if position_state == PositionState.LONG
                        else InternalActionType.COVER_SHORT
                    )
                    ext_dec, ext_sell = action.to_external_ui()
                    return PrecedenceEvaluationResult(
                        internal_action=action,
                        external_decision=ext_dec,
                        external_sell_type=ext_sell,
                        resolved_stage=DecisionStage.STAGE_1_DATA_AND_INSTRUMENT_ELIGIBILITY,
                        reason_code="DATA_OUTAGE_PROTECTIVE_LIQUIDATION",
                        message=(
                            f"Fail-safe protective liquidation: Outage duration {outage_duration_hours:.1f}h "
                            f"exceeded maximum tolerance {self.policy.data_quality.max_outage_tolerance_hours:.1f}h."
                        ),
                        metrics=stage1_meta,
                        stage_audit_log=audit_log,
                    )

        # =========================================================================
        # STAGE 2: Apply mandatory existing-position risk controls
        # =========================================================================
        if is_open_position:
            stage2_meta = {
                "stage": DecisionStage.STAGE_2_MANDATORY_EXISTING_POSITION_RISK.value,
                "position_state": position_state.value,
                "price": price_val,
                "stop_loss": stop_loss,
                "composite_score": composite_score,
                "has_critical_evidence_veto": has_critical_evidence_veto,
            }
            audit_log.append(stage2_meta)

            # Invariant: Mandatory risk controls CANNOT be blocked by a high score
            # 1. Stop Loss Breach
            if price_val is not None and stop_loss is not None:
                if position_state == PositionState.LONG and price_val <= stop_loss:
                    return PrecedenceEvaluationResult(
                        internal_action=InternalActionType.SELL_TO_CLOSE,
                        external_decision="SELL",
                        external_sell_type="LONG_EXIT",
                        resolved_stage=DecisionStage.STAGE_2_MANDATORY_EXISTING_POSITION_RISK,
                        reason_code="STOP_LOSS_BREACH",
                        message=f"Mandatory stop-loss breach: Price ${price_val:.2f} <= stop ${stop_loss:.2f}.",
                        metrics=stage2_meta,
                        stage_audit_log=audit_log,
                    )
                elif position_state == PositionState.SHORT and price_val >= stop_loss:
                    return PrecedenceEvaluationResult(
                        internal_action=InternalActionType.COVER_SHORT,
                        external_decision="BUY",
                        external_sell_type="COVER_SHORT",
                        resolved_stage=DecisionStage.STAGE_2_MANDATORY_EXISTING_POSITION_RISK,
                        reason_code="SHORT_STOP_LOSS_BREACH",
                        message=f"Mandatory short stop breach: Price ${price_val:.2f} >= stop ${stop_loss:.2f}.",
                        metrics=stage2_meta,
                        stage_audit_log=audit_log,
                    )

            # 2. Critical Qualitative Evidence Veto
            if has_critical_evidence_veto:
                action = (
                    InternalActionType.SELL_TO_CLOSE
                    if position_state == PositionState.LONG
                    else InternalActionType.COVER_SHORT
                )
                ext_dec, ext_sell = action.to_external_ui()
                return PrecedenceEvaluationResult(
                    internal_action=action,
                    external_decision=ext_dec,
                    external_sell_type=ext_sell,
                    resolved_stage=DecisionStage.STAGE_2_MANDATORY_EXISTING_POSITION_RISK,
                    reason_code="EVIDENCE_VETO_FORCED_EXIT",
                    message="Mandatory liquidation: Critical qualitative risk flag or regulatory investigation.",
                    metrics=stage2_meta,
                    stage_audit_log=audit_log,
                )

            # 3. Scheduled Binary Earnings Blackout Exit
            if (
                days_to_earnings is not None
                and days_to_earnings <= self.policy.events.earnings_blackout_sessions
                and self.policy.holding_guard.never_block_earnings_blackout
            ):
                action = (
                    InternalActionType.SELL_TO_CLOSE
                    if position_state == PositionState.LONG
                    else InternalActionType.COVER_SHORT
                )
                ext_dec, ext_sell = action.to_external_ui()
                return PrecedenceEvaluationResult(
                    internal_action=action,
                    external_decision=ext_dec,
                    external_sell_type=ext_sell,
                    resolved_stage=DecisionStage.STAGE_2_MANDATORY_EXISTING_POSITION_RISK,
                    reason_code="EARNINGS_BLACKOUT_FORCED_EXIT",
                    message=f"Mandatory liquidation: Earnings report in {days_to_earnings} session(s) is binary gap risk.",
                    metrics=stage2_meta,
                    stage_audit_log=audit_log,
                )

            # 4. Catastrophic Thesis Collapse (composite < 50 for long, > 60 for short)
            if composite_score is not None:
                if (
                    position_state == PositionState.LONG
                    and float(composite_score) < self.policy.holding_guard.catastrophic_composite_floor
                ):
                    return PrecedenceEvaluationResult(
                        internal_action=InternalActionType.SELL_TO_CLOSE,
                        external_decision="SELL",
                        external_sell_type="LONG_EXIT",
                        resolved_stage=DecisionStage.STAGE_2_MANDATORY_EXISTING_POSITION_RISK,
                        reason_code="THESIS_COLLAPSED",
                        message=(
                            f"Mandatory liquidation: Composite score {float(composite_score):.1f} collapsed "
                            f"below floor {self.policy.holding_guard.catastrophic_composite_floor:.1f}."
                        ),
                        metrics=stage2_meta,
                        stage_audit_log=audit_log,
                    )
                elif position_state == PositionState.SHORT and float(composite_score) > 60.0:
                    return PrecedenceEvaluationResult(
                        internal_action=InternalActionType.COVER_SHORT,
                        external_decision="BUY",
                        external_sell_type="COVER_SHORT",
                        resolved_stage=DecisionStage.STAGE_2_MANDATORY_EXISTING_POSITION_RISK,
                        reason_code="SHORT_THESIS_COLLAPSED",
                        message=f"Mandatory short cover: Composite score {float(composite_score):.1f} recovered above 60.0.",
                        metrics=stage2_meta,
                        stage_audit_log=audit_log,
                    )

        # =========================================================================
        # STAGE 3: Apply event, liquidity and compliance restrictions (for new entries)
        # =========================================================================
        new_entry_candidate = not is_open_position and candidate_decision.upper() in ("BUY", "SELL")
        if new_entry_candidate and price_val is not None:
            dir_eval = TradeDirection.LONG if candidate_decision.upper() == "BUY" else TradeDirection.SHORT
            atr_v = atr if (atr is not None and atr > 0) else (0.02 * price_val)
            stop_v = stop_loss if stop_loss is not None else (
                price_val - 1.5 * atr_v if dir_eval == TradeDirection.LONG else price_val + 1.5 * atr_v
            )

            g3_res = VolatilityEventLiquidityGate.evaluate(
                direction=dir_eval,
                price=price_val,
                stop_loss=stop_v,
                atr=atr_v,
                days_to_earnings=days_to_earnings,
                is_event_risk_blackout=is_event_risk_blackout,
                days_to_ex_dividend=days_to_ex_dividend,
                iv_pct=iv_pct,
                rvol_20d=rvol_20d,
                avg_dollar_volume_20d=avg_dollar_volume_20d,
                intended_order_value=intended_order_value,
                policy=self.policy,
            )
            audit_log.append({
                "stage": DecisionStage.STAGE_3_EVENT_LIQUIDITY_COMPLIANCE_RESTRICTIONS.value,
                "gate_3_passed": g3_res.passed,
                "reason_code": g3_res.reason_code,
                "metrics": g3_res.metrics,
            })

            if not g3_res.passed:
                return PrecedenceEvaluationResult(
                    internal_action=InternalActionType.HOLD,
                    external_decision="HOLD",
                    external_sell_type=None,
                    resolved_stage=DecisionStage.STAGE_3_EVENT_LIQUIDITY_COMPLIANCE_RESTRICTIONS,
                    reason_code=g3_res.reason_code or "RESTRICTION_ACTIVE",
                    message=f"New entry ineligible: {g3_res.message}",
                    metrics=g3_res.metrics,
                    stage_audit_log=audit_log,
                )

        # =========================================================================
        # STAGE 4: Evaluate existing-position exits and reductions
        # =========================================================================
        if is_open_position:
            stage4_meta = {
                "stage": DecisionStage.STAGE_4_EXISTING_POSITION_EXITS_AND_REDUCTIONS.value,
                "position_state": position_state.value,
                "price": price_val,
                "target_price": target_price,
                "candidate_decision": candidate_decision,
            }
            audit_log.append(stage4_meta)

            # 1. Profit Target Reached
            if price_val is not None and target_price is not None:
                if position_state == PositionState.LONG and price_val >= target_price:
                    return PrecedenceEvaluationResult(
                        internal_action=InternalActionType.SELL_TO_CLOSE,
                        external_decision="SELL",
                        external_sell_type="LONG_EXIT",
                        resolved_stage=DecisionStage.STAGE_4_EXISTING_POSITION_EXITS_AND_REDUCTIONS,
                        reason_code="TARGET_ACHIEVED",
                        message=f"Target achieved: Current price ${price_val:.2f} >= target ${target_price:.2f}.",
                        metrics=stage4_meta,
                        stage_audit_log=audit_log,
                    )
                elif position_state == PositionState.SHORT and price_val <= target_price:
                    return PrecedenceEvaluationResult(
                        internal_action=InternalActionType.COVER_SHORT,
                        external_decision="BUY",
                        external_sell_type="COVER_SHORT",
                        resolved_stage=DecisionStage.STAGE_4_EXISTING_POSITION_EXITS_AND_REDUCTIONS,
                        reason_code="SHORT_TARGET_ACHIEVED",
                        message=f"Short target achieved: Current price ${price_val:.2f} <= target ${target_price:.2f}.",
                        metrics=stage4_meta,
                        stage_audit_log=audit_log,
                    )

            # 2. Holding-Thesis Guard Evaluation (Prevents premature exits when thesis is intact)
            if position_state == PositionState.LONG and candidate_decision.upper() == "SELL":
                g4_exit = SellExitSymmetryGate.evaluate(
                    candidate_decision="SELL",
                    is_holding=True,
                    current_price=price_val or 0.0,
                    stop_loss=stop_loss,
                    target_price=target_price,
                    composite_score=composite_score,
                    has_critical_evidence_veto=has_critical_evidence_veto,
                    is_earnings_blackout=(
                        days_to_earnings is not None
                        and days_to_earnings <= self.policy.events.earnings_blackout_sessions
                    ),
                    policy=self.policy,
                )
                if not g4_exit.passed and g4_exit.reason_code == "HOLDING_THESIS_INTACT":
                    return PrecedenceEvaluationResult(
                        internal_action=InternalActionType.HOLD,
                        external_decision="HOLD",
                        external_sell_type=None,
                        resolved_stage=DecisionStage.STAGE_4_EXISTING_POSITION_EXITS_AND_REDUCTIONS,
                        reason_code="HOLDING_THESIS_INTACT",
                        message=g4_exit.message,
                        metrics=g4_exit.metrics,
                        stage_audit_log=audit_log,
                    )
                else:
                    return PrecedenceEvaluationResult(
                        internal_action=InternalActionType.SELL_TO_CLOSE,
                        external_decision="SELL",
                        external_sell_type="LONG_EXIT",
                        resolved_stage=DecisionStage.STAGE_4_EXISTING_POSITION_EXITS_AND_REDUCTIONS,
                        reason_code=g4_exit.reason_code or "DISCRETIONARY_EXIT",
                        message=g4_exit.message,
                        metrics=g4_exit.metrics,
                        stage_audit_log=audit_log,
                    )

            # If existing position was evaluated and no exit or target triggered, maintain position
            return PrecedenceEvaluationResult(
                internal_action=InternalActionType.HOLD,
                external_decision="HOLD",
                external_sell_type=None,
                resolved_stage=DecisionStage.STAGE_4_EXISTING_POSITION_EXITS_AND_REDUCTIONS,
                reason_code="POSITION_MAINTAINED",
                message=f"Active {position_state.value} position maintained: thesis intact, stop not breached.",
                metrics=stage4_meta,
                stage_audit_log=audit_log,
            )

        # =========================================================================
        # STAGE 5: Evaluate new long or short entries
        # =========================================================================
        stage5_meta = {
            "stage": DecisionStage.STAGE_5_NEW_ENTRIES.value,
            "candidate_decision": candidate_decision,
            "composite_score": composite_score,
        }

        # Branch 5A: New Long Entry (BUY)
        if candidate_decision.upper() == "BUY":
            # 1. Gate 1: BUY Floor
            g1_res = BuyFloorGate.evaluate(
                candidate_decision="BUY",
                composite_score=composite_score,
                catalyst_path=catalyst_path,
                rsi14=rsi14,
                drawdown_5d_pct=drawdown_5d_pct,
                valuation_history_score=valuation_history_score,
                is_value_trap=is_value_trap,
                policy=self.policy,
            )
            if not g1_res.passed:
                audit_log.append({"stage_5_gate": "Gate 1", "passed": False, "reason": g1_res.reason_code})
                return PrecedenceEvaluationResult(
                    internal_action=InternalActionType.HOLD,
                    external_decision="HOLD",
                    external_sell_type=None,
                    resolved_stage=DecisionStage.STAGE_5_NEW_ENTRIES,
                    reason_code=g1_res.reason_code or "COMPOSITE_BELOW_FLOOR",
                    message=g1_res.message,
                    metrics=g1_res.metrics,
                    stage_audit_log=audit_log,
                )

            # 2. Gate 2: Reward-to-Risk
            if price_val is not None and stop_loss is not None and target_price is not None:
                g2_res = RewardToRiskGate.evaluate(
                    direction=TradeDirection.LONG,
                    entry_price=price_val,
                    stop_loss=stop_loss,
                    target_price=target_price,
                    composite_score=composite_score,
                    policy=self.policy,
                    execution_costs=execution_costs,
                )
                if not g2_res.passed:
                    audit_log.append({"stage_5_gate": "Gate 2", "passed": False, "reason": g2_res.reason_code})
                    return PrecedenceEvaluationResult(
                        internal_action=InternalActionType.HOLD,
                        external_decision="HOLD",
                        external_sell_type=None,
                        resolved_stage=DecisionStage.STAGE_5_NEW_ENTRIES,
                        reason_code=g2_res.reason_code or "RR_TOO_LOW",
                        message=g2_res.message,
                        metrics=g2_res.metrics,
                        stage_audit_log=audit_log,
                    )

            is_certified_dip_buy = bool(g1_res.metrics.get("dip_certified", False))

            # 3. Gate 5: Gated Confidence
            g5_res = GatedConfidenceGate.evaluate(
                candidate_decision="BUY",
                composite_score=composite_score,
                pillar_scores=pillar_scores,
                data_completeness=data_completeness,
                policy=self.policy,
                is_certified_dip_buy=is_certified_dip_buy,
            )
            if not g5_res.passed:
                audit_log.append({"stage_5_gate": "Gate 5", "passed": False, "reason": g5_res.reason_code})
                return PrecedenceEvaluationResult(
                    internal_action=InternalActionType.HOLD,
                    external_decision="HOLD",
                    external_sell_type=None,
                    resolved_stage=DecisionStage.STAGE_5_NEW_ENTRIES,
                    reason_code=g5_res.reason_code or "CONFIDENCE_TOO_LOW",
                    message=g5_res.message,
                    metrics=g5_res.metrics,
                    stage_audit_log=audit_log,
                )

            # All gates passed -> Order Certified!
            return PrecedenceEvaluationResult(
                internal_action=InternalActionType.BUY,
                external_decision="BUY",
                external_sell_type=None,
                resolved_stage=DecisionStage.STAGE_5_NEW_ENTRIES,
                reason_code="BUY_ORDER_CERTIFIED",
                message="Long trade certified across all deterministic compliance gates.",
                metrics=stage5_meta,
                stage_audit_log=audit_log,
            )

        # Branch 5B: New Short Entry (OPEN_SHORT)
        elif candidate_decision.upper() == "SELL":
            # 1. Gate 4: Short Ceiling Check & Borrow Availability
            g4_short = SellExitSymmetryGate.evaluate(
                candidate_decision="SELL",
                is_holding=False,
                current_price=price_val or 0.0,
                stop_loss=stop_loss,
                target_price=target_price,
                composite_score=composite_score,
                borrow_available=borrow_available,
                borrow_status=borrow_status,
                borrow_fee_bps=borrow_fee_bps,
                available_borrow_shares=available_borrow_shares,
                rsi=rsi14,
                ema_20=ema_20,
                ema_50=ema_50,
                trend_score=trend_score,
                financial_quality_score=financial_quality_score,
                valuation_history_score=valuation_history_score,
                days_to_ex_dividend=days_to_ex_dividend,
                ticker=ticker,
                policy=self.policy,
            )
            if not g4_short.passed:
                audit_log.append({"stage_5_gate": "Gate 4 Short", "passed": False, "reason": g4_short.reason_code})
                return PrecedenceEvaluationResult(
                    internal_action=InternalActionType.HOLD,
                    external_decision="HOLD",
                    external_sell_type=None,
                    resolved_stage=DecisionStage.STAGE_5_NEW_ENTRIES,
                    reason_code=g4_short.reason_code or "SHORT_CEILING_EXCEEDED",
                    message=g4_short.message,
                    metrics=g4_short.metrics,
                    stage_audit_log=audit_log,
                )

            # 2. Gate 2: Short Reward-to-Risk
            if price_val is not None and stop_loss is not None and target_price is not None:
                g2_short = RewardToRiskGate.evaluate(
                    direction=TradeDirection.SHORT,
                    entry_price=price_val,
                    stop_loss=stop_loss,
                    target_price=target_price,
                    composite_score=composite_score,
                    policy=self.policy,
                    execution_costs=execution_costs,
                )
                if not g2_short.passed:
                    audit_log.append({"stage_5_gate": "Gate 2 Short", "passed": False, "reason": g2_short.reason_code})
                    return PrecedenceEvaluationResult(
                        internal_action=InternalActionType.HOLD,
                        external_decision="HOLD",
                        external_sell_type=None,
                        resolved_stage=DecisionStage.STAGE_5_NEW_ENTRIES,
                        reason_code=g2_short.reason_code or "RR_TOO_LOW",
                        message=g2_short.message,
                        metrics=g2_short.metrics,
                        stage_audit_log=audit_log,
                    )

            # 3. Gate 5: Gated Confidence
            g5_short = GatedConfidenceGate.evaluate(
                candidate_decision="SELL",
                composite_score=composite_score,
                pillar_scores=pillar_scores,
                data_completeness=data_completeness,
                policy=self.policy,
            )
            if not g5_short.passed:
                audit_log.append({"stage_5_gate": "Gate 5 Short", "passed": False, "reason": g5_short.reason_code})
                return PrecedenceEvaluationResult(
                    internal_action=InternalActionType.HOLD,
                    external_decision="HOLD",
                    external_sell_type=None,
                    resolved_stage=DecisionStage.STAGE_5_NEW_ENTRIES,
                    reason_code=g5_short.reason_code or "CONFIDENCE_TOO_LOW",
                    message=g5_short.message,
                    metrics=g5_short.metrics,
                    stage_audit_log=audit_log,
                )

            # All short gates passed -> Short Order Certified!
            return PrecedenceEvaluationResult(
                internal_action=InternalActionType.OPEN_SHORT,
                external_decision="SELL",
                external_sell_type="SHORT",
                resolved_stage=DecisionStage.STAGE_5_NEW_ENTRIES,
                reason_code="SHORT_ORDER_CERTIFIED",
                message="Directional short trade certified across all deterministic compliance gates.",
                metrics=stage5_meta,
                stage_audit_log=audit_log,
            )

        # =========================================================================
        # STAGE 6: Return HOLD or NO_TRADE if no valid action remains
        # =========================================================================
        audit_log.append({"stage": DecisionStage.STAGE_6_NO_TRADE_FALLBACK.value})
        return PrecedenceEvaluationResult(
            internal_action=InternalActionType.HOLD,
            external_decision="HOLD",
            external_sell_type=None,
            resolved_stage=DecisionStage.STAGE_6_NO_TRADE_FALLBACK,
            reason_code="NO_TRADE_DEFAULT",
            message="No valid entry or exit action certified; defaulting to HOLD.",
            metrics={"candidate_decision": candidate_decision},
            stage_audit_log=audit_log,
        )


def evaluate_decision_precedence(
    candidate_decision: str,
    position_state: Union[PositionState, str],
    current_price: Optional[float],
    stop_loss: Optional[float],
    target_price: Optional[float],
    atr: Optional[float],
    composite_score: Optional[float],
    pillar_scores: Optional[Dict[str, Optional[float]]] = None,
    data_completeness: float = 1.0,
    market_data_fresh: bool = True,
    ml_data_fresh: bool = True,
    outage_duration_hours: float = 0.0,
    catalyst_path: Optional[str] = None,
    rsi14: Optional[float] = None,
    drawdown_5d_pct: Optional[float] = None,
    valuation_history_score: Optional[float] = None,
    is_value_trap: bool = False,
    days_to_earnings: Optional[int] = None,
    is_event_risk_blackout: bool = False,
    days_to_ex_dividend: Optional[int] = None,
    iv_pct: Optional[float] = None,
    rvol_20d: Optional[float] = None,
    avg_dollar_volume_20d: Optional[float] = None,
    intended_order_value: float = 25_000.0,
    has_critical_evidence_veto: bool = False,
    execution_costs: Optional[ExecutionCostModel] = None,
    borrow_available: bool = True,
    policy: Optional[TradingPolicy] = None,
    borrow_status: str = "ETB",
    borrow_fee_bps: float = 0.0,
    available_borrow_shares: Optional[int] = None,
    ema_20: Optional[float] = None,
    ema_50: Optional[float] = None,
    trend_score: Optional[float] = None,
    financial_quality_score: Optional[float] = None,
    ticker: str = "TICKER",
) -> PrecedenceEvaluationResult:
    """Convenience wrapper for DecisionPrecedenceEngine evaluation."""
    engine = DecisionPrecedenceEngine(policy)
    return engine.evaluate(
        candidate_decision=candidate_decision,
        position_state=position_state,
        current_price=current_price,
        stop_loss=stop_loss,
        target_price=target_price,
        atr=atr,
        composite_score=composite_score,
        pillar_scores=pillar_scores,
        data_completeness=data_completeness,
        market_data_fresh=market_data_fresh,
        ml_data_fresh=ml_data_fresh,
        outage_duration_hours=outage_duration_hours,
        catalyst_path=catalyst_path,
        rsi14=rsi14,
        drawdown_5d_pct=drawdown_5d_pct,
        valuation_history_score=valuation_history_score,
        is_value_trap=is_value_trap,
        days_to_earnings=days_to_earnings,
        is_event_risk_blackout=is_event_risk_blackout,
        days_to_ex_dividend=days_to_ex_dividend,
        iv_pct=iv_pct,
        rvol_20d=rvol_20d,
        avg_dollar_volume_20d=avg_dollar_volume_20d,
        intended_order_value=intended_order_value,
        has_critical_evidence_veto=has_critical_evidence_veto,
        execution_costs=execution_costs,
        borrow_available=borrow_available,
        borrow_status=borrow_status,
        borrow_fee_bps=borrow_fee_bps,
        available_borrow_shares=available_borrow_shares,
        ema_20=ema_20,
        ema_50=ema_50,
        trend_score=trend_score,
        financial_quality_score=financial_quality_score,
        ticker=ticker,
    )
