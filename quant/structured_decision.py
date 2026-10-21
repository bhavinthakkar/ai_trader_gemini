"""
Deterministic Structured Decision Engine (Step 6.4).

Separates the investment decision from any rendered narrative or explanation.
Produces a pure, machine-readable, reproducible structured decision contract:
{
  "run_id": "example-run-id",
  "ticker": "NVDA",
  "decision": "HOLD",
  "composite_score": 58.4,
  "model_status": "IN_SCOPE",
  "passed_gates": [],
  "failed_gates": ["BUY_FLOOR"],
  "data_quality_status": "VALID",
  "position_action": "NO_ACTION",
  "reason_codes": ["SCORE_BELOW_BUY_THRESHOLD"]
}

Mandatory gates evaluated:
- DATA_QUALITY: Data freshness and outage surveillance.
- INSUFFICIENT_PILLARS: Minimum 3 of 5 pillars available.
- EVENT_RISK: Binary gap risk (earnings blackout, scheduled corporate events).
- LIQUIDITY: Tradeable volume (RVOL, ADV).
- BUY_FLOOR: Minimum score threshold for long entry (60.0 standard / 52.0 dip-buy).
- SHORT_CEILING: Maximum score threshold for short entry (<= 45.0).
- BORROW_AVAILABILITY: Locate & borrow availability for short sales.
- REWARD_TO_RISK: Valid price geometry, non-negative risk, threshold RR, and statistical edge.
- CONFIDENCE_HURDLE: Signal strength clearing directional entry hurdle (>= 0.65).
- STOP_LOSS_SURVEILLANCE: Inviolable stop loss risk control on open positions.
- HOLDING_THESIS: Inviolable holding-thesis guard protecting winning positions.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from typing import Any, Dict, List, Optional, Union
import uuid

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


class PositionAction(str, Enum):
    """Structured machine-readable action type for portfolio order management."""
    NO_ACTION = "NO_ACTION"
    OPEN_LONG = "OPEN_LONG"
    CLOSE_LONG = "CLOSE_LONG"
    OPEN_SHORT = "OPEN_SHORT"
    CLOSE_SHORT = "CLOSE_SHORT"
    TRIM_LONG = "TRIM_LONG"


class DataQualityStatus(str, Enum):
    """Categorical status of input data quality and integrity."""
    VALID = "VALID"
    PARTIAL = "PARTIAL"
    STALE = "STALE"
    OUTAGE = "OUTAGE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class ModelScopeStatus(str, Enum):
    """Categorical ML model scope status."""
    IN_SCOPE = "IN_SCOPE"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    STALE_SCORE = "STALE_SCORE"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
    LOOKAHEAD_VIOLATION = "LOOKAHEAD_VIOLATION"
    UNMODELED = "UNMODELED"


class StandardGateName:
    """Standardized identifier strings for all mandatory gates."""
    DATA_QUALITY = "DATA_QUALITY"
    INSUFFICIENT_PILLARS = "INSUFFICIENT_PILLARS"
    EVENT_RISK = "EVENT_RISK"
    LIQUIDITY = "LIQUIDITY"
    BUY_FLOOR = "BUY_FLOOR"
    SHORT_CEILING = "SHORT_CEILING"
    BORROW_AVAILABILITY = "BORROW_AVAILABILITY"
    REWARD_TO_RISK = "REWARD_TO_RISK"
    CONFIDENCE_HURDLE = "CONFIDENCE_HURDLE"
    STOP_LOSS_SURVEILLANCE = "STOP_LOSS_SURVEILLANCE"
    HOLDING_THESIS = "HOLDING_THESIS"
    PROFIT_TARGET = "PROFIT_TARGET"


class StandardReasonCode:
    """Standardized, machine-readable uppercase reason codes."""
    # Long Entry Gate Codes
    SCORE_BELOW_BUY_THRESHOLD = "SCORE_BELOW_BUY_THRESHOLD"
    DIP_CRITERIA_FAILED = "DIP_CRITERIA_FAILED"
    VALUE_TRAP_DETECTED = "VALUE_TRAP_DETECTED"
    BUY_ORDER_CERTIFIED = "BUY_ORDER_CERTIFIED"

    # Short Entry Gate Codes
    SHORT_CEILING_EXCEEDED = "SHORT_CEILING_EXCEEDED"
    BORROW_UNAVAILABLE = "BORROW_UNAVAILABLE"
    SHORT_ORDER_CERTIFIED = "SHORT_ORDER_CERTIFIED"

    # Geometry & Edge Codes
    REWARD_TO_RISK_INSUFFICIENT = "REWARD_TO_RISK_INSUFFICIENT"
    PRICE_ORDERING_INVALID = "PRICE_ORDERING_INVALID"
    NEGATIVE_OR_ZERO_RISK = "NEGATIVE_OR_ZERO_RISK"
    STATISTICAL_EDGE_INSUFFICIENT = "STATISTICAL_EDGE_INSUFFICIENT"

    # Event & Liquidity Codes
    EARNINGS_BLACKOUT_ACTIVE = "EARNINGS_BLACKOUT_ACTIVE"
    EVENT_RISK_BLACKOUT = "EVENT_RISK_BLACKOUT"
    EX_DIVIDEND_BLACKOUT = "EX_DIVIDEND_BLACKOUT"
    LIQUIDITY_INSUFFICIENT = "LIQUIDITY_INSUFFICIENT"
    STOP_INSIDE_VOLATILITY_NOISE = "STOP_INSIDE_VOLATILITY_NOISE"
    VOLATILITY_REGIME_EXTREME = "VOLATILITY_REGIME_EXTREME"

    # Position Management Codes
    STOP_LOSS_BREACHED = "STOP_LOSS_BREACHED"
    SHORT_STOP_BREACHED = "SHORT_STOP_BREACHED"
    PROFIT_TARGET_REACHED = "PROFIT_TARGET_REACHED"
    HOLDING_THESIS_INTACT = "HOLDING_THESIS_INTACT"
    THESIS_COLLAPSED = "THESIS_COLLAPSED"
    CRITICAL_EVIDENCE_VETO = "CRITICAL_EVIDENCE_VETO"

    # Data Quality & Fail-Safe Codes
    DATA_OUTAGE_BLOCK = "DATA_OUTAGE_BLOCK"
    DATA_OUTAGE_FAILSAFE_EXIT = "DATA_OUTAGE_FAILSAFE_EXIT"
    INSUFFICIENT_PILLARS = "INSUFFICIENT_PILLARS"
    CONFIDENCE_BELOW_HURDLE = "CONFIDENCE_BELOW_HURDLE"

    # LLM & Contract Safety Codes
    PROHIBITED_BROKER_COMMAND = "PROHIBITED_BROKER_COMMAND"
    UNSUPPORTED_ACTION = "UNSUPPORTED_ACTION"
    SCHEMA_VALIDATION_FAILED = "SCHEMA_VALIDATION_FAILED"

    # General Fallback
    NO_TRADE_DEFAULT = "NO_TRADE_DEFAULT"


@dataclass
class StructuredDecision:
    """
    Pure structured investment decision contract (Step 6.4).
    Contains no free-form narrative explanations; derived deterministically
    from mathematical components and policy configuration.
    """
    run_id: str
    ticker: str
    decision: str                                   # "BUY", "HOLD", "SELL"
    composite_score: Optional[float]                # e.g. 58.4
    model_status: str                              # e.g. "IN_SCOPE", "OUT_OF_SCOPE"
    passed_gates: List[str]                        # e.g. ["DATA_QUALITY", "EVENT_RISK"]
    failed_gates: List[str]                        # e.g. ["BUY_FLOOR"]
    data_quality_status: str                       # e.g. "VALID", "OUTAGE"
    position_action: str                           # e.g. "NO_ACTION", "OPEN_LONG", "CLOSE_LONG"
    reason_codes: List[str]                        # e.g. ["SCORE_BELOW_BUY_THRESHOLD"]
    timestamp: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if self.composite_score is not None:
            self.composite_score = round(float(self.composite_score), 2)

    def to_dict(self) -> Dict[str, Any]:
        """Returns clean serializable dictionary."""
        return {
            "run_id": str(self.run_id),
            "ticker": str(self.ticker).upper(),
            "decision": str(self.decision).upper(),
            "composite_score": self.composite_score,
            "model_status": str(self.model_status),
            "passed_gates": list(self.passed_gates),
            "failed_gates": list(self.failed_gates),
            "data_quality_status": str(self.data_quality_status),
            "position_action": str(self.position_action),
            "reason_codes": list(self.reason_codes),
            "timestamp": self.timestamp,
            "metrics": dict(self.metrics),
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serializes structured decision to deterministic JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


class StructuredDecisionEngine:
    """
    Deterministic engine that derives structured decisions independently of narratives.
    """

    def __init__(self, policy: Optional[TradingPolicy] = None):
        self.policy = policy or load_policy_config()

    def evaluate(
        self,
        ticker: str,
        candidate_decision: str,
        composite_score: Optional[float],
        position_state: str = "FLAT",
        current_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        target_price: Optional[float] = None,
        atr: Optional[float] = None,
        pillar_scores: Optional[Dict[str, Optional[float]]] = None,
        data_completeness: float = 1.0,
        market_data_fresh: bool = True,
        ml_data_fresh: bool = True,
        outage_duration_hours: float = 0.0,
        days_to_earnings: Optional[int] = None,
        is_event_risk_blackout: bool = False,
        days_to_ex_dividend: Optional[int] = None,
        iv_pct: Optional[float] = None,
        rvol_20d: Optional[float] = None,
        avg_dollar_volume_20d: Optional[float] = None,
        borrow_available: bool = True,
        has_critical_evidence_veto: bool = False,
        catalyst_path: Optional[str] = None,
        rsi14: Optional[float] = None,
        drawdown_5d_pct: Optional[float] = None,
        valuation_history_score: Optional[float] = None,
        is_value_trap: bool = False,
        model_status: Optional[str] = None,
        run_id: Optional[str] = None,
        intended_order_value: float = 25_000.0,
        execution_costs: Optional[ExecutionCostModel] = None,
        portfolio_equity: Optional[float] = None,
        risk_fraction: Optional[float] = None,
        mandate: Optional[Any] = None,
    ) -> StructuredDecision:
        """
        Evaluates the trade setup across all mandatory compliance gates and derives
        the exact structured decision without free-form natural language.
        """
        run_id_val = run_id or f"run-{uuid.uuid4().hex[:12]}"
        ticker_val = str(ticker).strip().upper()
        cand_dec = str(candidate_decision).strip().upper()

        passed_gates: List[str] = []
        failed_gates: List[str] = []
        reason_codes: List[str] = []
        metrics: Dict[str, Any] = {}

        # ---------------------------------------------------------------------
        # 1. Determine Model Status
        # ---------------------------------------------------------------------
        if model_status:
            eff_model_status = str(model_status).upper()
        elif not ml_data_fresh:
            eff_model_status = ModelScopeStatus.STALE_SCORE.value
        else:
            eff_model_status = ModelScopeStatus.IN_SCOPE.value

        # ---------------------------------------------------------------------
        # 2. Determine Data Quality Status
        # ---------------------------------------------------------------------
        price_val = float(current_price) if (current_price is not None and current_price > 0) else None
        atr_val = float(atr) if (atr is not None and atr > 0) else None

        # Check pillar count
        avail_pillars = 0
        if pillar_scores:
            avail_pillars = sum(1 for v in pillar_scores.values() if v is not None)

        if not market_data_fresh or price_val is None or (atr is not None and atr <= 0):
            data_quality_status = DataQualityStatus.OUTAGE.value
        elif pillar_scores is not None and avail_pillars < getattr(self.policy.data_quality, "min_valid_pillars", 3):
            data_quality_status = DataQualityStatus.INSUFFICIENT_DATA.value
        elif not ml_data_fresh or data_completeness < 1.0 or (pillar_scores is not None and avail_pillars < 5):
            data_quality_status = DataQualityStatus.PARTIAL.value
        else:
            data_quality_status = DataQualityStatus.VALID.value

        pos_str = str(position_state).strip().upper()
        is_open_long = pos_str in ("LONG", "HOLDING", "TRUE", "1")
        is_open_short = pos_str in ("SHORT", "BORROWED")
        is_open_position = is_open_long or is_open_short

        # ---------------------------------------------------------------------
        # STAGE 1: Data Freshness & Fail-Safe Surveillance
        # ---------------------------------------------------------------------
        data_outage = (data_quality_status == DataQualityStatus.OUTAGE.value)
        if data_outage:
            failed_gates.append(StandardGateName.DATA_QUALITY)
            if not is_open_position:
                reason_codes.append(StandardReasonCode.DATA_OUTAGE_BLOCK)
                return StructuredDecision(
                    run_id=run_id_val,
                    ticker=ticker_val,
                    decision="HOLD",
                    composite_score=composite_score,
                    model_status=eff_model_status,
                    passed_gates=passed_gates,
                    failed_gates=failed_gates,
                    data_quality_status=data_quality_status,
                    position_action=PositionAction.NO_ACTION.value,
                    reason_codes=reason_codes,
                    metrics={"outage": True},
                )
            else:
                # Open position fail-safe
                if price_val is not None and stop_loss is not None:
                    if is_open_long and price_val <= stop_loss:
                        failed_gates.append(StandardGateName.STOP_LOSS_SURVEILLANCE)
                        reason_codes.append(StandardReasonCode.DATA_OUTAGE_FAILSAFE_EXIT)
                        return StructuredDecision(
                            run_id=run_id_val,
                            ticker=ticker_val,
                            decision="SELL",
                            composite_score=composite_score,
                            model_status=eff_model_status,
                            passed_gates=passed_gates,
                            failed_gates=failed_gates,
                            data_quality_status=data_quality_status,
                            position_action=PositionAction.CLOSE_LONG.value,
                            reason_codes=reason_codes,
                            metrics={"failsafe_stop": True},
                        )
                    elif is_open_short and price_val >= stop_loss:
                        failed_gates.append(StandardGateName.STOP_LOSS_SURVEILLANCE)
                        reason_codes.append(StandardReasonCode.DATA_OUTAGE_FAILSAFE_EXIT)
                        return StructuredDecision(
                            run_id=run_id_val,
                            ticker=ticker_val,
                            decision="BUY",
                            composite_score=composite_score,
                            model_status=eff_model_status,
                            passed_gates=passed_gates,
                            failed_gates=failed_gates,
                            data_quality_status=data_quality_status,
                            position_action=PositionAction.CLOSE_SHORT.value,
                            reason_codes=reason_codes,
                            metrics={"failsafe_short_stop": True},
                        )
                if outage_duration_hours >= self.policy.data_quality.max_outage_tolerance_hours:
                    reason_codes.append(StandardReasonCode.DATA_OUTAGE_FAILSAFE_EXIT)
                    act = PositionAction.CLOSE_LONG.value if is_open_long else PositionAction.CLOSE_SHORT.value
                    dec = "SELL" if is_open_long else "BUY"
                    return StructuredDecision(
                        run_id=run_id_val,
                        ticker=ticker_val,
                        decision=dec,
                        composite_score=composite_score,
                        model_status=eff_model_status,
                        passed_gates=passed_gates,
                        failed_gates=failed_gates,
                        data_quality_status=data_quality_status,
                        position_action=act,
                        reason_codes=reason_codes,
                        metrics={"outage_hours": outage_duration_hours},
                    )
        else:
            passed_gates.append(StandardGateName.DATA_QUALITY)

        # Check pillar hurdle (>= 3 pillars required)
        if pillar_scores is not None and avail_pillars < getattr(self.policy.data_quality, "min_valid_pillars", 3):
            failed_gates.append(StandardGateName.INSUFFICIENT_PILLARS)
            reason_codes.append(StandardReasonCode.INSUFFICIENT_PILLARS)
            return StructuredDecision(
                run_id=run_id_val,
                ticker=ticker_val,
                decision="HOLD",
                composite_score=None,
                model_status=eff_model_status,
                passed_gates=passed_gates,
                failed_gates=failed_gates,
                data_quality_status=data_quality_status,
                position_action=PositionAction.NO_ACTION.value,
                reason_codes=reason_codes,
                metrics={"available_pillars": avail_pillars},
            )
        elif pillar_scores is not None:
            passed_gates.append(StandardGateName.INSUFFICIENT_PILLARS)

        # ---------------------------------------------------------------------
        # STAGE 2: Mandatory Existing-Position Risk Controls (NEVER OVERRIDABLE)
        # ---------------------------------------------------------------------
        if is_open_position:
            # 1. Stop loss surveillance
            if price_val is not None and stop_loss is not None:
                if is_open_long and price_val <= stop_loss:
                    failed_gates.append(StandardGateName.STOP_LOSS_SURVEILLANCE)
                    reason_codes.append(StandardReasonCode.STOP_LOSS_BREACHED)
                    return StructuredDecision(
                        run_id=run_id_val,
                        ticker=ticker_val,
                        decision="SELL",
                        composite_score=composite_score,
                        model_status=eff_model_status,
                        passed_gates=passed_gates,
                        failed_gates=failed_gates,
                        data_quality_status=data_quality_status,
                        position_action=PositionAction.CLOSE_LONG.value,
                        reason_codes=reason_codes,
                        metrics={"stop_loss": stop_loss, "price": price_val},
                    )
                elif is_open_short and price_val >= stop_loss:
                    failed_gates.append(StandardGateName.STOP_LOSS_SURVEILLANCE)
                    reason_codes.append(StandardReasonCode.SHORT_STOP_BREACHED)
                    return StructuredDecision(
                        run_id=run_id_val,
                        ticker=ticker_val,
                        decision="BUY",
                        composite_score=composite_score,
                        model_status=eff_model_status,
                        passed_gates=passed_gates,
                        failed_gates=failed_gates,
                        data_quality_status=data_quality_status,
                        position_action=PositionAction.CLOSE_SHORT.value,
                        reason_codes=reason_codes,
                        metrics={"short_stop": stop_loss, "price": price_val},
                    )

            # 2. Critical Qualitative Evidence Veto
            if has_critical_evidence_veto:
                reason_codes.append(StandardReasonCode.CRITICAL_EVIDENCE_VETO)
                act = PositionAction.CLOSE_LONG.value if is_open_long else PositionAction.CLOSE_SHORT.value
                dec = "SELL" if is_open_long else "BUY"
                return StructuredDecision(
                    run_id=run_id_val,
                    ticker=ticker_val,
                    decision=dec,
                    composite_score=composite_score,
                    model_status=eff_model_status,
                    passed_gates=passed_gates,
                    failed_gates=failed_gates,
                    data_quality_status=data_quality_status,
                    position_action=act,
                    reason_codes=reason_codes,
                    metrics={"critical_veto": True},
                )

            # 3. Scheduled Binary Earnings Blackout
            if days_to_earnings is not None and days_to_earnings <= self.policy.events.earnings_blackout_sessions:
                failed_gates.append(StandardGateName.EVENT_RISK)
                reason_codes.append(StandardReasonCode.EARNINGS_BLACKOUT_ACTIVE)
                act = PositionAction.CLOSE_LONG.value if is_open_long else PositionAction.CLOSE_SHORT.value
                dec = "SELL" if is_open_long else "BUY"
                return StructuredDecision(
                    run_id=run_id_val,
                    ticker=ticker_val,
                    decision=dec,
                    composite_score=composite_score,
                    model_status=eff_model_status,
                    passed_gates=passed_gates,
                    failed_gates=failed_gates,
                    data_quality_status=data_quality_status,
                    position_action=act,
                    reason_codes=reason_codes,
                    metrics={"days_to_earnings": days_to_earnings},
                )

            # 4. Thesis Collapse
            if composite_score is not None:
                if is_open_long and float(composite_score) < self.policy.holding_guard.catastrophic_composite_floor:
                    failed_gates.append(StandardGateName.HOLDING_THESIS)
                    reason_codes.append(StandardReasonCode.THESIS_COLLAPSED)
                    return StructuredDecision(
                        run_id=run_id_val,
                        ticker=ticker_val,
                        decision="SELL",
                        composite_score=composite_score,
                        model_status=eff_model_status,
                        passed_gates=passed_gates,
                        failed_gates=failed_gates,
                        data_quality_status=data_quality_status,
                        position_action=PositionAction.CLOSE_LONG.value,
                        reason_codes=reason_codes,
                        metrics={"composite_score": composite_score},
                    )
                elif is_open_short and float(composite_score) > 60.0:
                    failed_gates.append(StandardGateName.HOLDING_THESIS)
                    reason_codes.append(StandardReasonCode.THESIS_COLLAPSED)
                    return StructuredDecision(
                        run_id=run_id_val,
                        ticker=ticker_val,
                        decision="BUY",
                        composite_score=composite_score,
                        model_status=eff_model_status,
                        passed_gates=passed_gates,
                        failed_gates=failed_gates,
                        data_quality_status=data_quality_status,
                        position_action=PositionAction.CLOSE_SHORT.value,
                        reason_codes=reason_codes,
                        metrics={"composite_score": composite_score},
                    )

            # Passed mandatory position risk controls
            passed_gates.append(StandardGateName.STOP_LOSS_SURVEILLANCE)

            # Profit target check
            if price_val is not None and target_price is not None:
                if is_open_long and price_val >= target_price:
                    passed_gates.append(StandardGateName.PROFIT_TARGET)
                    reason_codes.append(StandardReasonCode.PROFIT_TARGET_REACHED)
                    return StructuredDecision(
                        run_id=run_id_val,
                        ticker=ticker_val,
                        decision="SELL",
                        composite_score=composite_score,
                        model_status=eff_model_status,
                        passed_gates=passed_gates,
                        failed_gates=failed_gates,
                        data_quality_status=data_quality_status,
                        position_action=PositionAction.CLOSE_LONG.value,
                        reason_codes=reason_codes,
                        metrics={"target_price": target_price, "price": price_val},
                    )
                elif is_open_short and price_val <= target_price:
                    passed_gates.append(StandardGateName.PROFIT_TARGET)
                    reason_codes.append(StandardReasonCode.PROFIT_TARGET_REACHED)
                    return StructuredDecision(
                        run_id=run_id_val,
                        ticker=ticker_val,
                        decision="BUY",
                        composite_score=composite_score,
                        model_status=eff_model_status,
                        passed_gates=passed_gates,
                        failed_gates=failed_gates,
                        data_quality_status=data_quality_status,
                        position_action=PositionAction.CLOSE_SHORT.value,
                        reason_codes=reason_codes,
                        metrics={"short_target": target_price, "price": price_val},
                    )

            # Holding thesis guard on winning long position
            if is_open_long and cand_dec == "SELL":
                g4 = SellExitSymmetryGate.evaluate(
                    candidate_decision="SELL",
                    is_holding=True,
                    current_price=price_val or 0.0,
                    stop_loss=stop_loss,
                    target_price=target_price,
                    composite_score=composite_score,
                    has_critical_evidence_veto=has_critical_evidence_veto,
                    policy=self.policy,
                )
                if not g4.passed and g4.reason_code == "HOLDING_THESIS_INTACT":
                    passed_gates.append(StandardGateName.HOLDING_THESIS)
                    reason_codes.append(StandardReasonCode.HOLDING_THESIS_INTACT)
                    return StructuredDecision(
                        run_id=run_id_val,
                        ticker=ticker_val,
                        decision="HOLD",
                        composite_score=composite_score,
                        model_status=eff_model_status,
                        passed_gates=passed_gates,
                        failed_gates=failed_gates,
                        data_quality_status=data_quality_status,
                        position_action=PositionAction.NO_ACTION.value,
                        reason_codes=reason_codes,
                        metrics=g4.metrics,
                    )
                else:
                    return StructuredDecision(
                        run_id=run_id_val,
                        ticker=ticker_val,
                        decision="SELL",
                        composite_score=composite_score,
                        model_status=eff_model_status,
                        passed_gates=passed_gates,
                        failed_gates=failed_gates,
                        data_quality_status=data_quality_status,
                        position_action=PositionAction.CLOSE_LONG.value,
                        reason_codes=[g4.reason_code or "DISCRETIONARY_EXIT"],
                        metrics=g4.metrics,
                    )

            # Position maintained
            return StructuredDecision(
                run_id=run_id_val,
                ticker=ticker_val,
                decision="HOLD",
                composite_score=composite_score,
                model_status=eff_model_status,
                passed_gates=passed_gates,
                failed_gates=failed_gates,
                data_quality_status=data_quality_status,
                position_action=PositionAction.NO_ACTION.value,
                reason_codes=[StandardReasonCode.HOLDING_THESIS_INTACT],
                metrics={"position_maintained": True},
            )

        # ---------------------------------------------------------------------
        # STAGE 3: Event, Liquidity, & Noise Restrictions (for Fresh Entries)
        # ---------------------------------------------------------------------
        if cand_dec in ("BUY", "SELL"):
            dir_eval = TradeDirection.LONG if cand_dec == "BUY" else TradeDirection.SHORT
            atr_calc = atr_val or (0.02 * (price_val or 100.0))
            stop_calc = stop_loss if stop_loss is not None else (
                (price_val or 100.0) - 1.5 * atr_calc if dir_eval == TradeDirection.LONG else (price_val or 100.0) + 1.5 * atr_calc
            )

            g3 = VolatilityEventLiquidityGate.evaluate(
                direction=dir_eval,
                price=price_val or 100.0,
                stop_loss=stop_calc,
                atr=atr_calc,
                days_to_earnings=days_to_earnings,
                is_event_risk_blackout=is_event_risk_blackout,
                days_to_ex_dividend=days_to_ex_dividend,
                iv_pct=iv_pct,
                rvol_20d=rvol_20d,
                avg_dollar_volume_20d=avg_dollar_volume_20d,
                intended_order_value=intended_order_value,
                policy=self.policy,
            )

            if not g3.passed:
                if any(k in (g3.reason_code or "") for k in ("BLACKOUT", "EVENT")):
                    gate_id = StandardGateName.EVENT_RISK
                elif any(k in (g3.reason_code or "") for k in ("RVOL", "VOLUME", "LIQUIDITY", "IMPACT")):
                    gate_id = StandardGateName.LIQUIDITY
                elif "NOISE" in (g3.reason_code or ""):
                    gate_id = StandardGateName.REWARD_TO_RISK
                else:
                    gate_id = StandardGateName.DATA_QUALITY
                failed_gates.append(gate_id)
                # Map reason code
                if g3.reason_code == "EARNINGS_BLACKOUT":
                    rc = StandardReasonCode.EARNINGS_BLACKOUT_ACTIVE
                elif g3.reason_code in ("EVENT_RISK_BLACKOUT", "EVENT_RISK"):
                    rc = StandardReasonCode.EVENT_RISK_BLACKOUT
                elif g3.reason_code == "EX_DIVIDEND_BLACKOUT":
                    rc = StandardReasonCode.EX_DIVIDEND_BLACKOUT
                elif g3.reason_code == "STOP_INSIDE_NOISE_ZONE":
                    rc = StandardReasonCode.STOP_INSIDE_VOLATILITY_NOISE
                elif g3.reason_code in ("LOW_RVOL", "LOW_DOLLAR_VOLUME", "LOW_LIQUIDITY", "LIQUIDITY_INSUFFICIENT"):
                    rc = StandardReasonCode.LIQUIDITY_INSUFFICIENT
                else:
                    rc = g3.reason_code or "RESTRICTION_ACTIVE"
                reason_codes.append(rc)

                return StructuredDecision(
                    run_id=run_id_val,
                    ticker=ticker_val,
                    decision="HOLD",
                    composite_score=composite_score,
                    model_status=eff_model_status,
                    passed_gates=passed_gates,
                    failed_gates=failed_gates,
                    data_quality_status=data_quality_status,
                    position_action=PositionAction.NO_ACTION.value,
                    reason_codes=reason_codes,
                    metrics=g3.metrics,
                )
            else:
                passed_gates.append(StandardGateName.EVENT_RISK)
                passed_gates.append(StandardGateName.LIQUIDITY)

        # ---------------------------------------------------------------------
        # STAGE 5: Directional Entry Certification
        # ---------------------------------------------------------------------
        if cand_dec == "BUY":
            # 1. Gate 1: BUY Floor & Value Trap Veto
            if is_value_trap:
                failed_gates.append(StandardGateName.BUY_FLOOR)
                reason_codes.append(StandardReasonCode.VALUE_TRAP_DETECTED)
                return StructuredDecision(
                    run_id=run_id_val,
                    ticker=ticker_val,
                    decision="HOLD",
                    composite_score=composite_score,
                    model_status=eff_model_status,
                    passed_gates=passed_gates,
                    failed_gates=failed_gates,
                    data_quality_status=data_quality_status,
                    position_action=PositionAction.NO_ACTION.value,
                    reason_codes=reason_codes,
                    metrics={"is_value_trap": True, "buy_vetoed": True},
                )

            g1 = BuyFloorGate.evaluate(
                candidate_decision="BUY",
                composite_score=composite_score,
                catalyst_path=catalyst_path,
                rsi14=rsi14,
                drawdown_5d_pct=drawdown_5d_pct,
                valuation_history_score=valuation_history_score,
                is_value_trap=is_value_trap,
                policy=self.policy,
            )
            if not g1.passed:
                failed_gates.append(StandardGateName.BUY_FLOOR)
                if catalyst_path == "DIP_BUY" and not g1.metrics.get("dip_certified", False):
                    rc = StandardReasonCode.DIP_CRITERIA_FAILED
                elif g1.reason_code == "VALUE_TRAP_CONFIRMED":
                    rc = StandardReasonCode.VALUE_TRAP_DETECTED
                else:
                    rc = StandardReasonCode.SCORE_BELOW_BUY_THRESHOLD
                reason_codes.append(rc)
                return StructuredDecision(
                    run_id=run_id_val,
                    ticker=ticker_val,
                    decision="HOLD",
                    composite_score=composite_score,
                    model_status=eff_model_status,
                    passed_gates=passed_gates,
                    failed_gates=failed_gates,
                    data_quality_status=data_quality_status,
                    position_action=PositionAction.NO_ACTION.value,
                    reason_codes=reason_codes,
                    metrics=g1.metrics,
                )
            passed_gates.append(StandardGateName.BUY_FLOOR)

            # 2. Gate 2: Reward-to-Risk
            if price_val is not None and stop_loss is not None and target_price is not None:
                g2 = RewardToRiskGate.evaluate(
                    direction=TradeDirection.LONG,
                    entry_price=price_val,
                    stop_loss=stop_loss,
                    target_price=target_price,
                    composite_score=composite_score,
                    policy=self.policy,
                    execution_costs=execution_costs,
                )
                if not g2.passed:
                    failed_gates.append(StandardGateName.REWARD_TO_RISK)
                    if g2.reason_code in (
                        "INVALID_PRICE_ORDERING",
                        "INVALID_ORDERING_LONG",
                        "ZERO_OR_NEGATIVE_REWARD",
                        "NON_POSITIVE_PRICE",
                    ):
                        rc = StandardReasonCode.PRICE_ORDERING_INVALID
                    elif g2.reason_code in ("ZERO_OR_NEGATIVE_RISK", "RISK_DENOMINATOR_ZERO"):
                        rc = StandardReasonCode.NEGATIVE_OR_ZERO_RISK
                    elif g2.reason_code in ("NO_STATISTICAL_EDGE", "INSUFFICIENT_STATISTICAL_EDGE"):
                        rc = StandardReasonCode.STATISTICAL_EDGE_INSUFFICIENT
                    else:
                        rc = StandardReasonCode.REWARD_TO_RISK_INSUFFICIENT
                    reason_codes.append(rc)
                    return StructuredDecision(
                        run_id=run_id_val,
                        ticker=ticker_val,
                        decision="HOLD",
                        composite_score=composite_score,
                        model_status=eff_model_status,
                        passed_gates=passed_gates,
                        failed_gates=failed_gates,
                        data_quality_status=data_quality_status,
                        position_action=PositionAction.NO_ACTION.value,
                        reason_codes=reason_codes,
                        metrics=g2.metrics,
                    )
                passed_gates.append(StandardGateName.REWARD_TO_RISK)

            is_certified_dip_buy = bool(g1.metrics.get("dip_certified", False))

            # 3. Gate 5: Gated Confidence
            g5 = GatedConfidenceGate.evaluate(
                candidate_decision="BUY",
                composite_score=composite_score,
                pillar_scores=pillar_scores,
                data_completeness=data_completeness,
                policy=self.policy,
                is_certified_dip_buy=is_certified_dip_buy,
            )
            if not g5.passed:
                failed_gates.append(StandardGateName.CONFIDENCE_HURDLE)
                reason_codes.append(StandardReasonCode.CONFIDENCE_BELOW_HURDLE)
                return StructuredDecision(
                    run_id=run_id_val,
                    ticker=ticker_val,
                    decision="HOLD",
                    composite_score=composite_score,
                    model_status=eff_model_status,
                    passed_gates=passed_gates,
                    failed_gates=failed_gates,
                    data_quality_status=data_quality_status,
                    position_action=PositionAction.NO_ACTION.value,
                    reason_codes=reason_codes,
                    metrics=g5.metrics,
                )
            passed_gates.append(StandardGateName.CONFIDENCE_HURDLE)

            # ALL GATES PASSED -> LONG ORDER CERTIFIED!
            reason_codes.append(StandardReasonCode.BUY_ORDER_CERTIFIED)
            sizing_metrics: Dict[str, Any] = {}
            if price_val is not None and stop_loss is not None:
                from quant.position_sizing import calculate_risk_based_position_size
                sizing_res = calculate_risk_based_position_size(
                    entry_price=price_val,
                    stop_loss=stop_loss,
                    ticker=ticker_val,
                    direction="LONG",
                    portfolio_equity=portfolio_equity,
                    risk_fraction=risk_fraction,
                    atr=atr,
                    avg_dollar_volume_20d=avg_dollar_volume_20d,
                    composite_score=composite_score,
                    mandate=mandate,
                    policy=self.policy,
                )
                sizing_metrics = sizing_res.to_dict()

            cert_metrics = {"certified": True}
            if sizing_metrics:
                cert_metrics["position_sizing"] = sizing_metrics

            return StructuredDecision(
                run_id=run_id_val,
                ticker=ticker_val,
                decision="BUY",
                composite_score=composite_score,
                model_status=eff_model_status,
                passed_gates=passed_gates,
                failed_gates=failed_gates,
                data_quality_status=data_quality_status,
                position_action=PositionAction.OPEN_LONG.value,
                reason_codes=reason_codes,
                metrics=cert_metrics,
            )

        elif cand_dec == "SELL":
            # 1. Gate 4: Short Ceiling & Borrow Availability
            g4_s = SellExitSymmetryGate.evaluate(
                candidate_decision="SELL",
                is_holding=False,
                current_price=price_val or 0.0,
                stop_loss=stop_loss,
                target_price=target_price,
                composite_score=composite_score,
                borrow_available=borrow_available,
                policy=self.policy,
            )
            if not g4_s.passed:
                if g4_s.reason_code == "BORROW_UNAVAILABLE":
                    failed_gates.append(StandardGateName.BORROW_AVAILABILITY)
                    rc = StandardReasonCode.BORROW_UNAVAILABLE
                else:
                    failed_gates.append(StandardGateName.SHORT_CEILING)
                    rc = StandardReasonCode.SHORT_CEILING_EXCEEDED
                reason_codes.append(rc)
                return StructuredDecision(
                    run_id=run_id_val,
                    ticker=ticker_val,
                    decision="HOLD",
                    composite_score=composite_score,
                    model_status=eff_model_status,
                    passed_gates=passed_gates,
                    failed_gates=failed_gates,
                    data_quality_status=data_quality_status,
                    position_action=PositionAction.NO_ACTION.value,
                    reason_codes=reason_codes,
                    metrics=g4_s.metrics,
                )
            passed_gates.append(StandardGateName.SHORT_CEILING)
            passed_gates.append(StandardGateName.BORROW_AVAILABILITY)

            # 2. Gate 2: Short Reward-to-Risk
            if price_val is not None and stop_loss is not None and target_price is not None:
                g2_s = RewardToRiskGate.evaluate(
                    direction=TradeDirection.SHORT,
                    entry_price=price_val,
                    stop_loss=stop_loss,
                    target_price=target_price,
                    composite_score=composite_score,
                    policy=self.policy,
                    execution_costs=execution_costs,
                )
                if not g2_s.passed:
                    failed_gates.append(StandardGateName.REWARD_TO_RISK)
                    if g2_s.reason_code in (
                        "INVALID_PRICE_ORDERING",
                        "INVALID_ORDERING_SHORT",
                        "ZERO_OR_NEGATIVE_REWARD",
                        "NON_POSITIVE_PRICE",
                    ):
                        rc = StandardReasonCode.PRICE_ORDERING_INVALID
                    elif g2_s.reason_code in ("ZERO_OR_NEGATIVE_RISK", "RISK_DENOMINATOR_ZERO"):
                        rc = StandardReasonCode.NEGATIVE_OR_ZERO_RISK
                    elif g2_s.reason_code in ("NO_STATISTICAL_EDGE", "INSUFFICIENT_STATISTICAL_EDGE"):
                        rc = StandardReasonCode.STATISTICAL_EDGE_INSUFFICIENT
                    else:
                        rc = StandardReasonCode.REWARD_TO_RISK_INSUFFICIENT
                    reason_codes.append(rc)
                    return StructuredDecision(
                        run_id=run_id_val,
                        ticker=ticker_val,
                        decision="HOLD",
                        composite_score=composite_score,
                        model_status=eff_model_status,
                        passed_gates=passed_gates,
                        failed_gates=failed_gates,
                        data_quality_status=data_quality_status,
                        position_action=PositionAction.NO_ACTION.value,
                        reason_codes=reason_codes,
                        metrics=g2_s.metrics,
                    )
                passed_gates.append(StandardGateName.REWARD_TO_RISK)

            # 3. Gate 5: Gated Confidence
            g5_s = GatedConfidenceGate.evaluate(
                candidate_decision="SELL",
                composite_score=composite_score,
                pillar_scores=pillar_scores,
                data_completeness=data_completeness,
                policy=self.policy,
            )
            if not g5_s.passed:
                failed_gates.append(StandardGateName.CONFIDENCE_HURDLE)
                reason_codes.append(StandardReasonCode.CONFIDENCE_BELOW_HURDLE)
                return StructuredDecision(
                    run_id=run_id_val,
                    ticker=ticker_val,
                    decision="HOLD",
                    composite_score=composite_score,
                    model_status=eff_model_status,
                    passed_gates=passed_gates,
                    failed_gates=failed_gates,
                    data_quality_status=data_quality_status,
                    position_action=PositionAction.NO_ACTION.value,
                    reason_codes=reason_codes,
                    metrics=g5_s.metrics,
                )
            passed_gates.append(StandardGateName.CONFIDENCE_HURDLE)

            # ALL GATES PASSED -> SHORT ORDER CERTIFIED!
            reason_codes.append(StandardReasonCode.SHORT_ORDER_CERTIFIED)
            sizing_metrics_s: Dict[str, Any] = {}
            if price_val is not None and stop_loss is not None:
                from quant.position_sizing import calculate_risk_based_position_size
                sizing_res_s = calculate_risk_based_position_size(
                    entry_price=price_val,
                    stop_loss=stop_loss,
                    ticker=ticker_val,
                    direction="SHORT",
                    portfolio_equity=portfolio_equity,
                    risk_fraction=risk_fraction,
                    atr=atr,
                    avg_dollar_volume_20d=avg_dollar_volume_20d,
                    composite_score=composite_score,
                    mandate=mandate,
                    policy=self.policy,
                )
                sizing_metrics_s = sizing_res_s.to_dict()

            cert_metrics_s = {"certified": True}
            if sizing_metrics_s:
                cert_metrics_s["position_sizing"] = sizing_metrics_s

            return StructuredDecision(
                run_id=run_id_val,
                ticker=ticker_val,
                decision="SELL",
                composite_score=composite_score,
                model_status=eff_model_status,
                passed_gates=passed_gates,
                failed_gates=failed_gates,
                data_quality_status=data_quality_status,
                position_action=PositionAction.OPEN_SHORT.value,
                reason_codes=reason_codes,
                metrics=cert_metrics_s,
            )

        # ---------------------------------------------------------------------
        # STAGE 6: Default Fallback
        # ---------------------------------------------------------------------
        reason_codes.append(StandardReasonCode.NO_TRADE_DEFAULT)
        return StructuredDecision(
            run_id=run_id_val,
            ticker=ticker_val,
            decision="HOLD",
            composite_score=composite_score,
            model_status=eff_model_status,
            passed_gates=passed_gates,
            failed_gates=failed_gates,
            data_quality_status=data_quality_status,
            position_action=PositionAction.NO_ACTION.value,
            reason_codes=reason_codes,
            metrics={"default_fallback": True},
        )


def build_structured_decision(
    ticker: str,
    candidate_decision: str,
    composite_score: Optional[float],
    position_state: str = "FLAT",
    current_price: Optional[float] = None,
    stop_loss: Optional[float] = None,
    target_price: Optional[float] = None,
    atr: Optional[float] = None,
    pillar_scores: Optional[Dict[str, Optional[float]]] = None,
    data_completeness: float = 1.0,
    market_data_fresh: bool = True,
    ml_data_fresh: bool = True,
    outage_duration_hours: float = 0.0,
    days_to_earnings: Optional[int] = None,
    is_event_risk_blackout: bool = False,
    days_to_ex_dividend: Optional[int] = None,
    iv_pct: Optional[float] = None,
    rvol_20d: Optional[float] = None,
    avg_dollar_volume_20d: Optional[float] = None,
    borrow_available: bool = True,
    has_critical_evidence_veto: bool = False,
    catalyst_path: Optional[str] = None,
    rsi14: Optional[float] = None,
    drawdown_5d_pct: Optional[float] = None,
    valuation_history_score: Optional[float] = None,
    is_value_trap: bool = False,
    model_status: Optional[str] = None,
    run_id: Optional[str] = None,
    portfolio_equity: Optional[float] = None,
    risk_fraction: Optional[float] = None,
    mandate: Optional[Any] = None,
    policy: Optional[TradingPolicy] = None,
) -> StructuredDecision:
    """Convenience helper to evaluate and build a StructuredDecision."""
    engine = StructuredDecisionEngine(policy)
    return engine.evaluate(
        ticker=ticker,
        candidate_decision=candidate_decision,
        composite_score=composite_score,
        position_state=position_state,
        current_price=current_price,
        stop_loss=stop_loss,
        target_price=target_price,
        atr=atr,
        pillar_scores=pillar_scores,
        data_completeness=data_completeness,
        market_data_fresh=market_data_fresh,
        ml_data_fresh=ml_data_fresh,
        outage_duration_hours=outage_duration_hours,
        days_to_earnings=days_to_earnings,
        is_event_risk_blackout=is_event_risk_blackout,
        days_to_ex_dividend=days_to_ex_dividend,
        iv_pct=iv_pct,
        rvol_20d=rvol_20d,
        avg_dollar_volume_20d=avg_dollar_volume_20d,
        borrow_available=borrow_available,
        has_critical_evidence_veto=has_critical_evidence_veto,
        catalyst_path=catalyst_path,
        rsi14=rsi14,
        drawdown_5d_pct=drawdown_5d_pct,
        valuation_history_score=valuation_history_score,
        is_value_trap=is_value_trap,
        model_status=model_status,
        run_id=run_id,
        portfolio_equity=portfolio_equity,
        risk_fraction=risk_fraction,
        mandate=mandate,
    )
