"""Order Execution Assumptions, Pre-Flight Validation, Operational Kill Switch, and Ledger Reconciliation (Step 7.3).

Implements the institutional execution layer:
1. Strict State Machine Separation:
   Maintains distinct states:
   - SIGNAL_GENERATED: Upstream trading signal produced, no order sent.
   - ORDER_SUBMITTED: Order constructed and dispatched to execution gateway.
   - BROKER_ACKNOWLEDGED: Broker acknowledged order with broker order ID.
   - PARTIALLY_FILLED: One or more execution fills recorded, remaining shares > 0.
   - FILLED: Order fully filled (filled_shares == requested_shares).
   - CANCELLED: Order cancelled by operator, time-in-force, or kill switch.
   - REJECTED: Order rejected by pre-flight validation or broker.
   Invariant: A BUY recommendation (SIGNAL_GENERATED) is NEVER recorded as an executed trade.

2. Pre-Flight Order Validation (PreFlightOrderValidator):
   Before an order is sent, verifies:
   - Tradability & market-data freshness (quote staleness threshold, trading halt status).
   - Position alignment & cash balance (free cash check, margin collateral, direction conflict).
   - Permitted order size relative to liquidity (1% ADV cap, square-root market impact <= 15 bps).
   - Stop and target price validity against latest available price:
     * Long: 0 < Stop < P_live < Target, and live Reward:Risk >= hurdle.
     * Short: 0 < Target < P_live < Stop, and live Reward:Risk >= short hurdle.
     * Slippage drift guard: |P_live - P_signal| / P_signal <= max_price_drift_pct.
   - Duplicate-order prevention and deterministic idempotency key.

3. Operational Kill Switch (OperationalKillSwitch):
   Fails closed for new risk entries (BUY, SELL_SHORT) while permitting defensive exits
   (SELL to close long, BUY_TO_COVER to close short) when:
   - Data quality fails (quote staleness, missing pricing, halted feeds).
   - Order reconciliation fails (ledger positions diverge from internal portfolio).
   - Risk monitoring fails (portfolio drawdown >= circuit breaker halt threshold).
   - Manual operator emergency stop.

4. Execution Ledger & Position Reconciliation (ExecutionLedger & PositionReconciler):
   - ExecutionLedger: Single source of truth for orders, fills, and transition events.
   - PositionReconciler: Derives positions from fill ledger and verifies zero discrepancy
     against internal PortfolioSnapshot positions and cash balance.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from quant.policy_config import TradingPolicy, get_default_policy
from quant.portfolio_controls import PortfolioSnapshot, Position


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _today_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# =============================================================================
# ENUMERATIONS & CONSTANTS
# =============================================================================

class OrderStatus(str, Enum):
    """Lifecycle states of an execution order."""
    SIGNAL_GENERATED = "SIGNAL_GENERATED"
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    BROKER_ACKNOWLEDGED = "BROKER_ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class OrderSide(str, Enum):
    """Trading order direction."""
    BUY = "BUY"
    SELL = "SELL"
    SELL_SHORT = "SELL_SHORT"
    BUY_TO_COVER = "BUY_TO_COVER"


class OrderType(str, Enum):
    """Order type classification."""
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    STOP_LIMIT = "STOP_LIMIT"
    STOP_MARKET = "STOP_MARKET"


class KillSwitchReason(str, Enum):
    """Standardized root cause trigger for operational kill switch engagement."""
    DATA_QUALITY_FAILURE = "DATA_QUALITY_FAILURE"
    ORDER_RECONCILIATION_FAILURE = "ORDER_RECONCILIATION_FAILURE"
    RISK_MONITORING_FAILURE = "RISK_MONITORING_FAILURE"
    MANUAL_OPERATOR_TRIP = "MANUAL_OPERATOR_TRIP"
    UNSPECIFIED = "UNSPECIFIED"


class PreFlightRejectionReason(str, Enum):
    """Pre-flight validation rejection codes."""
    KILL_SWITCH_ACTIVE = "KILL_SWITCH_ACTIVE"
    MARKET_DATA_STALE = "MARKET_DATA_STALE"
    MARKET_HALTED = "MARKET_HALTED"
    NON_POSITIVE_PRICE = "NON_POSITIVE_PRICE"
    INSUFFICIENT_CASH = "INSUFFICIENT_CASH"
    INSUFFICIENT_MARGIN = "INSUFFICIENT_MARGIN"
    LIQUIDITY_LIMIT_EXCEEDED = "LIQUIDITY_LIMIT_EXCEEDED"
    MARKET_IMPACT_EXCEEDED = "MARKET_IMPACT_EXCEEDED"
    INVALID_STOP_PRICE = "INVALID_STOP_PRICE"
    INVALID_TARGET_PRICE = "INVALID_TARGET_PRICE"
    EXCESSIVE_PRICE_DRIFT = "EXCESSIVE_PRICE_DRIFT"
    SUB_MINIMAL_REWARD_RISK = "SUB_MINIMAL_REWARD_RISK"
    DUPLICATE_ORDER_IDEMPOTENCY = "DUPLICATE_ORDER_IDEMPOTENCY"
    POSITION_DIRECTION_CONFLICT = "POSITION_DIRECTION_CONFLICT"
    EXCEEDS_HOLDING_QUANTITY = "EXCEEDS_HOLDING_QUANTITY"
    INVALID_ORDER_PARAMETERS = "INVALID_ORDER_PARAMETERS"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MarketQuote:
    """Point-in-time market quote for live pre-flight checks."""
    ticker: str
    last_price: float
    bid_price: float
    ask_price: float
    timestamp: Union[str, datetime]
    is_halted: bool = False
    adv_shares_20d: float = 1_000_000.0
    atr_14d: Optional[float] = None

    def get_parsed_timestamp(self) -> datetime:
        """Parses the timestamp to a timezone-aware or naive datetime object."""
        if isinstance(self.timestamp, datetime):
            return self.timestamp
        ts_str = str(self.timestamp).replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(ts_str)
        except ValueError:
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    return datetime.strptime(ts_str, fmt)
                except ValueError:
                    pass
            return _now_utc()

    def age_seconds(self, reference_time: Optional[datetime] = None) -> float:
        """Calculates age of the quote relative to reference time (defaults to now)."""
        ref = reference_time or datetime.now(timezone.utc)
        parsed = self.get_parsed_timestamp()
        if parsed.tzinfo is not None and ref.tzinfo is None:
            ref = ref.replace(tzinfo=timezone.utc)
        elif parsed.tzinfo is None and ref.tzinfo is not None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0.0, (ref - parsed).total_seconds())

    def is_stale(self, max_staleness_seconds: float, reference_time: Optional[datetime] = None) -> bool:
        """Returns True if the quote age exceeds max staleness."""
        return self.age_seconds(reference_time) > max_staleness_seconds


@dataclass
class Order:
    """Institutional Order record tracking full lifecycle progression."""
    order_id: str
    client_order_id: str
    idempotency_key: str
    ticker: str
    side: OrderSide
    requested_shares: int
    order_type: OrderType = OrderType.LIMIT
    status: OrderStatus = OrderStatus.ORDER_SUBMITTED
    signal_id: Optional[int] = None
    run_id: Optional[str] = None
    signal_price: float = 0.0
    limit_price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    target_price: Optional[float] = None
    filled_shares: int = 0
    avg_fill_price: float = 0.0
    total_commission: float = 0.0
    broker_order_id: Optional[str] = None
    rejection_reason: Optional[str] = None
    cancellation_reason: Optional[str] = None
    created_at: str = field(default_factory=_now_utc_str)
    updated_at: str = field(default_factory=_now_utc_str)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def remaining_shares(self) -> int:
        """Remaining unexecuted shares."""
        return max(0, self.requested_shares - self.filled_shares)

    @property
    def is_terminal(self) -> bool:
        """Whether the order has reached a final state."""
        return self.status in {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED}

    @property
    def is_open(self) -> bool:
        """Whether the order is actively open in the market."""
        return not self.is_terminal

    @property
    def is_entry(self) -> bool:
        """Whether order opens or expands risk exposure."""
        return self.side in {OrderSide.BUY, OrderSide.SELL_SHORT}

    @property
    def is_exit(self) -> bool:
        """Whether order reduces or closes risk exposure."""
        return self.side in {OrderSide.SELL, OrderSide.BUY_TO_COVER}

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the order to a dictionary."""
        d = asdict(self)
        d["side"] = self.side.value if isinstance(self.side, OrderSide) else str(self.side)
        d["order_type"] = self.order_type.value if isinstance(self.order_type, OrderType) else str(self.order_type)
        d["status"] = self.status.value if isinstance(self.status, OrderStatus) else str(self.status)
        d["remaining_shares"] = self.remaining_shares
        return d


@dataclass
class OrderFill:
    """Individual trade execution fill recorded from broker."""
    fill_id: str
    order_id: str
    ticker: str
    side: OrderSide
    shares: int
    price: float
    commission: float = 0.0
    timestamp: str = field(default_factory=_now_utc_str)
    execution_id: Optional[str] = None
    slippage_bps: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["side"] = self.side.value if isinstance(self.side, OrderSide) else str(self.side)
        return d


@dataclass
class ExecutionEvent:
    """Audit event for state transition logging."""
    event_id: str
    order_id: str
    event_type: str
    from_status: Optional[str]
    to_status: str
    details: str
    timestamp: str = field(default_factory=_now_utc_str)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PreFlightValidationResult:
    """Structured result of pre-flight order validation."""
    is_valid: bool
    rejection_reasons: List[str] = field(default_factory=list)
    permitted_shares: int = 0
    estimated_market_impact_bps: float = 0.0
    estimated_slippage_bps: float = 0.0
    required_cash_or_margin: float = 0.0
    free_cash_after_order: float = 0.0
    price_drift_pct: float = 0.0
    idempotency_key: str = ""
    live_reward_risk_ratio: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReconciliationDiscrepancy:
    """Individual mismatch detected during reconciliation."""
    ticker: Optional[str]
    discrepancy_type: str
    internal_value: Any
    ledger_value: Any
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReconciliationResult:
    """Result of internal position state vs. execution ledger reconciliation."""
    is_reconciled: bool
    discrepancies: List[ReconciliationDiscrepancy] = field(default_factory=list)
    internal_positions: Dict[str, int] = field(default_factory=dict)
    ledger_positions: Dict[str, int] = field(default_factory=dict)
    internal_cash: float = 0.0
    ledger_cash: float = 0.0
    evaluated_at: str = field(default_factory=_now_utc_str)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["discrepancies"] = [disc.to_dict() for disc in self.discrepancies]
        return d


# =============================================================================
# IDEMPOTENCY KEY GENERATOR
# =============================================================================

def generate_idempotency_key(
    ticker: str,
    side: Union[OrderSide, str],
    requested_shares: int,
    price: float,
    session_date: str,
    run_id: Optional[str] = None,
) -> str:
    """
    Computes a deterministic SHA-256 idempotency key.
    Guarantees duplicate order submissions within the same session/run are identified.
    """
    sym = (ticker or "").upper().strip()
    side_str = side.value if isinstance(side, OrderSide) else str(side).upper().strip()
    price_bucket = f"{price:.2f}"
    run_str = (run_id or "DEFAULT").strip()
    raw = f"{sym}|{side_str}|{requested_shares}|{price_bucket}|{session_date}|{run_str}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# =============================================================================
# OPERATIONAL KILL SWITCH
# =============================================================================

class OperationalKillSwitch:
    """
    Operational Kill Switch (Step 7.3).
    Provides emergency fail-closed protection by freezing all new risk entries
    (BUY, SELL_SHORT) while preserving the ability to execute defensive risk-reducing
    exits (SELL, BUY_TO_COVER).
    """

    def __init__(self):
        self._is_tripped: bool = False
        self._trip_reason: Optional[KillSwitchReason] = None
        self._trip_details: str = ""
        self._tripped_by: str = ""
        self._trip_timestamp: Optional[str] = None

    @property
    def is_tripped(self) -> bool:
        """Returns True if the kill switch is currently engaged."""
        return self._is_tripped

    @property
    def trip_reason(self) -> Optional[KillSwitchReason]:
        return self._trip_reason

    @property
    def trip_details(self) -> str:
        return self._trip_details

    @property
    def trip_timestamp(self) -> Optional[str]:
        return self._trip_timestamp

    @property
    def tripped_by(self) -> str:
        return self._tripped_by

    def trip(
        self,
        reason: KillSwitchReason,
        details: str,
        actor: str = "SYSTEM",
        db_path: Optional[str] = None,
    ) -> None:
        """
        Engages the kill switch, freezing all new entry orders.
        """
        self._is_tripped = True
        self._trip_reason = reason
        self._trip_details = str(details)
        self._tripped_by = str(actor)
        self._trip_timestamp = _now_utc_str()

        # Persist to database audit log if available
        try:
            from db import save_kill_switch_state
            save_kwargs = {"db_path": db_path} if db_path else {}
            save_kill_switch_state(
                is_tripped=True,
                reason=reason.value,
                details=details,
                actor=actor,
                **save_kwargs,
            )
        except Exception:
            pass

    def reset(
        self,
        actor: str,
        reason: str,
        db_path: Optional[str] = None,
    ) -> None:
        """
        Resets the kill switch to healthy operational state.
        """
        self._is_tripped = False
        details = f"Reset by {actor}: {reason}"
        self._trip_details = details
        self._tripped_by = str(actor)
        self._trip_timestamp = _now_utc_str()

        try:
            from db import save_kill_switch_state
            save_kwargs = {"db_path": db_path} if db_path else {}
            save_kill_switch_state(
                is_tripped=False,
                reason="OPERATOR_RESET",
                details=details,
                actor=actor,
                **save_kwargs,
            )
        except Exception:
            pass

    def can_submit_order(self, side: Union[OrderSide, str]) -> Tuple[bool, Optional[str]]:
        """
        Evaluates whether an order of given side is permitted under current kill switch state.

        RULE:
        - If NOT tripped: All orders permitted -> (True, None).
        - If TRIPPED:
          * Defensive exits (SELL, BUY_TO_COVER) ARE PERMITTED so the desk can reduce risk!
          * New risk entries (BUY, SELL_SHORT) ARE BLOCKED -> (False, reason).
        """
        if not self._is_tripped:
            return True, None

        side_enum = side if isinstance(side, OrderSide) else OrderSide(str(side).upper())
        if side_enum in {OrderSide.SELL, OrderSide.BUY_TO_COVER}:
            return True, None

        reason_str = self._trip_reason.value if self._trip_reason else "KILL_SWITCH_ACTIVE"
        return False, f"Operational kill switch active: {reason_str} - {self._trip_details}"

    def get_status(self) -> Dict[str, Any]:
        """Returns the current state dictionary of the kill switch."""
        return {
            "is_tripped": self._is_tripped,
            "trip_reason": self._trip_reason.value if self._trip_reason else None,
            "trip_details": self._trip_details,
            "tripped_by": self._tripped_by,
            "trip_timestamp": self._trip_timestamp,
        }


# Global operational kill switch instance
GLOBAL_KILL_SWITCH = OperationalKillSwitch()


# =============================================================================
# PRE-FLIGHT ORDER VALIDATOR
# =============================================================================

class PreFlightOrderValidator:
    """
    Pre-Flight Order Validation Engine (Step 7.3).
    Ensures every candidate trade satisfies all technical, market, liquidity,
    and portfolio safety requirements before transmission to broker execution.
    """

    def __init__(
        self,
        policy: Optional[TradingPolicy] = None,
        kill_switch: Optional[OperationalKillSwitch] = None,
    ):
        self.policy = policy or get_default_policy()
        self.kill_switch = kill_switch or GLOBAL_KILL_SWITCH

    def validate(
        self,
        ticker: str,
        side: Union[OrderSide, str],
        requested_shares: int,
        quote: Optional[MarketQuote],
        portfolio: Optional[PortfolioSnapshot] = None,
        signal_price: Optional[float] = None,
        limit_price: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        target_price: Optional[float] = None,
        session_date: Optional[str] = None,
        run_id: Optional[str] = None,
        existing_idempotency_keys: Optional[Set[str]] = None,
        reference_time: Optional[datetime] = None,
    ) -> PreFlightValidationResult:
        """
        Executes complete pre-flight validation on candidate order parameters.
        """
        rejection_reasons: List[str] = []
        details: Dict[str, Any] = {}

        sym = (ticker or "").upper().strip()
        side_enum = side if isinstance(side, OrderSide) else OrderSide(str(side).upper())
        s_date = session_date or _today_utc_str()
        effective_price = (
            limit_price
            if limit_price and limit_price > 0
            else (quote.last_price if quote and quote.last_price > 0 else (signal_price or 0.0))
        )

        idempotency_key = generate_idempotency_key(
            ticker=sym,
            side=side_enum,
            requested_shares=requested_shares,
            price=effective_price,
            session_date=s_date,
            run_id=run_id,
        )

        # ---------------------------------------------------------------------
        # 1. Parameter Sanity Check
        # ---------------------------------------------------------------------
        if requested_shares <= 0:
            rejection_reasons.append(PreFlightRejectionReason.INVALID_ORDER_PARAMETERS.value)
            details["parameter_error"] = f"requested_shares must be strictly positive, got {requested_shares}"

        # ---------------------------------------------------------------------
        # 2. Operational Kill Switch Check
        # ---------------------------------------------------------------------
        can_proceed, kill_reason = self.kill_switch.can_submit_order(side_enum)
        if not can_proceed:
            rejection_reasons.append(PreFlightRejectionReason.KILL_SWITCH_ACTIVE.value)
            details["kill_switch"] = kill_reason

        # ---------------------------------------------------------------------
        # 3. Market Data Freshness & Tradability
        # ---------------------------------------------------------------------
        if quote is None:
            rejection_reasons.append(PreFlightRejectionReason.NON_POSITIVE_PRICE.value)
            details["market_data"] = "Quote object is None"
        else:
            if quote.last_price <= 0.0:
                rejection_reasons.append(PreFlightRejectionReason.NON_POSITIVE_PRICE.value)
                details["quote_price"] = quote.last_price

            if quote.is_halted:
                rejection_reasons.append(PreFlightRejectionReason.MARKET_HALTED.value)
                details["halt_status"] = "Trading halt actively declared"

            max_stale = self.policy.execution.max_quote_staleness_seconds
            quote_age = quote.age_seconds(reference_time)
            details["quote_age_seconds"] = round(quote_age, 1)

            if quote.is_stale(max_stale, reference_time):
                rejection_reasons.append(PreFlightRejectionReason.MARKET_DATA_STALE.value)
                details["quote_stale_error"] = (
                    f"Quote age {quote_age:.1f}s exceeds maximum allowed staleness {max_stale:.1f}s"
                )
                if self.policy.execution.kill_switch_on_data_failure:
                    self.kill_switch.trip(
                        reason=KillSwitchReason.DATA_QUALITY_FAILURE,
                        details=f"Market data stale for {sym}: {quote_age:.1f}s > {max_stale:.1f}s",
                        actor="PRE_FLIGHT_VALIDATOR",
                    )

        # ---------------------------------------------------------------------
        # 4. Duplicate Order & Idempotency Prevention
        # ---------------------------------------------------------------------
        if self.policy.execution.enable_idempotency_enforcement and existing_idempotency_keys:
            if idempotency_key in existing_idempotency_keys:
                rejection_reasons.append(PreFlightRejectionReason.DUPLICATE_ORDER_IDEMPOTENCY.value)
                details["idempotency"] = f"Duplicate order detected with key {idempotency_key}"

        # ---------------------------------------------------------------------
        # 5. Position Alignment & Directional Conflicts
        # ---------------------------------------------------------------------
        if portfolio is not None:
            active_pos = portfolio.positions.get(sym)
            if active_pos is not None and active_pos.shares > 0:
                pos_dir = active_pos.direction.upper()
                if side_enum == OrderSide.BUY and pos_dir == "SHORT":
                    rejection_reasons.append(PreFlightRejectionReason.POSITION_DIRECTION_CONFLICT.value)
                    details["position_conflict"] = "Attempting BUY while currently holding SHORT position"
                elif side_enum == OrderSide.SELL_SHORT and pos_dir == "LONG":
                    rejection_reasons.append(PreFlightRejectionReason.POSITION_DIRECTION_CONFLICT.value)
                    details["position_conflict"] = "Attempting SELL_SHORT while currently holding LONG position"
                elif side_enum == OrderSide.SELL:
                    if pos_dir != "LONG":
                        rejection_reasons.append(PreFlightRejectionReason.POSITION_DIRECTION_CONFLICT.value)
                        details["position_conflict"] = "Cannot execute SELL to close non-LONG position"
                    elif requested_shares > active_pos.shares:
                        rejection_reasons.append(PreFlightRejectionReason.EXCEEDS_HOLDING_QUANTITY.value)
                        details["quantity_error"] = (
                            f"Requested sell shares ({requested_shares}) exceeds current long shares ({active_pos.shares})"
                        )
                elif side_enum == OrderSide.BUY_TO_COVER:
                    if pos_dir != "SHORT":
                        rejection_reasons.append(PreFlightRejectionReason.POSITION_DIRECTION_CONFLICT.value)
                        details["position_conflict"] = "Cannot execute BUY_TO_COVER on non-SHORT position"
                    elif requested_shares > active_pos.shares:
                        rejection_reasons.append(PreFlightRejectionReason.EXCEEDS_HOLDING_QUANTITY.value)
                        details["quantity_error"] = (
                            f"Requested cover shares ({requested_shares}) exceeds current short shares ({active_pos.shares})"
                        )
            else:
                # No active position
                if side_enum == OrderSide.SELL:
                    rejection_reasons.append(PreFlightRejectionReason.POSITION_DIRECTION_CONFLICT.value)
                    details["position_conflict"] = "Cannot execute SELL when holding 0 shares"
                elif side_enum == OrderSide.BUY_TO_COVER:
                    rejection_reasons.append(PreFlightRejectionReason.POSITION_DIRECTION_CONFLICT.value)
                    details["position_conflict"] = "Cannot execute BUY_TO_COVER when holding 0 short shares"

        # ---------------------------------------------------------------------
        # 6. Permitted Order Size Relative to Liquidity & Market Impact
        # ---------------------------------------------------------------------
        adv = quote.adv_shares_20d if (quote and quote.adv_shares_20d > 0) else 1_000_000.0
        max_part_pct = self.policy.execution.max_adv_participation_pct
        max_allowed_shares = max(1, int(adv * max_part_pct))

        if requested_shares > max_allowed_shares:
            rejection_reasons.append(PreFlightRejectionReason.LIQUIDITY_LIMIT_EXCEEDED.value)
            details["liquidity_limit"] = (
                f"Requested shares {requested_shares} exceeds {max_part_pct * 100:.1f}% ADV cap ({max_allowed_shares} shares)"
            )

        # Square-root market impact: 10000 * sigma * sqrt(shares / ADV)
        atr_pct = (quote.atr_14d / quote.last_price) if (quote and quote.atr_14d and quote.last_price > 0) else 0.02
        market_impact_bps = 10000.0 * atr_pct * math.sqrt(requested_shares / max(1.0, adv))
        details["estimated_market_impact_bps"] = round(market_impact_bps, 2)

        if market_impact_bps > self.policy.execution.max_market_impact_bps:
            rejection_reasons.append(PreFlightRejectionReason.MARKET_IMPACT_EXCEEDED.value)
            details["market_impact_error"] = (
                f"Estimated market impact {market_impact_bps:.1f} bps exceeds maximum allowed {self.policy.execution.max_market_impact_bps:.1f} bps"
            )

        # ---------------------------------------------------------------------
        # 7. Cash Balance & Margin Sufficiency
        # ---------------------------------------------------------------------
        required_funds = 0.0
        free_cash_after = 0.0
        if quote and quote.last_price > 0 and portfolio is not None:
            comm_per_share = self.policy.execution.default_commission_per_share
            min_comm = self.policy.execution.min_commission_per_order
            est_comm = max(min_comm, requested_shares * comm_per_share)
            slippage_mult = 1.0 + (self.policy.trade_quality.slippage_bps / 10000.0)

            if side_enum == OrderSide.BUY:
                exec_price = (quote.ask_price if quote.ask_price > 0 else quote.last_price) * slippage_mult
                required_funds = (requested_shares * exec_price) + est_comm
                free_cash_after = portfolio.cash - required_funds
                min_reserve = portfolio.portfolio_equity * self.policy.portfolio_risk.min_cash_reserve_pct
                if free_cash_after < min_reserve:
                    rejection_reasons.append(PreFlightRejectionReason.INSUFFICIENT_CASH.value)
                    details["cash_sufficiency"] = (
                        f"Post-trade cash ${free_cash_after:,.2f} violates minimum cash reserve ${min_reserve:,.2f}"
                    )
            elif side_enum == OrderSide.SELL_SHORT:
                exec_price = (quote.bid_price if quote.bid_price > 0 else quote.last_price)
                req_margin = (
                    requested_shares
                    * exec_price
                    * self.policy.portfolio_risk.short_margin_requirement_pct
                ) + est_comm
                required_funds = req_margin
                free_cash_after = portfolio.cash - required_funds
                min_reserve = portfolio.portfolio_equity * self.policy.portfolio_risk.min_cash_reserve_pct
                if free_cash_after < min_reserve:
                    rejection_reasons.append(PreFlightRejectionReason.INSUFFICIENT_MARGIN.value)
                    details["margin_sufficiency"] = (
                        f"Post-trade cash margin ${free_cash_after:,.2f} violates minimum cash reserve ${min_reserve:,.2f}"
                    )
            else:
                free_cash_after = portfolio.cash

        details["required_cash_or_margin"] = round(required_funds, 2)
        details["free_cash_after_order"] = round(free_cash_after, 2)

        # ---------------------------------------------------------------------
        # 8. Stop and Target Price Validity Against Latest Available Price
        # ---------------------------------------------------------------------
        live_rr: Optional[float] = None
        price_drift_pct: float = 0.0

        if quote and quote.last_price > 0:
            live_p = quote.last_price

            # Price drift guard relative to signal generation price
            if signal_price and signal_price > 0:
                price_drift_pct = abs(live_p - signal_price) / signal_price
                details["price_drift_pct"] = round(price_drift_pct, 4)
                if price_drift_pct > self.policy.execution.max_price_drift_pct:
                    rejection_reasons.append(PreFlightRejectionReason.EXCESSIVE_PRICE_DRIFT.value)
                    details["drift_error"] = (
                        f"Live price drift {price_drift_pct * 100:.2f}% exceeds tolerance {self.policy.execution.max_price_drift_pct * 100:.2f}%"
                    )

            # Inverted stop / target validation
            if side_enum == OrderSide.BUY:
                if stop_loss_price is not None:
                    if stop_loss_price <= 0.0:
                        rejection_reasons.append(PreFlightRejectionReason.INVALID_STOP_PRICE.value)
                        details["stop_geometry"] = f"Stop loss price {stop_loss_price} must be positive"
                    elif live_p <= stop_loss_price:
                        rejection_reasons.append(PreFlightRejectionReason.INVALID_STOP_PRICE.value)
                        details["stop_geometry"] = (
                            f"Live price ${live_p:.2f} is at or below stop loss ${stop_loss_price:.2f} (Trade dead-on-arrival)"
                        )

                if target_price is not None:
                    if target_price <= live_p:
                        rejection_reasons.append(PreFlightRejectionReason.INVALID_TARGET_PRICE.value)
                        details["target_geometry"] = (
                            f"Live price ${live_p:.2f} is at or above target price ${target_price:.2f} (Profit window passed)"
                        )

                if stop_loss_price and target_price and (live_p > stop_loss_price):
                    reward = target_price - live_p
                    risk = live_p - stop_loss_price
                    if risk > 0:
                        live_rr = reward / risk
                        details["live_reward_risk_ratio"] = round(live_rr, 2)
                        if live_rr < self.policy.trade_quality.reward_risk_minimum:
                            rejection_reasons.append(PreFlightRejectionReason.SUB_MINIMAL_REWARD_RISK.value)
                            details["rr_error"] = (
                                f"Live R:R {live_rr:.2f} below hurdle {self.policy.trade_quality.reward_risk_minimum:.2f}"
                            )

            elif side_enum == OrderSide.SELL_SHORT:
                if stop_loss_price is not None:
                    if live_p >= stop_loss_price:
                        rejection_reasons.append(PreFlightRejectionReason.INVALID_STOP_PRICE.value)
                        details["short_stop_geometry"] = (
                            f"Live price ${live_p:.2f} is at or above short stop loss ${stop_loss_price:.2f}"
                        )

                if target_price is not None:
                    if target_price <= 0.0 or live_p <= target_price:
                        rejection_reasons.append(PreFlightRejectionReason.INVALID_TARGET_PRICE.value)
                        details["short_target_geometry"] = (
                            f"Live price ${live_p:.2f} is at or below short target ${target_price:.2f}"
                        )

                if stop_loss_price and target_price and (stop_loss_price > live_p):
                    reward = live_p - target_price
                    risk = stop_loss_price - live_p
                    if risk > 0:
                        live_rr = reward / risk
                        details["live_reward_risk_ratio"] = round(live_rr, 2)
                        short_min_rr = self.policy.portfolio_risk.short_min_rr_threshold
                        if live_rr < short_min_rr:
                            rejection_reasons.append(PreFlightRejectionReason.SUB_MINIMAL_REWARD_RISK.value)
                            details["rr_error"] = (
                                f"Live short R:R {live_rr:.2f} below short hurdle {short_min_rr:.2f}"
                            )

        is_valid = len(rejection_reasons) == 0
        permitted_shares = requested_shares if is_valid else 0

        return PreFlightValidationResult(
            is_valid=is_valid,
            rejection_reasons=rejection_reasons,
            permitted_shares=permitted_shares,
            estimated_market_impact_bps=market_impact_bps,
            estimated_slippage_bps=self.policy.trade_quality.slippage_bps,
            required_cash_or_margin=required_funds,
            free_cash_after_order=free_cash_after,
            price_drift_pct=price_drift_pct,
            idempotency_key=idempotency_key,
            live_reward_risk_ratio=live_rr,
            details=details,
        )


# =============================================================================
# EXECUTION LEDGER
# =============================================================================

class ExecutionLedger:
    """
    Execution Ledger (Step 7.3).
    Immutable append-only ledger tracking all orders, fills, and lifecycle state events.
    Acts as the single source of truth for portfolio position reconciliation.
    """

    def __init__(self):
        self._orders: Dict[str, Order] = {}
        self._fills: List[OrderFill] = []
        self._events: List[ExecutionEvent] = []
        self._idempotency_index: Dict[str, str] = {}  # idempotency_key -> order_id

    @property
    def orders(self) -> Dict[str, Order]:
        return self._orders

    @property
    def fills(self) -> List[OrderFill]:
        return self._fills

    @property
    def events(self) -> List[ExecutionEvent]:
        return self._events

    def get_order(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    def get_all_orders(self) -> List[Order]:
        return list(self._orders.values())

    def get_open_orders(self) -> List[Order]:
        return [o for o in self._orders.values() if o.is_open]

    def get_fills_for_order(self, order_id: str) -> List[OrderFill]:
        return [f for f in self._fills if f.order_id == order_id]

    def create_order(
        self,
        ticker: str,
        side: Union[OrderSide, str],
        requested_shares: int,
        signal_price: float,
        limit_price: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        target_price: Optional[float] = None,
        order_type: OrderType = OrderType.LIMIT,
        signal_id: Optional[int] = None,
        run_id: Optional[str] = None,
        session_date: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Order:
        """
        Instantiates an Order object with deterministic identifiers.
        Initial status is set to ORDER_SUBMITTED (not executed!).
        """
        sym = ticker.upper().strip()
        side_enum = side if isinstance(side, OrderSide) else OrderSide(str(side).upper())
        s_date = session_date or _today_utc_str()
        effective_price = limit_price or signal_price

        idempotency_key = generate_idempotency_key(
            ticker=sym,
            side=side_enum,
            requested_shares=requested_shares,
            price=effective_price,
            session_date=s_date,
            run_id=run_id,
        )

        order_id = f"ORD-{uuid.uuid4().hex[:12].upper()}"
        client_order_id = f"CLO-{sym}-{uuid.uuid4().hex[:8].upper()}"

        order = Order(
            order_id=order_id,
            client_order_id=client_order_id,
            idempotency_key=idempotency_key,
            ticker=sym,
            side=side_enum,
            requested_shares=requested_shares,
            order_type=order_type,
            status=OrderStatus.ORDER_SUBMITTED,
            signal_id=signal_id,
            run_id=run_id,
            signal_price=signal_price,
            limit_price=limit_price,
            stop_loss_price=stop_loss_price,
            target_price=target_price,
            metadata=metadata or {},
        )

        self._orders[order_id] = order
        self._idempotency_index[idempotency_key] = order_id

        # Log creation/submission event
        event = ExecutionEvent(
            event_id=f"EVT-{uuid.uuid4().hex[:10].upper()}",
            order_id=order_id,
            event_type="SUBMITTED",
            from_status=OrderStatus.SIGNAL_GENERATED.value,
            to_status=OrderStatus.ORDER_SUBMITTED.value,
            details=f"Order created and submitted: {requested_shares} shares of {sym} ({side_enum.value})",
        )
        self._events.append(event)
        return order

    def acknowledge_order(
        self,
        order_id: str,
        broker_order_id: str,
        persist_db: bool = False,
        db_path: Optional[str] = None,
    ) -> Order:
        """
        Transitions order from ORDER_SUBMITTED to BROKER_ACKNOWLEDGED.
        """
        order = self._orders.get(order_id)
        if not order:
            raise KeyError(f"Order {order_id} not found in ledger")

        from_status = order.status.value
        order.broker_order_id = broker_order_id
        order.status = OrderStatus.BROKER_ACKNOWLEDGED
        order.updated_at = _now_utc_str()

        event = ExecutionEvent(
            event_id=f"EVT-{uuid.uuid4().hex[:10].upper()}",
            order_id=order_id,
            event_type="ACKNOWLEDGED",
            from_status=from_status,
            to_status=order.status.value,
            details=f"Broker acknowledged order with ID {broker_order_id}",
        )
        self._events.append(event)

        if persist_db:
            try:
                from db import save_execution_event, save_execution_order
                kwargs = {"db_path": db_path} if db_path else {}
                save_execution_order(order.to_dict(), **kwargs)
                save_execution_event(event.to_dict(), **kwargs)
            except Exception:
                pass

        return order

    def record_fill(
        self,
        fill: OrderFill,
        persist_db: bool = False,
        db_path: Optional[str] = None,
    ) -> Order:
        """
        Records an execution fill, updates average price and commission,
        and transitions order to PARTIALLY_FILLED or FILLED.
        """
        order = self._orders.get(fill.order_id)
        if not order:
            raise KeyError(f"Order {fill.order_id} not found in ledger")

        from_status = order.status.value
        old_filled = order.filled_shares
        new_filled = old_filled + fill.shares

        # Volume-weighted average fill price
        if new_filled > 0:
            order.avg_fill_price = (
                (order.avg_fill_price * old_filled) + (fill.price * fill.shares)
            ) / new_filled

        order.filled_shares = new_filled
        order.total_commission += fill.commission
        order.updated_at = _now_utc_str()

        # Determine new status
        if order.filled_shares >= order.requested_shares:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIALLY_FILLED

        self._fills.append(fill)

        event = ExecutionEvent(
            event_id=f"EVT-{uuid.uuid4().hex[:10].upper()}",
            order_id=order.order_id,
            event_type="FILL",
            from_status=from_status,
            to_status=order.status.value,
            details=(
                f"Fill recorded: {fill.shares} shares @ ${fill.price:.2f} "
                f"(Total filled: {order.filled_shares}/{order.requested_shares})"
            ),
        )
        self._events.append(event)

        if persist_db:
            try:
                from db import save_execution_event, save_execution_fill, save_execution_order
                kwargs = {"db_path": db_path} if db_path else {}
                save_execution_fill(fill.to_dict(), **kwargs)
                save_execution_order(order.to_dict(), **kwargs)
                save_execution_event(event.to_dict(), **kwargs)
            except Exception:
                pass

        return order

    def cancel_order(
        self,
        order_id: str,
        reason: str,
        persist_db: bool = False,
        db_path: Optional[str] = None,
    ) -> Order:
        """
        Cancels an open order.
        """
        order = self._orders.get(order_id)
        if not order:
            raise KeyError(f"Order {order_id} not found in ledger")

        from_status = order.status.value
        order.status = OrderStatus.CANCELLED
        order.cancellation_reason = str(reason)
        order.updated_at = _now_utc_str()

        event = ExecutionEvent(
            event_id=f"EVT-{uuid.uuid4().hex[:10].upper()}",
            order_id=order_id,
            event_type="CANCELLED",
            from_status=from_status,
            to_status=order.status.value,
            details=f"Order cancelled: {reason}",
        )
        self._events.append(event)

        if persist_db:
            try:
                from db import save_execution_event, save_execution_order
                kwargs = {"db_path": db_path} if db_path else {}
                save_execution_order(order.to_dict(), **kwargs)
                save_execution_event(event.to_dict(), **kwargs)
            except Exception:
                pass

        return order

    def reject_order(
        self,
        order_id: str,
        reason: str,
        persist_db: bool = False,
        db_path: Optional[str] = None,
    ) -> Order:
        """
        Rejects an order before execution.
        """
        order = self._orders.get(order_id)
        if not order:
            raise KeyError(f"Order {order_id} not found in ledger")

        from_status = order.status.value
        order.status = OrderStatus.REJECTED
        order.rejection_reason = str(reason)
        order.updated_at = _now_utc_str()

        event = ExecutionEvent(
            event_id=f"EVT-{uuid.uuid4().hex[:10].upper()}",
            order_id=order_id,
            event_type="REJECTED",
            from_status=from_status,
            to_status=order.status.value,
            details=f"Order rejected: {reason}",
        )
        self._events.append(event)

        if persist_db:
            try:
                from db import save_execution_event, save_execution_order
                kwargs = {"db_path": db_path} if db_path else {}
                save_execution_order(order.to_dict(), **kwargs)
                save_execution_event(event.to_dict(), **kwargs)
            except Exception:
                pass

        return order

    def calculate_ledger_positions(self) -> Dict[str, int]:
        """
        Derives net position quantities mathematically from the fill ledger:
        - BUY adds shares (+S)
        - SELL subtracts shares (-S)
        - SELL_SHORT subtracts shares (-S)
        - BUY_TO_COVER adds shares (+S)

        Returns:
            Dict mapping ticker -> net signed shares (+ for long, - for short).
        """
        positions: Dict[str, int] = {}
        for fill in self._fills:
            ticker = fill.ticker.upper()
            side = fill.side if isinstance(fill.side, OrderSide) else OrderSide(str(fill.side).upper())

            if side == OrderSide.BUY:
                positions[ticker] = positions.get(ticker, 0) + fill.shares
            elif side == OrderSide.SELL:
                positions[ticker] = positions.get(ticker, 0) - fill.shares
            elif side == OrderSide.SELL_SHORT:
                positions[ticker] = positions.get(ticker, 0) - fill.shares
            elif side == OrderSide.BUY_TO_COVER:
                positions[ticker] = positions.get(ticker, 0) + fill.shares

        return positions

    def calculate_ledger_cash(self, initial_cash: float) -> float:
        """
        Derives cash balance from the fill ledger starting from initial cash.
        - BUY reduces cash: -(shares * price + commission)
        - SELL increases cash: +(shares * price - commission)
        - SELL_SHORT provides gross cash proceeds: +(shares * price - commission)
        - BUY_TO_COVER reduces cash: -(shares * price + commission)
        """
        cash = initial_cash
        for fill in self._fills:
            side = fill.side if isinstance(fill.side, OrderSide) else OrderSide(str(fill.side).upper())
            trade_value = fill.shares * fill.price
            comm = fill.commission

            if side == OrderSide.BUY:
                cash -= (trade_value + comm)
            elif side == OrderSide.SELL:
                cash += (trade_value - comm)
            elif side == OrderSide.SELL_SHORT:
                cash += (trade_value - comm)
            elif side == OrderSide.BUY_TO_COVER:
                cash -= (trade_value + comm)

        return cash


# =============================================================================
# POSITION RECONCILER
# =============================================================================

class PositionReconciler:
    """
    Position State Reconciler (Step 7.3 Acceptance Gate).
    Reconciles internal portfolio position state with the execution ledger.
    Asserts zero discrepancy and automatically engages the operational kill switch
    if any mismatch is detected.
    """

    def __init__(
        self,
        policy: Optional[TradingPolicy] = None,
        kill_switch: Optional[OperationalKillSwitch] = None,
    ):
        self.policy = policy or get_default_policy()
        self.kill_switch = kill_switch or GLOBAL_KILL_SWITCH

    def reconcile(
        self,
        portfolio: PortfolioSnapshot,
        ledger: ExecutionLedger,
        initial_cash: Optional[float] = None,
        cash_tolerance: float = 0.01,
    ) -> ReconciliationResult:
        """
        Reconciles portfolio positions and cash against execution ledger.
        """
        discrepancies: List[ReconciliationDiscrepancy] = []

        # 1. Map internal portfolio positions to signed net share quantities
        internal_net_shares: Dict[str, int] = {}
        for ticker, pos in portfolio.positions.items():
            sym = ticker.upper()
            if pos.direction.upper() == "LONG":
                internal_net_shares[sym] = abs(pos.shares)
            elif pos.direction.upper() == "SHORT":
                internal_net_shares[sym] = -abs(pos.shares)
            else:
                internal_net_shares[sym] = 0

        # 2. Derive ledger positions
        ledger_net_shares = ledger.calculate_ledger_positions()

        # Combine all tickers
        all_tickers = set(internal_net_shares.keys()).union(set(ledger_net_shares.keys()))

        for ticker in sorted(all_tickers):
            int_shares = internal_net_shares.get(ticker, 0)
            ledg_shares = ledger_net_shares.get(ticker, 0)

            if int_shares != ledg_shares:
                if int_shares != 0 and ledg_shares == 0:
                    discrepancies.append(
                        ReconciliationDiscrepancy(
                            ticker=ticker,
                            discrepancy_type="PHANTOM_INTERNAL_POSITION",
                            internal_value=int_shares,
                            ledger_value=0,
                            description=f"Internal portfolio holds {int_shares} shares of {ticker} with zero fills in ledger",
                        )
                    )
                elif int_shares == 0 and ledg_shares != 0:
                    discrepancies.append(
                        ReconciliationDiscrepancy(
                            ticker=ticker,
                            discrepancy_type="PHANTOM_LEDGER_POSITION",
                            internal_value=0,
                            ledger_value=ledg_shares,
                            description=f"Ledger derived {ledg_shares} shares of {ticker} missing from internal portfolio",
                        )
                    )
                elif (int_shares > 0 and ledg_shares < 0) or (int_shares < 0 and ledg_shares > 0):
                    discrepancies.append(
                        ReconciliationDiscrepancy(
                            ticker=ticker,
                            discrepancy_type="DIRECTION_MISMATCH",
                            internal_value=int_shares,
                            ledger_value=ledg_shares,
                            description=f"Direction mismatch for {ticker}: internal={int_shares}, ledger={ledg_shares}",
                        )
                    )
                else:
                    discrepancies.append(
                        ReconciliationDiscrepancy(
                            ticker=ticker,
                            discrepancy_type="POSITION_SHARES_MISMATCH",
                            internal_value=int_shares,
                            ledger_value=ledg_shares,
                            description=f"Share quantity mismatch for {ticker}: internal={int_shares}, ledger={ledg_shares}",
                        )
                    )

        # 3. Reconcile cash balance if initial cash provided
        derived_cash = 0.0
        if initial_cash is not None:
            derived_cash = ledger.calculate_ledger_cash(initial_cash)
            cash_diff = abs(portfolio.cash - derived_cash)
            if cash_diff > cash_tolerance:
                discrepancies.append(
                    ReconciliationDiscrepancy(
                        ticker=None,
                        discrepancy_type="CASH_MISMATCH",
                        internal_value=round(portfolio.cash, 2),
                        ledger_value=round(derived_cash, 2),
                        description=(
                            f"Cash discrepancy detected: internal portfolio cash (${portfolio.cash:,.2f}) "
                            f"vs ledger cash (${derived_cash:,.2f}) diff=${cash_diff:,.2f}"
                        ),
                    )
                )

        is_reconciled = len(discrepancies) == 0

        # Trip operational kill switch if configured and mismatch detected
        if not is_reconciled and self.policy.execution.kill_switch_on_reconciliation_failure:
            disc_summaries = "; ".join(d.description for d in discrepancies[:3])
            self.kill_switch.trip(
                reason=KillSwitchReason.ORDER_RECONCILIATION_FAILURE,
                details=f"Reconciliation failure: {disc_summaries}",
                actor="POSITION_RECONCILER",
            )

        return ReconciliationResult(
            is_reconciled=is_reconciled,
            discrepancies=discrepancies,
            internal_positions=internal_net_shares,
            ledger_positions=ledger_net_shares,
            internal_cash=portfolio.cash,
            ledger_cash=derived_cash,
        )


# =============================================================================
# SIMULATED BROKER ADAPTER
# =============================================================================

class SimulatedBroker:
    """
    Simulated Broker Execution Adapter.
    Executes orders in testing and simulated environments with realistic
    slippage (spread + square-root market impact) and transaction commissions.
    """

    def __init__(self, policy: Optional[TradingPolicy] = None):
        self.policy = policy or get_default_policy()

    def execute_order(
        self,
        order: Order,
        quote: MarketQuote,
        ledger: ExecutionLedger,
        partial_fill_ratios: Optional[List[float]] = None,
        simulate_rejection: bool = False,
        rejection_reason: str = "BROKER_REJECTED",
        persist_db: bool = False,
        db_path: Optional[str] = None,
    ) -> Tuple[Order, List[OrderFill]]:
        """
        Simulates broker acknowledgement, realistic slippage execution,
        and fill event dispatching back to the execution ledger.
        """
        if simulate_rejection:
            updated_order = ledger.reject_order(
                order_id=order.order_id,
                reason=rejection_reason,
                persist_db=persist_db,
                db_path=db_path,
            )
            return updated_order, []

        # 1. Acknowledge Order
        broker_order_id = f"BRK-{uuid.uuid4().hex[:8].upper()}"
        ledger.acknowledge_order(
            order_id=order.order_id,
            broker_order_id=broker_order_id,
            persist_db=persist_db,
            db_path=db_path,
        )

        # 2. Compute Execution Price with Slippage
        # Spread half-cost + market impact
        spread = max(0.01, quote.ask_price - quote.bid_price) if quote.ask_price > quote.bid_price else 0.02
        half_spread = spread / 2.0
        adv = max(1.0, quote.adv_shares_20d)
        atr_pct = (quote.atr_14d / quote.last_price) if (quote.atr_14d and quote.last_price > 0) else 0.02
        market_impact_pct = atr_pct * math.sqrt(order.requested_shares / adv)

        total_slippage_dollars = half_spread + (quote.last_price * market_impact_pct)
        slippage_bps = (total_slippage_dollars / quote.last_price) * 10000.0 if quote.last_price > 0 else 5.0

        if order.side in {OrderSide.BUY, OrderSide.BUY_TO_COVER}:
            base_fill_price = quote.last_price + total_slippage_dollars
        else:
            base_fill_price = max(0.01, quote.last_price - total_slippage_dollars)

        # 3. Execute Fills
        ratios = partial_fill_ratios or [1.0]
        sum_ratios = sum(ratios)
        if sum_ratios <= 0:
            ratios = [1.0]
            sum_ratios = 1.0

        fills: List[OrderFill] = []
        shares_accounted = 0

        for i, ratio in enumerate(ratios):
            if i == len(ratios) - 1:
                fill_shares = order.requested_shares - shares_accounted
            else:
                fill_shares = int(order.requested_shares * (ratio / sum_ratios))

            if fill_shares <= 0:
                continue

            shares_accounted += fill_shares
            comm_per_share = self.policy.execution.default_commission_per_share
            min_comm = self.policy.execution.min_commission_per_order
            fill_comm = max(min_comm / len(ratios), fill_shares * comm_per_share)

            fill = OrderFill(
                fill_id=f"FIL-{uuid.uuid4().hex[:10].upper()}",
                order_id=order.order_id,
                ticker=order.ticker,
                side=order.side,
                shares=fill_shares,
                price=round(base_fill_price, 2),
                commission=round(fill_comm, 2),
                execution_id=f"EX-{uuid.uuid4().hex[:8].upper()}",
                slippage_bps=round(slippage_bps, 2),
            )
            ledger.record_fill(fill, persist_db=persist_db, db_path=db_path)
            fills.append(fill)

        return order, fills
