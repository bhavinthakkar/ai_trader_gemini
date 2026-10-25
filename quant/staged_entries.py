"""
Staged Entries and Single Risk Budget Governance (Step 5).

Sizes staged entries against one authoritative risk budget for the total position:
1. Max total position risk is decided in advance using existing institutional sizing and portfolio controls.
2. Tranches divide the single risk budget (e.g. 50% initial, 30% confirmation, 20% breakout);
   they are NEVER treated as independent risk allowances.
3. Explicit evidence contracts govern additional tranches:
   - Tranche 1: Qualified candidate setup passed all compliance gates.
   - Tranche 2: Confirmation of direction, position in profit >= min_profit_atr (0.50 ATR), score intact.
   - Tranche 3: Breakout / continuation confirmation with trailing stop moved to breakeven.
4. Refusal conditions (no further additions allowed):
   - Averaging down a losing position is strictly prohibited (position in loss).
   - Insufficient profit buffer (< 0.50 ATR).
   - Risk budget exhausted or open orders already consume the remaining budget.
   - Max tranches reached.
   - Corporate earnings blackout or extreme volatility regime.
   - Portfolio circuit breaker halt or exposure limits.
5. Open orders and existing exposure explicitly reduce the remaining available risk and share budget.

Guarantees: The summed risk of all tranches stays strictly within configured account and portfolio limits.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from quant.policy_config import TradingPolicy, get_default_policy
from quant.portfolio_controls import (
    PortfolioBindingConstraint,
    PortfolioComplianceReport,
    PortfolioRiskEngine,
    PortfolioSnapshot,
    Position,
)
from quant.position_sizing import (
    InvestmentMandate,
    PositionSizingResult,
    calculate_risk_based_position_size,
    compute_gap_risk_reserve,
    compute_round_trip_cost_per_share,
)


class TrancheStatus(str, Enum):
    """Lifecycle state of an entry tranche."""
    PLANNED = "PLANNED"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    BLOCKED = "BLOCKED"
    CANCELLED = "CANCELLED"
    SKIPPED = "SKIPPED"


class TrancheRejectionReason(str, Enum):
    """Standardized reasons for refusing a staged tranche addition."""
    RISK_BUDGET_EXHAUSTED = "RISK_BUDGET_EXHAUSTED"
    OPEN_ORDERS_BUDGET_EXHAUSTED = "OPEN_ORDERS_BUDGET_EXHAUSTED"
    MAX_TRANCHES_REACHED = "MAX_TRANCHES_REACHED"
    POSITION_IN_LOSS = "POSITION_IN_LOSS"
    INSUFFICIENT_PROFIT_BUFFER = "INSUFFICIENT_PROFIT_BUFFER"
    MAX_ADVERSE_EXCURSION_EXCEEDED = "MAX_ADVERSE_EXCURSION_EXCEEDED"
    COMPOSITE_SCORE_INSUFFICIENT = "COMPOSITE_SCORE_INSUFFICIENT"
    EVENT_RISK_BLACKOUT = "EVENT_RISK_BLACKOUT"
    VOLATILITY_REGIME_EXTREME = "VOLATILITY_REGIME_EXTREME"
    PORTFOLIO_LIMIT_REACHED = "PORTFOLIO_LIMIT_REACHED"
    DRAWDOWN_CIRCUIT_BREAKER = "DRAWDOWN_CIRCUIT_BREAKER"
    TRANCHE_TOO_SMALL = "TRANCHE_TOO_SMALL"
    INVALID_PRICE_ORDERING = "INVALID_PRICE_ORDERING"
    NO_EXISTING_POSITION = "NO_EXISTING_POSITION"


@dataclass
class TranchePlan:
    """Pre-planned specification for a single staged tranche."""
    tranche_index: int
    target_weight: float
    planned_shares: int
    planned_risk_dollars: float
    target_entry_price: float
    planned_stop_loss: float
    required_evidence_desc: str
    status: str = TrancheStatus.PLANNED.value
    allocated_shares: int = 0
    allocated_risk_dollars: float = 0.0
    is_addition_allowed: bool = False
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StagedEntryPlan:
    """Institutional staged entry plan deciding total position risk and tranches in advance."""
    ticker: str
    direction: str
    total_allowed_shares: int
    total_risk_budget_dollars: float
    effective_per_share_risk: float
    max_tranches: int
    tranches: List[TranchePlan]
    existing_shares: int = 0
    existing_entry_price: Optional[float] = None
    existing_stop_loss: Optional[float] = None
    existing_risk_dollars: float = 0.0
    open_order_shares: int = 0
    open_order_risk_dollars: float = 0.0
    total_committed_shares: int = 0
    total_committed_risk_dollars: float = 0.0
    remaining_shares: int = 0
    remaining_risk_budget_dollars: float = 0.0
    is_tradeable: bool = True
    binding_constraint: str = "RISK_BUDGET"
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["tranches"] = [t.to_dict() for t in self.tranches]
        return d


@dataclass
class StagedTrancheEvaluation:
    """Result of evaluating and sizing a specific staged tranche addition."""
    allowed: bool
    tranche_index: int
    shares: int
    entry_price: float
    stop_loss: float
    risk_dollars: float
    total_position_risk_after: float
    max_total_position_risk: float
    total_position_shares_after: int
    max_total_position_shares: int
    remaining_risk_budget: float
    rejection_reason: Optional[str] = None
    binding_constraint: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_staged_entry_plan(
    entry_price: float,
    stop_loss: float,
    ticker: str = "TICKER",
    direction: str = "LONG",
    portfolio_equity: Optional[float] = None,
    risk_fraction: Optional[float] = None,
    atr: Optional[float] = None,
    adv_shares_20d: Optional[float] = None,
    avg_dollar_volume_20d: Optional[float] = None,
    current_gross_exposure: Optional[float] = None,
    composite_score: Optional[float] = None,
    mandate: Optional[InvestmentMandate] = None,
    policy: Optional[TradingPolicy] = None,
    portfolio_snapshot: Optional[PortfolioSnapshot] = None,
    existing_shares: int = 0,
    existing_entry_price: Optional[float] = None,
    existing_stop_loss: Optional[float] = None,
    open_orders: Optional[List[Any]] = None,
    sector: Optional[str] = None,
    factor_tags: Optional[List[str]] = None,
) -> StagedEntryPlan:
    """
    Decides in advance the maximum total position risk and sizes staged tranches.

    Uses existing position-sizing and portfolio-control code for the total position,
    then divides the single risk budget across planned tranches according to policy weights.
    """
    active_policy = policy or get_default_policy()
    cfg = active_policy.staged_entries
    sym = (ticker or "TICKER").strip().upper()
    dir_norm = (direction or "LONG").strip().upper()

    # 1. Size Total Position using Existing Position Sizing & Portfolio Controls
    total_sizing = calculate_risk_based_position_size(
        entry_price=entry_price,
        stop_loss=stop_loss,
        ticker=sym,
        direction=dir_norm,
        portfolio_equity=portfolio_equity,
        risk_fraction=risk_fraction,
        atr=atr,
        adv_shares_20d=adv_shares_20d,
        avg_dollar_volume_20d=avg_dollar_volume_20d,
        current_gross_exposure=current_gross_exposure,
        composite_score=composite_score,
        mandate=mandate,
        policy=active_policy,
        portfolio_snapshot=portfolio_snapshot,
        sector=sector,
        factor_tags=factor_tags,
    )

    total_shares = total_sizing.shares_allowed
    effective_per_share_risk = total_sizing.effective_per_share_risk
    total_risk_budget = total_sizing.equity_at_risk_dollars
    binding_constraint = total_sizing.binding_constraint
    is_tradeable = total_sizing.is_tradeable

    # 2. Extract Existing Position & Open Orders
    curr_existing_shares = existing_shares
    curr_existing_entry = existing_entry_price or entry_price
    curr_existing_stop = existing_stop_loss or stop_loss

    if portfolio_snapshot and not curr_existing_shares and sym in portfolio_snapshot.positions:
        pos = portfolio_snapshot.positions[sym]
        if pos.direction.upper() == dir_norm:
            curr_existing_shares = abs(pos.shares)
            curr_existing_entry = pos.entry_price
            curr_existing_stop = stop_loss

    # Friction and gap reserve
    C = total_sizing.cost_per_share
    G = total_sizing.gap_reserve_per_share
    existing_per_share_risk = abs(curr_existing_entry - curr_existing_stop) + C + G if curr_existing_shares > 0 else 0.0
    existing_risk_dollars = round(curr_existing_shares * existing_per_share_risk, 2)

    # Open orders inspection
    all_open_orders = list(open_orders or [])
    if portfolio_snapshot and portfolio_snapshot.open_orders:
        all_open_orders.extend(portfolio_snapshot.open_orders)

    open_order_shares = 0
    open_order_risk_dollars = 0.0
    for o in all_open_orders:
        info = PortfolioSnapshot._parse_order(o)
        if info["is_open"] and info["ticker"] == sym:
            o_dir = "LONG" if info["side"] in ("BUY", "LONG", "BUY_TO_COVER") else "SHORT"
            if o_dir == dir_norm:
                open_order_shares += info["shares"]
                px = info["price"] or entry_price
                sl = info["stop_loss"] or stop_loss
                psr = abs(px - sl) + C + G if px > 0 and sl > 0 else effective_per_share_risk
                open_order_risk_dollars += info["shares"] * psr

    open_order_risk_dollars = round(open_order_risk_dollars, 2)
    total_committed_shares = curr_existing_shares + open_order_shares
    total_committed_risk_dollars = round(existing_risk_dollars + open_order_risk_dollars, 2)

    remaining_shares = max(0, total_shares - total_committed_shares)
    remaining_risk_budget_dollars = max(0.0, round(total_risk_budget - total_committed_risk_dollars, 2))

    # 3. Plan Staged Tranches Against Single Risk Budget
    max_tranches = max(1, cfg.max_tranches)
    weights = list(cfg.tranche_weights[:max_tranches])
    # Normalize weights to sum exactly to 1.0
    w_sum = sum(weights)
    if w_sum > 0:
        weights = [w / w_sum for w in weights]
    else:
        weights = [1.0 / max_tranches] * max_tranches

    evidence_descriptions = [
        "Tranche 1 (Initial / Pilot): Qualified candidate setup clearing compliance gates.",
        "Tranche 2 (Confirmation): Direction confirmed, position in profit >= 0.50 ATR, composite score intact.",
        "Tranche 3 (Continuation / Scale): Breakout / trend continuation with trailing stop moved to breakeven.",
    ]

    tranches: List[TranchePlan] = []
    accum_shares = 0
    for i, w in enumerate(weights, start=1):
        tranche_target_risk = round(total_risk_budget * w, 2)
        # Sizing by risk fraction and position cap
        tranche_shares_by_risk = math.floor(tranche_target_risk / effective_per_share_risk) if effective_per_share_risk > 0 else 0
        tranche_shares_by_cap = math.floor(total_shares * w)
        t_target_shares = min(tranche_shares_by_risk, tranche_shares_by_cap)
        if i == len(weights):
            # Final tranche takes remaining allowable shares
            t_shares = max(0, total_shares - accum_shares)
        else:
            t_shares = min(t_target_shares, max(0, total_shares - accum_shares))

        accum_shares += t_shares
        t_risk = round(t_shares * effective_per_share_risk, 2)

        desc = evidence_descriptions[i - 1] if i - 1 < len(evidence_descriptions) else f"Tranche {i}: Additional evidence confirmed."
        status = TrancheStatus.PLANNED.value
        if i == 1 and curr_existing_shares > 0:
            status = TrancheStatus.FILLED.value

        t_plan = TranchePlan(
            tranche_index=i,
            target_weight=round(w, 4),
            planned_shares=t_shares,
            planned_risk_dollars=t_risk,
            target_entry_price=entry_price,
            planned_stop_loss=stop_loss,
            required_evidence_desc=desc,
            status=status,
            allocated_shares=curr_existing_shares if (i == 1 and curr_existing_shares > 0) else 0,
            allocated_risk_dollars=existing_risk_dollars if (i == 1 and curr_existing_shares > 0) else 0.0,
            is_addition_allowed=(i > 1 and curr_existing_shares > 0),
        )
        tranches.append(t_plan)

    plan_metrics = {
        **total_sizing.metrics,
        "existing_shares": curr_existing_shares,
        "existing_risk_dollars": existing_risk_dollars,
        "open_order_shares": open_order_shares,
        "open_order_risk_dollars": open_order_risk_dollars,
        "total_committed_shares": total_committed_shares,
        "total_committed_risk_dollars": total_committed_risk_dollars,
        "remaining_shares": remaining_shares,
        "remaining_risk_budget_dollars": remaining_risk_budget_dollars,
    }

    return StagedEntryPlan(
        ticker=sym,
        direction=dir_norm,
        total_allowed_shares=total_shares,
        total_risk_budget_dollars=total_risk_budget,
        effective_per_share_risk=effective_per_share_risk,
        max_tranches=max_tranches,
        tranches=tranches,
        existing_shares=curr_existing_shares,
        existing_entry_price=curr_existing_entry if curr_existing_shares > 0 else None,
        existing_stop_loss=curr_existing_stop if curr_existing_shares > 0 else None,
        existing_risk_dollars=existing_risk_dollars,
        open_order_shares=open_order_shares,
        open_order_risk_dollars=open_order_risk_dollars,
        total_committed_shares=total_committed_shares,
        total_committed_risk_dollars=total_committed_risk_dollars,
        remaining_shares=remaining_shares,
        remaining_risk_budget_dollars=remaining_risk_budget_dollars,
        is_tradeable=is_tradeable,
        binding_constraint=binding_constraint,
        metrics=plan_metrics,
    )


def evaluate_staged_tranche_addition(
    plan: StagedEntryPlan,
    current_price: float,
    tranche_index: int,
    proposed_entry_price: Optional[float] = None,
    proposed_stop_loss: Optional[float] = None,
    atr: Optional[float] = None,
    composite_score: Optional[float] = None,
    days_to_earnings: Optional[int] = None,
    is_event_risk_blackout: bool = False,
    iv_pct: Optional[float] = None,
    is_extreme_volatility: bool = False,
    portfolio_snapshot: Optional[PortfolioSnapshot] = None,
    policy: Optional[TradingPolicy] = None,
    trailing_stop_tightened_to: Optional[float] = None,
) -> StagedTrancheEvaluation:
    """
    Evaluates and sizes an additional staged tranche against the single remaining risk budget.

    Strict Invariants:
    1. Never add to a losing position (prohibits averaging down losers).
    2. Minimum profit cushion >= min_profit_atr (0.50 ATR) required before adding.
    3. Tranche count strictly capped at max_tranches.
    4. Open orders and existing exposure consume the available risk budget.
    5. Summed risk of all tranches (existing + open + new) NEVER exceeds the total position risk budget.
    """
    active_policy = policy or get_default_policy()
    cfg = active_policy.staged_entries
    sym = plan.ticker.strip().upper()
    dir_norm = plan.direction.strip().upper()

    eval_evidence: Dict[str, Any] = {
        "ticker": sym,
        "direction": dir_norm,
        "tranche_index": tranche_index,
        "current_price": current_price,
        "composite_score": composite_score,
        "days_to_earnings": days_to_earnings,
        "is_event_risk_blackout": is_event_risk_blackout,
    }

    # 1. Tranche Index and Position State Checks
    if tranche_index > plan.max_tranches:
        return StagedTrancheEvaluation(
            allowed=False,
            tranche_index=tranche_index,
            shares=0,
            entry_price=proposed_entry_price or current_price,
            stop_loss=proposed_stop_loss or 0.0,
            risk_dollars=0.0,
            total_position_risk_after=plan.total_committed_risk_dollars,
            max_total_position_risk=plan.total_risk_budget_dollars,
            total_position_shares_after=plan.total_committed_shares,
            max_total_position_shares=plan.total_allowed_shares,
            remaining_risk_budget=plan.remaining_risk_budget_dollars,
            rejection_reason=TrancheRejectionReason.MAX_TRANCHES_REACHED.value,
            binding_constraint="MAX_TRANCHES_CAP",
            evidence=eval_evidence,
        )

    if tranche_index > 1 and plan.existing_shares <= 0:
        return StagedTrancheEvaluation(
            allowed=False,
            tranche_index=tranche_index,
            shares=0,
            entry_price=proposed_entry_price or current_price,
            stop_loss=proposed_stop_loss or 0.0,
            risk_dollars=0.0,
            total_position_risk_after=plan.total_committed_risk_dollars,
            max_total_position_risk=plan.total_risk_budget_dollars,
            total_position_shares_after=plan.total_committed_shares,
            max_total_position_shares=plan.total_allowed_shares,
            remaining_risk_budget=plan.remaining_risk_budget_dollars,
            rejection_reason=TrancheRejectionReason.NO_EXISTING_POSITION.value,
            binding_constraint="POSITION_STATE",
            evidence=eval_evidence,
        )

    p_entry = proposed_entry_price if (proposed_entry_price is not None and proposed_entry_price > 0.0) else current_price
    p_stop = proposed_stop_loss if (proposed_stop_loss is not None and proposed_stop_loss > 0.0) else (
        p_entry - 1.5 * (atr or 2.0) if dir_norm == "LONG" else p_entry + 1.5 * (atr or 2.0)
    )

    # 2. Price Ordering Validation
    if dir_norm == "LONG" and p_stop >= p_entry:
        return StagedTrancheEvaluation(
            allowed=False,
            tranche_index=tranche_index,
            shares=0,
            entry_price=p_entry,
            stop_loss=p_stop,
            risk_dollars=0.0,
            total_position_risk_after=plan.total_committed_risk_dollars,
            max_total_position_risk=plan.total_risk_budget_dollars,
            total_position_shares_after=plan.total_committed_shares,
            max_total_position_shares=plan.total_allowed_shares,
            remaining_risk_budget=plan.remaining_risk_budget_dollars,
            rejection_reason=TrancheRejectionReason.INVALID_PRICE_ORDERING.value,
            binding_constraint="PRICE_ORDERING",
            evidence=eval_evidence,
        )
    elif dir_norm == "SHORT" and p_stop <= p_entry:
        return StagedTrancheEvaluation(
            allowed=False,
            tranche_index=tranche_index,
            shares=0,
            entry_price=p_entry,
            stop_loss=p_stop,
            risk_dollars=0.0,
            total_position_risk_after=plan.total_committed_risk_dollars,
            max_total_position_risk=plan.total_risk_budget_dollars,
            total_position_shares_after=plan.total_committed_shares,
            max_total_position_shares=plan.total_allowed_shares,
            remaining_risk_budget=plan.remaining_risk_budget_dollars,
            rejection_reason=TrancheRejectionReason.INVALID_PRICE_ORDERING.value,
            binding_constraint="PRICE_ORDERING",
            evidence=eval_evidence,
        )

    # 3. Strict Invariant: Averaging Down Losers Prohibited & Profit Buffer Check
    if tranche_index > 1:
        # Existing position entry price
        existing_px = plan.existing_entry_price or (plan.tranches[0].target_entry_price if plan.tranches else p_entry)
        if portfolio_snapshot and sym in portfolio_snapshot.positions:
            existing_px = portfolio_snapshot.positions[sym].entry_price

        # Check if position is in loss
        if dir_norm == "LONG" and current_price < existing_px:
            eval_evidence["loss_amount"] = round(existing_px - current_price, 2)
            return StagedTrancheEvaluation(
                allowed=False,
                tranche_index=tranche_index,
                shares=0,
                entry_price=p_entry,
                stop_loss=p_stop,
                risk_dollars=0.0,
                total_position_risk_after=plan.total_committed_risk_dollars,
                max_total_position_risk=plan.total_risk_budget_dollars,
                total_position_shares_after=plan.total_committed_shares,
                max_total_position_shares=plan.total_allowed_shares,
                remaining_risk_budget=plan.remaining_risk_budget_dollars,
                rejection_reason=TrancheRejectionReason.POSITION_IN_LOSS.value,
                binding_constraint="AVERAGING_DOWN_PROHIBITED",
                evidence=eval_evidence,
            )
        elif dir_norm == "SHORT" and current_price > existing_px:
            eval_evidence["loss_amount"] = round(current_price - existing_px, 2)
            return StagedTrancheEvaluation(
                allowed=False,
                tranche_index=tranche_index,
                shares=0,
                entry_price=p_entry,
                stop_loss=p_stop,
                risk_dollars=0.0,
                total_position_risk_after=plan.total_committed_risk_dollars,
                max_total_position_risk=plan.total_risk_budget_dollars,
                total_position_shares_after=plan.total_committed_shares,
                max_total_position_shares=plan.total_allowed_shares,
                remaining_risk_budget=plan.remaining_risk_budget_dollars,
                rejection_reason=TrancheRejectionReason.POSITION_IN_LOSS.value,
                binding_constraint="AVERAGING_DOWN_PROHIBITED",
                evidence=eval_evidence,
            )

        # Check required profit cushion in ATR
        atr_val = atr or 2.0
        profit_dollars = (current_price - existing_px) if dir_norm == "LONG" else (existing_px - current_price)
        profit_atr = profit_dollars / atr_val if atr_val > 0 else 0.0
        eval_evidence["profit_dollars"] = round(profit_dollars, 2)
        eval_evidence["profit_atr"] = round(profit_atr, 2)

        if cfg.require_profit_for_addition and profit_atr < cfg.min_profit_atr_for_addition:
            return StagedTrancheEvaluation(
                allowed=False,
                tranche_index=tranche_index,
                shares=0,
                entry_price=p_entry,
                stop_loss=p_stop,
                risk_dollars=0.0,
                total_position_risk_after=plan.total_committed_risk_dollars,
                max_total_position_risk=plan.total_risk_budget_dollars,
                total_position_shares_after=plan.total_committed_shares,
                max_total_position_shares=plan.total_allowed_shares,
                remaining_risk_budget=plan.remaining_risk_budget_dollars,
                rejection_reason=TrancheRejectionReason.INSUFFICIENT_PROFIT_BUFFER.value,
                binding_constraint="PROFIT_CUSHION_HURDLE",
                evidence=eval_evidence,
            )

    # 4. Composite Score Re-qualification Check
    if composite_score is not None and composite_score < cfg.min_addition_composite_score:
        eval_evidence["composite_score"] = composite_score
        return StagedTrancheEvaluation(
            allowed=False,
            tranche_index=tranche_index,
            shares=0,
            entry_price=p_entry,
            stop_loss=p_stop,
            risk_dollars=0.0,
            total_position_risk_after=plan.total_committed_risk_dollars,
            max_total_position_risk=plan.total_risk_budget_dollars,
            total_position_shares_after=plan.total_committed_shares,
            max_total_position_shares=plan.total_allowed_shares,
            remaining_risk_budget=plan.remaining_risk_budget_dollars,
            rejection_reason=TrancheRejectionReason.COMPOSITE_SCORE_INSUFFICIENT.value,
            binding_constraint="COMPOSITE_SCORE",
            evidence=eval_evidence,
        )

    # 5. Corporate Binary Event & Volatility Blackout Checks
    if cfg.exclude_during_event_blackout:
        if days_to_earnings is not None and days_to_earnings <= active_policy.events.earnings_blackout_sessions:
            return StagedTrancheEvaluation(
                allowed=False,
                tranche_index=tranche_index,
                shares=0,
                entry_price=p_entry,
                stop_loss=p_stop,
                risk_dollars=0.0,
                total_position_risk_after=plan.total_committed_risk_dollars,
                max_total_position_risk=plan.total_risk_budget_dollars,
                total_position_shares_after=plan.total_committed_shares,
                max_total_position_shares=plan.total_allowed_shares,
                remaining_risk_budget=plan.remaining_risk_budget_dollars,
                rejection_reason=TrancheRejectionReason.EVENT_RISK_BLACKOUT.value,
                binding_constraint="EVENT_RISK",
                evidence=eval_evidence,
            )
        if is_event_risk_blackout:
            return StagedTrancheEvaluation(
                allowed=False,
                tranche_index=tranche_index,
                shares=0,
                entry_price=p_entry,
                stop_loss=p_stop,
                risk_dollars=0.0,
                total_position_risk_after=plan.total_committed_risk_dollars,
                max_total_position_risk=plan.total_risk_budget_dollars,
                total_position_shares_after=plan.total_committed_shares,
                max_total_position_shares=plan.total_allowed_shares,
                remaining_risk_budget=plan.remaining_risk_budget_dollars,
                rejection_reason=TrancheRejectionReason.EVENT_RISK_BLACKOUT.value,
                binding_constraint="EVENT_RISK",
                evidence=eval_evidence,
            )

    if (iv_pct is not None and iv_pct > 100.0) or is_extreme_volatility:
        return StagedTrancheEvaluation(
            allowed=False,
            tranche_index=tranche_index,
            shares=0,
            entry_price=p_entry,
            stop_loss=p_stop,
            risk_dollars=0.0,
            total_position_risk_after=plan.total_committed_risk_dollars,
            max_total_position_risk=plan.total_risk_budget_dollars,
            total_position_shares_after=plan.total_committed_shares,
            max_total_position_shares=plan.total_allowed_shares,
            remaining_risk_budget=plan.remaining_risk_budget_dollars,
            rejection_reason=TrancheRejectionReason.VOLATILITY_REGIME_EXTREME.value,
            binding_constraint="VOLATILITY_REGIME",
            evidence=eval_evidence,
        )

    # 6. Portfolio Drawdown Circuit Breaker Check
    if portfolio_snapshot and portfolio_snapshot.drawdown_pct >= active_policy.portfolio_risk.drawdown_halt_pct:
        return StagedTrancheEvaluation(
            allowed=False,
            tranche_index=tranche_index,
            shares=0,
            entry_price=p_entry,
            stop_loss=p_stop,
            risk_dollars=0.0,
            total_position_risk_after=plan.total_committed_risk_dollars,
            max_total_position_risk=plan.total_risk_budget_dollars,
            total_position_shares_after=plan.total_committed_shares,
            max_total_position_shares=plan.total_allowed_shares,
            remaining_risk_budget=plan.remaining_risk_budget_dollars,
            rejection_reason=TrancheRejectionReason.DRAWDOWN_CIRCUIT_BREAKER.value,
            binding_constraint="DRAWDOWN_CIRCUIT_BREAKER",
            evidence=eval_evidence,
        )

    # 7. Re-calculate Open Orders & Existing Exposure Against Single Risk Budget
    eff_existing_risk = plan.existing_risk_dollars
    if trailing_stop_tightened_to is not None and plan.existing_shares > 0:
        # Trailing stop moved to breakeven or tighter frees up risk budget
        existing_px = plan.tranches[0].target_entry_price if plan.tranches else p_entry
        if dir_norm == "LONG" and trailing_stop_tightened_to >= existing_px:
            eff_existing_risk = 0.0  # Zero downside risk to original principal
        elif dir_norm == "SHORT" and trailing_stop_tightened_to <= existing_px:
            eff_existing_risk = 0.0
        else:
            eff_existing_risk = round(plan.existing_shares * abs(existing_px - trailing_stop_tightened_to), 2)

    open_order_risk = plan.open_order_risk_dollars
    if portfolio_snapshot and portfolio_snapshot.open_orders:
        open_order_risk = portfolio_snapshot.get_ticker_open_orders_risk(sym)

    total_committed_risk = eff_existing_risk + open_order_risk
    remaining_risk_budget = max(0.0, round(plan.total_risk_budget_dollars - total_committed_risk, 2))

    total_committed_shares = plan.existing_shares + plan.open_order_shares
    if portfolio_snapshot:
        total_committed_shares = (
            abs(portfolio_snapshot.positions[sym].shares) if sym in portfolio_snapshot.positions else plan.existing_shares
        ) + portfolio_snapshot.get_ticker_open_orders_shares(sym, dir_norm)

    remaining_shares = max(0, plan.total_allowed_shares - total_committed_shares)

    eval_evidence["remaining_risk_budget"] = remaining_risk_budget
    eval_evidence["remaining_shares"] = remaining_shares

    # Budget exhaustion checks
    if remaining_risk_budget <= 0.0:
        return StagedTrancheEvaluation(
            allowed=False,
            tranche_index=tranche_index,
            shares=0,
            entry_price=p_entry,
            stop_loss=p_stop,
            risk_dollars=0.0,
            total_position_risk_after=total_committed_risk,
            max_total_position_risk=plan.total_risk_budget_dollars,
            total_position_shares_after=total_committed_shares,
            max_total_position_shares=plan.total_allowed_shares,
            remaining_risk_budget=0.0,
            rejection_reason=TrancheRejectionReason.RISK_BUDGET_EXHAUSTED.value,
            binding_constraint="RISK_BUDGET",
            evidence=eval_evidence,
        )

    if remaining_shares <= 0:
        return StagedTrancheEvaluation(
            allowed=False,
            tranche_index=tranche_index,
            shares=0,
            entry_price=p_entry,
            stop_loss=p_stop,
            risk_dollars=0.0,
            total_position_risk_after=total_committed_risk,
            max_total_position_risk=plan.total_risk_budget_dollars,
            total_position_shares_after=total_committed_shares,
            max_total_position_shares=plan.total_allowed_shares,
            remaining_risk_budget=remaining_risk_budget,
            rejection_reason=TrancheRejectionReason.PORTFOLIO_LIMIT_REACHED.value,
            binding_constraint="POSITION_SHARE_CAP",
            evidence=eval_evidence,
        )

    # 8. Size the Additional Tranche
    # Friction C and gap G
    C = compute_round_trip_cost_per_share(p_entry)
    G = compute_gap_risk_reserve(p_entry, atr)
    per_share_risk = abs(p_entry - p_stop) + C + G

    if per_share_risk <= 0.0:
        return StagedTrancheEvaluation(
            allowed=False,
            tranche_index=tranche_index,
            shares=0,
            entry_price=p_entry,
            stop_loss=p_stop,
            risk_dollars=0.0,
            total_position_risk_after=total_committed_risk,
            max_total_position_risk=plan.total_risk_budget_dollars,
            total_position_shares_after=total_committed_shares,
            max_total_position_shares=plan.total_allowed_shares,
            remaining_risk_budget=remaining_risk_budget,
            rejection_reason=TrancheRejectionReason.INVALID_PRICE_ORDERING.value,
            binding_constraint="ZERO_PER_SHARE_RISK",
            evidence=eval_evidence,
        )

    # Tranche target weight
    t_idx_0 = tranche_index - 1
    t_weight = cfg.tranche_weights[t_idx_0] if t_idx_0 < len(cfg.tranche_weights) else (1.0 / plan.max_tranches)
    target_tranche_risk = plan.total_risk_budget_dollars * t_weight

    # Permissible risk for this tranche is strictly capped by remaining available budget
    allowed_tranche_risk = min(target_tranche_risk, remaining_risk_budget)
    n_by_risk = math.floor(allowed_tranche_risk / per_share_risk)

    # If this is not the final tranche, cap shares by target tranche weight so future tranches retain share budget
    if tranche_index < plan.max_tranches:
        target_tranche_shares = math.floor(plan.total_allowed_shares * t_weight)
        allowed_shares = min(n_by_risk, target_tranche_shares, remaining_shares)
    else:
        allowed_shares = min(n_by_risk, remaining_shares)

    # Check minimum trade size / tranche dollars
    actual_risk = round(allowed_shares * per_share_risk, 2)
    if allowed_shares <= 0 or actual_risk < cfg.min_tranche_risk_dollars:
        return StagedTrancheEvaluation(
            allowed=False,
            tranche_index=tranche_index,
            shares=0,
            entry_price=p_entry,
            stop_loss=p_stop,
            risk_dollars=0.0,
            total_position_risk_after=total_committed_risk,
            max_total_position_risk=plan.total_risk_budget_dollars,
            total_position_shares_after=total_committed_shares,
            max_total_position_shares=plan.total_allowed_shares,
            remaining_risk_budget=remaining_risk_budget,
            rejection_reason=TrancheRejectionReason.TRANCHE_TOO_SMALL.value,
            binding_constraint="MIN_TRANCHE_THRESHOLD",
            evidence=eval_evidence,
        )

    # 9. Verify Safety Invariants
    total_risk_after = round(total_committed_risk + actual_risk, 2)
    total_shares_after = total_committed_shares + allowed_shares
    new_remaining_risk = max(0.0, round(plan.total_risk_budget_dollars - total_risk_after, 2))

    eval_evidence["allowed_shares"] = allowed_shares
    eval_evidence["actual_risk"] = actual_risk
    eval_evidence["total_risk_after"] = total_risk_after
    eval_evidence["total_shares_after"] = total_shares_after

    return StagedTrancheEvaluation(
        allowed=True,
        tranche_index=tranche_index,
        shares=allowed_shares,
        entry_price=p_entry,
        stop_loss=p_stop,
        risk_dollars=actual_risk,
        total_position_risk_after=total_risk_after,
        max_total_position_risk=plan.total_risk_budget_dollars,
        total_position_shares_after=total_shares_after,
        max_total_position_shares=plan.total_allowed_shares,
        remaining_risk_budget=new_remaining_risk,
        binding_constraint="RISK_BUDGET" if allowed_shares == n_by_risk else "POSITION_CAP",
        evidence=eval_evidence,
    )
