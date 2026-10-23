"""Risk-Based Position Sizing & Portfolio Risk Management (Step 7.1).

Implements deterministic position sizing derived strictly from the portfolio risk
budget and trade geometry, not score alone:

  E: current portfolio equity.
  r: permitted fraction of equity at risk for a trade.
  P: expected entry price.
  S: stop price.
  C: estimated round-trip trading costs per share.
  G: explicit reserve for gap risk, expressed in price units.

Formulas:
  1. Preliminary Risk Size:
     N_risk = floor( (E * r) / (|P - S| + C + G) )

  2. Additional Limit Caps:
     N_position_cap  = floor( (E * max_position_pct) / P )
     N_liquidity_cap = floor( ADV_shares * max_adv_pct )
     N_portfolio_cap = floor( max(0, E * max_gross_pct - current_gross) / P )

  3. Permissible Share Count:
     N_allowed = min(N_risk, N_position_cap, N_liquidity_cap, N_portfolio_cap)

Mandate Invariant:
  A score of 90 does NOT receive a larger risk budget than a score of 70 unless
  explicitly validated. Sizing is governed by the formal investment mandate.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from quant.policy_config import TradingPolicy, get_default_policy
from quant.portfolio_controls import (
    PortfolioSnapshot,
    PortfolioRiskEngine,
    PortfolioBindingConstraint,
    ShortEligibilityValidator,
)


class BindingConstraint(str, Enum):
    """Identifies which constraint strictly limits the permissible share count."""
    RISK_BUDGET = "RISK_BUDGET"
    POSITION_CAP = "POSITION_CAP"
    LIQUIDITY_CAP = "LIQUIDITY_CAP"
    PORTFOLIO_CAP = "PORTFOLIO_CAP"
    ZERO_ALLOCATION = "ZERO_ALLOCATION"
    INVALID_INPUTS = "INVALID_INPUTS"
    # Step 7.2 Portfolio-Level Constraints
    SECTOR_CAP = "SECTOR_CAP"
    FACTOR_CAP = "FACTOR_CAP"
    NET_EXPOSURE_CAP = "NET_EXPOSURE_CAP"
    CASH_RESERVE_CAP = "CASH_RESERVE_CAP"
    DRAWDOWN_CIRCUIT_BREAKER = "DRAWDOWN_CIRCUIT_BREAKER"
    SHORT_BORROW_UNAVAILABLE = "SHORT_BORROW_UNAVAILABLE"
    SHORT_FINANCING_PROHIBITIVE = "SHORT_FINANCING_PROHIBITIVE"
    SHORT_STRUCTURAL_INVALID = "SHORT_STRUCTURAL_INVALID"



@dataclass
class InvestmentMandate:
    """
    Formal institutional investment mandate and risk budgeting parameters.
    Governs portfolio-level capital allocation and single-name exposure constraints.
    """
    portfolio_equity: float = 100_000.0
    max_risk_fraction: float = 0.010        # r: 1.0% equity risk per trade
    max_position_pct: float = 0.10         # Max 10% equity in any single position
    max_adv_pct: float = 0.01              # Max 1% of 20d ADV
    max_gross_exposure_pct: float = 1.00   # Max 100% gross portfolio exposure
    current_gross_exposure: float = 0.0    # Current dollar capital deployed
    gap_risk_atr_multiple: float = 0.50    # G: 0.5x ATR gap reserve
    default_gap_risk_pct: float = 0.01     # Fallback gap reserve: 1.0% of price
    slippage_bps: float = 5.0              # 5 bps slippage per leg
    commission_bps: float = 1.0            # 1 bps commission per leg
    allow_score_budget_scaling: bool = False  # Invariant: score does not inflate risk budget
    min_trade_size_shares: int = 1         # Minimum executable share count
    round_to_lot_size: int = 1             # Rounding step (1 = odd lots allowed)

    def validate(self) -> None:
        """Validates internal mandate consistency."""
        if self.portfolio_equity <= 0.0:
            raise ValueError(f"portfolio_equity must be positive, got {self.portfolio_equity}")
        if not (0.0 < self.max_risk_fraction <= 0.10):
            raise ValueError(f"max_risk_fraction must be in (0.0, 0.10], got {self.max_risk_fraction}")
        if not (0.0 < self.max_position_pct <= 1.0):
            raise ValueError(f"max_position_pct must be in (0.0, 1.0], got {self.max_position_pct}")
        if not (0.0 < self.max_adv_pct <= 0.20):
            raise ValueError(f"max_adv_pct must be in (0.0, 0.20], got {self.max_adv_pct}")
        if self.max_gross_exposure_pct <= 0.0:
            raise ValueError(f"max_gross_exposure_pct must be positive, got {self.max_gross_exposure_pct}")
        if self.current_gross_exposure < 0.0:
            raise ValueError(f"current_gross_exposure cannot be negative, got {self.current_gross_exposure}")
        if self.gap_risk_atr_multiple < 0.0:
            raise ValueError(f"gap_risk_atr_multiple cannot be negative, got {self.gap_risk_atr_multiple}")

    @classmethod
    def from_policy(
        cls,
        policy: Optional[TradingPolicy] = None,
        portfolio_equity: Optional[float] = None,
        current_gross_exposure: float = 0.0,
    ) -> "InvestmentMandate":
        """Constructs an InvestmentMandate from an institutional TradingPolicy."""
        active_policy = policy or get_default_policy()
        pr = active_policy.portfolio_risk
        tq = active_policy.trade_quality
        equity = portfolio_equity if portfolio_equity is not None else pr.default_portfolio_equity
        return cls(
            portfolio_equity=equity,
            max_risk_fraction=pr.max_risk_fraction_per_trade,
            max_position_pct=pr.max_position_equity_pct,
            max_adv_pct=pr.max_adv_participation_pct,
            max_gross_exposure_pct=pr.max_gross_exposure_pct,
            current_gross_exposure=current_gross_exposure,
            gap_risk_atr_multiple=pr.gap_risk_atr_multiple,
            default_gap_risk_pct=pr.default_gap_risk_pct,
            slippage_bps=tq.slippage_bps,
            commission_bps=tq.commission_bps,
            allow_score_budget_scaling=pr.allow_score_budget_scaling,
            min_trade_size_shares=pr.min_trade_size_shares,
            round_to_lot_size=1,
        )


@dataclass
class PositionSizingResult:
    """
    Detailed audit result of risk-based position sizing and constraint resolution.
    """
    ticker: str
    direction: str
    shares_allowed: int
    n_risk: int
    n_position_cap: int
    n_liquidity_cap: int
    n_portfolio_cap: int
    binding_constraint: str
    entry_price: float
    stop_loss: float
    stop_distance: float
    cost_per_share: float
    gap_reserve_per_share: float
    effective_per_share_risk: float
    equity_at_risk_dollars: float
    position_market_value: float
    position_equity_pct: float
    total_risk_dollars: float
    realized_risk_pct: float
    is_tradeable: bool
    rejection_reason: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes result to a clean dictionary."""
        return asdict(self)


def compute_round_trip_cost_per_share(
    entry_price: float,
    slippage_bps: float = 5.0,
    commission_bps: float = 1.0,
) -> float:
    """
    Computes expected round-trip friction C per share (slippage + commissions on entry and exit):
        C = P * (slippage_bps + commission_bps) * 2 / 10,000
    """
    if entry_price <= 0.0:
        return 0.0
    total_friction_bps = max(0.0, (slippage_bps + commission_bps) * 2.0)
    return round(entry_price * (total_friction_bps / 10_000.0), 4)


def compute_gap_risk_reserve(
    entry_price: float,
    atr: Optional[float] = None,
    gap_risk_atr_multiple: float = 0.50,
    default_gap_risk_pct: float = 0.01,
) -> float:
    """
    Computes an explicit reserve G for gap risk, expressed in price units:
        G = gap_risk_atr_multiple * ATR (if ATR valid)
        G = entry_price * default_gap_risk_pct (fallback)
    """
    if entry_price <= 0.0:
        return 0.0
    if atr is not None and atr > 0.0 and not math.isnan(atr) and not math.isinf(atr):
        return round(gap_risk_atr_multiple * atr, 4)
    return round(entry_price * max(0.0, default_gap_risk_pct), 4)


def calculate_risk_based_position_size(
    entry_price: float,
    stop_loss: float,
    ticker: str = "TICKER",
    direction: str = "LONG",
    portfolio_equity: Optional[float] = None,
    risk_fraction: Optional[float] = None,
    cost_per_share: Optional[float] = None,
    gap_risk_reserve: Optional[float] = None,
    atr: Optional[float] = None,
    adv_shares_20d: Optional[float] = None,
    avg_dollar_volume_20d: Optional[float] = None,
    current_gross_exposure: Optional[float] = None,
    composite_score: Optional[float] = None,
    mandate: Optional[InvestmentMandate] = None,
    policy: Optional[TradingPolicy] = None,
    portfolio_snapshot: Optional[PortfolioSnapshot] = None,
    sector: Optional[str] = None,
    factor_tags: Optional[List[str]] = None,
    borrow_fee_bps: float = 0.0,
    borrow_available: bool = True,
    borrow_status: str = "ETB",
    available_borrow_shares: Optional[int] = None,
    rsi: Optional[float] = None,
    ema_20: Optional[float] = None,
    ema_50: Optional[float] = None,
    trend_score: Optional[float] = None,
    financial_quality_score: Optional[float] = None,
    valuation_history_score: Optional[float] = None,
    reward_risk_ratio: Optional[float] = None,
    days_to_ex_dividend: Optional[int] = None,
    days_to_earnings: Optional[int] = None,
    stop_inside_expected_move: bool = False,
) -> PositionSizingResult:
    """
    Calculates deterministic risk-based position size under formal investment mandate caps.

    Parameters:
        entry_price: Expected execution price P (> 0).
        stop_loss: Stop price S (> 0).
        ticker: Symbol string.
        direction: "LONG" or "SHORT".
        portfolio_equity: Current portfolio equity E.
        risk_fraction: Risk budget fraction r (e.g. 0.010 = 1.0%).
        cost_per_share: Round-trip trading cost C per share. If None, derived from mandate bps.
        gap_risk_reserve: Explicit gap risk reserve G in price units. If None, derived from ATR.
        atr: Average True Range (14-day).
        adv_shares_20d: 20-day Average Daily Volume in shares.
        avg_dollar_volume_20d: 20-day Average Daily Dollar Volume.
        current_gross_exposure: Current portfolio capital deployed.
        composite_score: Deterministic quantitative composite score (0-100).
        mandate: Optional explicit InvestmentMandate.
        policy: Optional TradingPolicy.

    Returns:
        PositionSizingResult with permissible share count and binding constraint.
    """
    # 1. Resolve Mandate & Baseline Equity
    active_mandate = mandate or InvestmentMandate.from_policy(
        policy=policy,
        portfolio_equity=portfolio_equity,
        current_gross_exposure=current_gross_exposure or 0.0,
    )
    if portfolio_equity is not None and portfolio_equity > 0.0:
        active_mandate.portfolio_equity = portfolio_equity
    if current_gross_exposure is not None and current_gross_exposure >= 0.0:
        active_mandate.current_gross_exposure = current_gross_exposure

    dir_norm = direction.upper().strip()
    E = active_mandate.portfolio_equity

    # Default metrics dictionary
    base_metrics: Dict[str, Any] = {
        "portfolio_equity": E,
        "direction": dir_norm,
        "composite_score": composite_score,
    }

    # 2. Input Validation (Fail-Closed)
    if entry_price is None or entry_price <= 0.0 or math.isnan(entry_price) or math.isinf(entry_price):
        return PositionSizingResult(
            ticker=ticker,
            direction=dir_norm,
            shares_allowed=0,
            n_risk=0,
            n_position_cap=0,
            n_liquidity_cap=0,
            n_portfolio_cap=0,
            binding_constraint=BindingConstraint.INVALID_INPUTS.value,
            entry_price=entry_price or 0.0,
            stop_loss=stop_loss or 0.0,
            stop_distance=0.0,
            cost_per_share=0.0,
            gap_reserve_per_share=0.0,
            effective_per_share_risk=0.0,
            equity_at_risk_dollars=0.0,
            position_market_value=0.0,
            position_equity_pct=0.0,
            total_risk_dollars=0.0,
            realized_risk_pct=0.0,
            is_tradeable=False,
            rejection_reason="Expected entry price P must be strictly positive and finite.",
            metrics=base_metrics,
        )

    if stop_loss is None or stop_loss <= 0.0 or math.isnan(stop_loss) or math.isinf(stop_loss):
        return PositionSizingResult(
            ticker=ticker,
            direction=dir_norm,
            shares_allowed=0,
            n_risk=0,
            n_position_cap=0,
            n_liquidity_cap=0,
            n_portfolio_cap=0,
            binding_constraint=BindingConstraint.INVALID_INPUTS.value,
            entry_price=entry_price,
            stop_loss=stop_loss or 0.0,
            stop_distance=0.0,
            cost_per_share=0.0,
            gap_reserve_per_share=0.0,
            effective_per_share_risk=0.0,
            equity_at_risk_dollars=0.0,
            position_market_value=0.0,
            position_equity_pct=0.0,
            total_risk_dollars=0.0,
            realized_risk_pct=0.0,
            is_tradeable=False,
            rejection_reason="Stop price S must be strictly positive and finite.",
            metrics=base_metrics,
        )

    if E <= 0.0 or math.isnan(E) or math.isinf(E):
        return PositionSizingResult(
            ticker=ticker,
            direction=dir_norm,
            shares_allowed=0,
            n_risk=0,
            n_position_cap=0,
            n_liquidity_cap=0,
            n_portfolio_cap=0,
            binding_constraint=BindingConstraint.INVALID_INPUTS.value,
            entry_price=entry_price,
            stop_loss=stop_loss,
            stop_distance=abs(entry_price - stop_loss),
            cost_per_share=0.0,
            gap_reserve_per_share=0.0,
            effective_per_share_risk=0.0,
            equity_at_risk_dollars=0.0,
            position_market_value=0.0,
            position_equity_pct=0.0,
            total_risk_dollars=0.0,
            realized_risk_pct=0.0,
            is_tradeable=False,
            rejection_reason="Portfolio equity E must be strictly positive and finite.",
            metrics=base_metrics,
        )

    # 3. Directional Stop Ordering & Distance
    if entry_price == stop_loss:
        return PositionSizingResult(
            ticker=ticker,
            direction=dir_norm,
            shares_allowed=0,
            n_risk=0,
            n_position_cap=0,
            n_liquidity_cap=0,
            n_portfolio_cap=0,
            binding_constraint=BindingConstraint.INVALID_INPUTS.value,
            entry_price=entry_price,
            stop_loss=stop_loss,
            stop_distance=0.0,
            cost_per_share=0.0,
            gap_reserve_per_share=0.0,
            effective_per_share_risk=0.0,
            equity_at_risk_dollars=0.0,
            position_market_value=0.0,
            position_equity_pct=0.0,
            total_risk_dollars=0.0,
            realized_risk_pct=0.0,
            is_tradeable=False,
            rejection_reason="Stop price cannot equal entry price; zero risk denominator.",
            metrics=base_metrics,
        )

    if dir_norm == "LONG" and stop_loss >= entry_price:
        return PositionSizingResult(
            ticker=ticker,
            direction=dir_norm,
            shares_allowed=0,
            n_risk=0,
            n_position_cap=0,
            n_liquidity_cap=0,
            n_portfolio_cap=0,
            binding_constraint=BindingConstraint.INVALID_INPUTS.value,
            entry_price=entry_price,
            stop_loss=stop_loss,
            stop_distance=0.0,
            cost_per_share=0.0,
            gap_reserve_per_share=0.0,
            effective_per_share_risk=0.0,
            equity_at_risk_dollars=0.0,
            position_market_value=0.0,
            position_equity_pct=0.0,
            total_risk_dollars=0.0,
            realized_risk_pct=0.0,
            is_tradeable=False,
            rejection_reason=f"Long stop loss {stop_loss} must be strictly below entry price {entry_price}.",
            metrics=base_metrics,
        )

    if dir_norm == "SHORT" and stop_loss <= entry_price:
        return PositionSizingResult(
            ticker=ticker,
            direction=dir_norm,
            shares_allowed=0,
            n_risk=0,
            n_position_cap=0,
            n_liquidity_cap=0,
            n_portfolio_cap=0,
            binding_constraint=BindingConstraint.INVALID_INPUTS.value,
            entry_price=entry_price,
            stop_loss=stop_loss,
            stop_distance=0.0,
            cost_per_share=0.0,
            gap_reserve_per_share=0.0,
            effective_per_share_risk=0.0,
            equity_at_risk_dollars=0.0,
            position_market_value=0.0,
            position_equity_pct=0.0,
            total_risk_dollars=0.0,
            realized_risk_pct=0.0,
            is_tradeable=False,
            rejection_reason=f"Short stop loss {stop_loss} must be strictly above entry price {entry_price}.",
            metrics=base_metrics,
        )

    stop_distance = abs(entry_price - stop_loss)
    if stop_distance <= 0.0:
        return PositionSizingResult(
            ticker=ticker,
            direction=dir_norm,
            shares_allowed=0,
            n_risk=0,
            n_position_cap=0,
            n_liquidity_cap=0,
            n_portfolio_cap=0,
            binding_constraint=BindingConstraint.INVALID_INPUTS.value,
            entry_price=entry_price,
            stop_loss=stop_loss,
            stop_distance=0.0,
            cost_per_share=0.0,
            gap_reserve_per_share=0.0,
            effective_per_share_risk=0.0,
            equity_at_risk_dollars=0.0,
            position_market_value=0.0,
            position_equity_pct=0.0,
            total_risk_dollars=0.0,
            realized_risk_pct=0.0,
            is_tradeable=False,
            rejection_reason="Stop price cannot equal entry price; zero risk denominator.",
            metrics=base_metrics,
        )

    # 4. Trading Costs C & Gap Risk Reserve G
    C = (
        cost_per_share
        if (cost_per_share is not None and cost_per_share >= 0.0)
        else compute_round_trip_cost_per_share(
            entry_price=entry_price,
            slippage_bps=active_mandate.slippage_bps,
            commission_bps=active_mandate.commission_bps,
        )
    )

    G = (
        gap_risk_reserve
        if (gap_risk_reserve is not None and gap_risk_reserve >= 0.0)
        else compute_gap_risk_reserve(
            entry_price=entry_price,
            atr=atr,
            gap_risk_atr_multiple=active_mandate.gap_risk_atr_multiple,
            default_gap_risk_pct=active_mandate.default_gap_risk_pct,
        )
    )

    # Effective per-share risk denominator: |P - S| + C + G
    effective_per_share_risk = stop_distance + C + G
    if effective_per_share_risk <= 0.0:
        return PositionSizingResult(
            ticker=ticker,
            direction=dir_norm,
            shares_allowed=0,
            n_risk=0,
            n_position_cap=0,
            n_liquidity_cap=0,
            n_portfolio_cap=0,
            binding_constraint=BindingConstraint.INVALID_INPUTS.value,
            entry_price=entry_price,
            stop_loss=stop_loss,
            stop_distance=stop_distance,
            cost_per_share=C,
            gap_reserve_per_share=G,
            effective_per_share_risk=effective_per_share_risk,
            equity_at_risk_dollars=0.0,
            position_market_value=0.0,
            position_equity_pct=0.0,
            total_risk_dollars=0.0,
            realized_risk_pct=0.0,
            is_tradeable=False,
            rejection_reason="Effective per-share risk denominator is non-positive.",
            metrics=base_metrics,
        )

    # 4B. Portfolio Snapshot Controls (Step 7.2)
    if portfolio_snapshot is not None:
        report = PortfolioRiskEngine.evaluate_portfolio_controls(
            ticker=ticker,
            direction=dir_norm,
            entry_price=entry_price,
            stop_loss=stop_loss,
            snapshot=portfolio_snapshot,
            policy=policy,
            sector=sector,
            factor_tags=factor_tags,
            adv_shares_20d=adv_shares_20d,
            cost_per_share=C,
            gap_risk_reserve=G,
            composite_score=composite_score,
            borrow_fee_bps=borrow_fee_bps,
            borrow_available=borrow_available,
            borrow_status=borrow_status,
            available_borrow_shares=available_borrow_shares,
            rsi=rsi,
            ema_20=ema_20,
            ema_50=ema_50,
            trend_score=trend_score,
            financial_quality_score=financial_quality_score,
            valuation_history_score=valuation_history_score,
            reward_risk_ratio=reward_risk_ratio,
            days_to_ex_dividend=days_to_ex_dividend,
        )
        n_allowed = report.allowed_shares
        b_constraint = report.binding_constraint
        pos_mv = round(n_allowed * entry_price, 2)
        pos_pct = round(pos_mv / E, 4) if E > 0 else 0.0
        tot_risk = round(n_allowed * effective_per_share_risk, 2)
        real_risk = round(tot_risk / E, 6) if E > 0 else 0.0
        ret_metrics = {**base_metrics, **report.metrics, "portfolio_report": report.to_dict()}

        return PositionSizingResult(
            ticker=ticker,
            direction=dir_norm,
            shares_allowed=n_allowed,
            n_risk=report.n_risk,
            n_position_cap=report.n_single_stock_cap,
            n_liquidity_cap=report.n_liquidity_cap,
            n_portfolio_cap=report.n_gross_cap,
            binding_constraint=b_constraint,
            entry_price=entry_price,
            stop_loss=stop_loss,
            stop_distance=round(stop_distance, 4),
            cost_per_share=round(C, 4),
            gap_reserve_per_share=round(G, 4),
            effective_per_share_risk=round(effective_per_share_risk, 4),
            equity_at_risk_dollars=round(E * report.effective_risk_fraction, 2),
            position_market_value=pos_mv,
            position_equity_pct=pos_pct,
            total_risk_dollars=tot_risk,
            realized_risk_pct=real_risk,
            is_tradeable=report.is_tradeable,
            rejection_reason=report.rejection_reason,
            metrics=ret_metrics,
        )

    # 4C. Short Eligibility Validation when Portfolio Snapshot is Absent (Step 7.2)
    if dir_norm == "SHORT":
        if not borrow_available or (borrow_status or "").upper() in ("UNAVAILABLE", "HARD_TO_BORROW_UNLOCATED"):
            return PositionSizingResult(
                ticker=ticker,
                direction=dir_norm,
                shares_allowed=0,
                n_risk=0,
                n_position_cap=0,
                n_liquidity_cap=0,
                n_portfolio_cap=0,
                binding_constraint=BindingConstraint.SHORT_BORROW_UNAVAILABLE.value,
                entry_price=entry_price,
                stop_loss=stop_loss,
                stop_distance=round(stop_distance, 4),
                cost_per_share=round(C, 4),
                gap_reserve_per_share=round(G, 4),
                effective_per_share_risk=round(effective_per_share_risk, 4),
                equity_at_risk_dollars=0.0,
                position_market_value=0.0,
                position_equity_pct=0.0,
                total_risk_dollars=0.0,
                realized_risk_pct=0.0,
                is_tradeable=False,
                rejection_reason="Short entry rejected: Shares are not borrowable or borrow pool is unavailable.",
                metrics={**base_metrics},
            )

        max_fee = (policy or get_default_policy()).portfolio_risk.short_borrow_fee_max_bps
        if borrow_fee_bps > max_fee:
            return PositionSizingResult(
                ticker=ticker,
                direction=dir_norm,
                shares_allowed=0,
                n_risk=0,
                n_position_cap=0,
                n_liquidity_cap=0,
                n_portfolio_cap=0,
                binding_constraint=BindingConstraint.SHORT_FINANCING_PROHIBITIVE.value,
                entry_price=entry_price,
                stop_loss=stop_loss,
                stop_distance=round(stop_distance, 4),
                cost_per_share=round(C, 4),
                gap_reserve_per_share=round(G, 4),
                effective_per_share_risk=round(effective_per_share_risk, 4),
                equity_at_risk_dollars=0.0,
                position_market_value=0.0,
                position_equity_pct=0.0,
                total_risk_dollars=0.0,
                realized_risk_pct=0.0,
                is_tradeable=False,
                rejection_reason=f"Short entry rejected: Annualized borrow fee {borrow_fee_bps:.1f} bps > {max_fee:.1f} bps.",
                metrics={**base_metrics},
            )

        if composite_score is not None:
            short_res = ShortEligibilityValidator.validate(
                ticker=ticker,
                entry_price=entry_price,
                stop_loss=stop_loss,
                composite_score=composite_score,
                policy=policy,
                borrow_available=borrow_available,
                borrow_status=borrow_status,
                borrow_fee_bps=borrow_fee_bps,
                available_borrow_shares=available_borrow_shares,
                shares_requested=1,
                rsi=rsi,
                ema_20=ema_20,
                ema_50=ema_50,
                trend_score=trend_score,
                financial_quality_score=financial_quality_score,
                valuation_history_score=valuation_history_score,
                reward_risk_ratio=reward_risk_ratio,
                days_to_ex_dividend=days_to_ex_dividend,
                portfolio_equity=E,
            )
            if not short_res.is_eligible:
                binding_map = {
                    "BORROW_UNAVAILABLE": BindingConstraint.SHORT_BORROW_UNAVAILABLE.value,
                    "BORROW_LOCATE_REQUIRED": BindingConstraint.SHORT_BORROW_UNAVAILABLE.value,
                    "INSUFFICIENT_BORROW_SHARES": BindingConstraint.SHORT_BORROW_UNAVAILABLE.value,
                    "BORROW_FEE_PROHIBITIVE": BindingConstraint.SHORT_FINANCING_PROHIBITIVE.value,
                    "SHORT_COLLATERAL_INSUFFICIENT": BindingConstraint.CASH_RESERVE_CAP.value,
                }
                b_const = binding_map.get(
                    short_res.reason_code or "", BindingConstraint.SHORT_STRUCTURAL_INVALID.value
                )
                return PositionSizingResult(
                    ticker=ticker,
                    direction=dir_norm,
                    shares_allowed=0,
                    n_risk=0,
                    n_position_cap=0,
                    n_liquidity_cap=0,
                    n_portfolio_cap=0,
                    binding_constraint=b_const,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    stop_distance=round(stop_distance, 4),
                    cost_per_share=round(C, 4),
                    gap_reserve_per_share=round(G, 4),
                    effective_per_share_risk=round(effective_per_share_risk, 4),
                    equity_at_risk_dollars=0.0,
                    position_market_value=0.0,
                    position_equity_pct=0.0,
                    total_risk_dollars=0.0,
                    realized_risk_pct=0.0,
                    is_tradeable=False,
                    rejection_reason=short_res.message,
                    metrics={**base_metrics, "short_validation": short_res.to_dict()},
                )

    # 5. Risk Budget Fraction r
    # Mandate Invariant: A score of 90 does NOT get a larger risk budget than a score of 70
    # unless validated scaling is explicitly permitted by policy configuration.
    r = risk_fraction if (risk_fraction is not None and risk_fraction > 0.0) else active_mandate.max_risk_fraction
    if active_mandate.allow_score_budget_scaling and composite_score is not None:
        # Strictly validated, clamped monotonic scaling if mandate explicitly permits
        score_val = max(0.0, min(100.0, float(composite_score)))
        # Conviction multiplier anchored at neutral 50.0: [0.8x, 1.2x]
        scale = 1.0 + (score_val - 50.0) * 0.004
        r = r * max(0.70, min(1.30, scale))

    # Pre-Earnings Event Sizing Haircut (Mode A: Event-Sized Pre-Earnings Trading)
    effective_policy = policy or get_default_policy()
    if (
        days_to_earnings is not None
        and days_to_earnings <= effective_policy.events.earnings_blackout_sessions
        and getattr(effective_policy.events, "allow_pre_earnings_trading", False)
    ):
        haircut = getattr(effective_policy.events, "earnings_risk_haircut", 0.50)
        r = r * haircut
        base_metrics["pre_earnings_event_mode"] = True
        base_metrics["earnings_risk_haircut"] = haircut

    # Stop Noise Dampener: If stop is within 1-sigma expected move, scale risk budget down
    if stop_inside_expected_move:
        r = r * 0.70
        base_metrics["stop_noise_dampener_applied"] = 0.70

    equity_at_risk_dollars = E * r

    # 6. Preliminary Risk Size: N_risk = floor( (E * r) / (|P - S| + C + G) )
    n_risk = math.floor(equity_at_risk_dollars / effective_per_share_risk)

    # 7. Additional Limit Caps
    # 7A. Maximum Position Cap: N_position_cap = floor( (E * max_position_pct) / P )
    n_position_cap = math.floor((E * active_mandate.max_position_pct) / entry_price)

    # 7B. Liquidity Cap: N_liquidity_cap = floor( ADV_shares * max_adv_pct )
    if adv_shares_20d is not None and adv_shares_20d > 0.0:
        n_liquidity_cap = math.floor(adv_shares_20d * active_mandate.max_adv_pct)
    elif avg_dollar_volume_20d is not None and avg_dollar_volume_20d > 0.0:
        adv_shares_calc = avg_dollar_volume_20d / entry_price
        n_liquidity_cap = math.floor(adv_shares_calc * active_mandate.max_adv_pct)
    else:
        # If no volume data is provided, liquidity cap is unconstrained (infinite)
        n_liquidity_cap = 10_000_000_000

    # 7C. Remaining Portfolio Exposure Cap:
    # N_portfolio_cap = floor( max(0, E * max_gross_pct - current_gross) / P )
    max_gross_dollars = E * active_mandate.max_gross_exposure_pct
    available_exposure = max(0.0, max_gross_dollars - active_mandate.current_gross_exposure)
    n_portfolio_cap = math.floor(available_exposure / entry_price)

    # 8. Permissible Share Count: N_allowed = min(N_risk, N_pos, N_liq, N_port)
    raw_allowed = min(n_risk, n_position_cap, n_liquidity_cap, n_portfolio_cap)
    n_allowed = max(0, raw_allowed)

    # Lot size rounding
    if active_mandate.round_to_lot_size > 1:
        n_allowed = (n_allowed // active_mandate.round_to_lot_size) * active_mandate.round_to_lot_size

    # 9. Binding Constraint Resolution
    if n_allowed <= 0:
        binding_constraint = BindingConstraint.ZERO_ALLOCATION.value
    elif n_allowed == n_risk:
        binding_constraint = BindingConstraint.RISK_BUDGET.value
    elif n_allowed == n_position_cap:
        binding_constraint = BindingConstraint.POSITION_CAP.value
    elif n_allowed == n_liquidity_cap:
        binding_constraint = BindingConstraint.LIQUIDITY_CAP.value
    elif n_allowed == n_portfolio_cap:
        binding_constraint = BindingConstraint.PORTFOLIO_CAP.value
    else:
        binding_constraint = BindingConstraint.RISK_BUDGET.value

    # 10. Financial Impact Metrics
    position_market_value = round(n_allowed * entry_price, 2)
    position_equity_pct = round(position_market_value / E, 4) if E > 0 else 0.0
    total_risk_dollars = round(n_allowed * effective_per_share_risk, 2)
    realized_risk_pct = round(total_risk_dollars / E, 6) if E > 0 else 0.0

    is_tradeable = n_allowed >= active_mandate.min_trade_size_shares
    rejection_reason = None
    if not is_tradeable:
        if n_allowed == 0:
            rejection_reason = f"Position size resolved to 0 shares (constrained by {binding_constraint})."
        else:
            rejection_reason = (
                f"Allowed shares {n_allowed} below minimum trade size {active_mandate.min_trade_size_shares}."
            )

    calc_metrics: Dict[str, Any] = {
        **base_metrics,
        "portfolio_equity": E,
        "risk_fraction_r": r,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "stop_distance": stop_distance,
        "trading_costs_c": C,
        "gap_reserve_g": G,
        "effective_per_share_risk": effective_per_share_risk,
        "n_risk": n_risk,
        "n_position_cap": n_position_cap,
        "n_liquidity_cap": n_liquidity_cap,
        "n_portfolio_cap": n_portfolio_cap,
        "binding_constraint": binding_constraint,
        "available_portfolio_capacity_dollars": available_exposure,
    }

    return PositionSizingResult(
        ticker=ticker,
        direction=dir_norm,
        shares_allowed=n_allowed,
        n_risk=n_risk,
        n_position_cap=n_position_cap,
        n_liquidity_cap=n_liquidity_cap,
        n_portfolio_cap=n_portfolio_cap,
        binding_constraint=binding_constraint,
        entry_price=entry_price,
        stop_loss=stop_loss,
        stop_distance=round(stop_distance, 4),
        cost_per_share=round(C, 4),
        gap_reserve_per_share=round(G, 4),
        effective_per_share_risk=round(effective_per_share_risk, 4),
        equity_at_risk_dollars=round(equity_at_risk_dollars, 2),
        position_market_value=position_market_value,
        position_equity_pct=position_equity_pct,
        total_risk_dollars=total_risk_dollars,
        realized_risk_pct=realized_risk_pct,
        is_tradeable=is_tradeable,
        rejection_reason=rejection_reason,
        metrics=calc_metrics,
    )
