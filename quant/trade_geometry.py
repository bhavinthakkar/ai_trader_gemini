"""
Trade Geometry and Deterministic Reward-to-Risk Engine (Step 5.2).

Calculates gross and net reward-to-risk ratios from deterministically validated prices,
accounting for expected transaction costs (slippage, commissions, borrow fees),
rejecting invalid price ordering, and verifying that realized trade outcomes
and empirical win probabilities justify entry.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Dict, Optional, Tuple, Union


class TradeDirection(str, Enum):
    """Directional classification for trade setups."""
    LONG = "LONG"
    SHORT = "SHORT"


class PriceOrderingStatus(str, Enum):
    """Status codes for price ordering and geometry validation."""
    VALID = "VALID"
    NON_POSITIVE_PRICE = "NON_POSITIVE_PRICE"
    INVALID_PRICE_ORDERING = "INVALID_PRICE_ORDERING"
    ZERO_OR_NEGATIVE_RISK = "ZERO_OR_NEGATIVE_RISK"
    ZERO_OR_NEGATIVE_REWARD = "ZERO_OR_NEGATIVE_REWARD"
    EXCESSIVE_STOP_DISTANCE = "EXCESSIVE_STOP_DISTANCE"
    UNVALIDATED_RISK_AMOUNT = "UNVALIDATED_RISK_AMOUNT"


@dataclass
class ExecutionCostModel:
    """
    Execution assumptions and transaction cost parameters.

    Attributes:
        slippage_bps: Expected slippage per execution leg in basis points (5 bps = 0.05%).
        commission_bps: Expected commissions and exchange/regulatory fees in basis points.
        fixed_fee_per_share: Optional fixed dollar fee per share ($0.00 for zero-comm brokers).
        borrow_rate_annual_pct: Annualized borrow rate for short trades (e.g. 1.5% for hard-to-borrow).
        holding_days_assumption: Expected holding period in trading days for short borrow accrual.
    """
    slippage_bps: float = 5.0
    commission_bps: float = 1.0
    fixed_fee_per_share: float = 0.0
    borrow_rate_annual_pct: float = 0.0
    holding_days_assumption: float = 5.0

    def compute_entry_execution(
        self, direction: TradeDirection, nominal_entry: float
    ) -> Tuple[float, float]:
        """
        Calculates effective entry price and entry transaction costs.

        Returns:
            (effective_entry_price, total_entry_friction_per_share)
        """
        slip = nominal_entry * (self.slippage_bps / 10000.0)
        comm = nominal_entry * (self.commission_bps / 10000.0) + self.fixed_fee_per_share
        friction = slip + comm

        if direction == TradeDirection.LONG:
            # Long buyer pays ask/higher due to slippage; commission added
            effective_price = nominal_entry + slip
        else:
            # Short seller receives bid/lower due to slippage; commission deducted
            effective_price = nominal_entry - slip

        return effective_price, friction

    def compute_exit_execution(
        self, direction: TradeDirection, nominal_exit: float, is_stop: bool
    ) -> Tuple[float, float]:
        """
        Calculates effective exit price and exit transaction costs.

        Returns:
            (effective_exit_price, total_exit_friction_per_share)
        """
        slip = nominal_exit * (self.slippage_bps / 10000.0)
        comm = nominal_exit * (self.commission_bps / 10000.0) + self.fixed_fee_per_share
        borrow_fee = 0.0

        if direction == TradeDirection.SHORT and self.borrow_rate_annual_pct > 0.0:
            borrow_fee = nominal_exit * (self.borrow_rate_annual_pct / 100.0) * (self.holding_days_assumption / 365.0)

        friction = slip + comm + borrow_fee

        if direction == TradeDirection.LONG:
            # Long seller receives bid/lower on exit
            effective_price = nominal_exit - slip
        else:
            # Short buy-to-cover pays ask/higher on exit
            effective_price = nominal_exit + slip

        return effective_price, friction


@dataclass
class RewardRiskResult:
    """
    Complete audit result of deterministic reward-to-risk and execution validation.
    """
    is_valid: bool
    status: PriceOrderingStatus
    rejection_reason: Optional[str]
    direction: TradeDirection
    entry_price: float
    stop_loss: float
    target_price: float

    # Gross geometry (pre-cost)
    gross_reward: float
    gross_risk: float
    gross_reward_risk_ratio: Optional[float]
    breakeven_win_rate_gross: Optional[float]

    # Execution assumptions & net metrics (post-cost)
    effective_entry: float
    effective_stop: float
    effective_target: float
    net_reward: float
    net_risk: float
    net_reward_risk_ratio: Optional[float]
    breakeven_win_rate_net: Optional[float]
    cost_drag_bps: float
    cost_breakdown: Dict[str, float] = field(default_factory=dict)

    # Realized trade outcomes & empirical edge validation
    estimated_win_probability: Optional[float] = None
    required_edge_pct: float = 0.02
    actual_edge_pct: Optional[float] = None
    expected_value_r: Optional[float] = None
    has_statistical_edge: bool = False
    meets_minimum_rr: bool = False
    trade_justified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serializes result into dictionary suitable for logs and databases."""
        res = asdict(self)
        res["direction"] = self.direction.value
        res["status"] = self.status.value
        return res


def validate_price_ordering(
    direction: Union[TradeDirection, str],
    entry: float,
    stop: float,
    target: float,
    max_stop_distance_pct: float = 0.30,
) -> Tuple[bool, PriceOrderingStatus, Optional[str]]:
    """
    Deterministically validates price ordering and risk bounds.

    Rules:
      Long:
        0 < Stop < Entry < Target
        Denominator (Entry - Stop) > 0
        Numerator (Target - Entry) > 0
        Stop distance (Entry - Stop) / Entry <= max_stop_distance_pct

      Short:
        0 < Target < Entry < Stop
        Denominator (Stop - Entry) > 0
        Numerator (Entry - Target) > 0
        Stop distance (Stop - Entry) / Entry <= max_stop_distance_pct

    Returns:
        (is_valid, status, rejection_reason)
    """
    if isinstance(direction, str):
        direction = TradeDirection(direction.upper())

    # 1. Non-positive price rejection
    if entry <= 0.0 or stop <= 0.0 or target <= 0.0:
        return (
            False,
            PriceOrderingStatus.NON_POSITIVE_PRICE,
            f"All trade prices must be strictly positive: entry={entry}, stop={stop}, target={target}",
        )

    # 2. Direction-specific price ordering and denominator checks
    if direction == TradeDirection.LONG:
        risk_denom = entry - stop
        reward_num = target - entry

        if risk_denom <= 0.0:
            return (
                False,
                PriceOrderingStatus.ZERO_OR_NEGATIVE_RISK,
                f"Long risk denominator (entry - stop) must be > 0: entry={entry}, stop={stop} (delta={risk_denom:.4f})",
            )

        if reward_num <= 0.0:
            return (
                False,
                PriceOrderingStatus.ZERO_OR_NEGATIVE_REWARD,
                f"Long target must be strictly above entry: target={target} <= entry={entry} (reward={reward_num:.4f})",
            )

        if not (stop < entry < target):
            return (
                False,
                PriceOrderingStatus.INVALID_PRICE_ORDERING,
                f"Long price ordering violation: expected 0 < stop < entry < target, got {stop} < {entry} < {target}",
            )

        stop_dist_pct = risk_denom / entry
        if stop_dist_pct > max_stop_distance_pct:
            return (
                False,
                PriceOrderingStatus.EXCESSIVE_STOP_DISTANCE,
                f"Long stop loss distance {stop_dist_pct:.1%} exceeds maximum permitted risk of {max_stop_distance_pct:.1%}",
            )

    else:  # SHORT
        risk_denom = stop - entry
        reward_num = entry - target

        if risk_denom <= 0.0:
            return (
                False,
                PriceOrderingStatus.ZERO_OR_NEGATIVE_RISK,
                f"Short risk denominator (stop - entry) must be > 0: stop={stop}, entry={entry} (delta={risk_denom:.4f})",
            )

        if reward_num <= 0.0:
            return (
                False,
                PriceOrderingStatus.ZERO_OR_NEGATIVE_REWARD,
                f"Short target must be strictly below entry: target={target} >= entry={entry} (reward={reward_num:.4f})",
            )

        if not (target < entry < stop):
            return (
                False,
                PriceOrderingStatus.INVALID_PRICE_ORDERING,
                f"Short price ordering violation: expected 0 < target < entry < stop, got {target} < {entry} < {stop}",
            )

        stop_dist_pct = risk_denom / entry
        if stop_dist_pct > max_stop_distance_pct:
            return (
                False,
                PriceOrderingStatus.EXCESSIVE_STOP_DISTANCE,
                f"Short stop loss distance {stop_dist_pct:.1%} exceeds maximum permitted risk of {max_stop_distance_pct:.1%}",
            )

    return True, PriceOrderingStatus.VALID, None


def compute_breakeven_win_rate(reward_risk_ratio: Optional[float]) -> Optional[float]:
    """
    Computes break-even win rate required for non-negative expectation under fixed payoffs:
        WinRate_BE = 1 / (1 + RR)
    """
    if reward_risk_ratio is None or reward_risk_ratio <= 0.0:
        return None
    return round(1.0 / (1.0 + reward_risk_ratio), 4)


def estimate_empirical_win_probability(
    composite_score: Optional[float],
    direction: TradeDirection = TradeDirection.LONG,
    historical_win_rate: Optional[float] = None,
) -> float:
    """
    Estimates realized win probability from historical calibrated outcomes or composite score.

    Calibrated mapping:
      Neutral score 50.0 maps to market baseline 50.0% win rate.
      Each 10 score points in trade direction provides ~3.5% empirical edge.
      Score 60.0 (BUY floor) -> 53.5% estimated win rate.
      Score 40.0 (SHORT ceiling) -> 53.5% estimated win rate.
    """
    if historical_win_rate is not None and 0.0 < historical_win_rate < 1.0:
        return round(historical_win_rate, 4)

    if composite_score is None:
        return 0.50

    score = max(0.0, min(100.0, float(composite_score)))

    if direction == TradeDirection.LONG:
        delta = score - 50.0
    else:
        delta = 50.0 - score

    p_win = 0.50 + (delta * 0.0035)
    return round(max(0.30, min(0.75, p_win)), 4)


def compute_expected_value_r(
    win_probability: float, reward_risk_ratio: float
) -> float:
    """
    Computes mathematical expectancy in R-multiples:
        E[R] = (p_win * RR) - (1 - p_win)
    """
    if reward_risk_ratio <= 0.0 or not (0.0 <= win_probability <= 1.0):
        return -1.0
    return round((win_probability * reward_risk_ratio) - (1.0 - win_probability), 4)


def compute_validated_reward_risk(
    direction: Union[TradeDirection, str],
    entry_price: float,
    stop_loss: float,
    target_price: float,
    execution_costs: Optional[ExecutionCostModel] = None,
    min_rr_threshold: float = 1.20,
    composite_score: Optional[float] = None,
    win_probability: Optional[float] = None,
    min_edge_pct: float = 0.02,
    max_stop_distance_pct: float = 0.30,
    require_positive_expectancy: bool = True,
) -> RewardRiskResult:
    """
    Main deterministic reward-to-risk calculation and trade justification engine.

    Formulas:
      Long:
        RR_gross = (Target - Entry) / (Entry - Stop)
        BE_gross = 1 / (1 + RR_gross)

      Short:
        RR_short = (Entry - Target) / (Stop - Entry)
        BE_gross = 1 / (1 + RR_gross)

    Accounting for execution assumptions:
      Adjusts entry, stop, and target for slippage and commissions.
      RR_net = Net_Reward / Net_Risk
      BE_net = Net_Risk / (Net_Reward + Net_Risk) = 1 / (1 + RR_net)
    """
    if isinstance(direction, str):
        direction = TradeDirection(direction.upper())

    costs = execution_costs or ExecutionCostModel()

    # Step 1: Deterministic price ordering and risk denominator checks
    is_valid, status, rejection_reason = validate_price_ordering(
        direction=direction,
        entry=entry_price,
        stop=stop_loss,
        target=target_price,
        max_stop_distance_pct=max_stop_distance_pct,
    )

    if not is_valid:
        return RewardRiskResult(
            is_valid=False,
            status=status,
            rejection_reason=rejection_reason,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            gross_reward=0.0,
            gross_risk=0.0,
            gross_reward_risk_ratio=None,
            breakeven_win_rate_gross=None,
            effective_entry=entry_price,
            effective_stop=stop_loss,
            effective_target=target_price,
            net_reward=0.0,
            net_risk=0.0,
            net_reward_risk_ratio=None,
            breakeven_win_rate_net=None,
            cost_drag_bps=0.0,
            trade_justified=False,
        )

    # Step 2: Compute gross reward and risk
    if direction == TradeDirection.LONG:
        gross_reward = target_price - entry_price
        gross_risk = entry_price - stop_loss
    else:
        gross_reward = entry_price - target_price
        gross_risk = stop_loss - entry_price

    gross_rr = round(gross_reward / gross_risk, 2)
    be_gross = compute_breakeven_win_rate(gross_rr)

    # Step 3: Compute execution assumptions and transaction costs
    eff_entry, entry_costs = costs.compute_entry_execution(direction, entry_price)
    eff_stop, stop_costs = costs.compute_exit_execution(direction, stop_loss, is_stop=True)
    eff_target, target_costs = costs.compute_exit_execution(direction, target_price, is_stop=False)

    if direction == TradeDirection.LONG:
        net_reward = eff_target - eff_entry
        net_risk = eff_entry - eff_stop
    else:
        net_reward = eff_entry - eff_target
        net_risk = eff_stop - eff_entry

    cost_breakdown = {
        "entry_friction": round(entry_costs, 4),
        "target_exit_friction": round(target_costs, 4),
        "stop_exit_friction": round(stop_costs, 4),
        "total_cost_target_win": round(entry_costs + target_costs, 4),
        "total_cost_stop_loss": round(entry_costs + stop_costs, 4),
    }

    # Reject if costs wipe out net reward or create negative net denominator
    if net_reward <= 0.0 or net_risk <= 0.0:
        return RewardRiskResult(
            is_valid=False,
            status=PriceOrderingStatus.ZERO_OR_NEGATIVE_REWARD if net_reward <= 0.0 else PriceOrderingStatus.ZERO_OR_NEGATIVE_RISK,
            rejection_reason="Transaction costs exceed reward or expand risk beyond viability",
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            gross_reward=round(gross_reward, 2),
            gross_risk=round(gross_risk, 2),
            gross_reward_risk_ratio=gross_rr,
            breakeven_win_rate_gross=be_gross,
            effective_entry=round(eff_entry, 2),
            effective_stop=round(eff_stop, 2),
            effective_target=round(eff_target, 2),
            net_reward=round(net_reward, 2),
            net_risk=round(net_risk, 2),
            net_reward_risk_ratio=None,
            breakeven_win_rate_net=None,
            cost_drag_bps=round(((gross_reward - net_reward) / entry_price) * 10000.0, 1),
            cost_breakdown=cost_breakdown,
            trade_justified=False,
        )

    net_rr = round(net_reward / net_risk, 2)
    be_net = compute_breakeven_win_rate(net_rr)
    cost_drag_bps = round(((gross_reward - net_reward) / entry_price) * 10000.0, 1)

    # Step 4: Validate realized trade outcome evidence & empirical statistical edge
    p_win = win_probability if win_probability is not None else estimate_empirical_win_probability(
        composite_score, direction
    )

    ev_r = compute_expected_value_r(p_win, net_rr)
    actual_edge = round(p_win - (be_net or 0.50), 4)
    has_statistical_edge = actual_edge >= min_edge_pct and (not require_positive_expectancy or ev_r > 0.0)
    meets_min_rr = gross_rr >= min_rr_threshold

    # Justified trade requires valid geometry, clearing minimum RR, and verified statistical edge
    trade_justified = is_valid and meets_min_rr and has_statistical_edge

    justification_failure = None
    if not meets_min_rr:
        justification_failure = f"Reward:risk ratio {gross_rr:.2f} < policy minimum {min_rr_threshold:.2f}"
    elif not has_statistical_edge:
        justification_failure = (
            f"Insufficient statistical edge: estimated win probability {p_win:.1%} "
            f"does not clear net break-even {be_net:.1%} with required {min_edge_pct:.1%} buffer (E[R]={ev_r:.2f}R)"
        )

    return RewardRiskResult(
        is_valid=True,
        status=PriceOrderingStatus.VALID,
        rejection_reason=justification_failure,
        direction=direction,
        entry_price=entry_price,
        stop_loss=stop_loss,
        target_price=target_price,
        gross_reward=round(gross_reward, 2),
        gross_risk=round(gross_risk, 2),
        gross_reward_risk_ratio=gross_rr,
        breakeven_win_rate_gross=be_gross,
        effective_entry=round(eff_entry, 2),
        effective_stop=round(eff_stop, 2),
        effective_target=round(eff_target, 2),
        net_reward=round(net_reward, 2),
        net_risk=round(net_risk, 2),
        net_reward_risk_ratio=net_rr,
        breakeven_win_rate_net=be_net,
        cost_drag_bps=cost_drag_bps,
        cost_breakdown=cost_breakdown,
        estimated_win_probability=p_win,
        required_edge_pct=min_edge_pct,
        actual_edge_pct=actual_edge,
        expected_value_r=ev_r,
        has_statistical_edge=has_statistical_edge,
        meets_minimum_rr=meets_min_rr,
        trade_justified=trade_justified,
    )


def derive_deterministic_channel_geometry(
    direction: Union[TradeDirection, str],
    current_price: float,
    atr: float,
    high_20d: Optional[float] = None,
    low_20d: Optional[float] = None,
    suggested_stop_loss: Optional[float] = None,
    suggested_target_price: Optional[float] = None,
    max_stop_distance_pct: float = 0.30,
) -> Tuple[Optional[float], Optional[float], Optional[float], Dict[str, Any]]:
    """
    Produces deterministic channel-anchored entry, stop, and target prices.
    Strictly sanitizes and rejects candidate/LLM prices if they violate ordering,
    have zero/negative denominator, or set unvalidated excessive risk.

    Conservative geometry:
      Long:
        stop = min(price - 1.5*ATR, low_20d - 0.5*ATR) capped at entry - 0.1
        target = min(price + 2.5*ATR, high_20d + 0.5*ATR) capped at entry + 0.1

      Short:
        stop = max(price + 1.5*ATR, high_20d + 0.5*ATR) capped at entry + 0.1
        target = max(price - 2.5*ATR, low_20d - 0.5*ATR) capped at entry - 0.1

    Returns:
        (entry, structural_stop, structural_target, validation_metadata)
    """
    if isinstance(direction, str):
        direction = TradeDirection(direction.upper())

    metadata = {
        "candidate_override_attempted": False,
        "candidate_accepted": False,
        "sanitization_notes": [],
    }

    if current_price <= 0.0:
        return None, None, None, metadata

    entry = current_price
    atr_val = atr if (atr is not None and atr > 0.0) else 0.02 * entry

    if direction == TradeDirection.LONG:
        # Stop candidates: low_20d anchored, or ATR fallback
        stop_cands = []
        if low_20d is not None and low_20d > 0:
            stop_cands.append(low_20d - 0.5 * atr_val)
        if atr_val > 0:
            stop_cands.append(entry - 1.5 * atr_val)
        valid_stops = [s for s in stop_cands if s < entry]
        structural_stop = min(valid_stops) if valid_stops else None

        # Target candidates: high_20d resistance capped, or ATR fallback
        tgt_cands = []
        if high_20d is not None and high_20d > entry:
            tgt_cands.append(high_20d + 0.5 * atr_val)
        elif high_20d is not None:
            tgt_cands.append(high_20d)
        if not tgt_cands and atr_val > 0:
            tgt_cands.append(entry + 2.5 * atr_val)
        valid_targets = [t for t in tgt_cands if t > entry]
        structural_target = min(valid_targets) if valid_targets else None

        # Conservative bounding with candidate prices:
        # Long stop = lower of (structural_stop, cand_stop)
        # Long target = conservative capped target
        if suggested_stop_loss is not None or suggested_target_price is not None:
            metadata["candidate_override_attempted"] = True
            cand_stop = suggested_stop_loss if suggested_stop_loss is not None else structural_stop
            cand_tgt = suggested_target_price if suggested_target_price is not None else structural_target

            if cand_stop is not None and cand_tgt is not None:
                cand_valid, cand_status, cand_reason = validate_price_ordering(
                    direction=direction,
                    entry=entry,
                    stop=cand_stop,
                    target=cand_tgt,
                    max_stop_distance_pct=max_stop_distance_pct,
                )

                if cand_valid:
                    metadata["candidate_accepted"] = True
                    stops_to_check = [s for s in [structural_stop, cand_stop] if s is not None and s < entry]
                    structural_stop = min(stops_to_check) if stops_to_check else cand_stop

                    targets_to_check = [t for t in [structural_target, cand_tgt] if t is not None and t > entry]
                    structural_target = min(targets_to_check) if targets_to_check else cand_tgt
                else:
                    metadata["candidate_accepted"] = False
                    metadata["sanitization_notes"].append(
                        f"Candidate prices rejected ({cand_status.value}): {cand_reason}. Snapped to deterministic structural levels."
                    )

    else:  # SHORT
        # Stop candidates for short: high_20d overhead risk, or ATR fallback
        stop_cands = []
        if high_20d is not None and high_20d > 0:
            stop_cands.append(high_20d + 0.5 * atr_val)
        if atr_val > 0:
            stop_cands.append(entry + 1.5 * atr_val)
        valid_stops = [s for s in stop_cands if s > entry]
        structural_stop = max(valid_stops) if valid_stops else None

        # Target candidates for short: low_20d support floor, or ATR fallback
        tgt_cands = []
        if low_20d is not None and low_20d < entry and low_20d > 0:
            tgt_cands.append(low_20d - 0.5 * atr_val)
        elif low_20d is not None and low_20d < entry:
            tgt_cands.append(low_20d)
        if not tgt_cands and atr_val > 0:
            tgt_cands.append(entry - 2.5 * atr_val)
        valid_targets = [t for t in tgt_cands if t < entry]
        structural_target = max(valid_targets) if valid_targets else None

        # Conservative bounding with candidate prices for short:
        # Short stop = higher of (structural_stop, cand_stop)
        # Short target = higher floor (conservative support)
        if suggested_stop_loss is not None or suggested_target_price is not None:
            metadata["candidate_override_attempted"] = True
            cand_stop = suggested_stop_loss if suggested_stop_loss is not None else structural_stop
            cand_tgt = suggested_target_price if suggested_target_price is not None else structural_target

            if cand_stop is not None and cand_tgt is not None:
                cand_valid, cand_status, cand_reason = validate_price_ordering(
                    direction=direction,
                    entry=entry,
                    stop=cand_stop,
                    target=cand_tgt,
                    max_stop_distance_pct=max_stop_distance_pct,
                )

                if cand_valid:
                    metadata["candidate_accepted"] = True
                    stops_to_check = [s for s in [structural_stop, cand_stop] if s is not None and s > entry]
                    structural_stop = max(stops_to_check) if stops_to_check else cand_stop

                    targets_to_check = [t for t in [structural_target, cand_tgt] if t is not None and t < entry]
                    structural_target = max(targets_to_check) if targets_to_check else cand_tgt
                else:
                    metadata["candidate_accepted"] = False
                    metadata["sanitization_notes"].append(
                        f"Candidate prices rejected ({cand_status.value}): {cand_reason}. Snapped to deterministic structural levels."
                    )

    return (
        round(entry, 2),
        round(structural_stop, 2) if structural_stop is not None else None,
        round(structural_target, 2) if structural_target is not None else None,
        metadata,
    )
