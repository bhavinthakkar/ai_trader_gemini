"""
Deterministic Policy Compliance Gates Engine (Step 5.3).

Explicitly implements the five compliance gates with unambiguous precedence,
conflict resolution, and mathematically rigorous controls:
1. BUY Floor Gate: Standard BUY vs separately eligible Dip-Buy branch; LLM override prohibition.
2. Reward-to-Risk Gate: Valid price ordering, non-negative risk denominators, cost-adjusted ratio, statistical edge.
3. Volatility, Events & Liquidity Gate: Options 1-sigma expected move, horizon, ATR fallback, earnings blackout, universal liquidity controls.
4. SELL and Exit Symmetry Gate: Long exits vs short entries; strict holding-thesis guard invariants.
5. Gated Confidence Gate: Exact formula, 5-pillar consensus dispersion, data completeness scaling, missing-data policy, and empirical threshold.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple, Union

from quant.policy_config import TradingPolicy, get_default_policy, load_policy_config
from quant.trade_geometry import (
    TradeDirection,
    PriceOrderingStatus,
    ExecutionCostModel,
    RewardRiskResult,
    validate_price_ordering,
    compute_validated_reward_risk,
)
from quant.signal_strength import (
    DispersionMetric,
    CombinationMethod,
    SignalStrengthWeights,
    calculate_signal_strength,
)
from quant.portfolio_controls import ShortEligibilityValidator



class GateStatus(str, Enum):
    """Execution status for an individual gate."""
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass
class GateResult:
    """Detailed audit record for an individual compliance gate evaluation."""
    gate_name: str
    passed: bool
    status: GateStatus
    reason_code: Optional[str] = None
    message: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["status"] = self.status.value
        return res


# =============================================================================
# GATE 1: BUY FLOOR GATE
# =============================================================================
class BuyFloorGate:
    """
    Gate 1: Composite Score Floor Gate.

    Semantics:
    1. Ordinary BUY branch: Requires deterministic composite score >= buy_minimum (60.0).
    2. Dip-Buy branch: If catalyst is DIP_BUY and strict structural dislocation checks hold
       (RSI [30, 45], 5d drop [5%, 15%], valuation history >= 60, no value trap),
       the floor drops to dip_buy_minimum (52.0). If any condition fails, reverts to 60.0.
    3. LLM Invariant: An LLM BUY advisory cannot override a failed score requirement.
    """

    @staticmethod
    def evaluate(
        candidate_decision: str,
        composite_score: Optional[float],
        catalyst_path: Optional[str] = None,
        rsi14: Optional[float] = None,
        drawdown_5d_pct: Optional[float] = None,
        valuation_history_score: Optional[float] = None,
        peer_valuation_score: Optional[float] = None,
        is_value_trap: bool = False,
        policy: Optional[TradingPolicy] = None,
    ) -> GateResult:
        active_policy = policy or get_default_policy()

        if candidate_decision.upper() != "BUY":
            return GateResult(
                gate_name="1. BUY Floor",
                passed=True,
                status=GateStatus.NOT_APPLICABLE,
                message="Candidate decision is not BUY; score floor not applicable.",
            )

        # Evaluate Dip-Buy eligibility
        is_dip_path = catalyst_path == "DIP_BUY"
        dip_certified = False
        dip_rejection_reasons = []

        if is_dip_path and active_policy.dip_buy.enabled:
            # RSI bounds
            if rsi14 is None or not (active_policy.dip_buy.min_rsi <= rsi14 <= active_policy.dip_buy.max_rsi):
                dip_rejection_reasons.append(
                    f"RSI {rsi14} outside [{active_policy.dip_buy.min_rsi}, {active_policy.dip_buy.max_rsi}]"
                )
            # Drawdown bounds (supports both positive magnitude e.g. 7.5% and signed drop e.g. -7.5%)
            dd_pct = abs(drawdown_5d_pct) if drawdown_5d_pct is not None else None
            if dd_pct is None or not (
                active_policy.dip_buy.min_drop_pct_5d <= dd_pct <= active_policy.dip_buy.max_drop_pct_5d
            ):
                dip_rejection_reasons.append(
                    f"5d drawdown {drawdown_5d_pct}% outside [{active_policy.dip_buy.min_drop_pct_5d}%, {active_policy.dip_buy.max_drop_pct_5d}%]"
                )
            # Valuation history anchor
            if valuation_history_score is None or valuation_history_score < active_policy.dip_buy.min_valuation_history:
                dip_rejection_reasons.append(
                    f"valuation history {valuation_history_score} < {active_policy.dip_buy.min_valuation_history}"
                )
            # Peer valuation anchor (if provided)
            if peer_valuation_score is not None and peer_valuation_score < active_policy.dip_buy.min_peer_valuation:
                dip_rejection_reasons.append(
                    f"peer valuation {peer_valuation_score} < {active_policy.dip_buy.min_peer_valuation}"
                )
            # Value trap prohibition
            if is_value_trap and active_policy.dip_buy.forbid_value_trap:
                dip_rejection_reasons.append("confirmed balance-sheet value trap")

            dip_certified = len(dip_rejection_reasons) == 0

        # Resolve precedence floor
        floor, rule_name = active_policy.resolve_buy_score_floor(is_certified_dip_buy=dip_certified)

        metrics = {
            "composite_score": composite_score,
            "required_floor": floor,
            "rule_name": rule_name,
            "is_dip_path": is_dip_path,
            "dip_certified": dip_certified,
            "dip_rejection_reasons": dip_rejection_reasons,
        }

        # Check composite against resolved floor
        if composite_score is None:
            return GateResult(
                gate_name="1. BUY Floor",
                passed=False,
                status=GateStatus.FAIL,
                reason_code="COMPOSITE_UNAVAILABLE",
                message="Deterministic composite score could not be computed from market data.",
                metrics=metrics,
            )

        if float(composite_score) < floor:
            return GateResult(
                gate_name="1. BUY Floor",
                passed=False,
                status=GateStatus.FAIL,
                reason_code="COMPOSITE_BELOW_FLOOR",
                message=f"Deterministic composite {float(composite_score):.1f}/100 < required floor {floor:.1f} ({rule_name}).",
                metrics=metrics,
            )

        return GateResult(
            gate_name="1. BUY Floor",
            passed=True,
            status=GateStatus.PASS,
            message=f"Deterministic composite {float(composite_score):.1f}/100 clears floor {floor:.1f} ({rule_name}).",
            metrics=metrics,
        )


# =============================================================================
# GATE 2: REWARD-TO-RISK GATE
# =============================================================================
class RewardToRiskGate:
    """
    Gate 2: Setup Geometry & Reward-to-Risk Gate.

    Semantics:
    1. Valid price ordering: 0 < Stop < Entry < Target (Long) or 0 < Target < Entry < Stop (Short).
    2. Positive risk denominator: (Entry - Stop) > 0 (Long) or (Stop - Entry) > 0 (Short).
    3. Minimum threshold: RR_gross >= reward_risk_minimum (1.20 for long, 1.50 for short).
    4. Transaction cost viability: Net RR after slippage and commissions must remain viable.
    5. Realized trade outcome justification: Estimated win probability must clear net break-even
       win rate with configured edge margin (min_edge_pct) and strictly positive expectancy.
    """

    @staticmethod
    def evaluate(
        direction: Union[TradeDirection, str],
        entry_price: float,
        stop_loss: float,
        target_price: float,
        composite_score: Optional[float] = None,
        policy: Optional[TradingPolicy] = None,
        execution_costs: Optional[ExecutionCostModel] = None,
        max_stop_distance_pct: Optional[float] = None,
    ) -> GateResult:
        active_policy = policy or get_default_policy()
        if isinstance(direction, str):
            direction = TradeDirection(direction.upper())

        min_rr = (
            1.50 if direction == TradeDirection.SHORT else active_policy.trade_quality.reward_risk_minimum
        )
        stop_dist_cap = (
            max_stop_distance_pct
            if max_stop_distance_pct is not None
            else active_policy.trade_quality.max_stop_distance_pct
        )

        rr_result = compute_validated_reward_risk(
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            execution_costs=execution_costs,
            min_rr_threshold=min_rr,
            composite_score=composite_score,
            min_edge_pct=active_policy.trade_quality.min_edge_pct,
            max_stop_distance_pct=stop_dist_cap,
            require_positive_expectancy=active_policy.trade_quality.require_positive_expectancy,
        )

        metrics = rr_result.to_dict()

        if not rr_result.is_valid:
            return GateResult(
                gate_name="2. Reward-to-Risk",
                passed=False,
                status=GateStatus.FAIL,
                reason_code=rr_result.status.value,
                message=f"Setup geometry rejected: {rr_result.rejection_reason}",
                metrics=metrics,
            )

        if not rr_result.meets_minimum_rr:
            return GateResult(
                gate_name="2. Reward-to-Risk",
                passed=False,
                status=GateStatus.FAIL,
                reason_code="RR_TOO_LOW",
                message=f"Reward:risk ratio {rr_result.gross_reward_risk_ratio:.2f} < configured threshold {min_rr:.2f}",
                metrics=metrics,
            )

        if not rr_result.has_statistical_edge:
            return GateResult(
                gate_name="2. Reward-to-Risk",
                passed=False,
                status=GateStatus.FAIL,
                reason_code="INSUFFICIENT_STATISTICAL_EDGE",
                message=(
                    f"Insufficient statistical edge: estimated win probability "
                    f"{(rr_result.estimated_win_probability or 0.0):.1%} does not clear "
                    f"net break-even rate {(rr_result.breakeven_win_rate_net or 0.0):.1%} with "
                    f"required {active_policy.trade_quality.min_edge_pct:.1%} buffer."
                ),
                metrics=metrics,
            )

        return GateResult(
            gate_name="2. Reward-to-Risk",
            passed=True,
            status=GateStatus.PASS,
            message=(
                f"Reward:risk {rr_result.gross_reward_risk_ratio:.2f} (Net: {rr_result.net_reward_risk_ratio:.2f}) "
                f"clears threshold {min_rr:.2f} with verified edge (E[R]={rr_result.expected_value_r:.2f}R)."
            ),
            metrics=metrics,
        )


# =============================================================================
# GATE 3: VOLATILITY, EVENTS & UNIVERSAL LIQUIDITY GATE
# =============================================================================
class VolatilityEventLiquidityGate:
    """
    Gate 3: Volatility Regimes, Event Risk, and Universal Liquidity Controls.

    Semantics:
    1. Earnings Blackout: Prohibits fresh entries within earnings_blackout_sessions (default 3 sessions).
    2. Scheduled Event Risk: Prohibits fresh entries into known corporate binary events.
    3. Ex-Dividend Blackout: Prohibits fresh short entries within 2 sessions of ex-dividend.
    4. Options 1-Sigma Move:
       - Source: ATM Implied Volatility (IV) from options chain; horizon = option_horizon_days (20d).
       - Fallback: Realized volatility equivalent: ATR_14 * sqrt(T) when options are missing.
       - Noise vs Risk: Stop distance must clear >= 0.85 * 1-sigma move to avoid price-noise shakeout,
         while per-share risk is capped at max_stop_distance_pct (30%). Position size scales inversely.
    5. Universal Liquidity Controls:
       - Maximum participation rate <= 1.0% of 20-day Average Daily Volume (ADV).
       - Square-root market impact <= max_market_impact_bps (15 bps).
    """

    @staticmethod
    def calculate_expected_move(
        price: float,
        iv_pct: Optional[float],
        atr: Optional[float],
        horizon_days: int = 20,
    ) -> Tuple[float, str]:
        """
        Calculates 1-standard-deviation (1-sigma) expected move over the trade horizon.

        Returns:
            (expected_move_dollars, move_source)
        """
        if price <= 0.0:
            return 0.0, "INVALID_PRICE"

        t_fraction = horizon_days / 252.0

        # Primary source: Options ATM Implied Volatility (if reliable and within crisis bounds)
        if iv_pct is not None and 5.0 <= iv_pct <= 130.0:
            sigma_annual = iv_pct / 100.0
            move_dollars = price * sigma_annual * math.sqrt(t_fraction)
            return round(move_dollars, 2), "OPTIONS_ATM_IMPLIED_VOLATILITY"

        # Fallback source: Realized volatility / ATR equivalent
        atr_val = atr if (atr is not None and atr > 0.0) else (0.02 * price)
        # 1-sigma move over T sessions from daily ATR: ATR * sqrt(T)
        realized_move = atr_val * math.sqrt(horizon_days)
        return round(realized_move, 2), "REALIZED_ATR_FALLBACK"

    @staticmethod
    def evaluate_market_impact(
        intended_order_value: float,
        avg_dollar_volume_20d: float,
        price: float,
        atr: float,
        max_participation_rate: float = 0.01,
        max_market_impact_bps: float = 15.0,
    ) -> Tuple[bool, float, float, Optional[str]]:
        """
        Evaluates universal liquidity and square-root market impact (Almgren-Chriss approximation).

        Returns:
            (passed, max_permitted_order_value, estimated_impact_bps, rejection_reason)
        """
        if avg_dollar_volume_20d <= 0.0 or price <= 0.0:
            return False, 0.0, 999.0, "Average daily dollar volume unavailable or zero"

        max_permitted_value = avg_dollar_volume_20d * max_participation_rate
        if intended_order_value > max_permitted_value:
            return (
                False,
                max_permitted_value,
                999.0,
                f"Intended order value ${intended_order_value:,.0f} exceeds {max_participation_rate*100:.1f}% "
                f"ADV cap (${max_permitted_value:,.0f})",
            )

        # Daily volatility estimate from ATR
        daily_vol = (atr / price) if price > 0 else 0.02
        # Square-root impact model: Impact = gamma * daily_vol * sqrt(Order / ADV)
        participation_fraction = intended_order_value / avg_dollar_volume_20d
        impact_fraction = 0.50 * daily_vol * math.sqrt(participation_fraction)
        impact_bps = round(impact_fraction * 10000.0, 1)

        if impact_bps > max_market_impact_bps:
            return (
                False,
                max_permitted_value,
                impact_bps,
                f"Estimated market impact {impact_bps:.1f} bps exceeds maximum permitted {max_market_impact_bps:.1f} bps",
            )

        return True, max_permitted_value, impact_bps, None

    @classmethod
    def evaluate(
        cls,
        direction: Union[TradeDirection, str],
        price: float,
        stop_loss: float,
        atr: float,
        days_to_earnings: Optional[int] = None,
        is_event_risk_blackout: bool = False,
        days_to_ex_dividend: Optional[int] = None,
        iv_pct: Optional[float] = None,
        rvol_20d: Optional[float] = None,
        avg_dollar_volume_20d: Optional[float] = None,
        intended_order_value: float = 25_000.0,
        policy: Optional[TradingPolicy] = None,
    ) -> GateResult:
        active_policy = policy or get_default_policy()
        if isinstance(direction, str):
            direction = TradeDirection(direction.upper())

        metrics: Dict[str, Any] = {}

        # 1. Earnings Blackout
        if days_to_earnings is not None and days_to_earnings <= active_policy.events.earnings_blackout_sessions:
            metrics["days_to_earnings"] = days_to_earnings
            return GateResult(
                gate_name="3. Volatility, Events & Liquidity",
                passed=False,
                status=GateStatus.FAIL,
                reason_code="EARNINGS_BLACKOUT",
                message=f"Earnings report in {days_to_earnings} day(s) is a binary gap-risk event.",
                metrics=metrics,
            )

        # 2. Scheduled Company Event Risk
        if is_event_risk_blackout and active_policy.events.event_risk_blackout:
            return GateResult(
                gate_name="3. Volatility, Events & Liquidity",
                passed=False,
                status=GateStatus.FAIL,
                reason_code="EVENT_RISK",
                message="Scheduled corporate binary event (regulatory/investor day) enforces mandatory HOLD.",
                metrics=metrics,
            )

        # 3. Ex-Dividend Short Blackout
        if (
            direction == TradeDirection.SHORT
            and days_to_ex_dividend is not None
            and days_to_ex_dividend <= active_policy.events.ex_dividend_short_blackout_days
        ):
            metrics["days_to_ex_dividend"] = days_to_ex_dividend
            return GateResult(
                gate_name="3. Volatility, Events & Liquidity",
                passed=False,
                status=GateStatus.FAIL,
                reason_code="EX_DIVIDEND_BLACKOUT",
                message=f"Short sales prohibited within {days_to_ex_dividend} day(s) of ex-dividend date.",
                metrics=metrics,
            )

        # 4. Options 1-Sigma Expected Move vs Stop Placement
        horizon = active_policy.volatility.option_horizon_days
        expected_move, move_source = cls.calculate_expected_move(price, iv_pct, atr, horizon_days=horizon)
        stop_distance = abs(price - stop_loss)
        min_required_stop = expected_move * active_policy.volatility.min_stop_noise_multiple

        metrics["expected_move_dollars"] = expected_move
        metrics["expected_move_source"] = move_source
        metrics["stop_distance_dollars"] = round(stop_distance, 2)
        metrics["min_required_noise_stop"] = round(min_required_stop, 2)

        if stop_distance < min_required_stop:
            return GateResult(
                gate_name="3. Volatility, Events & Liquidity",
                passed=False,
                status=GateStatus.FAIL,
                reason_code="STOP_INSIDE_NOISE_ZONE",
                message=(
                    f"Stop distance ${stop_distance:.2f} is inside the 1-sigma noise zone "
                    f"(${min_required_stop:.2f} required via {move_source}); high risk of routine noise exit."
                ),
                metrics=metrics,
            )

        # 5. Universal Liquidity Controls & Market Impact Sizing
        if rvol_20d is not None and rvol_20d < active_policy.trade_quality.min_rvol_20d:
            metrics["rvol_20d"] = rvol_20d
            return GateResult(
                gate_name="3. Volatility, Events & Liquidity",
                passed=False,
                status=GateStatus.FAIL,
                reason_code="LOW_LIQUIDITY",
                message=f"Relative volume {rvol_20d:.2f}x < required minimum {active_policy.trade_quality.min_rvol_20d:.2f}x.",
                metrics=metrics,
            )

        if (
            avg_dollar_volume_20d is not None
            and avg_dollar_volume_20d < active_policy.trade_quality.min_avg_dollar_volume
        ):
            metrics["avg_dollar_volume_20d"] = avg_dollar_volume_20d
            return GateResult(
                gate_name="3. Volatility, Events & Liquidity",
                passed=False,
                status=GateStatus.FAIL,
                reason_code="LOW_LIQUIDITY",
                message=(
                    f"20-day average dollar volume ${avg_dollar_volume_20d:,.0f} < "
                    f"required minimum ${active_policy.trade_quality.min_avg_dollar_volume:,.0f}."
                ),
                metrics=metrics,
            )

        adv = avg_dollar_volume_20d if avg_dollar_volume_20d is not None else 1_000_000.0
        liq_ok, max_perm_val, impact_bps, liq_reason = cls.evaluate_market_impact(
            intended_order_value=intended_order_value,
            avg_dollar_volume_20d=adv,
            price=price,
            atr=atr,
            max_participation_rate=active_policy.trade_quality.max_participation_rate,
            max_market_impact_bps=active_policy.trade_quality.max_market_impact_bps,
        )

        metrics["intended_order_value"] = intended_order_value
        metrics["max_permitted_order_value"] = max_perm_val
        metrics["estimated_market_impact_bps"] = impact_bps

        if not liq_ok:
            return GateResult(
                gate_name="3. Volatility, Events & Liquidity",
                passed=False,
                status=GateStatus.FAIL,
                reason_code="LIQUIDITY_INSUFFICIENT",
                message=f"Liquidity control failure: {liq_reason}",
                metrics=metrics,
            )

        return GateResult(
            gate_name="3. Volatility, Events & Liquidity",
            passed=True,
            status=GateStatus.PASS,
            message=(
                f"Volatility and liquidity verified: Stop outside 1-sigma noise (${expected_move:.2f} {move_source}), "
                f"market impact {impact_bps:.1f} bps within {active_policy.trade_quality.max_market_impact_bps:.0f} bps limit."
            ),
            metrics=metrics,
        )


# =============================================================================
# GATE 4: SELL AND EXIT SYMMETRY GATE
# =============================================================================
class SellExitSymmetryGate:
    """
    Gate 4: SELL and Exit Symmetry Gate.

    Semantics:
    1. Separation of Concerns:
       - Long Exits: Liquidating active holdings. Governed by stop breaches, target hits,
         critical evidence vetos, or fundamental thesis breakdowns.
       - Short Entries: Opening fresh bearish positions. Governed by structural breakdown
         score ceilings (composite <= 40), short RR >= 1.5, and borrow availability.
    2. Holding-Thesis Guard:
       - Protects intact long positions (composite >= 70) from premature liquidation.
       - MANDATORY FORCED EXITS (NEVER BLOCKED):
         * Stop loss breach (current_price <= stop_loss)
         * Critical evidence veto (SEC/DOJ probe, fraud, going-concern)
         * Scheduled binary earnings blackout
         * Catastrophic composite collapse (composite < 50.0)
    """

    @staticmethod
    def evaluate(
        candidate_decision: str,
        is_holding: bool,
        current_price: float,
        stop_loss: Optional[float],
        target_price: Optional[float],
        composite_score: Optional[float],
        has_critical_evidence_veto: bool = False,
        is_earnings_blackout: bool = False,
        borrow_available: bool = True,
        policy: Optional[TradingPolicy] = None,
        borrow_status: str = "ETB",
        borrow_fee_bps: float = 0.0,
        available_borrow_shares: Optional[int] = None,
        shares_requested: int = 1,
        rsi: Optional[float] = None,
        ema_20: Optional[float] = None,
        ema_50: Optional[float] = None,
        trend_score: Optional[float] = None,
        financial_quality_score: Optional[float] = None,
        valuation_history_score: Optional[float] = None,
        reward_risk_ratio: Optional[float] = None,
        days_to_ex_dividend: Optional[int] = None,
        available_cash: Optional[float] = None,
        portfolio_equity: Optional[float] = None,
        ticker: str = "TICKER",
    ) -> GateResult:
        active_policy = policy or get_default_policy()
        metrics = {
            "candidate_decision": candidate_decision,
            "is_holding": is_holding,
            "current_price": current_price,
            "stop_loss": stop_loss,
            "composite_score": composite_score,
            "has_critical_evidence_veto": has_critical_evidence_veto,
            "borrow_available": borrow_available,
            "borrow_status": borrow_status,
            "borrow_fee_bps": borrow_fee_bps,
        }

        # Branch A: Active Holding Exit Evaluation
        if is_holding:
            # 1. Stop Loss Breach (MANDATORY FORCED EXIT)
            if stop_loss is not None and current_price <= stop_loss:
                return GateResult(
                    gate_name="4. SELL & Exit Symmetry",
                    passed=True,
                    status=GateStatus.PASS,
                    reason_code="STOP_LOSS_BREACH",
                    message=f"Mandatory exit: Current price ${current_price:.2f} breached stop loss ${stop_loss:.2f}.",
                    metrics=metrics,
                )

            # 2. Critical Evidence Veto (MANDATORY FORCED EXIT)
            if has_critical_evidence_veto:
                return GateResult(
                    gate_name="4. SELL & Exit Symmetry",
                    passed=True,
                    status=GateStatus.PASS,
                    reason_code="EVIDENCE_VETO_FORCED_EXIT",
                    message="Mandatory exit: Critical qualitative risk flag or regulatory probe detected.",
                    metrics=metrics,
                )

            # 3. Scheduled Earnings Blackout (MANDATORY FORCED EXIT)
            if is_earnings_blackout and active_policy.holding_guard.never_block_earnings_blackout:
                return GateResult(
                    gate_name="4. SELL & Exit Symmetry",
                    passed=True,
                    status=GateStatus.PASS,
                    reason_code="EARNINGS_BLACKOUT_EXIT",
                    message="Mandatory exit: Position entering binary earnings risk event.",
                    metrics=metrics,
                )

            # 4. Catastrophic Composite Collapse (MANDATORY FORCED EXIT)
            if (
                composite_score is not None
                and float(composite_score) < active_policy.holding_guard.catastrophic_composite_floor
            ):
                return GateResult(
                    gate_name="4. SELL & Exit Symmetry",
                    passed=True,
                    status=GateStatus.PASS,
                    reason_code="THESIS_COLLAPSED",
                    message=(
                        f"Mandatory exit: Composite score {float(composite_score):.1f} collapsed below "
                        f"thesis maintenance floor {active_policy.holding_guard.catastrophic_composite_floor:.1f}."
                    ),
                    metrics=metrics,
                )

            # 5. Holding Thesis Guard Interception:
            # If thesis is intact (composite >= 70) and no fatal exit triggered, prevent premature sell
            is_intact = active_policy.is_holding_thesis_intact(
                is_holding=True,
                composite_score=composite_score,
                has_critical_risk=has_critical_evidence_veto,
                current_price=current_price,
                stop_loss=stop_loss,
                is_earnings_blackout=is_earnings_blackout,
            )
            if is_intact and candidate_decision.upper() == "SELL":
                return GateResult(
                    gate_name="4. SELL & Exit Symmetry",
                    passed=False,
                    status=GateStatus.FAIL,
                    reason_code="HOLDING_THESIS_INTACT",
                    message=(
                        f"Premature SELL intercepted by Holding-Thesis Guard: Composite "
                        f"{float(composite_score or 0):.1f} >= {active_policy.holding_guard.thesis_intact_score_minimum:.0f}; "
                        f"position converted to HOLD."
                    ),
                    metrics=metrics,
                )

        # Branch B: Directional Short Entry Evaluation
        if not is_holding and candidate_decision.upper() == "SELL":
            short_res = ShortEligibilityValidator.validate(
                ticker=ticker,
                entry_price=current_price,
                stop_loss=stop_loss or (current_price * 1.05),
                composite_score=composite_score,
                policy=active_policy,
                borrow_available=borrow_available,
                borrow_status=borrow_status,
                borrow_fee_bps=borrow_fee_bps,
                available_borrow_shares=available_borrow_shares,
                shares_requested=shares_requested,
                rsi=rsi,
                ema_20=ema_20,
                ema_50=ema_50,
                trend_score=trend_score,
                financial_quality_score=financial_quality_score,
                valuation_history_score=valuation_history_score,
                reward_risk_ratio=reward_risk_ratio,
                days_to_ex_dividend=days_to_ex_dividend,
                available_cash=available_cash,
                portfolio_equity=portfolio_equity,
            )
            metrics.update(short_res.metrics)
            if not short_res.is_eligible:
                return GateResult(
                    gate_name="4. SELL & Exit Symmetry",
                    passed=False,
                    status=GateStatus.FAIL,
                    reason_code=short_res.reason_code or "SHORT_STRUCTURAL_INVALID",
                    message=short_res.message or "Short eligibility and financing check failed.",
                    metrics=metrics,
                )

        return GateResult(
            gate_name="4. SELL & Exit Symmetry",
            passed=True,
            status=GateStatus.PASS,
            message="Exit and short symmetry conditions satisfied.",
            metrics=metrics,
        )


# =============================================================================
# GATE 5: GATED CONFIDENCE GATE
# =============================================================================
class GatedConfidenceGate:
    """
    Gate 5: Gated Confidence Formulation & Empirical Threshold.

    Exact Mathematical Specification:
      1. Distance: min(1.0, |S - 50.0| / 50.0)
      2. Pillar Consensus Dispersion:
         Mean = (1/K) sum(P_i)
         Dispersion = sum(|P_i - Mean|) / (K * 50.0)
         Agreement = 1.0 - min(1.0, Dispersion)
      3. Completeness: data_completeness in [0.0, 1.0]
      4. Edge Strength: 0.70 * Distance + 0.30 * (Agreement - 0.50)
      5. Base Confidence: 0.50 + 0.50 * Edge Strength
      6. Final Confidence: Base Confidence * (0.50 + 0.50 * Completeness)
      If decision == 'HOLD', confidence is capped at 0.60.

    Missing-Data Policy:
      If < 3 pillars available, deterministic composite is undefined (None),
      completeness is severely penalized, and directional entry is rejected.

    Empirically Justified Threshold:
      Requires Confidence >= 0.65 to authorize directional BUY or SHORT.
    """

    @staticmethod
    def calculate_confidence(
        composite_score: Optional[float],
        pillar_scores: Optional[Dict[str, Optional[float]]],
        data_completeness: float,
        decision: str = "BUY",
        policy: Optional[TradingPolicy] = None,
        is_certified_dip_buy: bool = False,
    ) -> Tuple[float, Dict[str, Any]]:
        active_policy = policy or get_default_policy()
        weights = SignalStrengthWeights(
            weight_decisiveness=active_policy.signal_strength.weight_decisiveness,
            weight_agreement=active_policy.signal_strength.weight_agreement,
            completeness_blend_weight=active_policy.signal_strength.completeness_blend_weight,
            linear_weight_decisiveness=active_policy.signal_strength.linear_weight_decisiveness,
            linear_weight_agreement=active_policy.signal_strength.linear_weight_agreement,
            linear_weight_completeness=active_policy.signal_strength.linear_weight_completeness,
            hold_ceiling=active_policy.signal_strength.hold_ceiling,
            min_threshold=active_policy.signal_strength.min_signal_strength_threshold,
            catalyst_confidence_boost=active_policy.dip_buy.catalyst_confidence_boost,
            catalyst_confidence_ceiling=active_policy.dip_buy.catalyst_confidence_ceiling,
        )
        res = calculate_signal_strength(
            composite_score=composite_score,
            pillar_scores=pillar_scores,
            data_completeness=data_completeness,
            decision=decision,
            weights=weights,
            method=CombinationMethod(active_policy.signal_strength.combination_method),
            dispersion_metric=DispersionMetric(active_policy.signal_strength.dispersion_metric),
            is_certified_dip_buy=is_certified_dip_buy,
        )
        meta = {
            "composite": float(composite_score) if composite_score is not None else 50.0,
            "distance": res.score_decisiveness,
            "valid_pillars_count": res.valid_pillars_count,
            "agreement": res.pillar_agreement,
            "dispersion": res.pillar_dispersion,
            "completeness": res.data_completeness,
            "edge_strength": res.metrics.get("edge_strength", 0.0),
            "raw_confidence": res.raw_signal_strength,
            "final_confidence": res.signal_strength,
            "signal_strength": res.signal_strength,
            "is_calibrated_probability": res.is_calibrated_probability,
            "label": res.label,
            "calibration_status": res.calibration_status,
            "is_certified_dip_buy": is_certified_dip_buy,
        }
        return res.signal_strength, meta

    @classmethod
    def evaluate(
        cls,
        candidate_decision: str,
        composite_score: Optional[float],
        pillar_scores: Optional[Dict[str, Optional[float]]],
        data_completeness: float,
        policy: Optional[TradingPolicy] = None,
        is_certified_dip_buy: bool = False,
    ) -> GateResult:
        active_policy = policy or get_default_policy()
        min_thresh = (
            active_policy.dip_buy.min_confidence_threshold
            if is_certified_dip_buy
            else active_policy.trade_quality.min_confidence_threshold
        )

        conf, meta = cls.calculate_confidence(
            composite_score=composite_score,
            pillar_scores=pillar_scores,
            data_completeness=data_completeness,
            decision=candidate_decision,
            policy=active_policy,
            is_certified_dip_buy=is_certified_dip_buy,
        )

        metrics = dict(meta)
        metrics["confidence_threshold"] = min_thresh
        metrics["is_certified_dip_buy"] = is_certified_dip_buy

        # Neutral HOLD is always acceptable
        if candidate_decision.upper() == "HOLD":
            return GateResult(
                gate_name="5. Gated Confidence",
                passed=True,
                status=GateStatus.PASS,
                message=f"Neutral HOLD confidence calibrated to {conf:.2f}.",
                metrics=metrics,
            )

        # Missing-data policy: < 3 available pillars rejects directional setup
        if meta["valid_pillars_count"] < 3:
            return GateResult(
                gate_name="5. Gated Confidence",
                passed=False,
                status=GateStatus.FAIL,
                reason_code="DATA_COVERAGE_INADEQUATE",
                message=(
                    f"Inadequate data coverage: Only {meta['valid_pillars_count']}/5 pillars available "
                    f"(minimum 3 required for directional conviction)."
                ),
                metrics=metrics,
            )

        # Empirical threshold verification
        if conf < min_thresh:
            return GateResult(
                gate_name="5. Gated Confidence",
                passed=False,
                status=GateStatus.FAIL,
                reason_code="CONFIDENCE_TOO_LOW",
                message=f"Gated confidence {conf:.2f} < empirically justified threshold {min_thresh:.2f}.",
                metrics=metrics,
            )

        return GateResult(
            gate_name="5. Gated Confidence",
            passed=True,
            status=GateStatus.PASS,
            message=f"Gated confidence {conf:.2f} satisfies empirical threshold {min_thresh:.2f}.",
            metrics=metrics,
        )


# =============================================================================
# UNIFIED COMPLIANCE POLICY ENGINE
# =============================================================================
@dataclass
class ComplianceReport:
    """Consolidated compliance report across all 5 gates."""
    all_passed: bool
    final_decision: str
    veto_gate: Optional[str]
    no_trade_reason: Optional[str]
    gate_results: List[GateResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "all_passed": self.all_passed,
            "final_decision": self.final_decision,
            "veto_gate": self.veto_gate,
            "no_trade_reason": self.no_trade_reason,
            "gate_results": [g.to_dict() for g in self.gate_results],
        }


class CompliancePolicyEngine:
    """
    Unified Orchestrator for the Five Deterministic Compliance Gates.
    Evaluates gates in strict hierarchical order.
    """

    def __init__(self, policy: Optional[TradingPolicy] = None):
        self.policy = policy or load_policy_config()

    def evaluate_setup(
        self,
        candidate_decision: str,
        is_holding: bool,
        current_price: float,
        stop_loss: Optional[float],
        target_price: Optional[float],
        atr: float,
        composite_score: Optional[float],
        pillar_scores: Optional[Dict[str, Optional[float]]] = None,
        data_completeness: float = 1.0,
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
    ) -> ComplianceReport:
        direction = TradeDirection.LONG if candidate_decision.upper() == "BUY" else TradeDirection.SHORT

        # Gate 1: BUY Floor
        g1 = BuyFloorGate.evaluate(
            candidate_decision=candidate_decision,
            composite_score=composite_score,
            catalyst_path=catalyst_path,
            rsi14=rsi14,
            drawdown_5d_pct=drawdown_5d_pct,
            valuation_history_score=valuation_history_score,
            is_value_trap=is_value_trap,
            policy=self.policy,
        )

        # Gate 2: Reward-to-Risk (only if candidate has geometry and is fresh entry)
        if candidate_decision.upper() in ("BUY", "SELL") and not is_holding and stop_loss and target_price:
            g2 = RewardToRiskGate.evaluate(
                direction=direction,
                entry_price=current_price,
                stop_loss=stop_loss,
                target_price=target_price,
                composite_score=composite_score,
                policy=self.policy,
            )
        else:
            g2 = GateResult(
                gate_name="2. Reward-to-Risk",
                passed=True,
                status=GateStatus.NOT_APPLICABLE,
                message="Reward-to-risk gate not applicable to holding exit or neutral setup.",
            )

        # Gate 3: Volatility, Events & Universal Liquidity
        if candidate_decision.upper() in ("BUY", "SELL") and not is_holding and stop_loss:
            g3 = VolatilityEventLiquidityGate.evaluate(
                direction=direction,
                price=current_price,
                stop_loss=stop_loss,
                atr=atr,
                days_to_earnings=days_to_earnings,
                is_event_risk_blackout=is_event_risk_blackout,
                days_to_ex_dividend=days_to_ex_dividend,
                iv_pct=iv_pct,
                rvol_20d=rvol_20d,
                avg_dollar_volume_20d=avg_dollar_volume_20d,
                intended_order_value=intended_order_value,
                policy=self.policy,
            )
        else:
            g3 = GateResult(
                gate_name="3. Volatility, Events & Liquidity",
                passed=True,
                status=GateStatus.NOT_APPLICABLE,
                message="Event and liquidity gates not applicable.",
            )

        # Gate 4: SELL and Exit Symmetry
        g4 = SellExitSymmetryGate.evaluate(
            candidate_decision=candidate_decision,
            is_holding=is_holding,
            current_price=current_price,
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

        # Gate 5: Gated Confidence
        is_certified_dip_buy = bool(g1.metrics.get("dip_certified", False))
        g5 = GatedConfidenceGate.evaluate(
            candidate_decision=candidate_decision,
            composite_score=composite_score,
            pillar_scores=pillar_scores,
            data_completeness=data_completeness,
            policy=self.policy,
            is_certified_dip_buy=is_certified_dip_buy,
        )

        all_gates = [g1, g2, g3, g4, g5]
        failed_gates = [g for g in all_gates if not g.passed]

        if not failed_gates:
            return ComplianceReport(
                all_passed=True,
                final_decision=candidate_decision,
                veto_gate=None,
                no_trade_reason=None,
                gate_results=all_gates,
            )

        first_failed = failed_gates[0]
        final_decision = "HOLD"
        no_trade_reason = first_failed.reason_code or "INSUFFICIENT_EVIDENCE"

        return ComplianceReport(
            all_passed=False,
            final_decision=final_decision,
            veto_gate=first_failed.gate_name,
            no_trade_reason=no_trade_reason,
            gate_results=all_gates,
        )
