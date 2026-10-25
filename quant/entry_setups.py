"""
Distinct Trend Pullback Candidate Setup Specification (Step 3).

Unlike adverse-news catalyst dip-buys which exploit temporary panic dislocations
in oversold territory (drop 5-15%, RSI 30-45, RVOL >= 1.2x), this module evaluates
an orderly consolidation or moving average retest within an established longer-term
uptrend without requiring any external news catalyst.

Defines configurable rules across 5 discrete condition areas:
1. Intact longer-term trend (composite floor, trend pillar, EMA50, EMA200)
2. Pullback area (anchor EMA distance in ATR, drop %, 20d range proximity)
3. Stabilization or reclaim (RSI cooling band, anchor floor hold, reward/risk)
4. Volume & relative strength (volume dry-up / normal RVOL, sector RS, market alpha)
5. Exclusions (maximum upward extension, earnings blackout, volatility, value trap)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from quant.policy_config import PullbackConfig, TradingPolicy, get_default_policy


def _clean_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    """Safely converts numeric or formatted string inputs to float."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    try:
        s = str(val).strip().replace("%", "").replace("$", "").replace(",", "")
        return float(s)
    except (ValueError, TypeError):
        return default


@dataclass
class PullbackSetupResult:
    """
    Deterministic result of a trend pullback candidate setup evaluation.
    Records exactly which conditions passed and failed for complete auditability.
    """
    qualified: bool = False
    candidate_action: str = "HOLD"
    candidate_type: str = "NO_SETUP"
    conditions_met: Dict[str, bool] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes result into dictionary format."""
        return {
            "qualified": self.qualified,
            "candidate_action": self.candidate_action,
            "candidate_type": self.candidate_type,
            "conditions_met": dict(self.conditions_met),
            "evidence": dict(self.evidence),
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
        }


def evaluate_pullback_setup(
    technical_data: Optional[Dict[str, Any]] = None,
    deterministic_scores: Optional[Dict[str, Any]] = None,
    policy: Optional[TradingPolicy] = None,
    # Explicit keyword parameters to supplement or override dictionary lookups
    current_price: Optional[float] = None,
    composite_score: Optional[float] = None,
    trend_score: Optional[float] = None,
    ema20: Optional[float] = None,
    ema50: Optional[float] = None,
    ema200: Optional[float] = None,
    atr: Optional[float] = None,
    rsi14: Optional[float] = None,
    pullback_drop_pct: Optional[float] = None,
    change_5d_pct: Optional[float] = None,
    drawdown_5d_pct: Optional[float] = None,
    high_20d: Optional[float] = None,
    low_20d: Optional[float] = None,
    rvol_20d: Optional[float] = None,
    sector_relative_score: Optional[float] = None,
    market_alpha_score: Optional[float] = None,
    reward_risk_ratio: Optional[float] = None,
    days_to_earnings: Optional[int] = None,
    vol_factor: Optional[float] = None,
    is_value_trap: bool = False,
) -> PullbackSetupResult:
    """
    Evaluates whether the asset qualifies as a normal trend pullback candidate.

    Returns a PullbackSetupResult containing `qualified`, `candidate_action`,
    `candidate_type`, `conditions_met`, `evidence`, and detailed `reasons`.
    """
    pol = policy or get_default_policy()
    cfg: PullbackConfig = getattr(pol, "pullback", PullbackConfig())

    tech = technical_data or {}
    scores = deterministic_scores or {}

    # Extract metrics prioritizing explicit parameters over dict lookups
    price = current_price if current_price is not None else _clean_float(tech.get("current_price"))
    comp_sc = composite_score if composite_score is not None else _clean_float(
        scores.get("composite_quantitative_score") or scores.get("composite") or tech.get("composite_score")
    )
    tr_sc = trend_score if trend_score is not None else _clean_float(
        scores.get("trend_score") or scores.get("trend") or tech.get("trend_score")
    )
    e20 = ema20 if ema20 is not None else _clean_float(tech.get("ema20"))
    e50 = ema50 if ema50 is not None else _clean_float(tech.get("ema50"))
    e200 = ema200 if ema200 is not None else _clean_float(tech.get("ema200"))
    atr_val = atr if atr is not None else _clean_float(tech.get("atr"))
    rsi = rsi14 if rsi14 is not None else _clean_float(tech.get("rsi14"))
    h20 = high_20d if high_20d is not None else _clean_float(tech.get("high_20d"))
    l20 = low_20d if low_20d is not None else _clean_float(tech.get("low_20d"))
    rvol = rvol_20d if rvol_20d is not None else _clean_float(tech.get("rvol_20d"))

    # Relative strength metrics
    sec_sc = sector_relative_score if sector_relative_score is not None else _clean_float(
        scores.get("sector_relative_score") or scores.get("sector") or tech.get("sector_relative_score")
    )
    alpha_sc = market_alpha_score if market_alpha_score is not None else _clean_float(
        scores.get("market_alpha_score") or scores.get("alpha") or tech.get("market_alpha_score")
    )
    rr = reward_risk_ratio if reward_risk_ratio is not None else _clean_float(
        tech.get("reward_risk_ratio") or tech.get("rr")
    )

    # Event, earnings and volatility metrics
    earnings_days = days_to_earnings if days_to_earnings is not None else tech.get("days_to_earnings")
    if earnings_days is not None:
        try:
            earnings_days = int(earnings_days)
        except (ValueError, TypeError):
            earnings_days = None

    vf = vol_factor if vol_factor is not None else _clean_float(
        tech.get("hv_20d") or tech.get("vol_factor")
    )

    # Derive pullback drop percentage
    drop_val = None
    if pullback_drop_pct is not None:
        drop_val = abs(float(pullback_drop_pct))
    elif drawdown_5d_pct is not None:
        drop_val = abs(float(drawdown_5d_pct))
    elif change_5d_pct is not None:
        c5 = _clean_float(change_5d_pct)
        if c5 is not None and c5 < 0:
            drop_val = abs(c5)
    elif tech.get("change_5d_pct") is not None:
        c5 = _clean_float(tech.get("change_5d_pct"))
        if c5 is not None and c5 < 0:
            drop_val = abs(c5)
    elif h20 is not None and price is not None and h20 > price and h20 > 0:
        drop_val = round(((h20 - price) / h20) * 100.0, 2)
    else:
        drop_val = 0.0

    # Derive range proximity: (price - low_20d) / (high_20d - low_20d)
    range_prox = 0.50
    if h20 is not None and l20 is not None and h20 > l20 and price is not None:
        range_prox = round((price - l20) / (h20 - l20), 4)

    # Anchor EMA selection
    anchor_name = str(cfg.anchor_ema).lower()
    anchor_val = e50 if anchor_name == "ema50" else e20
    if anchor_val is None:
        anchor_val = e20 if e20 is not None else e50

    # Distance to anchor in ATR units
    dist_anchor_atr = None
    if price is not None and anchor_val is not None:
        if atr_val is not None and atr_val > 0.0:
            dist_anchor_atr = round(abs(price - anchor_val) / atr_val, 2)
        elif price > 0.0:
            # Fallback estimation using 2% standard ATR
            dist_anchor_atr = round(abs(price - anchor_val) / (price * 0.02), 2)

    # Upward extension from anchor in ATR
    extension_atr = None
    if price is not None and anchor_val is not None:
        if atr_val is not None and atr_val > 0.0:
            extension_atr = round((price - anchor_val) / atr_val, 2)
        elif price > 0.0:
            extension_atr = round((price - anchor_val) / (price * 0.02), 2)

    metrics_snapshot = {
        "current_price": price,
        "composite_score": comp_sc,
        "trend_score": tr_sc,
        "ema20": e20,
        "ema50": e50,
        "ema200": e200,
        "anchor_ema": anchor_name,
        "anchor_value": anchor_val,
        "distance_to_anchor_atr": dist_anchor_atr,
        "pullback_drop_pct": drop_val,
        "range_proximity": range_prox,
        "rsi14": rsi,
        "rvol_20d": rvol,
        "sector_relative_score": sec_sc,
        "market_alpha_score": alpha_sc,
        "reward_risk_ratio": rr,
        "days_to_earnings": earnings_days,
        "vol_factor": vf,
        "is_value_trap": is_value_trap,
    }

    reasons: List[str] = []
    conditions_met: Dict[str, bool] = {
        "trend_intact": True,
        "pullback_area_tested": True,
        "stabilization_confirmed": True,
        "volume_and_relative_strength": True,
        "exclusions_passed": True,
    }

    if not cfg.enabled:
        return PullbackSetupResult(
            qualified=False,
            candidate_action="HOLD",
            candidate_type="NO_SETUP",
            conditions_met={k: False for k in conditions_met},
            evidence={"trigger": "pullback_rule_disabled"},
            reasons=["Pullback candidate rule is disabled in trading policy."],
            metrics=metrics_snapshot,
        )

    # -------------------------------------------------------------------------
    # 1. INTACT LONGER-TERM TREND EVALUATION
    # -------------------------------------------------------------------------
    trend_reasons: List[str] = []
    if comp_sc is None or comp_sc < cfg.composite_minimum:
        trend_reasons.append(
            f"Composite score {comp_sc} below minimum {cfg.composite_minimum}"
        )
    if tr_sc is not None and tr_sc < cfg.min_trend_score:
        trend_reasons.append(
            f"Trend pillar score {tr_sc} below minimum {cfg.min_trend_score}"
        )
    if cfg.require_above_ema50:
        if price is None:
            trend_reasons.append("Current price unavailable for EMA50 trend evaluation")
        elif e50 is not None and price < e50:
            trend_reasons.append(
                f"Price (${price:.2f}) trading below EMA50 (${e50:.2f}) - intermediate trend broken"
            )
    if cfg.require_ema50_above_ema200:
        if e50 is not None and e200 is not None and e200 > 0.0 and e50 < e200:
            trend_reasons.append(
                f"EMA50 (${e50:.2f}) trading below EMA200 (${e200:.2f}) - structural macro downtrend"
            )

    if trend_reasons:
        conditions_met["trend_intact"] = False
        reasons.extend(trend_reasons)

    # -------------------------------------------------------------------------
    # 2. PULLBACK AREA DEFINITION (Moving Average & Price Geometry)
    # -------------------------------------------------------------------------
    area_reasons: List[str] = []
    if anchor_val is None and (cfg.max_distance_to_anchor_atr > 0):
        # Only fail if no moving average anchor could be resolved
        area_reasons.append(f"Anchor moving average ({cfg.anchor_ema}) not available")
    elif dist_anchor_atr is not None and dist_anchor_atr > cfg.max_distance_to_anchor_atr:
        area_reasons.append(
            f"Distance to {cfg.anchor_ema} ({dist_anchor_atr:.2f} ATR) exceeds maximum {cfg.max_distance_to_anchor_atr:.2f} ATR"
        )

    if drop_val is not None:
        if drop_val < cfg.min_pullback_drop_pct:
            area_reasons.append(
                f"Pullback drop {drop_val:.2f}% below minimum {cfg.min_pullback_drop_pct:.2f}% (no meaningful pullback)"
            )
        elif drop_val > cfg.max_pullback_drop_pct:
            area_reasons.append(
                f"Pullback drop {drop_val:.2f}% exceeds maximum {cfg.max_pullback_drop_pct:.2f}% (excessive drop / breakdown)"
            )

    if range_prox < cfg.min_range_proximity:
        area_reasons.append(
            f"Range proximity {range_prox:.2f} below support threshold {cfg.min_range_proximity:.2f} (near 20d breakdown)"
        )
    elif range_prox > cfg.max_range_proximity:
        area_reasons.append(
            f"Range proximity {range_prox:.2f} above pullback threshold {cfg.max_range_proximity:.2f} (extended near 20d highs)"
        )

    if area_reasons:
        conditions_met["pullback_area_tested"] = False
        reasons.extend(area_reasons)

    # -------------------------------------------------------------------------
    # 3. STABILIZATION / RECLAIM CONFIRMATION
    # -------------------------------------------------------------------------
    stab_reasons: List[str] = []
    if rsi is None:
        stab_reasons.append("RSI(14) unavailable for pullback stabilization check")
    else:
        if rsi < cfg.rsi_min:
            stab_reasons.append(
                f"RSI {rsi:.1f} below minimum {cfg.rsi_min:.1f} (momentum breakdown / severe oversold)"
            )
        elif rsi > cfg.rsi_max:
            stab_reasons.append(
                f"RSI {rsi:.1f} exceeds maximum {cfg.rsi_max:.1f} (not in orderly cooling zone)"
            )

    if cfg.require_stabilization and price is not None and anchor_val is not None:
        buffer = (0.50 * atr_val) if (atr_val is not None and atr_val > 0.0) else (0.015 * price)
        anchor_floor = anchor_val - buffer
        if price < anchor_floor:
            stab_reasons.append(
                f"Price (${price:.2f}) failed stabilization below {cfg.anchor_ema} floor (${anchor_floor:.2f})"
            )

    if rr is not None and rr < cfg.min_reward_risk:
        stab_reasons.append(
            f"Reward/risk ratio {rr:.2f} below minimum {cfg.min_reward_risk:.2f}"
        )

    if stab_reasons:
        conditions_met["stabilization_confirmed"] = False
        reasons.extend(stab_reasons)

    # -------------------------------------------------------------------------
    # 4. VOLUME & RELATIVE-STRENGTH CONDITIONS
    # -------------------------------------------------------------------------
    vol_rs_reasons: List[str] = []
    if rvol is not None and rvol > cfg.max_pullback_rvol:
        vol_rs_reasons.append(
            f"Pullback RVOL {rvol:.2f} exceeds {cfg.max_pullback_rvol:.2f} (abnormal distribution volume)"
        )
    if sec_sc is not None and sec_sc < cfg.min_sector_relative_score:
        vol_rs_reasons.append(
            f"Sector relative score {sec_sc:.1f} below minimum {cfg.min_sector_relative_score:.1f}"
        )
    if alpha_sc is not None and alpha_sc < cfg.min_market_alpha_score:
        vol_rs_reasons.append(
            f"Market alpha score {alpha_sc:.1f} below minimum {cfg.min_market_alpha_score:.1f}"
        )

    if vol_rs_reasons:
        conditions_met["volume_and_relative_strength"] = False
        reasons.extend(vol_rs_reasons)

    # -------------------------------------------------------------------------
    # 5. MAXIMUM EXTENSION, EVENT & VOLATILITY EXCLUSIONS
    # -------------------------------------------------------------------------
    excl_reasons: List[str] = []
    if extension_atr is not None and extension_atr > cfg.max_extension_atr:
        excl_reasons.append(
            f"Extension above {cfg.anchor_ema} ({extension_atr:.2f} ATR) exceeds maximum {cfg.max_extension_atr:.2f} ATR"
        )

    if cfg.exclude_earnings_blackout and earnings_days is not None:
        blackout_days = getattr(pol.events, "earnings_blackout_sessions", 3) if hasattr(pol, "events") else 3
        if 0 <= earnings_days <= blackout_days:
            excl_reasons.append(
                f"Scheduled earnings in {earnings_days} day(s) inside blackout window ({blackout_days}d)"
            )

    if vf is not None and vf > cfg.max_vol_factor:
        excl_reasons.append(
            f"Volatility factor {vf:.2f} exceeds ceiling {cfg.max_vol_factor:.2f}"
        )

    if cfg.forbid_value_trap and is_value_trap:
        excl_reasons.append("Security flagged as value trap with deteriorating fundamentals")

    if excl_reasons:
        conditions_met["exclusions_passed"] = False
        reasons.extend(excl_reasons)

    # -------------------------------------------------------------------------
    # FINAL SETUP QUALIFICATION
    # -------------------------------------------------------------------------
    is_qualified = all(conditions_met.values())

    if is_qualified:
        candidate_action = "BUY"
        candidate_type = "PULLBACK_BUY"
        evidence_trigger = "pullback_in_uptrend"
        final_reasons = [
            f"Orderly trend pullback qualified: intact bull trend, {cfg.anchor_ema} tested, "
            f"stabilization confirmed (RSI {rsi}), normal volume (RVOL {rvol or 1.0}), exclusions cleared."
        ]
    else:
        candidate_action = "HOLD"
        candidate_type = "NO_SETUP"
        evidence_trigger = "pullback_criteria_not_met"
        final_reasons = reasons

    evidence = {
        "trigger": evidence_trigger,
        "setup_type": "PULLBACK_BUY",
        "composite_score": comp_sc,
        "rsi14": rsi,
        "reward_risk_ratio": rr,
        "pullback_drop_pct": drop_val,
        "anchor_ema": cfg.anchor_ema,
        "distance_to_anchor_atr": dist_anchor_atr,
        "conditions_met": conditions_met,
        "metrics": metrics_snapshot,
        "reasons": final_reasons,
    }

    return PullbackSetupResult(
        qualified=is_qualified,
        candidate_action=candidate_action,
        candidate_type=candidate_type,
        conditions_met=conditions_met,
        evidence=evidence,
        reasons=final_reasons,
        metrics=metrics_snapshot,
    )
