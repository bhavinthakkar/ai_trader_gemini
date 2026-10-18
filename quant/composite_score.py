"""Complete 7-Step Deterministic Quantitative Composite Pipeline."""

import math
from typing import Any, Dict, Optional, Tuple
from quant.schemas import (
    PillarResult,
    PillarStatus,
    FinancialQualityResult,
    QuantCompositeResult,
)
from quant.pillar_scores import (
    calculate_trend_pillar,
    calculate_sector_pillar,
    calculate_alpha_pillar,
    calculate_valuation_history_pillar,
    calculate_peer_valuation_pillar,
)
from quant.quality_adjustments import (
    evaluate_financial_quality,
    apply_quality_adjustment,
)
from quant.volatility import (
    compute_vol_factor,
    apply_volatility_dampener,
)
from quant.normalization import clamp


BASE_WEIGHTS: Dict[str, float] = {
    "trend": 0.25,
    "sector": 0.20,
    "alpha": 0.20,
    "valuation_history": 0.15,
    "peer_valuation": 0.20,
}

MIN_REQUIRED_PILLARS: int = 3


def compute_deterministic_composite(
    technical_data: Optional[Dict[str, Any]],
    gloomberb_payload: Optional[Dict[str, Any]],
    sector_bench: Optional[Dict[str, Any]] = None,
    use_score_penalty: bool = False,  # False = Design B (Hard Gate); True = Design A
) -> QuantCompositeResult:
    """
    Executes the 7-step deterministic quantitative scoring transformation:

    1. Validate required inputs:
       - Price must be positive.
       - A minimum of 3 of 5 pillars must be valid (measured or partial).
       - If conditions fail, aborts with INSUFFICIENT_EVIDENCE (composite = None).
    2. Calculate pillar scores:
       - 5 explicit 0-100 scores with input tracking and status.
    3. Calculate the weighted raw score:
       - Re-normalizes base weights dynamically across available pillars:
         W_A = sum(w_i for i in avail); Q_raw = sum(w_i * S_i) / W_A.
    4. Apply the documented quality adjustment:
       - Compounder bonus (+3.0 pts if financial quality >= 80).
       - Under Design A: subtracts -15.0 pts if value trap. Under Design B: pure.
    5. Clamp the score to 0–100.
    6. Apply centred volatility dampening:
       - Q_final = 50.0 + (Q_adjusted - 50.0) * vol_factor.
    7. Pass the result to the next decision stage:
       - Under Design B, value traps set is_buy_eligible = False (Hard Gate veto).
    """
    tech = technical_data or {}
    gloom = gloomberb_payload or {}
    sec = sector_bench or gloom.get("sector_benchmark", {})

    # --- Step 1 & 2: Calculate Pillar Scores & Financial Quality ---
    p_trend = calculate_trend_pillar(tech)
    p_sector = calculate_sector_pillar(tech, sec)
    p_alpha = calculate_alpha_pillar(tech)
    p_val_hist = calculate_valuation_history_pillar(gloom)
    p_peer_val = calculate_peer_valuation_pillar(gloom)
    p_fq = evaluate_financial_quality(gloom, use_score_penalty=use_score_penalty)

    pillars = {
        "trend": p_trend,
        "sector": p_sector,
        "alpha": p_alpha,
        "valuation_history": p_val_hist,
        "peer_valuation": p_peer_val,
    }

    # Available pillars (score is not None)
    avail_pillars = {k: p for k, p in pillars.items() if p.score is not None}
    avail_count = len(avail_pillars)
    total_count = len(pillars)

    # --- Step 1 (Validation Gate): Minimum Information Hurdle ---
    # Must have >= 3 available pillars and trend must be measured or partial
    price = tech.get("current_price")
    has_valid_price = price is not None and float(price) > 0.0

    if avail_count < MIN_REQUIRED_PILLARS or not has_valid_price or p_trend.score is None:
        return QuantCompositeResult(
            composite_quantitative_score=None,
            raw_composite=None,
            vol_factor=1.0,
            quant_signal="Unavailable Quant Signal (Insufficient pillar data: < 3 available)",
            is_tradeable=False,
            no_trade_reason="INSUFFICIENT_EVIDENCE",
            available_pillars_count=avail_count,
            total_pillars_count=total_count,
            pillars=pillars,
            financial_quality=p_fq,
            active_weights={},
        )

    # --- Step 3: Weighted Raw Score with Dynamic Weight Re-normalization ---
    avail_weight_sum = sum(BASE_WEIGHTS[k] for k in avail_pillars)
    active_weights = {k: round(BASE_WEIGHTS[k] / avail_weight_sum, 4) for k in avail_pillars}

    raw_composite = sum(active_weights[k] * avail_pillars[k].score for k in avail_pillars)

    # --- Step 4: Apply Documented Quality Adjustment ---
    adjusted_score = apply_quality_adjustment(raw_composite, p_fq, use_score_penalty=use_score_penalty)

    # --- Step 5: Clamp Score to 0-100 ---
    clamped_score = clamp(adjusted_score, 0.0, 100.0)

    # --- Step 6: Centred Volatility Dampening ---
    atr = tech.get("atr")
    vol_factor = compute_vol_factor(atr, price)
    final_score = round(apply_volatility_dampener(clamped_score, vol_factor), 2)
    raw_composite_out = round(raw_composite, 2)

    # --- Step 7: Classification & Hard Eligibility Rules ---
    if p_fq.is_value_trap and not use_score_penalty:
        # Design B: Hard Eligibility Gate blocks BUY
        is_tradeable = True
        no_trade_reason = "VALUE_TRAP"
        signal = "Value Trap Detected (BUY mechanically vetoed; HOLD/SELL only)"
    else:
        is_tradeable = True
        no_trade_reason = None
        if final_score >= 60.0:
            signal = "Strong Bullish Quant Signal (BUY candidate if fundamental RAG confirms)"
        elif final_score >= 45.0:
            signal = "Neutral / Mixed Quant Signal (HOLD candidate; BUY requires exceptional fundamental catalyst)"
        else:
            signal = "Bearish / Overvalued Quant Signal (SELL / Caution candidate)"

    return QuantCompositeResult(
        composite_quantitative_score=final_score,
        raw_composite=raw_composite_out,
        vol_factor=vol_factor,
        quant_signal=signal,
        is_tradeable=is_tradeable,
        no_trade_reason=no_trade_reason,
        available_pillars_count=avail_count,
        total_pillars_count=total_count,
        pillars=pillars,
        financial_quality=p_fq,
        active_weights=active_weights,
    )


def blend_quant_and_ml_scores(
    det_quant_score: Optional[float],
    ml_context: Optional[Dict[str, Any]],
    default_ml_weight: float = 0.20,
) -> Tuple[Optional[float], float, Optional[float]]:
    """
    Blends deterministic quant score and ML signal score enforcing the Step 3.5 contract.

    Guarantees:
      - If ml_context has model_status != 'IN_SCOPE' or in_scope is False: ml_weight is strictly 0.0, ml_score is None.
      - Decision composite is 100% pure deterministic quant score.
      - No artificial 50.0 score or accidental ML weighting is ever assigned.

    Returns:
        (decision_composite, ml_weight, ml_score)
    """
    if det_quant_score is None:
        return None, 0.0, None

    det_val = float(det_quant_score)
    if not isinstance(ml_context, dict):
        return round(det_val, 2), 0.0, None

    model_status = ml_context.get("model_status")
    is_in_scope = (
        model_status == "IN_SCOPE"
        or (
            bool(ml_context.get("in_scope", False))
            and model_status not in ("OUT_OF_SCOPE", "STALE_SCORE", "SCHEMA_MISMATCH", "FILE_NOT_FOUND", "LOOKAHEAD_VIOLATION")
        )
    )

    raw_ml = ml_context.get("ml_score")
    if raw_ml is None and is_in_scope:
        raw_ml = ml_context.get("score_0_100")

    if not is_in_scope or raw_ml is None:
        return round(det_val, 2), 0.0, None

    try:
        ml_score = float(raw_ml)
        if math.isnan(ml_score) or math.isinf(ml_score):
            return round(det_val, 2), 0.0, None
    except (ValueError, TypeError):
        return round(det_val, 2), 0.0, None

    configured_w = ml_context.get("ml_weight")
    ml_w = float(configured_w) if configured_w is not None and float(configured_w) > 0.0 else default_ml_weight
    ml_w = max(0.0, min(1.0, ml_w))

    composite = round(max(0.0, min(100.0, (1.0 - ml_w) * det_val + ml_w * ml_score)), 2)
    return composite, ml_w, ml_score

