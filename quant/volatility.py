"""Deterministic Volatility Dampening and Position Sizing Functions."""

import math
from typing import Optional
from quant.normalization import clamp


def compute_vol_factor(atr: Optional[float], price: Optional[float]) -> float:
    """
    Returns a deterministic volatility factor based on daily ATR as a percentage of price:
        ATR% = (ATR14 / Price) * 100.0

    Bands:
      <= 1.5%  -> 1.00  (Calm)
      <= 2.5%  -> 0.95  (Normal)
      <= 4.0%  -> 0.90  (Elevated)
      >  4.0%  -> 0.85  (Extreme Volatility)
    """
    if atr is None or price is None:
        return 1.0

    try:
        f_atr = float(atr)
        f_price = float(price)
        if math.isnan(f_atr) or math.isnan(f_price) or math.isinf(f_atr) or math.isinf(f_price):
            return 1.0
        if f_price <= 0.0 or f_atr < 0.0:
            return 1.0
        atr_pct = (f_atr / f_price) * 100.0
    except (TypeError, ValueError, ZeroDivisionError):
        return 1.0

    if atr_pct <= 1.5:
        return 1.0
    elif atr_pct <= 2.5:
        return 0.95
    elif atr_pct <= 4.0:
        return 0.90
    return 0.85


def apply_volatility_dampener(adjusted_score: float, vol_factor: float) -> float:
    """
    Applies a centred volatility dampening transformation around the neutral midpoint (50.0):
        Q_final = 50.0 + (Q_adjusted - 50.0) * vol_factor

    Constraints:
    - Neutral Invariance: If Q_adjusted == 50.0, Q_final == 50.0 for all vol_factor values.
    - Symmetrically reduces conviction for both long (>50) and short (<50) signals.
    - Strictly bounded in [0.0, 100.0].
    """
    if adjusted_score is None:
        return 50.0
    try:
        score_val = float(adjusted_score)
        vf_val = float(vol_factor) if vol_factor is not None else 1.0
    except (TypeError, ValueError):
        return 50.0

    if math.isnan(score_val) or math.isinf(score_val):
        return 50.0
    if math.isnan(vf_val) or math.isinf(vf_val):
        vf_val = 1.0

    if not (0.0 <= score_val <= 100.0):
        raise ValueError(f"Score must be between 0 and 100, got {adjusted_score}")
    if not (0.0 <= vf_val <= 1.0):
        raise ValueError(f"Volatility factor must be between 0 and 1, got {vol_factor}")

    score = 50.0 + (score_val - 50.0) * vf_val
    return clamp(score, 0.0, 100.0)


def compute_volatility_adjusted_position_size(
    base_weight: float,
    atr_pct: float,
    target_atr_pct: float = 2.5,
    max_weight_cap: float = 0.15,
) -> float:
    """
    Computes volatility-based position sizing without distorting the directional score:
        w_i = base_weight * min(1.0, target_atr_pct / atr_pct)
    Preserves pure directional ranking while reducing capital allocation to high-ATR names.
    """
    if atr_pct <= 0.0:
        return min(base_weight, max_weight_cap)

    scaling = min(1.0, target_atr_pct / atr_pct)
    adjusted_weight = base_weight * scaling
    return min(adjusted_weight, max_weight_cap)
