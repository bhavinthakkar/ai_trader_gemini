"""Pure factor extraction and calculation functions."""

import math
from typing import Any, Dict, List, Optional, Tuple


def clean_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    """Safely parses string percentages (e.g. '+4.64%', 'N/A') or numerics to float."""
    if val is None:
        return default
    if isinstance(val, bool):
        return default
    if isinstance(val, (int, float)):
        if math.isnan(val) or math.isinf(val):
            return default
        return float(val)
    try:
        s = str(val).replace("%", "").replace("+", "").replace("$", "").replace(",", "").strip()
        if s.upper() in {"N/A", "NONE", "NAN", "NULL", ""}:
            return default
        return float(s)
    except Exception:
        return default


def compute_ema_alignment(
    price: float,
    ema20: Optional[float],
    ema50: Optional[float],
    ema200: Optional[float] = None,
    atr14: Optional[float] = None,
    ema20_prev5: Optional[float] = None,
) -> Tuple[float, str]:
    """
    Computes hierarchical EMA structural alignment and slope score in [0.0, 100.0].

    Hierarchical Alignment:
    - Bullish Stack: Price > EMA20 > EMA50 > EMA200 (Base = 85.0)
    - Golden Cross: Price > EMA20 > EMA50, EMA50 <= EMA200 (Base = 65.0)
    - Mixed / Range: Price oscillating between EMA20 and EMA50 (Base = 50.0)
    - Breakdown: Price < EMA20 or EMA20 < EMA50 (Base = 35.0)
    - Bearish Stack: Price < EMA20 < EMA50 < EMA200 (Base = 15.0)

    Slope Adjustment:
    - Upward EMA20 slope normalized by ATR: adds up to +15.0 / -15.0 pts.
    """
    if price <= 0.0 or ema20 is None or ema20 <= 0.0 or ema50 is None or ema50 <= 0.0:
        return 50.0, "insufficient_data"

    has_ema200 = ema200 is not None and ema200 > 0.0

    if has_ema200:
        if price > ema20 > ema50 > ema200:
            base = 85.0
            desc = "bullish_stack_above_ema200"
        elif price > ema20 > ema50:
            base = 65.0
            desc = "golden_cross_below_or_near_ema200"
        elif price < ema20 < ema50 < ema200:
            base = 15.0
            desc = "bearish_stack_below_ema200"
        elif price < ema20 and ema20 < ema50:
            base = 35.0
            desc = "bearish_alignment"
        else:
            base = 50.0
            desc = "mixed_consolidation"
    else:
        if price > ema20 > ema50:
            base = 75.0
            desc = "bullish_alignment_no_ema200"
        elif price < ema20 < ema50:
            base = 25.0
            desc = "bearish_alignment_no_ema200"
        elif price > ema20:
            base = 60.0
            desc = "above_ema20"
        else:
            base = 40.0
            desc = "below_ema20"

    # Slope adjustment: (EMA20_t - EMA20_t-5) / (5 * ATR14)
    slope_adj = 0.0
    if ema20_prev5 is not None and atr14 is not None and atr14 > 0.0:
        raw_slope = (ema20 - ema20_prev5) / (5.0 * atr14)
        slope_adj = max(-15.0, min(15.0, raw_slope * 15.0))

    score = max(0.0, min(100.0, base + slope_adj))
    return score, desc


def compute_rsi_velocity(
    rsi14: Optional[float],
    rsi14_prev5: Optional[float] = None,
) -> Tuple[float, Optional[float], str]:
    """
    Computes RSI-14 momentum and 5-session rate-of-change velocity in [0.0, 100.0].

    Velocity:
        RSI_Velocity_5 = (RSI14_t - RSI14_t-5) / 5.0  (pts / session)
    """
    if rsi14 is None or rsi14 < 0.0:
        return 50.0, None, "missing_rsi"

    velocity = None
    if rsi14_prev5 is not None:
        velocity = (rsi14 - rsi14_prev5) / 5.0

    # 1. Bullish Momentum Acceleration
    if 50.0 <= rsi14 <= 68.0:
        base = 65.0 + ((rsi14 - 50.0) / 18.0) * 15.0
        vel_bonus = max(0.0, min(15.0, velocity * 3.0)) if velocity is not None else 0.0
        return max(0.0, min(100.0, base + vel_bonus)), velocity, "bullish_momentum_expansion"

    # 2. Mean-Reversion Oversold Bounce
    if rsi14 < 32.0:
        if velocity is not None and velocity >= 1.5:
            # Positive inflection out of oversold
            score = 45.0 + (velocity * 5.0)
            return max(0.0, min(100.0, score)), velocity, "oversold_positive_inflection"
        # Severe unreclaimed washout
        score = max(5.0, rsi14 * 0.8)
        return score, velocity, "severe_oversold_washout"

    # 3. Overbought Exhaustion / Blow-Off
    if rsi14 > 75.0:
        score = max(20.0, 85.0 - (rsi14 - 75.0) * 4.0)
        return score, velocity, "overbought_exhaustion"

    # 4. Neutral / Drift
    if 40.0 <= rsi14 < 50.0:
        score = 50.0 + ((rsi14 - 45.0) * 1.5)
        vel_adj = max(-10.0, min(10.0, velocity * 2.0)) if velocity is not None else 0.0
        return max(0.0, min(100.0, score + vel_adj)), velocity, "neutral_accumulation"

    return 50.0, velocity, "neutral_range"


GICS_SECTOR_ETF_MAP: Dict[str, str] = {
    "Technology": "XLK",
    "Information Technology": "XLK",
    "Communication Services": "XLC",
    "Consumer Cyclical": "XLY",
    "Consumer Discretionary": "XLY",
    "Consumer Defensive": "XLP",
    "Consumer Staples": "XLP",
    "Financial Services": "XLF",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Health Care": "XLV",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}


def compute_sector_excess(
    stock_ret_5d: Optional[float],
    sector_ret_5d: Optional[float],
    stock_ret_20d: Optional[float] = None,
    sector_ret_20d: Optional[float] = None,
    stock_ret_60d: Optional[float] = None,
    sector_ret_60d: Optional[float] = None,
) -> Optional[float]:
    """
    Computes multi-horizon sector-relative excess return:
        Sector_Excess = 0.20 * Delta_5d + 0.50 * Delta_20d + 0.30 * Delta_60d
    If multi-horizon data is unavailable, falls back gracefully to available horizons.
    Returns composite excess return in decimal format (e.g. +0.035 = +3.5%).
    """
    excess_components = []
    weights = []

    # 20-session core swing horizon (50% weight)
    if stock_ret_20d is not None and sector_ret_20d is not None:
        excess_components.append((stock_ret_20d - sector_ret_20d) * 0.50)
        weights.append(0.50)

    # 60-session trend context (30% weight)
    if stock_ret_60d is not None and sector_ret_60d is not None:
        excess_components.append((stock_ret_60d - sector_ret_60d) * 0.30)
        weights.append(0.30)

    # 5-session tactical horizon (20% weight)
    if stock_ret_5d is not None and sector_ret_5d is not None:
        excess_components.append((stock_ret_5d - sector_ret_5d) * 0.20)
        weights.append(0.20)

    if not weights:
        return None

    return sum(excess_components) / sum(weights)


def compute_market_alpha(
    stock_ret_20d: Optional[float],
    spy_ret_20d: Optional[float],
    beta_60d: Optional[float] = 1.0,
    rf_20d: float = 0.0016,  # ~2.0% annual / 12
    stock_ret_5d: Optional[float] = None,
    spy_ret_5d: Optional[float] = None,
    idiosyncratic_vol_60d: Optional[float] = None,
) -> Optional[float]:
    """
    Computes Market Alpha metric combining Jensen's Alpha and multi-session SPY excess return.
        alpha_20d = R_i,20d - [rf + beta * (R_spy,20d - rf)]
        excess_20d = R_i,20d - R_spy,20d
        excess_5d = R_i,5d - R_spy,5d
    Returns the composite alpha in decimal format.
    """
    if stock_ret_20d is None or spy_ret_20d is None:
        # Fallback to 5-day excess if 20-day is absent
        if stock_ret_5d is not None and spy_ret_5d is not None:
            return stock_ret_5d - spy_ret_5d
        return None

    beta = beta_60d if beta_60d is not None and beta_60d > 0.0 else 1.0
    jensen_alpha_20d = stock_ret_20d - (rf_20d + beta * (spy_ret_20d - rf_20d))
    excess_20d = stock_ret_20d - spy_ret_20d

    if stock_ret_5d is not None and spy_ret_5d is not None:
        excess_5d = stock_ret_5d - spy_ret_5d
        composite_alpha = 0.50 * jensen_alpha_20d + 0.35 * excess_20d + 0.15 * excess_5d
    else:
        composite_alpha = 0.60 * jensen_alpha_20d + 0.40 * excess_20d

    return composite_alpha


def compute_earnings_yield(eps_ttm: Optional[float], price: float) -> Optional[float]:
    """Computes Earnings Yield (E/P). Continuous across negative values without singularities."""
    clean_eps = clean_float(eps_ttm)
    clean_p = clean_float(price)
    if clean_eps is None or clean_p is None or clean_p <= 0.0:
        return None
    return clean_eps / clean_p


def compute_ebitda_yield(
    ebitda_ttm: Optional[float],
    enterprise_value: Optional[float],
    market_cap: Optional[float] = None,
) -> Optional[float]:
    """
    Computes EBITDA Yield (EBITDA / EV).
    Handles cash-heavy companies with EV <= 0 by falling back to Market Cap.
    """
    clean_ebitda = clean_float(ebitda_ttm)
    if clean_ebitda is None:
        return None

    clean_ev = clean_float(enterprise_value)
    clean_mc = clean_float(market_cap)

    denom = clean_ev
    if denom is None or denom <= 0.0:
        denom = clean_mc

    if denom is None or denom <= 0.0:
        return None

    return clean_ebitda / denom


def compute_fcf_yield(
    fcf_ttm: Optional[float],
    enterprise_value: Optional[float],
    market_cap: Optional[float] = None,
) -> Optional[float]:
    """Computes Free Cash Flow Yield (FCF / EV or FCF / Market Cap)."""
    clean_fcf = clean_float(fcf_ttm)
    if clean_fcf is None:
        return None

    clean_ev = clean_float(enterprise_value)
    clean_mc = clean_float(market_cap)

    denom = clean_ev if (clean_ev is not None and clean_ev > 0.0) else clean_mc
    if denom is None or denom <= 0.0:
        return None

    return clean_fcf / denom
