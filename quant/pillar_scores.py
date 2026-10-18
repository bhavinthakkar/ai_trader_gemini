"""Individual Pillar Scoring Routines (Pillars 1 to 5)."""

from typing import Any, Dict, List, Optional
from quant.schemas import PillarResult, PillarStatus, InputTracking
from quant.factors import (
    clean_float,
    compute_ema_alignment,
    compute_rsi_velocity,
    compute_sector_excess,
    compute_market_alpha,
    compute_earnings_yield,
    compute_ebitda_yield,
    compute_fcf_yield,
    GICS_SECTOR_ETF_MAP,
)
from quant.normalization import (
    clamp,
    logistic_scale,
    normal_cdf_scale,
    cross_sectional_robust_zscore,
    time_series_percentile_score,
)


def track_input(name: str, value: Any, is_valid: Optional[bool] = None, timestamp: Optional[str] = None) -> InputTracking:
    valid = is_valid if is_valid is not None else (value is not None and str(value).upper() not in {"N/A", "NONE", "NAN", ""})
    return InputTracking(
        name=name,
        value=value if valid else None,
        available=bool(valid),
        freshness="fresh" if valid else "unavailable",
        timestamp=timestamp,
    )


def calculate_trend_pillar(technical_data: Optional[Dict[str, Any]]) -> PillarResult:
    """
    Evaluates Historical Trend Pillar (Base Weight: 25%):
    - Inputs: current_price, ema20, ema50, ema200, rsi14, change_5d_pct, atr
    - Combines EMA structural alignment (60%) and RSI momentum velocity (40%).
    """
    data = technical_data or {}
    price = clean_float(data.get("current_price"), 0.0)
    ema20 = clean_float(data.get("ema20"), None)
    ema50 = clean_float(data.get("ema50"), None)
    ema200 = clean_float(data.get("ema200"), None)
    rsi14 = clean_float(data.get("rsi14"), None)
    atr = clean_float(data.get("atr"), None)

    has_price = price is not None and price > 0.0
    has_ema20 = ema20 is not None and ema20 > 0.0
    has_ema50 = ema50 is not None and ema50 > 0.0
    has_rsi = rsi14 is not None and rsi14 >= 0.0

    tracking = {
        "current_price": track_input("current_price", price, has_price),
        "ema20": track_input("ema20", ema20, has_ema20),
        "ema50": track_input("ema50", ema50, has_ema50),
        "ema200": track_input("ema200", ema200, ema200 is not None),
        "rsi14": track_input("rsi14", rsi14, has_rsi),
        "atr": track_input("atr", atr, atr is not None),
    }

    if not has_price or (not has_ema20 and not has_rsi):
        return PillarResult(
            name="trend",
            score=None,
            status=PillarStatus.UNAVAILABLE,
            inputs=tracking,
            weight=0.25,
            description="Insufficient price and technical indicator data",
        )

    ema_score, ema_desc = compute_ema_alignment(price, ema20, ema50, ema200, atr)
    rsi_score, _, rsi_desc = compute_rsi_velocity(rsi14)

    if has_ema20 and has_rsi:
        final_score = 0.60 * ema_score + 0.40 * rsi_score
        status = PillarStatus.MEASURED if (has_ema50 and ema200 is not None) else PillarStatus.PARTIAL
    elif has_ema20:
        final_score = ema_score
        status = PillarStatus.PARTIAL
    else:
        final_score = rsi_score
        status = PillarStatus.PARTIAL

    return PillarResult(
        name="trend",
        score=clamp(final_score),
        status=status,
        inputs=tracking,
        weight=0.25,
        description=f"EMA: {ema_desc} ({ema_score:.1f}) | RSI: {rsi_desc} ({rsi_score:.1f})",
    )


def calculate_sector_pillar(
    technical_data: Optional[Dict[str, Any]],
    sector_bench: Optional[Dict[str, Any]],
) -> PillarResult:
    """
    Evaluates Sector Relative Performance Pillar (Base Weight: 20%):
    - Inputs: stock trailing returns (5d, 20d, 60d), sector ETF benchmark returns.
    - Logistic mapping centred at 50.0.
    """
    t_data = technical_data or {}
    s_data = sector_bench or {}

    stock_5d = clean_float(t_data.get("change_5d_pct"), None)
    sector_5d = clean_float(s_data.get("sector_change_pct"), None)

    stock_20d = clean_float(t_data.get("change_20d_pct"), None)
    sector_20d = clean_float(s_data.get("sector_change_20d_pct"), None)

    stock_60d = clean_float(t_data.get("change_60d_pct"), None)
    sector_60d = clean_float(s_data.get("sector_change_60d_pct"), None)

    sec_sym = s_data.get("sector_etf") or s_data.get("sector_symbol")

    tracking = {
        "stock_change_5d": track_input("stock_change_5d", stock_5d, stock_5d is not None),
        "sector_change_5d": track_input("sector_change_5d", sector_5d, sector_5d is not None),
        "stock_change_20d": track_input("stock_change_20d", stock_20d, stock_20d is not None),
        "sector_change_20d": track_input("sector_change_20d", sector_20d, sector_20d is not None),
        "sector_symbol": track_input("sector_symbol", sec_sym, bool(sec_sym)),
    }

    # Convert percentage (e.g. 4.5) to decimal (0.045) for formula
    def to_dec(val):
        return val / 100.0 if val is not None else None

    excess = compute_sector_excess(
        to_dec(stock_5d), to_dec(sector_5d),
        to_dec(stock_20d), to_dec(sector_20d),
        to_dec(stock_60d), to_dec(sector_60d),
    )

    if excess is None:
        return PillarResult(
            name="sector",
            score=None,
            status=PillarStatus.UNAVAILABLE,
            inputs=tracking,
            weight=0.20,
            description="Sector benchmark or stock return data unavailable",
        )

    score = logistic_scale(excess, k=25.0, midpoint=0.0)
    status = PillarStatus.MEASURED if (stock_20d is not None and sector_20d is not None) else PillarStatus.PARTIAL

    return PillarResult(
        name="sector",
        score=clamp(score),
        status=status,
        inputs=tracking,
        weight=0.20,
        description=f"Sector Excess ({sec_sym or 'ETF'}): {excess*100:+.2f}% -> Score {score:.1f}",
    )


def calculate_alpha_pillar(technical_data: Optional[Dict[str, Any]]) -> PillarResult:
    """
    Evaluates Market Alpha vs. SPY Pillar (Base Weight: 20%):
    - Inputs: relative_alpha_5d, relative_alpha_20d, beta_60d.
    - Standard normal CDF scaling centred at 50.0.
    """
    data = technical_data or {}
    alpha_5d = clean_float(data.get("relative_alpha_5d"), None)
    alpha_20d = clean_float(data.get("relative_alpha_20d"), None)
    beta = clean_float(data.get("beta_60d"), 1.0)
    stock_5d = clean_float(data.get("change_5d_pct"), None)
    spy_5d = clean_float(data.get("market_spy_5d_pct"), None)

    tracking = {
        "relative_alpha_5d": track_input("relative_alpha_5d", alpha_5d, alpha_5d is not None),
        "relative_alpha_20d": track_input("relative_alpha_20d", alpha_20d, alpha_20d is not None),
        "beta_60d": track_input("beta_60d", beta, beta is not None),
    }

    def to_dec(val):
        return val / 100.0 if val is not None else None

    # Derive alpha decimal
    alpha_metric = None
    if alpha_20d is not None:
        alpha_metric = to_dec(alpha_20d)
    elif alpha_5d is not None:
        alpha_metric = to_dec(alpha_5d)
    elif stock_5d is not None and spy_5d is not None:
        alpha_metric = to_dec(stock_5d - spy_5d)

    if alpha_metric is None:
        return PillarResult(
            name="alpha",
            score=None,
            status=PillarStatus.UNAVAILABLE,
            inputs=tracking,
            weight=0.20,
            description="Market benchmark (SPY) relative performance unavailable",
        )

    # Transform: logistic scale with k=25.0 (0.0 excess = 50.0; +4.0% excess = 73.1; -4.0% = 26.9)
    score = logistic_scale(alpha_metric, k=25.0, midpoint=0.0)
    status = PillarStatus.MEASURED if alpha_20d is not None else PillarStatus.PARTIAL

    return PillarResult(
        name="alpha",
        score=clamp(score),
        status=status,
        inputs=tracking,
        weight=0.20,
        description=f"Market Alpha vs SPY: {alpha_metric*100:+.2f}% -> Score {score:.1f}",
    )


def calculate_valuation_history_pillar(gloomberb_payload: Optional[Dict[str, Any]]) -> PillarResult:
    """
    Evaluates Valuation vs History Pillar (Base Weight: 15%):
    - Inputs: historical yield distributions, forward/trailing P/E expansion, revenue growth.
    - Yield Space calculations avoid divide-by-zero on negative earnings.
    """
    payload = gloomberb_payload or {}
    fin = payload.get("financials", {})
    ratios = fin.get("key_ratios", {})

    fwd_pe = clean_float(ratios.get("forward_pe"), None)
    trl_pe = clean_float(ratios.get("trailing_pe"), None)
    rev_growth = clean_float(ratios.get("revenue_growth"), None)
    ey_hist = ratios.get("historical_earnings_yields") or []

    has_pe_pair = (fwd_pe is not None and trl_pe is not None and trl_pe > 0.0)
    has_rev_growth = rev_growth is not None
    has_hist = len(ey_hist) >= 8

    tracking = {
        "forward_pe": track_input("forward_pe", fwd_pe, fwd_pe is not None),
        "trailing_pe": track_input("trailing_pe", trl_pe, trl_pe is not None),
        "revenue_growth": track_input("revenue_growth", rev_growth, has_rev_growth),
        "historical_yield_count": track_input("historical_yield_count", len(ey_hist), has_hist),
    }

    if not has_pe_pair and not has_rev_growth and not has_hist:
        return PillarResult(
            name="valuation_history",
            score=None,
            status=PillarStatus.UNAVAILABLE,
            inputs=tracking,
            weight=0.15,
            description="Historical valuation multiples unavailable",
        )

    score = 50.0
    if has_hist and fwd_pe and fwd_pe > 0:
        current_ey = 1.0 / fwd_pe
        pr = time_series_percentile_score(current_ey, ey_hist)
        score = pr
        status = PillarStatus.MEASURED
    else:
        # Heuristic fallback when long time-series is unavailable
        if has_pe_pair:
            if fwd_pe < trl_pe:
                # Earnings expanding faster than price -> Multiple compression
                score += 20.0
            else:
                score -= 10.0
        if has_rev_growth:
            if rev_growth > 0.15:
                score += 15.0
            elif rev_growth > 0.05:
                score += 5.0
            elif rev_growth < 0.0:
                score -= 10.0
        status = PillarStatus.PARTIAL

    return PillarResult(
        name="valuation_history",
        score=clamp(score),
        status=status,
        inputs=tracking,
        weight=0.15,
        description=f"Valuation History: Score {score:.1f}",
    )


def calculate_peer_valuation_pillar(gloomberb_payload: Optional[Dict[str, Any]]) -> PillarResult:
    """
    Evaluates Valuation vs Direct Peers Pillar (Base Weight: 20%):
    - Inputs: stock forward_pe, stock ev_to_ebitda, peer benchmarks.
    - Yield Space robust z-scores prevent non-positive EBITDA / negative P/E singularities.
    """
    payload = gloomberb_payload or {}
    peer_data = payload.get("peer_valuation", {})
    ratios = payload.get("financials", {}).get("key_ratios", {})
    benchmarks = peer_data.get("direct_peer_benchmarks", [])

    stock_fwd_pe = clean_float(ratios.get("forward_pe"), None)
    stock_ev_ebitda = clean_float(peer_data.get("ev_to_ebitda"), None)

    peer_pes = [clean_float(p.get("forward_pe"), None) for p in benchmarks if clean_float(p.get("forward_pe"), None) is not None]
    peer_evs = [clean_float(p.get("ev_to_ebitda"), None) for p in benchmarks if clean_float(p.get("ev_to_ebitda"), None) is not None]

    clean_peer_pes = [p for p in peer_pes if p > 0.0]
    clean_peer_evs = [e for e in peer_evs if e > 0.0]

    has_pe = (stock_fwd_pe is not None and stock_fwd_pe > 0.0 and len(clean_peer_pes) > 0)
    has_ev = (stock_ev_ebitda is not None and stock_ev_ebitda > 0.0 and len(clean_peer_evs) > 0)

    tracking = {
        "stock_forward_pe": track_input("stock_forward_pe", stock_fwd_pe, stock_fwd_pe is not None),
        "stock_ev_ebitda": track_input("stock_ev_ebitda", stock_ev_ebitda, stock_ev_ebitda is not None),
        "peer_count": track_input("peer_count", len(benchmarks), len(benchmarks) > 0),
    }

    if not has_pe and not has_ev:
        return PillarResult(
            name="peer_valuation",
            score=None,
            status=PillarStatus.UNAVAILABLE,
            inputs=tracking,
            weight=0.20,
            description="Peer valuation benchmarks unavailable",
        )

    # Invert to Yield Space: Earnings Yield = 1/PE, EBITDA Yield = 1/(EV/EBITDA)
    # Higher yield = cheaper = higher score
    z_components = []
    if has_pe:
        stock_ey = 1.0 / stock_fwd_pe
        peer_eys = [1.0 / p for p in clean_peer_pes]
        z_pe = cross_sectional_robust_zscore(stock_ey, peer_eys)
        z_components.append(z_pe)

    if has_ev:
        stock_ebitday = 1.0 / stock_ev_ebitda
        peer_ebitdays = [1.0 / e for e in clean_peer_evs]
        z_ev = cross_sectional_robust_zscore(stock_ebitday, peer_ebitdays)
        z_components.append(z_ev)

    avg_z = sum(z_components) / len(z_components)
    score = normal_cdf_scale(avg_z, scale=1.0)
    status = PillarStatus.MEASURED if (has_pe and has_ev) else PillarStatus.PARTIAL

    return PillarResult(
        name="peer_valuation",
        score=clamp(score),
        status=status,
        inputs=tracking,
        weight=0.20,
        description=f"Peer Yield z-score: {avg_z:+.2f} -> Score {score:.1f}",
    )
