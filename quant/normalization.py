"""Normalization and monotonic score transformation functions."""

import math
from typing import Sequence, Optional


def clamp(value: float, min_val: float = 0.0, max_val: float = 100.0) -> float:
    """Clamps numeric value into [min_val, max_val]."""
    default_val = max(min_val, min(max_val, 50.0))
    if value is None:
        return default_val
    try:
        val = float(value)
        if math.isnan(val):
            return default_val
        return max(min_val, min(max_val, val))
    except (TypeError, ValueError):
        return default_val


def logistic_scale(val: float, k: float = 25.0, midpoint: float = 0.0) -> float:
    """
    Monotonically maps an unbounded continuous metric (e.g. excess return) to [0.0, 100.0]
    using a logistic sigmoid centred at midpoint:
        S = 100.0 / (1.0 + exp(-k * (val - midpoint)))
    """
    if val is None:
        return 50.0
    try:
        f_val = float(val)
        if math.isnan(f_val):
            return 50.0
    except (TypeError, ValueError):
        return 50.0
    diff = f_val - midpoint
    # Avoid overflow
    if k * diff > 40.0:
        return 100.0
    if k * diff < -40.0:
        return 0.0
    return clamp(100.0 / (1.0 + math.exp(-k * diff)))


def normal_cdf(z: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def normal_cdf_scale(z: float, scale: float = 1.0) -> float:
    """
    Monotonically maps a z-score or standardized appraisal ratio to [0.0, 100.0]
    using the standard normal CDF:
        S = 100.0 * Phi(z / scale)
    """
    if z is None:
        return 50.0
    try:
        fz = float(z)
        if math.isnan(fz):
            return 50.0
    except (TypeError, ValueError):
        return 50.0
    if scale <= 0.0:
        scale = 1.0
    scaled_z = fz / scale
    return clamp(100.0 * normal_cdf(scaled_z))


def cross_sectional_robust_zscore(
    target_val: float,
    peer_vals: Sequence[float],
) -> float:
    """
    Computes a robust z-score using the Median and Median Absolute Deviation (MAD):
        z = (x - median) / (1.4826 * MAD)
    Protects against extreme outliers and multi-segment business distortion.
    """
    clean_peers = [p for p in peer_vals if p is not None and not math.isnan(p)]
    if not clean_peers:
        return 0.0

    sorted_peers = sorted(clean_peers)
    n = len(sorted_peers)
    mid = n // 2
    med = (sorted_peers[mid] if n % 2 != 0 else (sorted_peers[mid - 1] + sorted_peers[mid]) / 2.0)

    deviations = sorted([abs(p - med) for p in clean_peers])
    mad = (deviations[mid] if n % 2 != 0 else (deviations[mid - 1] + deviations[mid]) / 2.0)
    sigma_robust = max(1.4826 * mad, 0.01)

    z = (target_val - med) / sigma_robust
    return max(-3.0, min(3.0, z))


def time_series_percentile_score(
    current_val: float,
    history: Sequence[float],
) -> float:
    """
    Computes point-in-time percentile rank in [0.0, 100.0] within trailing history:
        PR = (count(history_i <= current_val) / N) * 100.0
    """
    clean_hist = [h for h in history if h is not None and not math.isnan(h)]
    if not clean_hist:
        return 50.0

    count_below = sum(1 for h in clean_hist if h <= current_val)
    return clamp((count_below / float(len(clean_hist))) * 100.0)
