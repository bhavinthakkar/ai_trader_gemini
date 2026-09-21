"""
Strict output schema for the master-trader LLM signal.

The decision fields are validated here, once, against known enums/ranges:
- buy/hold/sell must be numeric in [0,1] and sum to ~1.0
- decision / primary_driver must match the closed enum
- quantitative scores/pillars must be in [0,100], completeness in [0,1]

validate_signal_json() returns (normalized, errors). The only "repairs" it
performs are safe normalizations: enum upper-casing, mono-whitespace collapse,
and probability clamp+re-normalization when the sum is within tolerance.
Anything else is reported as an error so the caller can issue one corrective
retry instead of silently coercing critical decision fields.
"""

VALID_DECISIONS = ("BUY", "SELL", "HOLD")
VALID_PRIMARY_DRIVERS = ("QUANT_STRUCTURE", "FUNDAMENTAL", "NEWS_CATALYST", "MACRO_EVENT", "EARNINGS_CATALYST")

NO_TRADE_REASONS = (
    "INSUFFICIENT_EVIDENCE",
    "EARNINGS_BLACKOUT",
    "LOW_LIQUIDITY",
    "RR_TOO_LOW",
    "SCHEMA_INVALID",
)

PROBABILITY_SUM_TOLERANCE = 0.05

PILLAR_KEYS = ("trend", "sector", "alpha", "valuation_history", "peer_valuation")

LIST_KEYS = (
    "bull_case", "bear_case", "key_risks", "missing_information", "key_risk",
    "catalyst_analysis", "investor_questions",
)
OBJECT_KEYS = ("thesis_assumptions", "valuation_assessment", "price_level_map")


def _to_float(val, default=None):
    if val is None:
        return default
    if isinstance(val, bool):
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def normalize_probabilities(buy, hold, sell):
    """Clamp probabilities to [0,1] and re-normalize to sum 1.0 (when sum is sane)."""
    vals = [max(0.0, min(1.0, float(v))) for v in (buy, hold, sell)]
    total = sum(vals)
    if total <= 0.0:
        return None
    return [round(v / total, 4) for v in vals]


def validate_signal_json(obj):
    """
    Validate an extracted LLM signal object against the strict schema.

    Returns (normalized, errors):
      - normalized: dict safe for normalize_master_trader_json (enums cleaned,
        probabilities clamped + re-normalized when within tolerance)
      - errors: list of human-readable violations (empty == valid)
    """
    errors = []
    if not isinstance(obj, dict):
        return {}, ["response is not a JSON object"]

    norm = dict(obj)
    clean = lambda s: " ".join(str(s).strip().split())

    # --- Decision enum ----------------------------------------------------
    raw_dec = norm.get("decision") or norm.get("recommendation") or norm.get("signal")
    if raw_dec is None:
        errors.append("missing required field 'decision'")
    else:
        dec = clean(raw_dec).upper()
        if "BUY" in dec:
            dec = "BUY"
        elif "SELL" in dec:
            dec = "SELL"
        elif dec == "HOLD":
            dec = "HOLD"
        elif dec in VALID_DECISIONS:
            dec = dec
        else:
            errors.append(f"invalid decision value '{raw_dec}'")
        norm["decision"] = dec

    # --- primary_driver enum ---------------------------------------------
    raw_driver = norm.get("primary_driver")
    if raw_driver is None:
        errors.append("missing required field 'primary_driver'")
    else:
        driver = "_".join(clean(raw_driver).upper().split())
        if driver not in VALID_PRIMARY_DRIVERS:
            errors.append(f"invalid primary_driver value '{raw_driver}'")
        norm["primary_driver"] = driver

    # --- Probabilities: numeric, in-range, ~sum to 1 ----------------------
    missing_probs = [k for k in ("buy_score", "hold_score", "sell_score") if k not in norm]
    if missing_probs:
        errors.append("missing required probability field(s): " + ", ".join(missing_probs))
    else:
        b = _to_float(norm.get("buy_score"))
        h = _to_float(norm.get("hold_score"))
        s = _to_float(norm.get("sell_score"))
        if None in (b, h, s):
            errors.append("buy_score/hold_score/sell_score must be numeric")
        elif any(v < 0.0 or v > 1.0 for v in (b, h, s)):
            errors.append(f"probabilities out of [0,1]: buy={b} hold={h} sell={s}")
        elif b + h + s <= 0.0:
            errors.append("probabilities are all zero")
        else:
            diff = abs(b + h + s - 1.0)
            if diff > PROBABILITY_SUM_TOLERANCE:
                errors.append(
                    f"probabilities sum to {b + h + s:.3f} (tolerance ±{PROBABILITY_SUM_TOLERANCE}); "
                    f"expected ~1.0"
                )
            else:
                repaired = normalize_probabilities(b, h, s)
                if repaired is None:
                    errors.append("probabilities are all zero")
                else:
                    norm["buy_score"], norm["hold_score"], norm["sell_score"] = repaired

    # --- Quantitative scores / pillars --------------------------------
    q = _to_float(norm.get("quant_score"))
    if q is not None and not (0.0 <= q <= 100.0):
        errors.append(f"quant_score {q} out of [0,100]")

    raw_pillars = norm.get("pillar_scores")
    if raw_pillars is not None and not isinstance(raw_pillars, dict):
        errors.append("pillar_scores must be an object")
    elif isinstance(raw_pillars, dict):
        for k in PILLAR_KEYS:
            v = _to_float(raw_pillars.get(k))
            if v is not None and not (0.0 <= v <= 100.0):
                errors.append(f"pillar_scores.{k} {v} out of [0,100]")

    dc = _to_float(norm.get("data_completeness"))
    if dc is not None and not (0.0 <= dc <= 1.0):
        errors.append(f"data_completeness {dc} out of [0,1]")

    # --- horizon_days -------------------------------------------------
    try:
        hd = int(norm.get("horizon_days"))
        if hd <= 0:
            errors.append("horizon_days must be > 0")
    except (TypeError, ValueError):
        if "horizon_days" in norm:
            errors.append("horizon_days must be an integer")

    # --- List fields must be lists if present -------------------------
    for key in LIST_KEYS:
        if key in norm and norm[key] is not None and not isinstance(norm[key], (list, tuple, str)):
            errors.append(f"{key} must be a list")

    for key in OBJECT_KEYS:
        if key in norm and norm[key] is not None and not isinstance(norm[key], dict):
            errors.append(f"{key} must be an object")

    return norm, errors
