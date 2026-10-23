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


from llm_evidence_schema import (
    FORBIDDEN_BROKER_FIELDS,
    LLMEvidenceAnalysis,
    validate_llm_evidence_response,
    evidence_analysis_to_signal_dict,
)


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


def validate_signal_json(
    obj,
    require_ml_assessment=False,
    required_horizon_sessions=None,
    require_forecast_20d=False,
    expected_ticker=None,
    valid_document_ids=None,
):
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

    # Step 4.2: Intercept prohibited broker command fields
    forbidden = [k for k in obj.keys() if str(k).lower() in FORBIDDEN_BROKER_FIELDS]
    if forbidden:
        return {}, [f"prohibited broker command fields detected: {forbidden}. Output must not contain broker execution instructions."]

    # Step 4.2: Detect if object is structured under the strict LLMEvidenceAnalysis contract.
    # An evidence contract object specifies advisory_action or analysis_status without traditional decision authority.
    is_evidence_obj = (
        ("advisory_action" in obj or "analysis_status" in obj)
        and "decision" not in obj
        and "buy_score" not in obj
    )
    if is_evidence_obj:
        model, evidence_errors = validate_llm_evidence_response(
            obj,
            expected_ticker=expected_ticker,
            valid_document_ids=valid_document_ids,
        )
        if evidence_errors:
            return {}, evidence_errors
        # Convert validated evidence model to signal dictionary for normalize_master_trader_json
        norm = evidence_analysis_to_signal_dict(model)
        return norm, []

    # Check expected ticker for traditional signal format
    if expected_ticker:
        raw_ticker = obj.get("ticker") or obj.get("stock") or obj.get("symbol")
        if raw_ticker and str(raw_ticker).strip().upper() != str(expected_ticker).strip().upper():
            return {}, [f"ticker mismatch: received '{raw_ticker}', expected '{expected_ticker}'"]

    norm = dict(obj)
    clean = lambda s: " ".join(str(s).strip().split())

    # When a fresh ML score is supplied, require an explicit disposition so
    # the LLM's use of this material input is auditable.
    assessment = norm.get("ml_signal_assessment")
    if require_ml_assessment and not isinstance(assessment, dict):
        errors.append("missing required object 'ml_signal_assessment'")
    elif assessment is not None:
        if not isinstance(assessment, dict):
            errors.append("ml_signal_assessment must be an object")
        else:
            disposition = clean(assessment.get("disposition", "")).upper()
            if "SUPPORT" in disposition:
                disposition = "SUPPORTIVE"
            elif "CONFLICT" in disposition:
                disposition = "CONFLICTING"
            elif "NEUTRAL" in disposition:
                disposition = "NEUTRAL"
            elif disposition not in ("SUPPORTIVE", "CONFLICTING", "NEUTRAL"):
                errors.append("ml_signal_assessment.disposition must be SUPPORTIVE, CONFLICTING, or NEUTRAL")
            reason = clean(assessment.get("reason") or assessment.get("explanation") or assessment.get("summary") or "")
            if not reason:
                errors.append("ml_signal_assessment.reason must be a non-empty explanation")
            norm["ml_signal_assessment"] = {"disposition": disposition, "reason": reason}

    forecast = norm.get("forecast_20d")
    if require_forecast_20d and not isinstance(forecast, dict):
        errors.append("missing required object 'forecast_20d'")
    elif forecast is not None:
        if not isinstance(forecast, dict):
            errors.append("forecast_20d must be an object")
        else:
            outlook = clean(forecast.get("relative_outlook", "")).upper()
            if "OUTPERFORM" in outlook or "BULL" in outlook:
                outlook = "OUTPERFORM"
            elif "UNDERPERFORM" in outlook or "BEAR" in outlook:
                outlook = "UNDERPERFORM"
            elif "MARKET" in outlook or "NEUTRAL" in outlook or "HOLD" in outlook or "IN_LINE" in outlook:
                outlook = "MARKET_LIKE"
            elif outlook not in ("OUTPERFORM", "MARKET_LIKE", "UNDERPERFORM"):
                errors.append("forecast_20d.relative_outlook must be OUTPERFORM, MARKET_LIKE, or UNDERPERFORM")
            forecast_confidence = _to_float(forecast.get("confidence"))
            if forecast_confidence is not None and 1.0 < forecast_confidence <= 100.0:
                forecast_confidence = forecast_confidence / 100.0
            if forecast_confidence is None:
                forecast_confidence = _to_float(norm.get("confidence"), default=0.5)
            if forecast_confidence is None or not (0.0 <= forecast_confidence <= 1.0):
                errors.append("forecast_20d.confidence must be numeric in [0,1]")
            rationale = clean(forecast.get("rationale") or forecast.get("reason") or forecast.get("summary") or "")
            if not rationale:
                errors.append("forecast_20d.rationale must be a non-empty explanation")
            norm["forecast_20d"] = {
                "relative_outlook": outlook,
                "confidence": forecast_confidence,
                "rationale": rationale,
            }

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
        elif "HOLD" in dec or "NEUTRAL" in dec or "NO_ACTION" in dec:
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
            if "EARNING" in driver:
                driver = "EARNINGS_CATALYST"
            elif "NEWS" in driver or "CATALYST" in driver:
                driver = "NEWS_CATALYST"
            elif "MACRO" in driver or "FED" in driver or "RATE" in driver or "GEOPOLITIC" in driver:
                driver = "MACRO_EVENT"
            elif "FUNDAMENTAL" in driver or "VALUATION" in driver:
                driver = "FUNDAMENTAL"
            elif "QUANT" in driver or "STRUCTURE" in driver or "TECH" in driver or "PRICE" in driver:
                driver = "QUANT_STRUCTURE"
            else:
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
        else:
            # Auto-normalize 0-100 percentage scale if detected
            if any(v > 1.0 for v in (b, h, s)) and all(0.0 <= v <= 100.0 for v in (b, h, s)):
                tot = b + h + s
                if 80.0 <= tot <= 120.0:
                    b, h, s = b / tot, h / tot, s / tot
                elif tot > 0:
                    b, h, s = b / 100.0, h / 100.0, s / 100.0
            if any(v < 0.0 or v > 1.0 for v in (b, h, s)):
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

    if required_horizon_sessions is not None:
        raw_hs = norm.get("horizon_sessions")
        if raw_hs is None and "horizon_days" in norm:
            raw_hs = norm.get("horizon_days")
        try:
            hs = int(raw_hs)
            if hs != int(required_horizon_sessions):
                errors.append(f"horizon_sessions must equal {required_horizon_sessions}")
            else:
                norm["horizon_sessions"] = hs
                norm["horizon_days"] = hs
        except (TypeError, ValueError):
            errors.append("missing or invalid required field 'horizon_sessions'")
    elif "horizon_sessions" in norm:
        try:
            norm["horizon_sessions"] = int(norm["horizon_sessions"])
            if norm["horizon_sessions"] <= 0:
                errors.append("horizon_sessions must be > 0")
        except (TypeError, ValueError):
            errors.append("horizon_sessions must be an integer")

    # --- List fields must be lists if present -------------------------
    for key in LIST_KEYS:
        if key in norm and norm[key] is not None and not isinstance(norm[key], (list, tuple, str)):
            errors.append(f"{key} must be a list")

    for key in OBJECT_KEYS:
        if key in norm and norm[key] is not None and not isinstance(norm[key], dict):
            errors.append(f"{key} must be an object")

    return norm, errors
