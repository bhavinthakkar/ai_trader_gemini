import sys
import argparse
import os
import json
import time
import requests
from dotenv import load_dotenv

from llm_service import query_llm, get_model_label, extract_json, uses_compact_prompt_profile
from llamacpp_server import managed_llamacpp_server
from db import init_db, save_results, save_portfolio_review, update_signal_outcomes
from market_agent import MarketAgent
from institutional_agent import InstitutionalDataAgent
from macro_agent import MacroDataAgent
from news_agent import NewsAgent
from risk_agent import RiskAgent
from analyst_agent import AnalystAgent

from gloomberb_service import GloomberbService
from institutional_data_service import InstitutionalDataService
from rag_service import RAGService
from quantitative_scoring_service import QuantitativeScoringService
from catalyst_service import (
    DIP_BUY,
    EXTENSION_SELL,
    DIP_MIN_COMPOSITE,
    EXTENSION_MAX_COMPOSITE,
    SHORT_MAX_COMPOSITE,
    CATALYST_CONFIDENCE_BOOST,
    CATALYST_CONFIDENCE_CEILING,
    build_relevance_needles,
    evaluate_catalyst_setup,
)
from signal_schema import validate_signal_json
from ticker_resolver import resolve_symbol
from ml_signal_service import format_ml_context, get_ml_context
from quant.composite_score import format_quant_and_ml_context, blend_quant_and_ml_scores
from danelfin_service import DanelfinService
from policy_config import TradingPolicy, load_policy_config
from decision_precedence import (
    InternalActionType,
    DecisionStage,
    PositionState,
    evaluate_decision_precedence,
)
from signal_strength import (
    DispersionMetric,
    CombinationMethod,
    calculate_signal_strength,
)
from structured_decision import (
    PositionAction,
    DataQualityStatus,
    ModelScopeStatus,
    StandardGateName,
    StandardReasonCode,
    StructuredDecision,
    build_structured_decision,
    qualify_candidate_setup,
)
from quant.entry_setups import (
    PullbackSetupResult,
    evaluate_pullback_setup,
)

_POLICY = load_policy_config()
WATCHLIST = ["000660.KS"]
MIN_REWARD_RISK_RATIO = float(os.getenv("MIN_REWARD_RISK_RATIO", str(_POLICY.trade_quality.reward_risk_minimum)))
BUY_COMPOSITE_FLOOR = float(os.getenv("BUY_COMPOSITE_FLOOR", str(_POLICY.scores.buy_minimum)))

load_dotenv()

MASTER_TRADER_INSTRUCTION = """
You are a senior Master Trader & Portfolio Manager.

You receive short-term intelligence collected by specialized sub-agents and quantitative engines:
1. Quantitative Scoring Engine: Deterministic Quant Score (0-100 across 5 pillars: Trend, Sector, Alpha, Valuation History, Peer Valuation) and 20-Session ML Alpha Score (0-100) blended into the Decision Composite.
2. Market Agent: Price action, 5-day velocity, RSI, EMA20/50, ATR, Forward P/E, profit margins.
3. Institutional Data Agent: SEC EDGAR regulatory filings (Form 4 insider transactions, 10-K/10-Q reports, 8-K material events, 13D/13G/13F institutional ownership).
4. Macro Data Agent (marco_data): FRED macroeconomic indicators (10Y Treasury Yield, 10Y-2Y yield curve spread, Fed Funds Rate, Unemployment) and CFTC Commitment of Traders (COT) futures positioning (S&P 500, Nasdaq 100, VIX).
5. News Agent: Today's real-time headlines, news summary, sentiment, and short-term impact.
6. Risk Agent: Volatility ATR%, technical trend breakdown risk, RSI extreme risk, and stop-loss boundaries.
7. Analyst Agent: Wall Street consensus, mean target price, and recent rating actions from major investment banks (BofA, Barclays, Credit Suisse, DB, Evercore, Goldman, JPM, Morgan Stanley, UBS).

Your Task:
Synthesize the quantitative scores and insights from all specialized sub-agents to forecast relative performance over the next 20 trading sessions (about one month) and assess whether a trade is actionable.

Rules & Guidelines:
1. Ignore stale long-term macro narratives that are already priced in, but incorporate macro yield curve & CFTC positioning context.
2. Ground your decision in the Deterministic Quant Score and ML 20-Session Score (Decision Composite).
3. Synthesize Quantitative Scores + Institutional/SEC filings + Macro FRED/COT data + News Agent impact + Market Agent momentum + Risk Agent factors + Analyst Agent bank ratings.
3. Return a valid JSON object matching the requested schema.

Schema:
{
  "stock": "Ticker Symbol",
  "decision": "BUY|SELL|HOLD",
  "confidence": 0.85,
  "decision_reasoning": "Detailed expert rationale explaining exactly why this decision (BUY, SELL, HOLD) was reached, synthesizing quantitative scores, technical action, risk factors, and catalysts",
  "reason": "Short-term swing setup rationale synthesizing Market, Institutional, Macro, News, Risk & Analyst Agent insights",
  "institutional_data": "Summary of SEC EDGAR insider & institutional filing activity from Institutional Data Agent",
  "macro_data": "Summary of FRED yield curve/interest rates & CFTC COT positioning from Macro Data Agent",
  "news": "Today's key news summary & catalyst impact from News Agent",
  "investment_bank_coverage": "Recent rating actions & consensus from Analyst Agent",
  "risk_assessment": "Risk level & stop-loss boundary from Risk Agent",
  "push_notification": "TRUE|FALSE",
  "PE_and_PEG": "Forward PE ratio value"
}
"""


def _ensure_str(val, default="") -> str:
    if val is None:
        return default
    if isinstance(val, (list, tuple)):
        return ", ".join(str(x) for x in val)
    if isinstance(val, dict):
        return json.dumps(val)
    return str(val)


def _to_list_of_strings(val, default_str="None") -> list:
    if isinstance(val, list):
        items = []
        for item in val:
            if item is not None and str(item).strip():
                items.append(str(item).strip())
        return items if items else [default_str]
    elif isinstance(val, str) and val.strip():
        parts = [p.strip(" •-*") for p in val.split("\n") if p.strip(" •-*")]
        return parts if parts else [val.strip()]
    return [default_str]


def format_danelfin_context(danelfin_scores: dict | None) -> str:
    """Formats Danelfin AI model scores and sub-scores for LLM context injection."""
    if not danelfin_scores or not danelfin_scores.get("aiscore"):
        return ""

    aiscore = danelfin_scores.get("aiscore")
    tech = danelfin_scores.get("technical", "N/A")
    fund = danelfin_scores.get("fundamental", "N/A")
    sent = danelfin_scores.get("sentiment", "N/A")
    low_risk = danelfin_scores.get("low_risk", "N/A")
    date_str = danelfin_scores.get("date", "latest")

    tier = (
        "HIGH CONVICTION OUTPERFORM (AI Score >= 8/10)"
        if (isinstance(aiscore, (int, float)) and aiscore >= 8)
        else "MODERATE / NEUTRAL (AI Score 5-7/10)"
        if (isinstance(aiscore, (int, float)) and aiscore >= 5)
        else "UNDERPERFORM / HIGH RISK (AI Score < 5/10)"
    )

    return f"""
================ 7. DANELFIN AI QUANTITATIVE INTELLIGENCE ================
As of: {date_str}
Danelfin Overall AI Score: {aiscore}/10 [{tier}]
  - Fundamental Pillar Score: {fund}/10
  - Technical Pillar Score:   {tech}/10
  - Sentiment Pillar Score:   {sent}/10
  - Low Risk Pillar Score:    {low_risk}/10
Guidance: Factor Danelfin's independent predictive score into your synthesis. Pillar concordance provides additional confirmation.
"""


def days_from_earnings_date(date_str, today=None) -> int:
    """
    Parses a 'YYYY-MM-DD' earnings_date string (e.g. from the Gloomberb earnings
    channel) and returns calendar days until that date. Returns None when the date
    is missing, unparsable, or in the past (earnings already reported).
    """
    if not date_str or str(date_str).strip().upper() in ("N/A", "NONE", ""):
        return None
    try:
        import datetime
        target_dt = datetime.date.fromisoformat(str(date_str)[:10])
        now = today or datetime.date.today()
        diff = (target_dt - now).days
        return diff if diff >= 0 else None
    except Exception:
        return None


def gated_confidence(composite, pillar_scores: dict, data_completeness: float = 0.87, decision: str = "BUY") -> float:
    """
    Formalized Signal Strength and Decision Quality Index (0..1) (Step 5.5):
    - Score Decisiveness D = |S - 50.0| / 50.0 in [0.0, 1.0]
    - Pillar Agreement A from normalized cross-pillar dispersion
    - Data Completeness C from valid, fresh inputs
    Returns a float rounded to 2 decimals. Explicitly designated as Signal Strength
    (not the empirical probability that a trade will be profitable).
    """
    res = calculate_signal_strength(
        composite_score=composite,
        pillar_scores=pillar_scores,
        data_completeness=data_completeness,
        decision=decision,
    )
    return res.signal_strength


def normalize_master_trader_json(
    data: dict,
    symbol: str,
    m_data: dict = None,
    macro_data: dict = None,
    gloomberb_payload: dict = None,
    policy: TradingPolicy = None,
) -> dict:
    active_policy = policy or load_policy_config()
    if not isinstance(data, dict):
        data = {}
    if not isinstance(m_data, dict):
        m_data = {}
    if not isinstance(macro_data, dict):
        macro_data = {}
    if not isinstance(gloomberb_payload, dict):
        gloomberb_payload = {}

    stock = data.get("stock") or data.get("symbol") or symbol

    def _to_float(val, default=0.0):
        try:
            return round(float(val), 2)
        except (TypeError, ValueError):
            return default

    buy_score = _to_float(data.get("buy_score"), 0.0)
    hold_score = _to_float(data.get("hold_score"), 0.0)
    sell_score = _to_float(data.get("sell_score"), 0.0)

    # Normalize Decision (5-State Taxonomy: BUY, DEEP BUY, HOLD, BOOK PROFIT, SELL)
    raw_dec = str(data.get("decision") or data.get("recommendation") or data.get("signal") or "HOLD").upper()
    if "DEEP BUY" in raw_dec or "DIP BUY" in raw_dec or "DEEP_BUY" in raw_dec or "DIP_BUY" in raw_dec:
        decision = "DEEP BUY"
    elif "BOOK PROFIT" in raw_dec or "TAKE PROFIT" in raw_dec or "BOOK_PROFIT" in raw_dec or "TAKE_PROFIT" in raw_dec or "PROFIT_TAKING" in raw_dec:
        decision = "BOOK PROFIT"
    elif "BUY" in raw_dec:
        decision = "BUY"
    elif "SELL" in raw_dec:
        decision = "SELL"
    else:
        decision = "HOLD"
    original_decision = decision
    model_decision = decision  # Preserved as explicit separate model suggestion (Step 2 Contract)
    decision_origin = "MODEL"

    # Normalize model confidence -- kept as a reference; the authoritative `confidence`
    # is overwritten below by the deterministic gated_confidence() signal-quality formula.
    raw_conf = data.get("confidence") or buy_score or 0.70
    model_confidence = _to_float(raw_conf, 0.70)
    if model_confidence > 1.0:
        model_confidence = round(model_confidence / 100.0, 2) if model_confidence <= 100 else 0.70

    try:
        horizon_days = int(data.get("horizon_days") or 20)
    except (ValueError, TypeError):
        horizon_days = 20
    try:
        horizon_sessions = int(data.get("horizon_sessions") or horizon_days)
    except (ValueError, TypeError):
        horizon_sessions = horizon_days
    forecast_20d = data.get("forecast_20d") if isinstance(data.get("forecast_20d"), dict) else None

    # Quantitative 5-Pillar Scores. The deterministic composite calculated from real market data is
    # authoritative for decisions and confidence. The model's echoed values are stored separately
    # (model_quant_score / model_pillar_scores) for reference only and never drive the signal.
    deterministic_scores = m_data.get("deterministic_5pillar_scores", {})
    det_quant = deterministic_scores.get("composite_quantitative_score")
    ml_context = m_data.get("ml_context")
    model_status = ml_context.get("model_status") if isinstance(ml_context, dict) else None
    is_in_scope = (
        model_status == "IN_SCOPE"
        or (
            bool(ml_context.get("in_scope", False))
            and model_status not in ("OUT_OF_SCOPE", "STALE_SCORE", "SCHEMA_MISMATCH", "FILE_NOT_FOUND", "LOOKAHEAD_VIOLATION")
        )
    ) if isinstance(ml_context, dict) else False

    ml_score = _to_float(ml_context.get("score_0_100"), None) if (isinstance(ml_context, dict) and is_in_scope) else None
    if is_in_scope and ml_score is not None and det_quant is not None:
        configured_weight = _to_float(ml_context.get("ml_weight"), 0.20)
        ml_weight = configured_weight if (configured_weight is not None and configured_weight > 0.0) else 0.20
    else:
        ml_weight = 0.0

    if det_quant is not None and ml_score is not None and ml_weight > 0.0:
        decision_composite = round((1.0 - ml_weight) * float(det_quant) + ml_weight * ml_score, 2)
    else:
        decision_composite = _to_float(det_quant, None)
    quant_score = decision_composite if decision_composite is not None else _to_float(data.get("quant_score"), None)
    if quant_score is None:
        quant_score = 50.0  # neutral fallback for display only; BUY gate uses the deterministic composite
    model_quant_score = _to_float(data.get("quant_score"), None)

    raw_pillars = data.get("pillar_scores") or {}
    if not isinstance(raw_pillars, dict):
        raw_pillars = {}

    pillar_status = deterministic_scores.get("pillar_status") or {}
    pillar_input_tracking = deterministic_scores.get("pillar_input_tracking") or {}
    fq_info = deterministic_scores.get("financial_quality") or {}
    fq_score = deterministic_scores.get("financial_quality_score")
    is_value_trap = bool(deterministic_scores.get("is_value_trap", False) or m_data.get("is_value_trap", False))
    value_trap_reasons = deterministic_scores.get("value_trap_reasons", [])

    pillar_scores = {}
    model_pillar_scores = {}
    for det_key, model_key, out_key in (
        ("trend_score", "trend", "trend"),
        ("sector_relative_score", "sector", "sector"),
        ("market_alpha_score", "alpha", "alpha"),
        ("valuation_history_score", "valuation_history", "valuation_history"),
        ("peer_valuation_score", "peer_valuation", "peer_valuation"),
    ):
        if det_key in deterministic_scores and deterministic_scores.get(det_key) is not None:
            det_val = deterministic_scores.get(det_key)
            pillar_scores[out_key] = _to_float(det_val, None) if det_val is not None else None
        elif model_key in deterministic_scores and deterministic_scores.get(model_key) is not None:
            det_val = deterministic_scores.get(model_key)
            pillar_scores[out_key] = _to_float(det_val, None) if det_val is not None else None
        elif model_key in raw_pillars and raw_pillars.get(model_key) is not None:
            model_val = raw_pillars.get(model_key)
            pillar_scores[out_key] = _to_float(model_val, None) if model_val is not None else None
        elif deterministic_scores:
            # Deterministic scores dictionary was provided but this pillar was missing/excluded
            pillar_scores[out_key] = None
        else:
            # Deterministic pipeline was not run; fall back to raw model pillars
            model_val = raw_pillars.get(model_key)
            pillar_scores[out_key] = _to_float(model_val, None) if model_val is not None else None
        model_pillar_scores[out_key] = _to_float(raw_pillars.get(model_key), None)

    # Array Fields
    bull_case = _to_list_of_strings(data.get("bull_case"), "Bullish trade setup and vector RAG catalysts evaluated.")
    bear_case = _to_list_of_strings(data.get("bear_case") or data.get("risk_assessment"), "Bearish trade setup and downside risks evaluated.")
    key_risks = _to_list_of_strings(data.get("key_risks") or data.get("key_risk"), "Volatile market conditions and stop-loss boundaries.")

    # Model Decision Reasoning: capture the model's explicit rationale for why this decision was chosen
    raw_dec_reasoning = (
        data.get("decision_reasoning")
        or data.get("decision_rationale")
        or (forecast_20d.get("rationale") if isinstance(forecast_20d, dict) else None)
        or data.get("reason")
    )
    if not raw_dec_reasoning or (isinstance(raw_dec_reasoning, str) and not raw_dec_reasoning.strip()):
        if original_decision in ("BUY", "DEEP BUY"):
            raw_dec_reasoning = "; ".join(bull_case)
        elif original_decision in ("SELL", "BOOK PROFIT"):
            raw_dec_reasoning = "; ".join(bear_case)
        else:
            raw_dec_reasoning = (forecast_20d.get("rationale") if isinstance(forecast_20d, dict) else None) or "; ".join(bull_case)
    decision_reasoning = str(raw_dec_reasoning or "No specific decision reasoning recorded.").strip()
    missing_info = _to_list_of_strings(data.get("missing_information"), "None")
    catalyst_analysis = data.get("catalyst_analysis") if isinstance(data.get("catalyst_analysis"), list) else []
    biggest_surprise_catalyst = str(data.get("biggest_surprise_catalyst") or "")
    thesis_assumptions = data.get("thesis_assumptions") if isinstance(data.get("thesis_assumptions"), dict) else {}
    investor_questions = _to_list_of_strings(data.get("investor_questions"), "None")
    valuation_assessment = data.get("valuation_assessment") if isinstance(data.get("valuation_assessment"), dict) else {}
    price_level_map = data.get("price_level_map") if isinstance(data.get("price_level_map"), dict) else {}

    # Data completeness is computed DETERMINISTICALLY by the pipeline from live provider
    # availability (QuantitativeScoringService.compute_data_coverage) and attached to
    # m_data["deterministic_data_completeness"]. The model's stated value is kept only as a
    # reference (model_data_completeness); an absent provider set is treated as 0.0, never
    # invented (the old 0.87 default could let a malformed BUY pass the coverage bar).
    model_data_completeness = _to_float(data.get("data_completeness"), None)
    deterministic_completeness = _to_float(m_data.get("deterministic_data_completeness"), None)
    if deterministic_completeness is not None:
        data_completeness = deterministic_completeness
    else:
        data_completeness = _to_float(model_data_completeness, 0.0)

    # Mechanistic confidence: a function of composite decisiveness, pillar agreement,
    # and data completeness -- NOT the model's stated number. Deterministic pillars are
    # authoritative for confidence. A HOLD never carries high conviction, so cap it.
    conf_composite = decision_composite
    if conf_composite is None:
        conf_composite = quant_score or 50.0
    confidence = gated_confidence(conf_composite, pillar_scores, data_completeness)
    if decision == "HOLD":
        confidence = round(min(confidence, 0.60), 2)

    price_v = _to_float(m_data.get("current_price"), 0.0)
    atr_v = _to_float(m_data.get("atr"), 0.0)
    stop_v = _to_float(m_data.get("suggested_stop_loss"), 0.0)
    target_v = _to_float(m_data.get("suggested_target_price"), 0.0)

    # Reward:Risk setup geometry -- deterministic, computed from market data, not the model.
    # Channel-anchored so the ratio varies with price position inside the 20-day range.
    rr_info = QuantitativeScoringService.compute_channel_reward_risk(
        price_v,
        atr_v,
        m_data.get("high_20d"),
        m_data.get("low_20d"),
        stop_v,
        target_v
    )
    analyst_rr_info = QuantitativeScoringService.compute_reward_risk(
        price_v,
        stop_v,
        m_data.get("analyst_target_price")
    )
    rr = rr_info.get("reward_risk_ratio")
    breakeven = rr_info.get("breakeven_win_rate")
    analyst_rr = analyst_rr_info.get("reward_risk_ratio")
    dist_resistance = rr_info.get("distance_to_resistance_atr")

    # Short Reward:Risk setup geometry -- deterministic, channel-anchored
    short_rr_info = QuantitativeScoringService.compute_channel_short_reward_risk(
        price_v,
        atr_v,
        m_data.get("high_20d"),
        m_data.get("low_20d"),
        m_data.get("suggested_short_stop_loss"),
        m_data.get("suggested_short_target_price")
    )
    short_rr = short_rr_info.get("reward_risk_ratio")
    short_breakeven = short_rr_info.get("breakeven_win_rate")
    structural_short_stop = short_rr_info.get("structural_stop")
    structural_short_target = short_rr_info.get("structural_target")
    dist_short_resistance = short_rr_info.get("distance_to_resistance_atr")
    dist_short_support = short_rr_info.get("distance_to_support_atr")

    # Volatility risk profile -- deterministic ATR dampener, mirrors RiskAgent's HIGH threshold.
    vol_factor = QuantitativeScoringService.compute_vol_factor(atr_v, price_v)
    try:
        atr_pct = round((float(atr_v) / float(price_v if price_v > 0 else 1.0)) * 100.0, 2)
    except (TypeError, ValueError):
        atr_pct = 0.0

    # Earnings recency -- primary source is MarketAgent's yfinance calendar; fall back to the
    # Gloomberb earnings_date string when the calendar is unavailable.
    days_to_earnings = m_data.get("days_to_earnings")
    if days_to_earnings is None:
        days_to_earnings = days_from_earnings_date(
            gloomberb_payload.get("earnings", {}).get("earnings_date")
        )

    # Options Implied Volatility & Expected Move metrics
    options_data = gloomberb_payload.get("options") or {}
    iv_val = options_data.get("implied_volatility_pct") or m_data.get("implied_volatility_pct") or m_data.get("iv_pct")
    if iv_val is None:
        iv_val = QuantitativeScoringService._clean_float(
            options_data.get("implied_volatility") or m_data.get("implied_volatility"), None
        )
    expected_move = QuantitativeScoringService._clean_float(
        options_data.get("expected_move") or m_data.get("expected_move"), None
    )
    expected_move_pct = QuantitativeScoringService._clean_float(
        options_data.get("expected_move_pct") or m_data.get("expected_move_pct"), None
    )
    dte_opt = options_data.get("days_to_expiration") or horizon_days or 20
    if expected_move is None and iv_val is not None and iv_val > 0.0 and price_v > 0.0:
        em_calc = QuantitativeScoringService.compute_expected_move(
            price_v, iv_val, dte=dte_opt, atm_straddle=options_data.get("atm_straddle")
        )
        expected_move = em_calc.get("expected_move")
        expected_move_pct = em_calc.get("expected_move_pct")

    # Setup risk: Expected move vs Stop distance
    em_eval = QuantitativeScoringService.evaluate_expected_move_vs_stop(price_v, stop_v, expected_move)
    stop_distance = em_eval.get("stop_distance")
    stop_distance_pct = em_eval.get("stop_distance_pct")
    em_stop_ratio = em_eval.get("ratio")
    stop_inside_expected_move = bool(em_eval.get("stop_inside_expected_move"))

    # Short setup risk: Expected move vs overhead Short Stop distance
    short_em_eval = QuantitativeScoringService.evaluate_short_expected_move_vs_stop(
        price_v, structural_short_stop, expected_move
    )
    short_stop_distance = short_em_eval.get("stop_distance")
    short_stop_distance_pct = short_em_eval.get("stop_distance_pct")
    short_em_stop_ratio = short_em_eval.get("ratio")
    short_stop_inside_expected_move = bool(short_em_eval.get("stop_inside_expected_move"))

    # Scheduled company events beyond earnings (dividends, splits, investor days)
    events_payload = gloomberb_payload.get("events") or {}
    days_to_next_event = m_data.get("days_to_next_event")
    if days_to_next_event is None:
        days_to_next_event = events_payload.get("days_to_next_event")
    next_event_type = m_data.get("next_event_type") or events_payload.get("next_event_type") or "COMPANY_EVENT"
    next_event_date = m_data.get("next_event_date") or events_payload.get("next_event_date")

    days_to_ex_div = m_data.get("days_to_ex_dividend") or events_payload.get("days_to_ex_dividend")
    if days_to_ex_div is not None and (days_to_next_event is None or days_to_ex_div < days_to_next_event):
        days_to_next_event = days_to_ex_div
        next_event_type = "EX_DIVIDEND"
        next_event_date = m_data.get("ex_dividend_date") or events_payload.get("ex_dividend_date")

    event_risk_info = QuantitativeScoringService.evaluate_event_risk(
        days_to_earnings=days_to_earnings,
        days_to_next_event=days_to_next_event,
        next_event_type=next_event_type,
        next_event_date=next_event_date
    )

    # Multi-source Volatility Regime (combining ATR, IV, HV)
    hv_20d = _to_float(m_data.get("hv_20d"), None)
    vol_regime_info = QuantitativeScoringService.evaluate_volatility_regime(
        atr_v, price_v, implied_volatility=iv_val, historical_volatility=hv_20d
    )
    vol_regime = vol_regime_info.get("vol_regime", "NORMAL")

    # Decision origin: a trade's edge must come from price structure + composite + setup geometry,
    # never from a headline. News/macro/geopolitical events can confirm or veto, but cannot initiate.
    primary_driver = str(data.get("primary_driver") or "QUANT_STRUCTURE").upper()
    gates_applied = False

    def _append_risk(note):
        if note not in key_risks:
            key_risks.append(note)

    # Explicit risk checks beyond earnings blackout (unconditionally documented in key_risks)
    if stop_inside_expected_move:
        _append_risk(
            f"Option-implied expected move (${expected_move:.2f} / {expected_move_pct:.1f}%) "
            f"exceeds stop distance (${stop_distance:.2f} / {stop_distance_pct:.1f}%); "
            f"stop is placed inside expected market volatility noise."
        )
    elif em_stop_ratio is not None and em_stop_ratio >= 0.8:
        _append_risk(
            f"Elevated volatility buffer compression: stop distance (${stop_distance:.2f}) "
            f"is close to option-implied expected move (${expected_move:.2f}, ratio {em_stop_ratio:.2f}x)."
        )

    if event_risk_info.get("is_blackout") and event_risk_info.get("reason_code") == "EVENT_RISK":
        _append_risk(f"Scheduled company event risk: {event_risk_info.get('description')}.")
    elif days_to_next_event is not None and days_to_next_event <= 7:
        _append_risk(f"Upcoming scheduled company event: '{next_event_type}' in {days_to_next_event} day(s).")

    if vol_regime_info.get("is_extreme"):
        _append_risk(
            f"Extreme volatility regime detected ({'; '.join(vol_regime_info.get('reasons', []))}); "
            f"market pricing severe tail risk."
        )
    elif vol_regime == "ELEVATED" and vol_regime_info.get("reasons"):
        _append_risk(
            f"Elevated volatility regime ({'; '.join(vol_regime_info.get('reasons', []))}); "
            f"option implied volatility premium is heightened."
        )

    # Missing data visibility notes (unconditionally documented in key_risks)
    unavail_pillars = [k for k, st in pillar_status.items() if st == "unavailable"]
    if unavail_pillars:
        _append_risk(f"Missing data visibility: {', '.join(unavail_pillars)} pillar(s) unavailable; excluded from quantitative composite.")

    # Financial quality & value-trap alerts (unconditionally documented in key_risks)
    if is_value_trap:
        _append_risk(f"Value-Trap Alert: {'; '.join(value_trap_reasons or ['low valuation multiple masks balance-sheet solvency or cash-burn risk'])}.")
    elif fq_info.get("flags"):
        _append_risk(f"Financial quality notice: {'; '.join(fq_info.get('flags'))}.")

    # Step 4.2: Evidence Risk Flags & Deterministic Veto Triggers
    raw_risk_flags = data.get("risk_flags") or []
    evidence_veto_reasons = []
    if isinstance(raw_risk_flags, list):
        for rf in raw_risk_flags:
            if isinstance(rf, dict):
                r_name = rf.get("risk_name") or rf.get("flag_id", "Risk")
                r_sev = str(rf.get("severity", "LOW")).upper()
                r_veto = bool(rf.get("veto_trigger", False))
                r_cat = rf.get("category", "RISK")
                r_docs = rf.get("supporting_document_ids", [])
                doc_str = f" [Doc: {', '.join(r_docs)}]" if r_docs else ""

                if r_veto or r_sev == "CRITICAL":
                    note = f"Critical Evidence Risk Flag: [{r_cat}] {r_name} (Severity: {r_sev}){doc_str}"
                    _append_risk(note)
                    evidence_veto_reasons.append(note)
                elif r_sev == "HIGH":
                    _append_risk(f"High-Severity Evidence Risk: [{r_cat}] {r_name}{doc_str}.")

    # NO_TRADE reason codes: set whenever a deterministic gate vetoes a directional call.
    # None means the decision is the model's own signal, not a forced downgrade.
    no_trade_reason = None

    # Defense-in-depth: a directional call needs real probabilities. If none were provided
    # (missing/non-numeric/all-zero), the model cannot initiate a BUY/SELL -- degrade to HOLD.
    # Schema validation rejects this upstream in the pipeline; this catches any other path.
    if decision != "HOLD":
        prob_fields_present = all(
            data.get(k) is not None and isinstance(data.get(k), (int, float))
            for k in ("buy_score", "hold_score", "sell_score")
        )
        if not prob_fields_present or (buy_score == 0.0 and hold_score == 0.0 and sell_score == 0.0):
            decision = "HOLD"
            gates_applied = True
            no_trade_reason = "INSUFFICIENT_EVIDENCE"
            _append_risk(
                "Directional decision downgraded to HOLD: no valid buy/hold/sell probabilities were provided."
            )

    # Deterministic liquidity bar (BUY-only): a long needs tradeable turnover.
    LIQUIDITY_MIN_RVOL_20D = active_policy.trade_quality.min_rvol_20d
    LIQUIDITY_MIN_AVG_DOLLAR_VOL_20D = active_policy.trade_quality.min_avg_dollar_volume
    rvol_20d_val = _to_float(m_data.get("rvol_20d"), None)
    avg_dollar_vol_20d_val = _to_float(m_data.get("avg_dollar_vol_20d"), None)
    liquidity_ok = (
        (rvol_20d_val is None or rvol_20d_val >= LIQUIDITY_MIN_RVOL_20D)
        and (avg_dollar_vol_20d_val is None or avg_dollar_vol_20d_val >= LIQUIDITY_MIN_AVG_DOLLAR_VOL_20D)
    )

    # Catalyst overlay (additive). A headline alone still cannot initiate a trade: the certificate
    # must also prove a measured price/fundamental dislocation. Computed unconditionally for audit.
    catalyst_setup = evaluate_catalyst_setup(
        m_data,
        deterministic_scores,
        gloomberb_payload.get("news", []) or [],
        build_relevance_needles(symbol),
    )
    catalyst_qualified = bool(catalyst_setup.get("qualified"))
    catalyst_direction = catalyst_setup.get("direction")

    # Dedicated Trend Pullback Candidate Setup (Step 3: Qualified independently without news)
    pullback_setup = evaluate_pullback_setup(
        technical_data=m_data,
        deterministic_scores=deterministic_scores,
        policy=active_policy,
        is_value_trap=is_value_trap,
    )

    # A certificate only unlocks the direction it proves: DIP_BUY unlocks BUY/DEEP BUY; EXTENSION_SELL unlocks SELL/BOOK PROFIT.
    catalyst_allows = {
        DIP_BUY: ("BUY", "DEEP BUY"),
        EXTENSION_SELL: ("SELL", "BOOK PROFIT"),
    }.get(catalyst_direction, ())

    # Decouple deterministic catalyst setup generation from model's directional opinion (Step 4)
    catalyst_certificate_verified = (
        catalyst_qualified
        and (catalyst_direction in (DIP_BUY, EXTENSION_SELL))
        and primary_driver != "MACRO_EVENT"
    )

    raw_model_intent = str(data.get("decision") or "").upper()
    opposing_intent = (
        (raw_model_intent in ("SELL", "BOOK PROFIT") and catalyst_direction == DIP_BUY)
        or (raw_model_intent in ("BUY", "DEEP BUY") and catalyst_direction == EXTENSION_SELL)
    )
    is_catalyst_eligible_driver = primary_driver in ("NEWS_CATALYST", None, "", "CATALYST")

    if catalyst_certificate_verified and not opposing_intent and is_catalyst_eligible_driver:
        eff_catalyst_path = catalyst_direction
    else:
        eff_catalyst_path = None

    # MACRO_EVENT is never unlocks-able: the certificate is built from ticker-specific news, so a
    # macro-tagged trade is a misclassification of a move this system most wants to refuse.
    catalyst_path = (
        catalyst_certificate_verified
        and decision in catalyst_allows
        and primary_driver == "NEWS_CATALYST"
    )

    if decision in ("BUY", "DEEP BUY", "SELL", "BOOK PROFIT") and primary_driver in ("NEWS_CATALYST", "MACRO_EVENT"):
        if catalyst_path:
            _append_risk(
                f"Catalyst certificate '{catalyst_direction}' unlocked this {decision} despite "
                f"primary_driver '{primary_driver}': {catalyst_setup.get('reasons', [''])[0]} "
                f"All non-catalyst gates still apply."
            )
        else:
            requested = decision
            decision = "HOLD"
            gates_applied = True
            no_trade_reason = "INSUFFICIENT_EVIDENCE"
            catalyst_refusal = (catalyst_setup.get("reasons") or ["no certificate"])[0]
            _append_risk(
                f"{requested} capped to HOLD: primary_driver '{primary_driver}' is an external "
                f"event; news/geopolitical/macro headlines can confirm or veto but cannot initiate "
                f"a directional trade. Catalyst certificate check: {catalyst_refusal} "
                f"Re-anchor the trade on price structure + composite + setup geometry."
            )

    # Bounded lift: a catalyst may corroborate, never manufacture high conviction.
    if catalyst_path and confidence < CATALYST_CONFIDENCE_CEILING:
        confidence = round(
            min(confidence + CATALYST_CONFIDENCE_BOOST, CATALYST_CONFIDENCE_CEILING), 2
        )
        _append_risk(
            f"Confidence lifted to {confidence} by verified catalyst certificate '{catalyst_direction}' "
            f"(capped at {CATALYST_CONFIDENCE_CEILING})."
        )

    # Determine SELL intention: short-term bearish directional trade (SHORT) vs exiting/trimming long holding (LONG_EXIT)
    raw_sell_type = str(data.get("sell_type") or m_data.get("sell_type") or "").upper()
    is_explicit_holding = (
        bool(m_data.get("is_holding"))
        or _to_float(m_data.get("shares"), 0.0) > 0.0
        or str(m_data.get("position", "")).upper() in ("LONG", "HOLDING")
        or str(data.get("position_type", "")).upper() in ("LONG", "HOLDING")
        or str(data.get("action_type", "")).upper() in ("EXIT", "TRIM", "CLOSE", "REDUCE", "SELL_LONG", "TAKE_PROFIT")
    )
    if raw_sell_type in ("SHORT", "SHORT_SALE", "BEARISH_TRADE"):
        sell_type = "SHORT"
    elif raw_sell_type in ("LONG_EXIT", "EXIT", "TRIM", "REDUCE", "SELL_LONG", "TAKE_PROFIT"):
        sell_type = "LONG_EXIT"
    elif is_explicit_holding:
        sell_type = "LONG_EXIT"
    elif catalyst_path and catalyst_direction == EXTENSION_SELL and stop_v > 0 and stop_v < price_v:
        # Certified extension sell with long stops provided represents profit taking on long exposure
        sell_type = "LONG_EXIT"
    else:
        sell_type = "SHORT"

    if decision == "SELL" and sell_type == "SHORT":
        if short_stop_inside_expected_move:
            _append_risk(
                f"Option-implied expected move (${expected_move:.2f} / {expected_move_pct:.1f}%) "
                f"exceeds short overhead stop distance (${short_stop_distance:.2f} / {short_stop_distance_pct:.1f}%); "
                f"short overhead stop is placed inside expected market volatility noise."
            )
        elif short_em_stop_ratio is not None and short_em_stop_ratio >= 0.8:
            _append_risk(
                f"Elevated volatility buffer compression: short stop distance (${short_stop_distance:.2f}) "
                f"is close to option-implied expected move (${expected_move:.2f}, ratio {short_em_stop_ratio:.2f}x)."
            )
        if days_to_ex_div is not None and days_to_ex_div <= 7:
            _append_risk(
                f"Upcoming ex-dividend date in {days_to_ex_div} day(s); short sellers incur dividend liability and borrow recall risk."
            )

    # Deterministic BUY / DEEP BUY gates: momentum/valuation anchors cannot override setup geometry or risk.
    if decision in ("BUY", "DEEP BUY"):
        det_composite = decision_composite
        if det_composite is None:
            det_composite = deterministic_scores.get("raw_composite")
        # A certified catalyst dip buys a discount the composite cannot score, because the composite
        # reads the selloff itself as weak trend/alpha. That path carries its own, distinct floor
        # resolved deterministically via the policy precedence hierarchy.
        is_dip_req = (decision == "DEEP BUY") or catalyst_path
        buy_composite_floor, rule_name = active_policy.resolve_buy_score_floor(is_dip_req)
        try:
            composite_ok = det_composite is not None and float(det_composite) >= buy_composite_floor
        except (TypeError, ValueError):
            composite_ok = False

        price_v = _to_float(m_data.get("current_price"), 0.0)
        atr_v = _to_float(m_data.get("atr"), 0.0)
        stop_v = _to_float(m_data.get("suggested_stop_loss"), 0.0)
        target_v = _to_float(m_data.get("suggested_target_price"), 0.0)
        market_data_ok = (
            price_v > 0.0 and atr_v > 0.0 and stop_v > 0.0 and target_v > 0.0
            and (m_data.get("rsi14") is not None or m_data.get("rvol_20d") is not None)
        )

        # Adequate data coverage: reported completeness >= policy bar AND a deterministic composite was
        # actually computed (the model cannot certify a BUY on zero deterministic pillars).
        coverage_ok = (
            data_completeness >= active_policy.trade_quality.min_data_completeness
            and deterministic_scores.get("composite_quantitative_score") is not None
        )

        # Reward:risk must exist and clear the minimum bar (rr is None only if stop/target degenerate).
        rr_threshold = active_policy.trade_quality.reward_risk_minimum
        rr_ok = rr is not None and rr >= rr_threshold

        gate_reasons = []
        if not composite_ok:
            if det_composite is None:
                gate_reasons.append("deterministic composite could not be computed from market data")
            else:
                gate_reasons.append(f"deterministic composite {float(det_composite):.1f}/100 < {buy_composite_floor:.0f}")
        if not market_data_ok:
            gate_reasons.append("invalid or stale market data (price, ATR, stop, target must all be positive)")
        if not coverage_ok:
            if not deterministic_scores.get("composite_quantitative_score"):
                gate_reasons.append("data coverage inadequate: no deterministic composite could be computed from market data")
            else:
                gate_reasons.append(f"data coverage {data_completeness * 100:.0f}% < {active_policy.trade_quality.min_data_completeness * 100:.0f}% bar")
        if not rr_ok:
            if rr is None:
                gate_reasons.append("reward:risk ratio unavailable from setup geometry")
            else:
                gate_reasons.append(f"reward:risk ratio {rr:.2f} < {rr_threshold:.2f}")

        if is_value_trap:
            gate_reasons.append(f"value-trap risk ({'; '.join(value_trap_reasons or ['low valuation multiple masks balance-sheet solvency or cash-burn risk'])})")

        if evidence_veto_reasons:
            gate_reasons.append(f"deterministic evidence veto ({'; '.join(evidence_veto_reasons)})")

        if gate_reasons:
            decision = "HOLD"
            gates_applied = True
            if evidence_veto_reasons:
                no_trade_reason = "EVIDENCE_VETO"
            elif is_value_trap:
                no_trade_reason = "VALUE_TRAP"
            elif any("reward:risk" in r for r in gate_reasons):
                no_trade_reason = "RR_TOO_LOW"
            else:
                no_trade_reason = "INSUFFICIENT_EVIDENCE"
            _append_risk(
                "BUY downgraded to HOLD (deterministic BUY eligibility): " + "; ".join(gate_reasons) + "."
            )
        elif days_to_earnings is not None and days_to_earnings <= active_policy.events.earnings_blackout_sessions:
            if getattr(active_policy.events, "allow_pre_earnings_trading", False):
                _append_risk(
                    f"Pre-earnings event trading mode active ({days_to_earnings} session(s) to earnings report). "
                    f"50% position sizing haircut applied to mitigate discrete overnight gap risk."
                )
            else:
                decision = "HOLD"
                gates_applied = True
                no_trade_reason = "EARNINGS_BLACKOUT"
                note = (
                    f"{decision} downgraded to HOLD: earnings report in {days_to_earnings} day(s) is a "
                    f"binary gap-risk event; do not initiate a fresh position into it."
                )
                _append_risk(note)
        elif event_risk_info.get("is_blackout") and event_risk_info.get("reason_code") == "EVENT_RISK":
            decision = "HOLD"
            gates_applied = True
            no_trade_reason = "EVENT_RISK"
            note = (
                f"{decision} downgraded to HOLD: {event_risk_info.get('description')}; "
                f"do not initiate a fresh position into a scheduled company event."
            )
            _append_risk(note)
        elif vol_factor < active_policy.volatility.min_vol_factor or (active_policy.volatility.reject_extreme_regime and vol_regime_info.get("is_extreme")):
            decision = "HOLD"
            gates_applied = True
            no_trade_reason = "VOLATILITY_REGIME" if vol_regime_info.get("is_extreme") and vol_factor >= active_policy.volatility.min_vol_factor else "INSUFFICIENT_EVIDENCE"
            reasons_str = "; ".join(vol_regime_info.get("reasons", [])) if vol_regime_info.get("reasons") else f"ATR {atr_pct}% of price, vol factor {vol_factor:.2f}"
            note = (
                f"{decision} downgraded to HOLD: extreme volatility ({reasons_str}); "
                f"composite is dampened to {quant_score}/100."
            )
            _append_risk(note)
        elif stop_inside_expected_move:
            # Downgraded from hard veto to position sizing risk note & sizing dampener
            _append_risk(
                f"Elevated volatility noise: option-implied expected move (${expected_move:.2f} / {expected_move_pct:.1f}%) "
                f"exceeds stop distance (${stop_distance:.2f} / {stop_distance_pct:.1f}%). "
                f"Position size scaled down by 30% to account for normal volatility noise."
            )
        elif not liquidity_ok:
            decision = "HOLD"
            gates_applied = True
            no_trade_reason = "LOW_LIQUIDITY"
            note = (
                f"{decision} downgraded to HOLD: insufficient liquidity "
                f"(RVOL {rvol_20d_val if rvol_20d_val is not None else 'N/A'}x, "
                f"20d avg dollar volume "
                f"${avg_dollar_vol_20d_val if avg_dollar_vol_20d_val is not None else 'N/A'}); "
                f"entries/exits unreliable below "
                f"{LIQUIDITY_MIN_RVOL_20D}x RVOL / ${LIQUIDITY_MIN_AVG_DOLLAR_VOL_20D / 1_000_000:.0f}M."
            )
            _append_risk(note)
        elif analyst_rr is not None and analyst_rr < 1.0:
            # Downgraded from hard veto to confidence penalty & risk note
            confidence = round(max(0.50, confidence - 0.10), 2)
            _append_risk(
                f"Analyst consensus caution: Wall Street mean target "
                f"${_to_float(m_data.get('analyst_target_price'), 0.0):.2f} is below entry "
                f"${_to_float(m_data.get('current_price'), 0.0):.2f} (analyst RR {analyst_rr:.2f}); "
                f"confidence dampened by -0.10 to {confidence}."
            )
        elif evidence_veto_reasons:
            decision = "HOLD"
            gates_applied = True
            no_trade_reason = "EVIDENCE_VETO"
            note = (
                f"{decision} downgraded to HOLD by deterministic evidence veto: {'; '.join(evidence_veto_reasons)}."
            )
            _append_risk(note)

    elif decision == "BOOK PROFIT":
        # First-class profit-taking action (Step 5.4)
        sell_type = "LONG_EXIT"
        target_hit = target_v > 0.0 and price_v >= target_v
        overbought = (m_data.get("rsi14") is not None and float(m_data.get("rsi14")) >= 70.0)
        _append_risk(
            f"BOOK PROFIT certified: Locking in gains into extended strength/resistance "
            f"(Price ${price_v:.2f}, Target ${target_v:.2f}, RSI {m_data.get('rsi14')})."
        )

    elif decision == "SELL":
        det_composite = decision_composite
        if det_composite is None:
            det_composite = deterministic_scores.get("raw_composite")

        if sell_type == "LONG_EXIT":
            # EXIT / TRIM OF AN EXISTING LONG HOLDING
            # Allowed when stop is breached, target is reached, thesis is broken, extension certified, or de-risking.
            stop_hit = stop_v > 0.0 and price_v <= stop_v
            target_hit = target_v > 0.0 and price_v >= target_v
            thesis_broken = (
                (det_composite is not None and float(det_composite) < 50.0)
                or (pillar_scores.get("trend") is not None and float(pillar_scores.get("trend")) < 40.0)
            )
            extension_exit = catalyst_path and catalyst_direction == EXTENSION_SELL
            event_derisking = (days_to_earnings is not None and days_to_earnings <= 3) or (
                event_risk_info.get("is_blackout") and event_risk_info.get("reason_code") == "EVENT_RISK"
            )

            if not (stop_hit or target_hit or thesis_broken or extension_exit or event_derisking):
                # Holding thesis is fully intact -- prevent premature liquidation of a winning holding
                has_critical_evidence = bool(evidence_veto_reasons)
                if active_policy.is_holding_thesis_intact(is_explicit_holding, det_composite, has_critical_risk=has_critical_evidence):
                    decision = "HOLD"
                    gates_applied = True
                    no_trade_reason = "HOLDING_THESIS_INTACT"
                    _append_risk(
                        f"SELL (LONG_EXIT) downgraded to HOLD: holding thesis is intact "
                        f"(deterministic composite {float(det_composite):.1f} >= {active_policy.scores.holding_thesis_minimum:.1f}, "
                        f"price ${price_v:.2f} above stop ${stop_v:.2f}, target ${target_v:.2f} not reached). "
                        f"Maintain position."
                    )
        else:
            # SHORT-TERM BEARISH DIRECTIONAL TRADE (SHORT SALE)
            # Symmetric deterministic eligibility gates to BUY:
            short_composite_ceiling = (
                EXTENSION_MAX_COMPOSITE
                if (catalyst_path and catalyst_direction == EXTENSION_SELL)
                else min(active_policy.scores.short_maximum, SHORT_MAX_COMPOSITE)
            )
            try:
                composite_ok = det_composite is not None and float(det_composite) <= short_composite_ceiling
            except (TypeError, ValueError):
                composite_ok = False

            market_data_ok = (
                price_v > 0.0 and atr_v > 0.0
                and structural_short_stop is not None and structural_short_stop > price_v
                and structural_short_target is not None and structural_short_target < price_v
                and (m_data.get("rsi14") is not None or m_data.get("rvol_20d") is not None)
            )

            coverage_ok = (
                data_completeness >= active_policy.trade_quality.min_data_completeness
                and deterministic_scores.get("composite_quantitative_score") is not None
            )

            short_rr_threshold = active_policy.trade_quality.reward_risk_minimum
            short_rr_ok = short_rr is not None and short_rr >= short_rr_threshold

            gate_reasons = []
            if not composite_ok:
                if det_composite is None:
                    gate_reasons.append("deterministic composite could not be computed from market data")
                else:
                    gate_reasons.append(
                        f"deterministic composite {float(det_composite):.1f}/100 > {short_composite_ceiling:.0f} ceiling "
                        f"(requires structural breakdown <= {short_composite_ceiling:.0f} or certified extension <= {EXTENSION_MAX_COMPOSITE:.0f})"
                    )
            if not market_data_ok:
                gate_reasons.append("invalid or stale market data for short setup (valid price, ATR, short stop > entry, target < entry required)")
            if not coverage_ok:
                if not deterministic_scores.get("composite_quantitative_score"):
                    gate_reasons.append("data coverage inadequate: no deterministic composite could be computed from market data")
                else:
                    gate_reasons.append(f"data coverage {data_completeness * 100:.0f}% < {active_policy.trade_quality.min_data_completeness * 100:.0f}% bar")
            if not short_rr_ok:
                if short_rr is None:
                    gate_reasons.append("short reward:risk ratio unavailable from setup geometry")
                else:
                    gate_reasons.append(f"short reward:risk ratio {short_rr:.2f} < {short_rr_threshold:.2f}")

            if gate_reasons:
                decision = "HOLD"
                gates_applied = True
                no_trade_reason = (
                    "RR_TOO_LOW" if any("reward:risk" in r for r in gate_reasons) else "INSUFFICIENT_EVIDENCE"
                )
                _append_risk(
                    "SELL (SHORT) downgraded to HOLD (deterministic SHORT eligibility): " + "; ".join(gate_reasons) + "."
                )
            elif days_to_earnings is not None and days_to_earnings <= 3:
                decision = "HOLD"
                gates_applied = True
                no_trade_reason = "EARNINGS_BLACKOUT"
                note = (
                    f"SELL (SHORT) downgraded to HOLD: earnings report in {days_to_earnings} day(s) is a "
                    f"binary upside gap-risk event; do not initiate a fresh short position into it."
                )
                _append_risk(note)
            elif days_to_ex_div is not None and days_to_ex_div <= 2:
                decision = "HOLD"
                gates_applied = True
                no_trade_reason = "EVENT_RISK"
                note = (
                    f"SELL (SHORT) downgraded to HOLD: ex-dividend date in {days_to_ex_div} day(s); "
                    f"short sellers incur mandatory dividend payment liabilities and lender recall risk."
                )
                _append_risk(note)
            elif event_risk_info.get("is_blackout") and event_risk_info.get("reason_code") == "EVENT_RISK":
                decision = "HOLD"
                gates_applied = True
                no_trade_reason = "EVENT_RISK"
                note = (
                    f"SELL (SHORT) downgraded to HOLD: {event_risk_info.get('description')}; "
                    f"do not initiate a fresh short position into a scheduled company event."
                )
                _append_risk(note)
            elif vol_factor < 0.80 or vol_regime_info.get("is_extreme"):
                decision = "HOLD"
                gates_applied = True
                no_trade_reason = "VOLATILITY_REGIME" if vol_regime_info.get("is_extreme") and vol_factor >= 0.80 else "INSUFFICIENT_EVIDENCE"
                reasons_str = "; ".join(vol_regime_info.get("reasons", [])) if vol_regime_info.get("reasons") else f"ATR {atr_pct}% of price, vol factor {vol_factor:.2f}"
                note = (
                    f"SELL (SHORT) downgraded to HOLD: extreme volatility ({reasons_str}); "
                    f"short squeeze and margin tail risks are heightened."
                )
                _append_risk(note)
            elif short_stop_inside_expected_move:
                decision = "HOLD"
                gates_applied = True
                no_trade_reason = "EXPECTED_MOVE_EXCEEDS_STOP"
                note = (
                    f"SELL (SHORT) downgraded to HOLD: option-implied expected move (${expected_move:.2f} / {expected_move_pct:.1f}%) "
                    f"exceeds short overhead stop distance (${short_stop_distance:.2f} / {short_stop_distance_pct:.1f}%); "
                    f"overhead stop is placed inside normal volatility noise, creating high stop-out probability."
                )
                _append_risk(note)
            elif not liquidity_ok:
                decision = "HOLD"
                gates_applied = True
                no_trade_reason = "LOW_LIQUIDITY"
                note = (
                    f"SELL (SHORT) downgraded to HOLD: insufficient liquidity "
                    f"(RVOL {rvol_20d_val if rvol_20d_val is not None else 'N/A'}x, "
                    f"20d avg dollar volume "
                    f"${avg_dollar_vol_20d_val if avg_dollar_vol_20d_val is not None else 'N/A'}); "
                    f"short locates and exits are unreliable below "
                    f"{LIQUIDITY_MIN_RVOL_20D}x RVOL / ${LIQUIDITY_MIN_AVG_DOLLAR_VOL_20D / 1_000_000:.0f}M."
                )
                _append_risk(note)

    # Decision-probability consistency (Point 4): unless a hard gate overrode the model, the decision
    # must equal the argmax of its own buy/hold/sell probabilities. A unique max is required; ties keep
    # the model's stated decision.
    if not gates_applied:
        prob_map = {"BUY": buy_score, "HOLD": hold_score, "SELL": sell_score}
        top_score = max(prob_map.values())
        top_choices = [k for k, v in prob_map.items() if v == top_score]
        if len(top_choices) == 1 and top_choices[0] != decision:
            _append_risk(
                f"Decision realigned: probabilities imply {top_choices[0]} (buy {buy_score}/hold "
                f"{hold_score}/sell {sell_score}) but model stated {decision}."
            )
            decision = top_choices[0]
            decision_origin = "PROBABILITY_REALIGNED"

    if gates_applied:
        decision_origin = "GATE"

    # A gate/realignment that lands on HOLD must not carry the conviction of a directional call.
    if decision == "HOLD":
        confidence = round(min(confidence, 0.60), 2)

    # Map resolved decision to 5 internal action types (Step 5.4)
    if decision in ("BUY", "DEEP BUY"):
        if sell_type == "COVER_SHORT":
            internal_action = InternalActionType.COVER_SHORT.value
        else:
            internal_action = InternalActionType.BUY.value
    elif decision == "BOOK PROFIT":
        internal_action = InternalActionType.SELL_TO_CLOSE.value
    elif decision == "SELL":
        if sell_type == "LONG_EXIT":
            internal_action = InternalActionType.SELL_TO_CLOSE.value
        else:
            internal_action = InternalActionType.OPEN_SHORT.value
    else:
        internal_action = InternalActionType.HOLD.value

    # Market snapshot metrics for database and outcome tracking
    entry_price = m_data.get("current_price")
    stop_loss_price = m_data.get("suggested_stop_loss")
    target_price = m_data.get("target_price") or m_data.get("suggested_target_price")
    rsi14 = m_data.get("rsi14")
    rvol_20d = m_data.get("rvol_20d")

    # Macro & sentiment snapshots
    us_10y_yield = macro_data.get("us_10y_yield")
    yield_spread_10y2y = macro_data.get("yield_curve_spread_10y2y")
    macro_summary = macro_data.get("summary", "")
    fear_greed_score = gloomberb_payload.get("macro_econ", {}).get("fear_greed_score")

    # Summaries for backward compatibility in SQLite
    news_items = gloomberb_payload.get("news", [])
    news_summary = "; ".join(f"[{n.get('source', '')}] {n.get('title', '')}" for n in news_items[:3]) if news_items else "No news source available."
    
    analyst_info = gloomberb_payload.get("analyst_ratings", {})
    bank_cov = f"Consensus: {analyst_info.get('recommendation_rating', 'N/A')}, Target: ${analyst_info.get('mean_target_price', 'N/A')}"
    
    filings = gloomberb_payload.get("filings", [])
    inst_summary = "; ".join(f"{f.get('form')}: {f.get('summary')[:80]}" for f in filings[:2]) if filings else "Filings monitored."

    forward_pe = m_data.get("forward_pe", "N/A")

    # Explicit Candidate Qualification & Structured Decision Contract (Step 2)
    position_state = "LONG" if is_explicit_holding else "FLAT"

    # For short trades, use structural short stop and short target if available
    cand_short_stop = structural_short_stop if (structural_short_stop is not None and structural_short_stop > price_v) else (_to_float(m_data.get("suggested_short_stop_loss"), None) or stop_v)
    cand_short_target = structural_short_target if (structural_short_target is not None and structural_short_target < price_v) else (_to_float(m_data.get("suggested_short_target_price"), None) or target_v)

    # 5d drawdown
    dd_5d = _to_float(m_data.get("drawdown_5d_pct"), None)
    if dd_5d is None and m_data.get("change_5d_pct") is not None:
        try:
            dd_5d = abs(float(str(m_data.get("change_5d_pct")).replace("%", "")))
        except (ValueError, TypeError):
            dd_5d = None

    # eff_catalyst_path is computed above with decoupling and driver consistency checks
    cand_act, cand_typ, cand_ev = qualify_candidate_setup(
        ticker=stock,
        composite_score=decision_composite,
        position_state=position_state,
        current_price=price_v,
        stop_loss=stop_v,
        target_price=target_v,
        atr=atr_v,
        rsi14=rsi14,
        drawdown_5d_pct=dd_5d,
        pillar_scores=pillar_scores,
        reward_risk_ratio=rr,
        short_reward_risk_ratio=short_rr,
        catalyst_path=eff_catalyst_path,
        catalyst_setup=catalyst_setup,
        is_value_trap=is_value_trap,
        model_decision=model_decision,
        primary_driver=primary_driver,
        policy=active_policy,
        technical_data=m_data,
        pullback_setup=pullback_setup.to_dict(),
    )
    candidate_action = cand_act
    candidate_type = cand_typ
    setup_evidence = cand_ev

    # Determine effective stop and target for the candidate action
    if candidate_action == "SELL" and not is_explicit_holding:
        eval_stop = cand_short_stop
        eval_target = cand_short_target
    else:
        eval_stop = stop_v
        eval_target = target_v

    run_id_val = str(data.get("run_id") or m_data.get("run_id") or f"run-{stock}-{int(time.time())}")
    structured_dec = build_structured_decision(
        ticker=stock,
        candidate_decision=candidate_action,
        composite_score=decision_composite,
        position_state=position_state,
        current_price=price_v,
        stop_loss=eval_stop,
        target_price=eval_target,
        atr=atr_v,
        pillar_scores=pillar_scores,
        data_completeness=data_completeness,
        market_data_fresh=(price_v > 0 and atr_v > 0),
        ml_data_fresh=is_in_scope if ml_context else True,
        days_to_earnings=days_to_earnings,
        is_event_risk_blackout=bool(event_risk_info.get("is_blackout")),
        days_to_ex_dividend=days_to_ex_div,
        iv_pct=iv_val,
        rvol_20d=rvol_20d_val,
        avg_dollar_volume_20d=avg_dollar_vol_20d_val,
        borrow_available=True,
        has_critical_evidence_veto=bool(evidence_veto_reasons),
        catalyst_path=eff_catalyst_path,
        rsi14=rsi14,
        drawdown_5d_pct=dd_5d,
        valuation_history_score=pillar_scores.get("valuation_history"),
        is_value_trap=is_value_trap,
        model_status=model_status or ("IN_SCOPE" if is_in_scope else "OUT_OF_SCOPE"),
        run_id=run_id_val,
        policy=active_policy,
        portfolio_equity=_to_float(m_data.get("portfolio_equity", data.get("portfolio_equity")), None),
        risk_fraction=active_policy.portfolio_risk.max_risk_fraction_per_trade,
        candidate_action=candidate_action,
        candidate_type=candidate_type,
        model_decision=model_decision,
        setup_evidence=setup_evidence,
        is_extreme_volatility=bool(vol_regime_info.get("is_extreme")),
    )
    if no_trade_reason and no_trade_reason == "SCHEMA_INVALID":
        structured_dec.failed_gates = ["SCHEMA_VALIDATION"]
        structured_dec.reason_codes = ["SCHEMA_VALIDATION_FAILED"]
        structured_dec.rejection_reasons = ["SCHEMA_VALIDATION_FAILED"]
        structured_dec.decision = "HOLD"
        structured_dec.position_action = "NO_ACTION"

    # One Authoritative Decision Contract: Structured Decision is Final Action
    final_decision = structured_dec.decision
    decision = final_decision
    position_action = structured_dec.position_action
    recommendation = (
        "BUY" if final_decision in ("BUY", "DEEP BUY")
        else "SELL" if final_decision in ("SELL", "BOOK PROFIT")
        else "HOLD"
    )
    rejection_reasons = list(structured_dec.rejection_reasons)
    gate_results = {
        "passed_gates": list(structured_dec.passed_gates),
        "failed_gates": list(structured_dec.failed_gates),
        "reason_codes": list(structured_dec.reason_codes),
        "data_quality_status": str(structured_dec.data_quality_status),
    }

    if structured_dec.failed_gates:
        decision_origin = "GATE"
        if not no_trade_reason and structured_dec.reason_codes:
            rc = structured_dec.reason_codes[0]
            if rc in ("VOLATILITY_REGIME", StandardReasonCode.VOLATILITY_REGIME):
                no_trade_reason = "VOLATILITY_REGIME"
            elif rc in ("EVENT_RISK_BLACKOUT", StandardReasonCode.EVENT_RISK_BLACKOUT):
                no_trade_reason = "EVENT_RISK"
            elif rc in ("EARNINGS_BLACKOUT_ACTIVE", StandardReasonCode.EARNINGS_BLACKOUT_ACTIVE):
                no_trade_reason = "EARNINGS_BLACKOUT"
            elif rc in ("CONFIDENCE_BELOW_HURDLE", StandardReasonCode.CONFIDENCE_BELOW_HURDLE):
                no_trade_reason = "CONFIDENCE_HURDLE"
            else:
                no_trade_reason = rc
    elif candidate_type.startswith("MODEL_DISCRETIONARY"):
        decision_origin = "MODEL"
    elif final_decision == "HOLD" and candidate_type == "NO_SETUP":
        decision_origin = "NO_SETUP"
        if not no_trade_reason:
            no_trade_reason = "INSUFFICIENT_EVIDENCE"
    else:
        decision_origin = "QUANT_DETERMINISTIC"

    if final_decision == "HOLD":
        confidence = round(min(confidence, 0.60), 2)

    # Realign internal action with authoritative decision
    if final_decision in ("BUY", "DEEP BUY"):
        internal_action = InternalActionType.COVER_SHORT.value if (is_explicit_holding and sell_type == "COVER_SHORT") else InternalActionType.BUY.value
    elif final_decision == "BOOK PROFIT":
        internal_action = InternalActionType.SELL_TO_CLOSE.value
    elif final_decision == "SELL":
        internal_action = InternalActionType.SELL_TO_CLOSE.value if (is_explicit_holding or sell_type == "LONG_EXIT") else InternalActionType.OPEN_SHORT.value
    else:
        internal_action = InternalActionType.HOLD.value

    # Backward compatibility for SQLite DB string storage
    staged_entry_plan = None
    if final_decision in ("BUY", "DEEP BUY"):
        equity_for_plan = _to_float(m_data.get("portfolio_equity", data.get("portfolio_equity")), None)
        if equity_for_plan and equity_for_plan > 0 and price_v > 0 and eval_stop and eval_stop < price_v:
            from quant.staged_entries import build_staged_entry_plan

            staged_entry_plan = build_staged_entry_plan(
                entry_price=price_v,
                stop_loss=eval_stop,
                ticker=stock,
                portfolio_equity=equity_for_plan,
                risk_fraction=active_policy.portfolio_risk.max_risk_fraction_per_trade,
                atr=atr_v if atr_v > 0 else None,
                avg_dollar_volume_20d=avg_dollar_vol_20d_val,
                composite_score=decision_composite,
                policy=active_policy,
            ).to_dict()

    return {
        "stock": _ensure_str(stock),
        "recommendation": recommendation,
        "decision": final_decision,
        "final_decision": final_decision,
        "model_decision": model_decision,
        "candidate_action": candidate_action,
        "candidate_type": candidate_type,
        "setup_evidence": setup_evidence,
        "gate_results": gate_results,
        "rejection_reasons": rejection_reasons,
        "position_action": position_action,
        "staged_entry_plan": staged_entry_plan,
        "decision_origin": decision_origin,
        "confidence": confidence,
        "signal_strength": confidence,
        "is_calibrated_probability": False,
        "model_confidence": model_confidence,
        "buy_score": buy_score,
        "hold_score": hold_score,
        "sell_score": sell_score,
        "horizon_days": horizon_days,
        "horizon_sessions": horizon_sessions,
        "forecast_20d": forecast_20d,
        "quant_score": quant_score,
        "model_quant_score": model_quant_score,
        "no_trade_reason": no_trade_reason,
        "deterministic_data_completeness": deterministic_completeness,
        "model_data_completeness": model_data_completeness,
        "ml_signal": ml_context,
        "ml_signal_assessment": data.get("ml_signal_assessment"),
        "deterministic_quant_score": _to_float(det_quant, None),
        "ml_score_0_100": ml_score,
        "ml_weight": ml_weight,
        "combined_quant_score": decision_composite,
        "raw_composite": deterministic_scores.get("raw_composite"),
        "vol_factor": vol_factor,
        "atr_pct": atr_pct,
        "primary_driver": primary_driver,
        "falsification_bull": str(data.get("falsification_bull") or ""),
        "pillar_scores": pillar_scores,
        "model_pillar_scores": model_pillar_scores,
        "pillar_status": pillar_status,
        "pillar_input_tracking": pillar_input_tracking,
        "financial_quality": fq_info,
        "financial_quality_score": fq_score,
        "is_value_trap": is_value_trap,
        "value_trap_reasons": value_trap_reasons,
        "reward_risk_ratio": rr,
        "breakeven_win_rate": breakeven,
        "analyst_target_rr": analyst_rr,
        "structural_stop_price": rr_info.get("structural_stop"),
        "structural_target_price": rr_info.get("structural_target"),
        "distance_to_resistance_atr": dist_resistance,
        "bull_case": bull_case,
        "bear_case": bear_case,
        "key_risks": key_risks,
        "missing_information": missing_info,
        # These richer prompt outputs are retained in raw_json for downstream consumers.
        "catalyst_analysis": catalyst_analysis,
        "biggest_surprise_catalyst": biggest_surprise_catalyst,
        "thesis_assumptions": thesis_assumptions,
        "investor_questions": investor_questions,
        "valuation_assessment": valuation_assessment,
        "price_level_map": price_level_map,
        "data_completeness": data_completeness,
        "entry_price": entry_price,
        "stop_loss_price": stop_loss_price,
        "target_price": target_price,
        "rsi14": rsi14,
        "rvol_20d": rvol_20d,
        "us_10y_yield": us_10y_yield,
        "yield_spread_10y2y": yield_spread_10y2y,
        "fear_greed_score": fear_greed_score,
        "days_to_earnings": days_to_earnings,
        "implied_volatility": iv_val,
        "expected_move": expected_move,
        "expected_move_pct": expected_move_pct,
        "stop_distance": stop_distance,
        "stop_distance_pct": stop_distance_pct,
        "expected_move_stop_ratio": em_stop_ratio,
        "volatility_regime": vol_regime,
        "days_to_next_event": days_to_next_event,
        "next_event_type": next_event_type,
        "next_event_date": next_event_date,
        "internal_action": internal_action,
        "sell_type": sell_type if (original_decision == "SELL" or decision == "SELL") else None,
        "short_reward_risk_ratio": short_rr,
        "structural_short_stop_price": structural_short_stop,
        "structural_short_target_price": structural_short_target,
        "short_stop_distance": short_stop_distance,
        "short_stop_distance_pct": short_stop_distance_pct,
        "short_expected_move_stop_ratio": short_em_stop_ratio,
        # Step 4.2: Evidence & qualitative risk analysis schema outputs
        "catalysts": data.get("catalysts", []),
        "risk_flags": data.get("risk_flags", []),
        "contradictions": data.get("contradictions", []),
        "supporting_document_ids": data.get("supporting_document_ids", []),
        "summary": data.get("summary", ""),
        "advisory_action": data.get("advisory_action"),
        "analysis_status": data.get("analysis_status"),
        "as_of": data.get("as_of"),
        # Step 6.4: Structured Decision Output (Independent of rendered narrative)
        "run_id": structured_dec.run_id,
        "model_status": structured_dec.model_status,
        "passed_gates": structured_dec.passed_gates,
        "failed_gates": structured_dec.failed_gates,
        "data_quality_status": structured_dec.data_quality_status,
        "position_action": structured_dec.position_action,
        "reason_codes": structured_dec.reason_codes,
        "structured_decision": structured_dec.to_dict(),
        # Step 7.1: Risk-Based Position Sizing (Independent of score)
        "position_sizing": structured_dec.metrics.get("position_sizing", {}),
        # Danelfin AI Intelligence metrics
        "danelfin": m_data.get("danelfin", {}),
        "danelfin_score": m_data.get("danelfin", {}).get("aiscore"),
        "danelfin_signal": m_data.get("danelfin", {}).get("signal", ""),
        # Explicit Model Decision Reasoning & backward compatibility
        "decision_reasoning": decision_reasoning,
        "reason": decision_reasoning,
        "risk_assessment": "; ".join(key_risks),
        "institutional_data": inst_summary,
        "macro_data": macro_summary,
        "news": news_summary,
        "investment_bank_coverage": bank_cov,
        "PE_and_PEG": str(forward_pe)
    }





def run_portfolio_review(llm_choice: str, temperature: float = None, reasoning_budget: int = None, reasoning_effort: str = None) -> dict:
    """
    Portfolio risk-review mode: reads the terminal portfolio (raw, as-is), injects the live
    macro/Fed rate outlook (mirroring the stock workflow's interest_rate_outlook enrichment),
    builds the portfolio-risk prompt, runs the chosen LLM, persists the review, and returns the parsed JSON.
    """
    from rag_service import RAGService
    from macro_agent import MacroDataAgent

    init_db()

    print("[Portfolio] Fetching terminal portfolio via Gloomberb CLI...")
    portfolio = GloomberbService().get_portfolio()
    if not portfolio:
        print("[Portfolio] No portfolio data available from Gloomberb CLI. Nothing to review.")
        return {}

    # Enrich the portfolio payload with live Fed/rate data (FRED + CFTC), matching the
    # interest_rate_outlook block the single-stock workflow injects. Best-effort: if the
    # macro feed fails, the payload simply omits it and the model reports it as missing.
    try:
        macro_data = MacroDataAgent().analyze()
        interest_rate_outlook = {
            "yield_10y": macro_data.get("us_10y_yield", "N/A"),
            "yield_2y": macro_data.get("us_2y_yield", "N/A"),
            "yield_curve_spread_2y10y": macro_data.get("yield_curve_spread_10y2y", "N/A"),
            "yield_curve_status": macro_data.get("yield_curve_status", "Unknown"),
            "yield_10y_real": macro_data.get("us_10y_real_yield", "N/A"),
            "yield_10y_5d_change": macro_data.get("us_10y_yield_5d_change", "N/A"),
            "fed_funds_rate": macro_data.get("fed_funds_rate", "N/A"),
            "unemployment_rate": macro_data.get("unemployment_rate", "N/A"),
            "cftc_cot": macro_data.get("cftc_cot_summary", "")
        }
        if isinstance(portfolio, dict):
            portfolio = dict(portfolio)
            portfolio.setdefault("macro_econ", {})["interest_rate_outlook"] = interest_rate_outlook
            print("[Portfolio] Injected live Fed/rate macro outlook into portfolio payload.")
    except Exception as e:
        print(f"[Portfolio] Warning: could not inject macro rate data ({e}); the model will report rates as missing.")

    rag_service = RAGService()
    prompt_profile = "compact" if uses_compact_prompt_profile(llm_choice) else "full"
    payload = rag_service.get_portfolio_analysis_payload(
        portfolio,
        prompt_profile=prompt_profile,
    )

    print(f"[Portfolio] Executing portfolio risk review with {get_model_label(llm_choice)}...")
    res_content = query_llm(
        system_instruction=payload["system_instruction"],
        user_prompt=payload["user_prompt"],
        model_choice=llm_choice,
        temperature=temperature,
        reasoning_budget=reasoning_budget,
        reasoning_effort=reasoning_effort
    )

    try:
        parsed = extract_json(res_content)
        if not isinstance(parsed, dict):
            parsed = {"review": parsed, "raw": res_content}
    except Exception as e:
        print(f"[Portfolio] Failed to parse LLM output as JSON: {e}")
        parsed = {"parse_error": str(e), "raw": res_content}

    review_id = save_portfolio_review(parsed, portfolio, model_used=get_model_label(llm_choice))
    print(f"[Portfolio] Saved portfolio review #{review_id} to database.")
    print("\n================ Portfolio Risk Review ================")
    print(json.dumps(parsed, indent=2))
    return parsed


def main():
    parser = argparse.ArgumentParser(
        description="6-Agent Stock Swing Trading Analysis Pipeline (Powered by Recursive: Flint (free) via OpenRouter)",
        usage="python main.py [flint] [ticker] | python main.py portfolio"
    )
    parser.add_argument(
        "model_arg",
        nargs="?",
        default=None,
        help="Model short name (default: 'flint' - Recursive: Flint (free)) or stock ticker symbol"
    )
    parser.add_argument(
        "ticker_arg",
        nargs="?",
        default=None,
        help="Optional stock ticker symbol (e.g. AAPL, NVDA) or comma-separated list of tickers"
    )
    parser.add_argument(
        "--model", "-m",
        dest="model_opt",
        default=None,
        help="Model short name (default: 'flint' - Recursive: Flint (free))"
    )
    parser.add_argument(
        "--ticker", "-t",
        dest="ticker_opt",
        default=None,
        help="Stock ticker symbol (e.g. AAPL, NVDA) or comma-separated list of tickers"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Model sampling temperature (default: 0.2)"
    )
    parser.add_argument(
        "--reasoning-budget",
        type=int,
        default=None,
        help="Internal reasoning token budget"
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        choices=["low", "medium", "high", "max"],
        default=None,
        help="Reasoning effort level"
    )
    parser.add_argument(
        "--market",
        choices=["US", "EU"],
        default="US",
        help="Target market: US or EU (Europe - gettex / XETRA) (default: US)"
    )
    parser.add_argument(
        "--eu",
        action="store_true",
        help="Shorthand for European market (--market EU)"
    )
    parser.add_argument(
        "--prefer-exchange",
        choices=["DE", "MU", "F", "HA", "TG"],
        default="DE",
        help="Preferred European exchange suffix (DE=XETRA, MU=gettex, HA=Hannover/EIX) (default: DE)"
    )
    args = parser.parse_args()

    raw_first = args.model_arg
    raw_second = args.ticker_arg
    raw_model_opt = args.model_opt
    raw_ticker_opt = args.ticker_opt

    # If first argument is 'portfolio', run portfolio review
    if (raw_first and raw_first.lower() == "portfolio") or (raw_model_opt and raw_model_opt.lower() == "portfolio"):
        llm_choice = raw_model_opt or "flint"
        with managed_llamacpp_server(llm_choice):
            run_portfolio_review(
                llm_choice,
                temperature=args.temperature,
                reasoning_budget=args.reasoning_budget,
                reasoning_effort=args.reasoning_effort
            )
        return

    known_model_names = {
        "flint", "flint-v1.0", "recursive/flint-v1.0:free", "recursive", "flint-free",
        "free", "openrouter/free", "openrouter", "auto", "or",
        "xingchen", "xing", "xing4", "xingchen-agi/xing4.0-29b-a4b:free",
        "kimi", "kimi-k3", "nemotron", "ultra", "super",
        "gemini", "ling", "qwen", "llamacpp",
        "haiku", "haiku-5.5", "haike", "haike-5.5", "claude", "claude-haiku", "claude-haiku-5-5", "claude-5-5-haiku", "anthropic"
    }

    if raw_first and raw_second:
        if raw_first.lower() in known_model_names:
            model_choice = raw_first.lower()
        else:
            model_choice = "flint"
        raw_ticker = raw_ticker_opt or raw_second
    elif raw_first and not raw_second:
        if raw_first.lower() in known_model_names:
            model_choice = raw_first.lower()
            raw_ticker = raw_ticker_opt
        else:
            # e.g. python main.py AAPL
            model_choice = "flint"
            raw_ticker = raw_first
    else:
        model_choice = raw_model_opt or "flint"
        raw_ticker = raw_ticker_opt

    if not raw_ticker:
        print("\nUsage: python main.py [flint] <TICKER> (e.g. python main.py AAPL)")
        print("       python main.py portfolio")
        print("Model: Recursive: Flint (free) from OpenRouter (recursive/flint-v1.0:free)\n")
        sys.exit(1)

    market = "EU" if args.eu else args.market.upper()
    force_eu = (market == "EU")
    # DE is the CLI's EU preference default. Keep US tickers on their US listing
    # unless the caller explicitly selects the EU market.
    resolve_prefer_exchange = args.prefer_exchange if force_eu else "AUTO"
    if raw_ticker:
        raw_list = [t.strip().upper() for t in raw_ticker.split(",") if t.strip()]
        watchlist = [
            resolve_symbol(t, prefer_exchange=resolve_prefer_exchange, force_european=force_eu)["symbol"]
            for t in raw_list
        ]
    else:
        watchlist = WATCHLIST

    with managed_llamacpp_server(model_choice):
        model_label = get_model_label(model_choice)

        print(f"=== Initializing Gloomberb RAG & Technical Analysis Pipeline (Model: {model_label}) ===")
        print(f"Target Watchlist: {', '.join(watchlist)}")
        market_agent = MarketAgent()
        macro_agent = MacroDataAgent()
        gloomberb_service = GloomberbService()
        institutional_service = InstitutionalDataService()
        rag_service = RAGService()
        danelfin_service = DanelfinService()

        all_results = []
        prompt_profile = "compact" if uses_compact_prompt_profile(model_choice) else "full"

        for idx, symbol in enumerate(watchlist, 1):
            print(f"\n--- [{idx}/{len(watchlist)}] Processing {symbol} ---")

            # 1. Technical Data Collection (RSI / EMA / ATR / Volume / RVOL / Channels)
            m_data = market_agent.analyze(
                symbol,
                prefer_exchange=resolve_prefer_exchange,
                force_european=force_eu,
            )
            if not m_data:
                print(f"Skipping {symbol}: Insufficient price data.")
                continue

            # Danelfin AI Intelligence (Cached, quota-protected)
            danelfin_scores = danelfin_service.get_scores(symbol)
            if danelfin_scores and danelfin_scores.get("aiscore") is not None:
                m_data["danelfin"] = danelfin_scores
                print(f"[Danelfin] AI Score for {symbol}: {danelfin_scores.get('aiscore')}/10 (Source: {danelfin_scores.get('source')})")

            # 2. Macro Data Stream (FRED Yield Curve, Spreads, 10Y Real Yield, 5d Velocity, Fed Funds, CFTC COT)
            macro_data = macro_agent.analyze(symbol)

            # 3. Gloomberb Data Source Stream (News, Filings, Financials, Options, Insiders, Peer Valuation)
            gloomberb_payload = gloomberb_service.get_all_gloomberb_data(symbol)
            if macro_data:
                macro_econ = gloomberb_payload.setdefault("macro_econ", {})
                iro = macro_econ.setdefault("interest_rate_outlook", {})
                if iro.get("yield_10y") in ["N/A", None] and macro_data.get("us_10y_yield") != "N/A":
                    iro["yield_10y"] = macro_data.get("us_10y_yield")
                if iro.get("yield_2y") in ["N/A", None] and macro_data.get("us_2y_yield") != "N/A":
                    iro["yield_2y"] = macro_data.get("us_2y_yield")
                if iro.get("yield_curve_spread_2y10y") in ["N/A", None] and macro_data.get("yield_curve_spread_10y2y") != "N/A":
                    iro["yield_curve_spread_2y10y"] = macro_data.get("yield_curve_spread_10y2y")
                if iro.get("yield_curve_status") in ["N/A", None] and macro_data.get("yield_curve_status") != "N/A":
                    iro["yield_curve_status"] = macro_data.get("yield_curve_status")
                if macro_data.get("us_10y_real_yield"):
                    iro["yield_10y_real"] = macro_data.get("us_10y_real_yield")
                if macro_data.get("us_10y_yield_5d_change"):
                    iro["yield_10y_5d_change"] = macro_data.get("us_10y_yield_5d_change")
                if iro.get("fed_funds_rate") in ["N/A", None] and macro_data.get("fed_funds_rate") != "N/A":
                    iro["fed_funds_rate"] = macro_data.get("fed_funds_rate")
                macro_econ["cftc_cot"] = macro_data.get("cftc_cot_summary", "")

            # 4. Institutional Multi-Source Data Stream (IR, SEC direct, Earnings calls, Press releases, Reputable news)
            institutional_payload = institutional_service.get_all_institutional_data(symbol)

            # 4.1 ML Context & Quantitative Scoring setup
            ml_context = get_ml_context(symbol)
            if isinstance(m_data, dict):
                m_data = dict(m_data)
                m_data["ml_context"] = ml_context

            # 5. Dense Vector Embedding RAG & Prompt Payload Assembly
            context = rag_service.get_nemotron_payload(
                gloomberb_payload,
                technical_data=m_data,
                institutional_data=institutional_payload,
                prompt_profile=prompt_profile,
            )

            # Compute and attach deterministic 5-pillar scores to m_data so normalize can fall back to them
            # instead of defaults (and so gated_confidence uses real pillar agreement).
            quant_service = QuantitativeScoringService()
            sector_bench = gloomberb_payload.get("sector_benchmark", {})
            deterministic_scores = quant_service.compute_5pillar_scores(m_data, gloomberb_payload, sector_bench)

            if isinstance(m_data, dict):
                m_data["deterministic_5pillar_scores"] = deterministic_scores
                # Deterministic data completeness: computed from live provider availability once.
                # The LLM's own claim (model_data_completeness) is stored separately and never authoritative.
                m_data["deterministic_data_completeness"] = QuantitativeScoringService.compute_data_coverage(
                    m_data, macro_data, gloomberb_payload, institutional_payload
                )

            # 5.1 Inject authoritatively formatted Quantitative Intelligence (Deterministic Quant Score + ML 20-Session Score + Blended Composite)
            context["user_prompt"] += format_quant_and_ml_context(
                deterministic_scores=deterministic_scores,
                ml_context=ml_context,
                m_data=m_data,
                symbol=symbol,
            )
            if danelfin_scores:
                context["user_prompt"] += format_danelfin_context(danelfin_scores)

            # 6. Model Reasoning Core & Market Analysis
            print(f"[{model_label}] Executing market analysis for {symbol}...")

            try:
                res_content = query_llm(
                    system_instruction=context["system_instruction"],
                    user_prompt=context["user_prompt"],
                    model_choice=model_choice,
                    temperature=args.temperature,
                    reasoning_budget=args.reasoning_budget,
                    reasoning_effort=args.reasoning_effort
                )
                parsed = extract_json(res_content)

                res_obj = None
                if isinstance(parsed, dict):
                    res_obj = parsed
                elif isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
                    res_obj = parsed[0]

                schema_failed = False
                retrieved_doc_ids = {
                    r.get("document_id") for r in context.get("evidence_records", [])
                    if isinstance(r, dict) and r.get("document_id")
                } or None

                if res_obj:
                    validated, errors = validate_signal_json(
                        res_obj,
                        require_ml_assessment=bool(ml_context and ml_context.get("in_scope", True)),
                        required_horizon_sessions=20,
                        require_forecast_20d=True,
                        expected_ticker=symbol,
                        valid_document_ids=retrieved_doc_ids,
                    )

                    # One constrained corrective retry on strict schema violations.
                    if errors:
                        print(f"[{model_label}] Schema validation failed for {symbol}; issuing one corrective retry...")
                        ml_req_str = "- ml_signal_assessment: object with 'disposition' ('SUPPORTIVE', 'CONFLICTING', or 'NEUTRAL') and 'reason' (explanation string)\n" if (ml_context and ml_context.get("in_scope", True)) else ""
                        retry_prompt = (
                            f"{context['user_prompt']}\n\n"
                            "=== SCHEMA VALIDATION FAILED -- RESUBMIT ONLY THE FIXED JSON ===\n"
                            "Your previous response failed strict schema validation. Correct EVERY "
                            "violation below and return ONLY a single valid JSON object with the required schema fields:\n"
                            f"- stock: \"{symbol}\"\n"
                            "- decision: \"BUY\", \"HOLD\", or \"SELL\"\n"
                            "- decision_reasoning: detailed explanation string explaining why this decision was reached\n"
                            "- primary_driver: \"QUANT_STRUCTURE\", \"FUNDAMENTAL\", \"NEWS_CATALYST\", \"MACRO_EVENT\", or \"EARNINGS_CATALYST\"\n"
                            "- confidence: float between 0.0 and 1.0\n"
                            "- buy_score, hold_score, sell_score: floats between 0.0 and 1.0 summing to 1.0\n"
                            "- horizon_days: 20\n"
                            "- horizon_sessions: 20\n"
                            "- quant_score: float between 0.0 and 100.0\n"
                            "- forecast_20d: object with 'relative_outlook' ('OUTPERFORM', 'MARKET_LIKE', or 'UNDERPERFORM'), 'confidence' (float 0.0 to 1.0), and 'rationale' (string)\n"
                            "- pillar_scores: object with 'trend', 'sector', 'alpha', 'valuation_history', 'peer_valuation' (each 0.0 to 100.0)\n"
                            "- data_completeness: float between 0.0 and 1.0\n"
                            "- falsification_bull: string\n"
                            "- falsification_bear: string\n"
                            "- bull_case: list of strings\n"
                            "- bear_case: list of strings\n"
                            "- key_risks: list of strings\n"
                            "- missing_information: list of strings\n"
                            f"{ml_req_str}"
                            "Do not include unauthorized broker execution commands.\n"
                            f"Violations to correct:\n- " + "\n- ".join(errors) + "\n"
                        )
                        retry_content = query_llm(
                            system_instruction=context["system_instruction"],
                            user_prompt=retry_prompt,
                            model_choice=model_choice,
                            temperature=args.temperature,
                            reasoning_budget=args.reasoning_budget,
                            reasoning_effort=args.reasoning_effort
                        )
                        retry_parsed = extract_json(retry_content)
                        retry_obj = None
                        if isinstance(retry_parsed, dict):
                            retry_obj = retry_parsed
                        elif isinstance(retry_parsed, list) and len(retry_parsed) > 0 and isinstance(retry_parsed[0], dict):
                            retry_obj = retry_parsed[0]
                        if retry_obj:
                            validated, errors = validate_signal_json(
                                retry_obj,
                                require_ml_assessment=bool(ml_context and ml_context.get("in_scope", True)),
                                required_horizon_sessions=20,
                                require_forecast_20d=True,
                                expected_ticker=symbol,
                                valid_document_ids=retrieved_doc_ids,
                            )

                    if not errors:
                        normalized_obj = normalize_master_trader_json(
                            validated,
                            symbol,
                            m_data=m_data,
                            macro_data=macro_data,
                            gloomberb_payload=gloomberb_payload
                        )
                    else:
                        # Audit path: a still-invalid signal never passes through normalize unvalidated.
                        # Record a deterministic HOLD so the event is visible, not silently dropped.
                        schema_failed = True
                        fallback_res = {
                            "decision": "HOLD",
                            "primary_driver": "QUANT_STRUCTURE",
                            "buy_score": 0.0,
                            "hold_score": 1.0,
                            "sell_score": 0.0,
                            "horizon_days": 20,
                            "horizon_sessions": 20,
                            "quant_score": None,
                            "data_completeness": None,
                            "missing_information": [
                                "Model output failed strict schema validation; reverted to HOLD."
                            ],
                            "key_risks": [
                                "Schema validation failed after one corrective retry: " + "; ".join(errors)
                            ],
                        }
                        normalized_obj = normalize_master_trader_json(
                            fallback_res,
                            symbol,
                            m_data=m_data,
                            macro_data=macro_data,
                            gloomberb_payload=gloomberb_payload
                        )
                        normalized_obj["no_trade_reason"] = "SCHEMA_INVALID"
                        normalized_obj["decision"] = "HOLD"
                        normalized_obj["decision_origin"] = "SCHEMA_INVALID"

                    # Step 4.1: Attach auditable evidence records to normalized signal
                    normalized_obj["evidence_records"] = context.get("evidence_records", [])

                    all_results.append(normalized_obj)
                    if schema_failed:
                        print(f"[{model_label}] Schema failure for {symbol}; recorded deterministic HOLD (SCHEMA_INVALID).")
                    else:
                        print(f"[{model_label}] Final Decision for {symbol}: {normalized_obj.get('decision')} (Conf: {normalized_obj.get('confidence')})")
                else:
                    print(f"Warning: Model returned empty output for {symbol}.")

            except Exception as e:
                print(f"Model synthesis error for {symbol}: {e}")

    if not all_results:
        print("\nNo analysis results generated.")
        sys.exit(1)

    print("\n================ Aggregated Analysis Output ================")
    print(json.dumps(all_results, indent=2))

    # Auto-save results to SQLite DB
    save_results(all_results, model_used=model_label)

    # Evaluate forward outcomes for past historical signals
    try:
        evaluated = update_signal_outcomes()
        if evaluated:
            print(f"[Outcome Tracker] Evaluated {evaluated} historical trade signals.")
    except Exception as e:
        print(f"[Outcome Tracker] Notice: {e}")



if __name__ == "__main__":
    main()
