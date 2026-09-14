import sys
import argparse
import os
import json
import time
import requests
from dotenv import load_dotenv

from llm_service import query_llm, get_model_label, extract_json
from db import init_db, save_results, update_signal_outcomes
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
from signal_schema import validate_signal_json

WATCHLIST = ["000660.KS"]

load_dotenv()
CHAT_ID = os.getenv("TELEGRAM_CHANNEL_CHAT_ID", "969601315")
telegram_token = os.getenv("TELEGRAM_CHANNEL_API_TOKEN")

MASTER_TRADER_INSTRUCTION = """
You are a senior Master Trader & Portfolio Manager.

You receive short-term intelligence collected by six specialized sub-agents:
1. Market Agent: Price action, 5-day velocity, RSI, EMA20/50, ATR, Forward P/E, profit margins.
2. Institutional Data Agent: SEC EDGAR regulatory filings (Form 4 insider transactions, 10-K/10-Q reports, 8-K material events, 13D/13G/13F institutional ownership).
3. Macro Data Agent (marco_data): FRED macroeconomic indicators (10Y Treasury Yield, 10Y-2Y yield curve spread, Fed Funds Rate, Unemployment) and CFTC Commitment of Traders (COT) futures positioning (S&P 500, Nasdaq 100, VIX).
4. News Agent: Today's real-time headlines, news summary, sentiment, and short-term impact.
5. Risk Agent: Volatility ATR%, technical trend breakdown risk, RSI extreme risk, and stop-loss boundaries.
6. Analyst Agent: Wall Street consensus, mean target price, and recent rating actions from major investment banks (BofA, Barclays, Credit Suisse, DB, Evercore, Goldman, JPM, Morgan Stanley, UBS).

Your Task:
Synthesize the insights from all six agents to deliver a definitive short-term swing trade decision (1 to 10 trading days horizon).

Rules & Guidelines:
1. Ignore stale long-term macro narratives that are already priced in, but incorporate macro yield curve & CFTC positioning context.
2. Synthesize Institutional/SEC filings + Macro FRED/COT data + News Agent impact + Market Agent momentum + Risk Agent factors + Analyst Agent bank ratings.
3. Return a valid JSON object matching the requested schema.

Schema:
{
  "stock": "Ticker Symbol",
  "decision": "BUY|SELL|HOLD",
  "confidence": 0.85,
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


def gated_confidence(composite, pillar_scores: dict, data_completeness: float = 0.87) -> float:
    """
    Mechanistic decision confidence (0..1) derived from deterministic signal quality:
    - distance of the composite from neutral 50: the more decisive the score, the higher the base.
    - agreement among the 5 pillars: heavy dispersion (pillars contradict) drags confidence down.
    - data completeness: missing data trims confidence.

    Returns a float rounded to 2 decimals. Fully deterministic -- the model's own
    stated confidence is recorded separately as model_confidence.
    """
    def _f(v, default=50.0):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    comp = _f(composite, 50.0)
    keys = ["trend", "sector", "alpha", "valuation_history", "peer_valuation"]
    values = [_f((pillar_scores or {}).get(k), 50.0) for k in keys]

    distance = max(0.0, min(1.0, abs(comp - 50.0) / 50.0))
    mean_p = sum(values) / len(values)
    dispersion = sum(abs(v - mean_p) for v in values) / (len(values) * 50.0)
    agreement = 1.0 - max(0.0, min(1.0, dispersion))

    try:
        completeness = max(0.0, min(1.0, float(data_completeness)))
    except (TypeError, ValueError):
        completeness = 0.87

    edge_strength = 0.7 * distance + 0.3 * (agreement - 0.5)
    base = 0.5 + 0.5 * edge_strength
    conf = base * (0.5 + 0.5 * completeness)
    return round(max(0.0, min(1.0, conf)), 2)


def normalize_master_trader_json(data: dict, symbol: str, m_data: dict = None, macro_data: dict = None, gloomberb_payload: dict = None) -> dict:
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

    # Normalize Decision
    raw_dec = str(data.get("decision") or data.get("recommendation") or data.get("signal") or "HOLD").upper()
    if "BUY" in raw_dec:
        decision = "BUY"
    elif "SELL" in raw_dec:
        decision = "SELL"
    else:
        decision = "HOLD"

    # Normalize model confidence -- kept as a reference; the authoritative `confidence`
    # is overwritten below by the deterministic gated_confidence() signal-quality formula.
    raw_conf = data.get("confidence") or buy_score or 0.70
    model_confidence = _to_float(raw_conf, 0.70)
    if model_confidence > 1.0:
        model_confidence = round(model_confidence / 100.0, 2) if model_confidence <= 100 else 0.70

    try:
        horizon_days = int(data.get("horizon_days") or 10)
    except (ValueError, TypeError):
        horizon_days = 10

    # Quantitative 5-Pillar Scores. The deterministic composite calculated from real market data is
    # authoritative for decisions and confidence. The model's echoed values are stored separately
    # (model_quant_score / model_pillar_scores) for reference only and never drive the signal.
    deterministic_scores = m_data.get("deterministic_5pillar_scores", {})
    det_quant = deterministic_scores.get("composite_quantitative_score")
    quant_score = _to_float(det_quant if det_quant is not None else data.get("quant_score"), None)
    if quant_score is None:
        quant_score = 50.0  # neutral fallback for display only; BUY gate uses the deterministic composite
    model_quant_score = _to_float(data.get("quant_score"), None)

    raw_pillars = data.get("pillar_scores") or {}
    if not isinstance(raw_pillars, dict):
        raw_pillars = {}

    pillar_scores = {}
    model_pillar_scores = {}
    for det_key, model_key, out_key in (
        ("trend_score", "trend", "trend"),
        ("sector_relative_score", "sector", "sector"),
        ("market_alpha_score", "alpha", "alpha"),
        ("valuation_history_score", "valuation_history", "valuation_history"),
        ("peer_valuation_score", "peer_valuation", "peer_valuation"),
    ):
        det_val = deterministic_scores.get(det_key)
        # Deterministic pillar wins whenever it was computed; the model value is only a fallback
        # when the deterministic score is absent (e.g., standalone normalize calls).
        pillar_scores[out_key] = _to_float(det_val if det_val is not None else raw_pillars.get(model_key), 50.0)
        model_pillar_scores[out_key] = _to_float(raw_pillars.get(model_key), None)

    # Array Fields
    bull_case = _to_list_of_strings(data.get("bull_case") or data.get("reason"), "Bullish trade setup and vector RAG catalysts evaluated.")
    bear_case = _to_list_of_strings(data.get("bear_case") or data.get("risk_assessment"), "Bearish trade setup and downside risks evaluated.")
    key_risks = _to_list_of_strings(data.get("key_risks") or data.get("key_risk"), "Volatile market conditions and stop-loss boundaries.")
    missing_info = _to_list_of_strings(data.get("missing_information"), "None")

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
    conf_composite = deterministic_scores.get("composite_quantitative_score")
    if conf_composite is None:
        conf_composite = quant_score or 50.0
    confidence = gated_confidence(conf_composite, pillar_scores, data_completeness)
    if decision == "HOLD":
        confidence = round(min(confidence, 0.60), 2)

    # Reward:Risk setup geometry -- deterministic, computed from market data, not the model.
    # Channel-anchored so the ratio varies with price position inside the 20-day range.
    rr_info = QuantitativeScoringService.compute_channel_reward_risk(
        m_data.get("current_price"),
        m_data.get("atr"),
        m_data.get("high_20d"),
        m_data.get("low_20d"),
        m_data.get("suggested_stop_loss"),
        m_data.get("suggested_target_price")
    )
    analyst_rr_info = QuantitativeScoringService.compute_reward_risk(
        m_data.get("current_price"),
        m_data.get("suggested_stop_loss"),
        m_data.get("analyst_target_price")
    )
    rr = rr_info.get("reward_risk_ratio")
    breakeven = rr_info.get("breakeven_win_rate")
    analyst_rr = analyst_rr_info.get("reward_risk_ratio")
    dist_resistance = rr_info.get("distance_to_resistance_atr")

    # Volatility risk profile -- deterministic ATR dampener, mirrors RiskAgent's HIGH threshold.
    vol_factor = QuantitativeScoringService.compute_vol_factor(m_data.get("atr"), m_data.get("current_price"))
    try:
        atr_pct = round((float(m_data.get("atr") or 0.0) / float(m_data.get("current_price") or 1.0)) * 100.0, 2)
    except (TypeError, ValueError):
        atr_pct = 0.0

    # Earnings recency -- primary source is MarketAgent's yfinance calendar; fall back to the
    # Gloomberb earnings_date string when the calendar is unavailable.
    days_to_earnings = m_data.get("days_to_earnings")
    if days_to_earnings is None:
        days_to_earnings = days_from_earnings_date(
            gloomberb_payload.get("earnings", {}).get("earnings_date")
        )

    # Decision origin: a trade's edge must come from price structure + composite + setup geometry,
    # never from a headline. News/macro/geopolitical events can confirm or veto, but cannot initiate.
    primary_driver = str(data.get("primary_driver") or "QUANT_STRUCTURE").upper()
    gates_applied = False

    def _append_risk(note):
        if note not in key_risks:
            key_risks.append(note)

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

    # Deterministic liquidity bar (BUY-only): a long needs tradeable turnover. RVOL below 0.5x
    # or 20d avg dollar volume below $1M means entries/exits are unreliable -- veto the BUY.
    LIQUIDITY_MIN_RVOL_20D = 0.5
    LIQUIDITY_MIN_AVG_DOLLAR_VOL_20D = 1_000_000.0
    rvol_20d_val = _to_float(m_data.get("rvol_20d"), None)
    avg_dollar_vol_20d_val = _to_float(m_data.get("avg_dollar_vol_20d"), None)
    liquidity_ok = (
        (rvol_20d_val is None or rvol_20d_val >= LIQUIDITY_MIN_RVOL_20D)
        and (avg_dollar_vol_20d_val is None or avg_dollar_vol_20d_val >= LIQUIDITY_MIN_AVG_DOLLAR_VOL_20D)
    )

    if decision in ("BUY", "SELL") and primary_driver in ("NEWS_CATALYST", "MACRO_EVENT"):
        decision = "HOLD"
        gates_applied = True
        no_trade_reason = "INSUFFICIENT_EVIDENCE"
        _append_risk(
            f"{'BUY' if decision == 'HOLD' and buy_score > sell_score else 'SELL'} capped to HOLD: "
            f"primary_driver '{primary_driver}' is an external event; news/geopolitical/macro "
            f"headlines can confirm or veto but cannot initiate a directional trade. "
            f"Re-anchor the trade on price structure + composite + setup geometry."
        )

    # Deterministic BUY gates: momentum/valuation anchors cannot override setup geometry or risk.
    # Hard post-model rule: the model's BUY is advisory and is only certified when the
    # DETERMINISTIC composite (computed from market data, not echoed by the model) is >= 70,
    # the market snapshot is valid/fresh, data coverage is adequate, and reward:risk >= 1.5.
    if decision == "BUY":
        det_composite = deterministic_scores.get("composite_quantitative_score")
        if det_composite is None:
            det_composite = deterministic_scores.get("raw_composite")
        try:
            composite_ok = det_composite is not None and float(det_composite) >= 70.0
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

        # Adequate data coverage: reported completeness >= 80% AND a deterministic composite was
        # actually computed (the model cannot certify a BUY on zero deterministic pillars).
        coverage_ok = (
            data_completeness >= 0.80
            and deterministic_scores.get("composite_quantitative_score") is not None
        )

        # Reward:risk must exist and clear the 1.5 minimum (rr is None only if stop/target degenerate).
        rr_ok = rr is not None and rr >= 1.5

        gate_reasons = []
        if not composite_ok:
            if det_composite is None:
                gate_reasons.append("deterministic composite could not be computed from market data")
            else:
                gate_reasons.append(f"deterministic composite {float(det_composite):.1f}/100 < 70")
        if not market_data_ok:
            gate_reasons.append("invalid or stale market data (price, ATR, stop, target must all be positive)")
        if not coverage_ok:
            if not deterministic_scores.get("composite_quantitative_score"):
                gate_reasons.append("data coverage inadequate: no deterministic composite could be computed from market data")
            else:
                gate_reasons.append(f"data coverage {data_completeness * 100:.0f}% < 80% bar")
        if not rr_ok:
            if rr is None:
                gate_reasons.append("reward:risk ratio unavailable from setup geometry")
            else:
                gate_reasons.append(f"reward:risk ratio {rr:.2f} < 1.5")

        if gate_reasons:
            decision = "HOLD"
            gates_applied = True
            no_trade_reason = (
                "RR_TOO_LOW" if any("reward:risk" in r for r in gate_reasons) else "INSUFFICIENT_EVIDENCE"
            )
            _append_risk(
                "BUY downgraded to HOLD (deterministic BUY eligibility): " + "; ".join(gate_reasons) + "."
            )
        elif days_to_earnings is not None and days_to_earnings <= 3:
            decision = "HOLD"
            gates_applied = True
            no_trade_reason = "EARNINGS_BLACKOUT"
            note = (
                f"BUY downgraded to HOLD: earnings report in {days_to_earnings} day(s) is a "
                f"binary gap-risk event; do not initiate a fresh position into it."
            )
            _append_risk(note)
        elif vol_factor < 0.85:
            decision = "HOLD"
            gates_applied = True
            no_trade_reason = "INSUFFICIENT_EVIDENCE"
            note = (
                f"BUY downgraded to HOLD: extreme volatility (ATR {atr_pct}% of price, "
                f"vol factor {vol_factor:.2f}); composite is dampened to {quant_score}/100."
            )
            _append_risk(note)
        elif not liquidity_ok:
            decision = "HOLD"
            gates_applied = True
            no_trade_reason = "LOW_LIQUIDITY"
            note = (
                f"BUY downgraded to HOLD: insufficient liquidity "
                f"(RVOL {rvol_20d_val if rvol_20d_val is not None else 'N/A'}x, "
                f"20d avg dollar volume "
                f"${avg_dollar_vol_20d_val if avg_dollar_vol_20d_val is not None else 'N/A'}); "
                f"entries/exits unreliable below "
                f"{LIQUIDITY_MIN_RVOL_20D}x RVOL / ${LIQUIDITY_MIN_AVG_DOLLAR_VOL_20D / 1_000_000:.0f}M."
            )
            _append_risk(note)
        elif analyst_rr is not None and analyst_rr < 1.0:
            decision = "HOLD"
            gates_applied = True
            no_trade_reason = "INSUFFICIENT_EVIDENCE"
            note = (
                f"BUY downgraded to HOLD: Wall Street mean target "
                f"${_to_float(m_data.get('analyst_target_price'), 0.0):.2f} is below entry "
                f"${_to_float(m_data.get('current_price'), 0.0):.2f} (analyst RR {analyst_rr:.2f})."
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

    # A gate/realignment that lands on HOLD must not carry the conviction of a directional call.
    if decision == "HOLD":
        confidence = round(min(confidence, 0.60), 2)

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

    return {
        "stock": _ensure_str(stock),
        "decision": decision,
        "confidence": confidence,
        "model_confidence": model_confidence,
        "buy_score": buy_score,
        "hold_score": hold_score,
        "sell_score": sell_score,
        "horizon_days": horizon_days,
        "quant_score": quant_score,
        "model_quant_score": model_quant_score,
        "no_trade_reason": no_trade_reason,
        "deterministic_data_completeness": deterministic_completeness,
        "model_data_completeness": model_data_completeness,
        "raw_composite": deterministic_scores.get("raw_composite"),
        "vol_factor": vol_factor,
        "atr_pct": atr_pct,
        "primary_driver": primary_driver,
        "falsification_bull": str(data.get("falsification_bull") or ""),
        "falsification_bear": str(data.get("falsification_bear") or ""),
        "pillar_scores": pillar_scores,
        "model_pillar_scores": model_pillar_scores,
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
        # Backward compatibility for SQLite DB string storage
        "reason": "; ".join(bull_case),
        "risk_assessment": "; ".join(key_risks),
        "institutional_data": inst_summary,
        "macro_data": macro_summary,
        "news": news_summary,
        "investment_bank_coverage": bank_cov,
        "PE_and_PEG": str(forward_pe)
    }


def format_telegram_digest(results, model_label="Nemotron-3 Super 120B"):
    lines = [f"⚡ *Gloomberb Multi-Pillar RAG Digest ({model_label})* ⚡\n"]
    emoji_map = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}

    for item in results:
        stock = item.get("stock", "N/A")
        decision = str(item.get("decision", "HOLD")).upper()
        confidence = item.get("confidence", 0.70)
        buy_score = item.get("buy_score", 0.0)
        hold_score = item.get("hold_score", 0.0)
        sell_score = item.get("sell_score", 0.0)
        horizon = item.get("horizon_days", 10)
        quant_score = item.get("quant_score", 65.0)
        pillars = item.get("pillar_scores", {})
        completeness = item.get("data_completeness", 0.85)
        emoji = emoji_map.get(decision, "⚪")

        lines.append(f"{emoji} *{stock}* | *{decision}* (Conf: {confidence} | {horizon}d Horizon)")
        no_trade_reason = item.get("no_trade_reason")
        if no_trade_reason:
            lines.append(f"• *No-Trade:* `{no_trade_reason}`")
        vol_factor = item.get("vol_factor", 1.0)
        atr_pct = item.get("atr_pct", 0.0)
        lines.append(f"• *Quant Score:* `{quant_score}/100` | *Vol Factor:* `{vol_factor}` (ATR {atr_pct}%) | *Data Coverage:* `{int(completeness * 100)}%`")

        rr = item.get("reward_risk_ratio")
        breakeven = item.get("breakeven_win_rate")
        if rr is not None:
            be_str = f"{breakeven * 100:.0f}%" if isinstance(breakeven, (int, float)) else "N/A"
            lines.append(f"• *Reward:Risk:* `{rr}` (breakeven win rate: `{be_str}`)")
        lines.append(f"• *Pillars:* Trend: {pillars.get('trend', 0)} | Sector: {pillars.get('sector', 0)} | Alpha: {pillars.get('alpha', 0)} | ValHist: {pillars.get('valuation_history', 0)} | PeerVal: {pillars.get('peer_valuation', 0)}")
        lines.append(f"• *Probabilities:* Buy: {buy_score} | Hold: {hold_score} | Sell: {sell_score}")
        driver = item.get("primary_driver")
        if driver and str(driver).strip() and str(driver).strip().upper() != "NONE":
            lines.append(f"• *Primary Driver:* `{driver}`")

        bull_items = item.get("bull_case", [])
        if bull_items:
            lines.append("• *Bull Case:*")
            for b in bull_items:
                lines.append(f"  - _{b}_")

        bear_items = item.get("bear_case", [])
        if bear_items:
            lines.append("• *Bear Case:*")
            for b in bear_items:
                lines.append(f"  - _{b}_")

        risk_items = item.get("key_risks", [])
        if risk_items:
            lines.append("• *Key Risks:*")
            for r in risk_items:
                lines.append(f"  - _{r}_")

        missing_items = item.get("missing_information", [])
        if missing_items and missing_items != ["None"]:
            lines.append("• *Missing Info:*")
            for m in missing_items:
                lines.append(f"  - _{m}_")
        lines.append("")

    return "\n".join(lines)


def send_telegram_digest(token, chat_id, text):
    url = f"https://api.telegram.org/bot{token}/sendMessage"

    if len(text) <= 4000:
        res = requests.post(
            url,
            json={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "Markdown"
            }
        )
        if res.status_code == 200:
            print("\nTelegram digest message sent successfully!")
        else:
            print(f"\nTelegram error: {res.status_code} {res.text}")
    else:
        blocks = text.split("\n\n")
        current_chunk = ""
        chunks = []
        for block in blocks:
            if len(current_chunk) + len(block) + 2 > 3800:
                chunks.append(current_chunk)
                current_chunk = block
            else:
                current_chunk = (current_chunk + "\n\n" + block).strip()
        if current_chunk:
            chunks.append(current_chunk)

        for idx, chunk in enumerate(chunks):
            res = requests.post(
                url,
                json={
                    "chat_id": chat_id,
                    "text": chunk,
                    "parse_mode": "Markdown"
                }
            )
            if res.status_code == 200:
                print(f"Telegram digest part {idx+1}/{len(chunks)} sent.")
            else:
                print(f"Telegram error on part {idx+1}: {res.status_code} {res.text}")
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(
        description="6-Agent Stock Swing Trading Analysis Pipeline",
        usage="python main.py {nemotron|ultra|kimi|super|gemini|openrouter|twostage|gemma|qwen} [ticker]"
    )
    parser.add_argument(
        "model_arg",
        nargs="?",
        default=None,
        help="Required model short name: 'nemotron' / 'ultra' (Nemotron-3 Ultra 550B), 'kimi' (Moonshot AI Kimi-K3), 'super' (Nemotron-3 Super 120B), 'gemini' (Gemini 3.1 Pro), 'openrouter' (OpenRouter Free Models Router - openrouter/free), 'twostage' (Qwen2.5 14B + DeepSeek-R1 14B), 'gemma' (gemma4:12b), or 'qwen' (qwen2.5:14b)"
    )
    parser.add_argument(
        "ticker_arg",
        nargs="?",
        default=None,
        help="Optional stock ticker symbol (e.g. 000660.KS, NVDA) or comma-separated list of tickers"
    )
    parser.add_argument(
        "--model", "-m",
        dest="model_opt",
        default=None,
        help="Model short name: 'nemotron', 'ultra', 'kimi', 'super', 'gemini', 'openrouter', 'free', 'twostage', 'gemma', or 'qwen'"
    )
    parser.add_argument(
        "--ticker", "-t",
        dest="ticker_opt",
        default=None,
        help="Stock ticker symbol (e.g. 000660.KS, NVDA) or comma-separated list of tickers"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Model sampling temperature (e.g. 0.2 for strict determinism, 0.6-0.8 for deep reasoning CoT)"
    )
    parser.add_argument(
        "--reasoning-budget",
        type=int,
        default=None,
        help="Internal reasoning token budget for reasoning models (default: 16000)"
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        choices=["low", "medium", "high", "max"],
        default=None,
        help="Reasoning effort level ('low', 'medium', 'high', 'max', default: 'high')"
    )
    args = parser.parse_args()

    raw_model = args.model_opt or args.model_arg
    if not raw_model:
        print("\n❌ ERROR: Model argument is required!")
        print("Usage: python main.py {nemotron|ultra|kimi|super|gemini|openrouter|twostage|gemma|qwen} [ticker]")
        print("  - nemotron / ultra : Cloud Nemotron-3 Ultra 550B (NVIDIA)")
        print("  - kimi             : Moonshot AI Kimi-K3 (NVIDIA)")
        print("  - super            : Cloud Nemotron-3 Super 120B (NVIDIA)")
        print("  - gemini           : Cloud Gemini 3.1 Pro")
        print("  - openrouter       : OpenRouter Free Models Router (openrouter/free)")
        print("  - twostage         : 2-Stage Local (qwen2.5:14b extraction + qwen3:30b-a3b-instruct-2507-q4_K_M reasoning)")
        print("  - gemma            : Local Ollama gemma4:12b")
        print("  - qwen             : Local Ollama qwen2.5:14b\n")
        sys.exit(1)

    model_choice = str(raw_model).strip().lower()
    valid_models = [
        "nemotron", "nvidia", "ultra", "nemotron-ultra", "550b",
        "kimi", "kimi-k3", "k3", "moonshot",
        "super", "nemotron-super", "120b",
        "gemini", "openrouter", "free", "openrouter/free",
        "minimax", "minimax-m3", "minimax_m3", "m3",
        "twostage", "gemma", "qwen", "local"
    ]
    if model_choice not in valid_models:
        print(f"\n❌ ERROR: Invalid model choice '{raw_model}'!")
        print("Supported choices are: 'nemotron' (Ultra 550B), 'kimi' (Kimi-K3), 'super' (120B), 'gemini', 'openrouter', 'twostage', 'gemma', 'qwen'\n")
        sys.exit(1)

    raw_ticker = args.ticker_opt or args.ticker_arg
    if raw_ticker:
        watchlist = [t.strip().upper() for t in raw_ticker.split(",") if t.strip()]
    else:
        watchlist = WATCHLIST

    model_label = get_model_label(model_choice)

    print(f"=== Initializing Gloomberb RAG & Technical Analysis Pipeline (Model: {model_label}) ===")
    print(f"Target Watchlist: {', '.join(watchlist)}")
    market_agent = MarketAgent()
    macro_agent = MacroDataAgent()
    gloomberb_service = GloomberbService()
    institutional_service = InstitutionalDataService()
    rag_service = RAGService()

    all_results = []

    for idx, symbol in enumerate(watchlist, 1):
        print(f"\n--- [{idx}/{len(watchlist)}] Processing {symbol} ---")

        # 1. Technical Data Collection (RSI / EMA / ATR / Volume / RVOL / Channels)
        m_data = market_agent.analyze(symbol)
        if not m_data:
            print(f"Skipping {symbol}: Insufficient price data.")
            continue

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

        # 5. Dense Vector Embedding RAG & Prompt Payload Assembly
        context = rag_service.get_nemotron_payload(gloomberb_payload, technical_data=m_data, institutional_data=institutional_payload)

        # Attach the deterministic 5-pillar scores to m_data so normalize can fall back to them
        # instead of defaults (and so gated_confidence uses real pillar agreement).
        if isinstance(m_data, dict):
            m_data = dict(m_data)
            m_data["deterministic_5pillar_scores"] = QuantitativeScoringService().compute_5pillar_scores(
                m_data, gloomberb_payload, gloomberb_payload.get("sector_benchmark", {})
            )
            # Deterministic data completeness: computed from live provider availability once.
            # The LLM's own claim (model_data_completeness) is stored separately and never authoritative.
            m_data["deterministic_data_completeness"] = QuantitativeScoringService.compute_data_coverage(
                m_data, macro_data, gloomberb_payload, institutional_payload
            )

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
            if res_obj:
                validated, errors = validate_signal_json(res_obj)

                # One constrained corrective retry on strict schema violations.
                if errors:
                    print(f"[{model_label}] Schema validation failed for {symbol}; issuing one corrective retry...")
                    retry_prompt = (
                        f"{context['user_prompt']}\n\n"
                        "=== SCHEMA VALIDATION FAILED -- RESUBMIT ONLY THE FIXED JSON ===\n"
                        "Your previous response failed strict schema validation. Correct EVERY "
                        "violation below and return ONLY a single valid JSON object with the exact "
                        "same golden keys (decision, primary_driver, buy_score, hold_score, "
                        "sell_score, horizon_days, quant_score, pillar_scores, data_completeness, "
                        "falsification_bull, falsification_bear, bull_case, bear_case, key_risks, "
                        "missing_information). buy_score/hold_score/sell_score must each be in "
                        "[0,1] and sum to ~1.0.\n"
                        f"Violations:\n- " + "\n- ".join(errors) + "\n"
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
                        validated, errors = validate_signal_json(retry_obj)

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
                        "horizon_days": 10,
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
        return

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

    # Send Telegram alerts
    if all_results and telegram_token:
        digest_message = format_telegram_digest(all_results, model_label=model_label)
        send_telegram_digest(telegram_token, CHAT_ID, digest_message)


if __name__ == "__main__":
    main()
