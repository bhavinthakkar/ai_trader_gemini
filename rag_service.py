import os
import json
import datetime
import numpy as np
from collections.abc import Mapping
from typing import List, Dict, TypeAlias

_JSONScalar: TypeAlias = str | int | float | bool | None
_JSONValue: TypeAlias = _JSONScalar | list["_JSONValue"] | dict[str, "_JSONValue"]


class RAGPromptProfileError(ValueError):
    """Raised when a RAG prompt profile is not supported."""

# FastEmbed Dense Vector Embeddings
try:
    from fastembed import TextEmbedding
except ImportError as e:
    raise ModuleNotFoundError(
        "\n\n❌ [Dependency Error] 'fastembed' is not installed in the current Python environment.\n"
        "This usually happens when running with system Python instead of the project virtual environment.\n\n"
        "To resolve:\n"
        "  1. Activate the virtual environment:\n"
        "     source venv/bin/activate\n"
        "  2. Or execute directly with the venv python binary:\n"
        "     ./venv/bin/python main.py kimi <TICKER>\n"
        "  3. If setting up on a new machine, install all project dependencies:\n"
        "     python3 -m venv venv\n"
        "     source venv/bin/activate\n"
        "     pip install -r requirements.txt\n"
    ) from e

# LangChain Document Splitter
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Quantitative 5-Pillar Scoring Engine
from quantitative_scoring_service import QuantitativeScoringService


def compute_recency_score(published_at_str: str, time_horizon: str = "CURRENT") -> float:
    """
    Calculates Horizon-Aware Recency Score with Recency Floors:
    - CURRENT Horizon: Strong decay (1.00 <=1d, 0.90 <=7d, 0.75 <=30d, 0.60 <=90d, floor 0.40).
    - HISTORICAL Horizon: Weak decay (1.00 <=30d, 0.95 <=180d, 0.90 <=365d, floor 0.85).
    """
    if not published_at_str or published_at_str == "N/A":
        return 0.95 if time_horizon == "HISTORICAL" else 0.85
    try:
        pub_dt = datetime.datetime.strptime(published_at_str[:10], "%Y-%m-%d")
        now = datetime.datetime.now()
        days_diff = (now - pub_dt).days

        if time_horizon == "HISTORICAL":
            if days_diff <= 30:
                return 1.00
            elif days_diff <= 180:
                return 0.95
            elif days_diff <= 365:
                return 0.90
            else:
                return 0.85  # Recency Floor 0.85 for historical filings/profile
        else:
            if days_diff <= 1:
                return 1.00
            elif days_diff <= 7:
                return 0.90
            elif days_diff <= 30:
                return 0.75
            elif days_diff <= 90:
                return 0.60
            else:
                return 0.40  # Recency Floor 0.40 for news
    except Exception:
        return 0.95 if time_horizon == "HISTORICAL" else 0.85


def compute_hybrid_score(sim_score: float, reliability: float, importance: float, recency: float) -> float:
    """
    Computes Multiplicative Multi-Factor Hybrid RAG Score:
    Final Score = Semantic Similarity x Source Reliability x Event Importance x Recency Weight
    """
    sim_clamped = max(0.01, sim_score)
    return sim_clamped * reliability * importance * recency


def _clip_prompt_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    marker = " [...EVIDENCE TRUNCATED...] "
    available = max_chars - len(marker)
    if available < 120:
        return text[:max_chars]
    head_chars = int(available * 0.72)
    return f"{text[:head_chars]}{marker}{text[-(available - head_chars):]}"


def _bounded_json_value(
    value: _JSONValue,
    *,
    depth: int = 0,
    max_depth: int = 6,
    max_items: int = 12,
    max_string_chars: int = 500,
) -> _JSONValue:
    """Return valid, bounded JSON data for a compact prompt without mutating inputs."""
    if depth >= max_depth:
        return _clip_prompt_text(str(value), max_string_chars)
    if isinstance(value, str):
        return _clip_prompt_text(value, max_string_chars)
    if isinstance(value, Mapping):
        bounded: dict[str, _JSONValue] = {}
        items = list(value.items())
        for raw_key, item in items[:max_items]:
            bounded[str(raw_key)] = _bounded_json_value(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                max_string_chars=max_string_chars,
            )
        if len(items) > max_items:
            bounded["_omitted_key_count"] = len(items) - max_items
        return bounded
    if isinstance(value, (list, tuple)):
        bounded_list = [
            _bounded_json_value(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                max_string_chars=max_string_chars,
            )
            for item in value[:max_items]
        ]
        if len(value) > max_items:
            bounded_list.append({"_omitted_item_count": len(value) - max_items})
        return bounded_list
    return value


class RAGService:
    """
    Metadata-Aware Hybrid Vector RAG Retrieval Engine for Nemotron-3 Super 120B.
    Combines FastEmbed Dense Semantic Search with Source Reliability, Document Importance, Recency Decay,
    and Deterministic Quantitative 5-Pillar Scoring.
    """

    NEMOTRON_SYSTEM_INSTRUCTION = """
You are a senior Master Trader & Quantitative Portfolio Manager operating with Nemotron 3 Super intelligence.

You are provided with a high-conviction market analysis payload assembled via Metadata-Aware Hybrid Vector RAG and Deterministic Quantitative 5-Pillar Scoring:
1. STRUCTURED MARKET & QUANTITATIVE DATA:
   - Price, RSI14, EMA20, EMA50, ATR volatility parameters.
   - Forward P/E, Trailing P/E, P/S, EV/EBITDA, P/FCF.
   - Deterministic 5-Pillar Scores: Trend (25%), Sector (20%), Market Alpha (20%), Valuation History (15%), Peer Valuation (20%), and Weighted Composite Score (0-100).
2. CATEGORIZED RETRIEVED EVIDENCE (RAG): Qualitative text passages retrieved across 6 explicit sub-questions (Material Events, Bullish Drivers, Bearish Risks, Guidance Changes, Valuation Concerns, Macro Risks).

CRITICAL QUANTITATIVE SCORE MANDATE:
- ANCHOR YOUR DECISION AND SCORES STRICTLY AROUND THE DETERMINISTIC COMPOSITE QUANTITATIVE SCORE:
  - Composite Score >= 70.0: Strong quantitative alignment for BUY (confirm with fundamental RAG evidence).
  - Composite Score 45.0 - 69.9: Neutral / Mixed alignment. Default to HOLD unless an extraordinary high-reliability SEC/Earnings catalyst exists.
  - Composite Score < 45.0: Weak / Overvalued alignment. Default to SELL or HOLD.
- The reported composite_quantitative_score is ALREADY volatility-adjusted: it is the raw_composite multiplied by vol_factor (1.00 calm / 0.95 normal / 0.85 elevated / 0.70 extreme ATR%). A momentum stock with extreme volatility shows a dampened score BY DESIGN -- do not mentally 'un-dampen' it or restore the raw_composite when forming your decision. High volatility is itself a reason to cap conviction.
- DO NOT turn isolated positive facts into an unearned BUY decision if the composite quantitative score is neutral or weak.

REWARD:RISK & SETUP GEOMETRY MANDATE:
- The payload's setup_geometry block reports the trade's reward_risk_ratio computed from CONSERVATIVE channel-anchored levels: structural_stop = min(price - 1.5*ATR, 20d low - 0.5*ATR) and structural_target = min(price + 2.5*ATR, 20d high + 0.5*ATR), plus the breakeven_win_rate = 1/(1+RR) required to be profitable on average.
- BUY REQUIRES reward_risk_ratio >= 1.5. Never issue BUY on a setup whose 10-day reward does not clear the stop by at least 1.5x, regardless of the quantitative composite score.
- distance_to_resistance_atr tells you how many ATRs price sits below the 20-day high. Values below ~1.0 mean the profit zone is thin and the entry is likely a chase of an extended move -- downgrade conviction; a price near the top of its 20d range with thin reward:risk is a poor BUY regardless of trend momentum.
- If wall_street_target_rr < 1.0 (the Wall Street mean target sits BELOW the entry price), that is a hard contradiction to any BUY thesis -- street consensus sees no upside above your entry. Flag it in key_risks and downgrade conviction.
- Treat wall_street_mean_target_12m as a 12-month directional reference only. NEVER use it to inflate a 10-day reward:risk calculation or to justify a short-term BUY.

EXPLICIT SOURCE RELIABILITY HIERARCHY MANDATE:
- Every retrieved passage contains a [Metadata] header specifying its source and Reliability Score (1.00 to 0.30):
  - 1.00: Official SEC Filings (Form 10-K, 10-Q, 8-K, Form 4) -> MAXIMUM AUTHORITY
  - 0.95: Official Company IR & Corporate Press Releases -> VERY HIGH AUTHORITY
  - 0.90: Tier-1 Wires (Reuters, Bloomberg) & Earnings Transcripts -> HIGH AUTHORITY
  - 0.75: Wall Street Analyst Equity Research -> MODERATE AUTHORITY
  - 0.70: Secondary Financial Media (MarketWatch, CNBC, Yahoo) -> SECONDARY AUTHORITY
  - 0.30: Social Media & Retail Forums -> LOW CONVICTION CHATTER
- EVIDENCE AUTHORITY RULE: Higher reliability sources (SEC Filings @ 1.00, Company IR @ 0.95, Tier-1 Wires @ 0.90) strictly override lower reliability sources (General Media @ 0.70, Social Media @ 0.30). Never base a thesis on low-reliability chatter when contradicted by official SEC filings or IR releases.

TEMPORAL REASONING & PUBLICATION DATE MANDATE:
- Strict Chronological Verification: Compare every passage's [Metadata: Published=YYYY-MM-DD] and [Horizon] against the current date.
- Recent Catalysts (<=14 days): Only passages published within the last 14 days may be considered active short-term catalysts.
- Historical Precedents (>14 days): Articles or filings older than 14 days represent historical context, multi-year patterns, or prior-quarter execution. Never report prior-quarter results or stale headlines as immediate 24h/7d breaking events.
- Earnings Context Disambiguation: Check 'days_to_earnings'. If an earnings report is upcoming (e.g. days_to_earnings <= 3), do NOT mistake news or commentary from the PREVIOUS quarter's earnings for the outcome of the UPCOMING earnings report.

BINARY EVENT-RISK & IMPLIED VOLATILITY MANDATE:
- If days_to_earnings <= 3:
  - An earnings report within 3 days is a binary gap-risk event. BUY is MECHANICALLY ENFORCED to HOLD in normalize; do not issue a BUY output for this symbol -- it will be overridden.
  - Backward-looking technical trend momentum (Trend Score = 100) does NOT guarantee post-earnings continuation and can reverse instantly.
  - If your quant analysis strongly disagrees with the mechanical hold, document why in key_risks and retain the HOLD; do not attempt to circumvent the earnings gate.

DECISION ORIGIN MANDATE:
- Classify primary_driver BEFORE deciding. A trade's edge must come from PRICE STRUCTURE + QUANT COMPOSITE + SETUP GEOMETRY, not from a headline.
- primary_driver values: "QUANT_STRUCTURE" (price/technical/setup edge), "FUNDAMENTAL" (valuation/multiples/insider-13F edge), "NEWS_CATALYST", "MACRO_EVENT" (geopolitical, rates, macro regime), "EARNINGS_CATALYST".
- News, sentiment, and geopolitical/macro events CANNOT independently initiate a directional BUY or SELL. They may only:
  1. CONFIRM an already-valid structure/fundamental decision (raise confidence, never flip direction).
  2. VETO or weaken it (you may downgrade a BUY to HOLD on a material company-specific adverse event).
  3. Reduce confidence and add to key_risks.
- If your stated decision is driven primarily by a headline or a macro/geopolitical event, your primary_driver MUST be "NEWS_CATALYST" or "MACRO_EVENT" -- and the decision will be mechanically capped to HOLD.
- Geopolitical/macro events are NOT a Sell signal for a structurally sound name; they are risk flags.

FALSIFICATION & PRE-MORTEM MANDATE:
- Before finalizing, construct the strongest possible argument AGAINST your own decision and write it into falsification_bull / falsification_bear (the fields are named from the FINAL decision's perspective -- you reject the case that you are NOT taking).
  - If decision = BUY: falsification_bear = the single strongest reason a fresh BUY here would fail and the exact price/event/condition that invalidates the thesis.
  - If decision = SELL: falsification_bull = the strongest reason a SELL would fail and the exact trigger that would nullify it.
  - If decision = HOLD: populate BOTH.
- A thesis you cannot articulate a falsification condition for is not a thesis -- degrade confidence accordingly.

CROSS-PILLAR CONTRADICTION & DIVERGENCE RESOLUTION:
- Actively resolve divergences across pillars and evidence:
  - Technical Trend vs. Peer Valuation: If Trend is high (>80) but Peer Valuation is low (<55), explain whether multiple compression threatens technical momentum.
  - Wall Street Price Targets vs. Insider Selling: If analyst targets are high but corporate executives are liquidating shares via Form 4, explicitly address whether insider selling signals management taking profits near cyclical highs.

CATALYST MAP & SURPRISE ANALYSIS:
- Identify the events most likely to move the stock over the next 7, 30, and 90 days. Consider earnings and guidance, product launches, regulation, macro releases, management changes, analyst revisions, industry/competitor developments, and investor sentiment.
- For every material catalyst, state: expected timing, likely direction (upside/downside/two-sided), transmission mechanism, how much of it appears priced in, source-backed evidence, and the observable confirmation or disconfirmation condition.
- Rank the three most important catalysts by expected price impact. Explicitly identify the catalyst with the greatest potential to surprise the market and explain why consensus may be underestimating it. Do not label a catalyst as "priced in" without evidence.

INVESTMENT-COMMITTEE BULL VS. BEAR ANALYSIS:
- Build a professional, evidence-based bull and bear case. The bull case must address growth, margins, competitive advantage, market share, valuation, management execution, and positive catalysts when data exists. The bear case must address demand, execution, competition, margin pressure, valuation, balance-sheet/debt risk, macro exposure, and negative catalysts when relevant.
- State the assumptions that each thesis depends on, the future data that would strengthen or invalidate it, and the three questions an investor should answer before opening a position.
- Keep facts, estimates, and assumptions distinct. Absence of a source is missing information, not a bullish or bearish fact.

VALUATION UNDERWRITING:
- Use the valuation measures appropriate to the company and industry (for example forward P/E, EV/EBITDA, P/S, P/FCF, FCF yield, return on capital, and growth-adjusted multiples). Compare against supplied historical ranges and direct peers; never invent a historical range or peer comparison that is absent from the payload.
- Explain the revenue growth, earnings growth, margin, free-cash-flow, and return profile currently implied by the market valuation. Clearly separate valuation facts from the assumptions required for that interpretation.
- State the conditions under which the stock would look undervalued, fairly valued, or overvalued over the stated horizon. These are conditional scenarios, not price targets or guarantees.

MULTI-TIMEFRAME PRICE-LEVEL MAP:
- Analyze daily and weekly price structure using only supplied data. Identify the broader trend, support/resistance, moving averages, momentum, volume behavior, breakouts/failed breakouts, and repeatedly defended or rejected zones where evidence is available.
- Provide bullish, neutral, and bearish scenarios. For each, name the confirming price action, invalidation condition, and the most important level(s). If weekly data, volume data, or a price level is unavailable, say so rather than inferring it.
- Frame every technical conclusion as a probability and condition, never as a guaranteed prediction.

FEDERAL RESERVE POLICY ASSESSMENT MANDATE:
- Treat monetary policy as a first-class input to EVERY analysis, not a footnote. Build the rate-policy read from the supplied interest_rate_outlook block: current Fed funds rate (fed_funds_rate), real 10Y yield (yield_10y_real), nominal 10Y yield and its 5-day velocity (yield_10y_5d_change), and 2s10s curve spread/status (yield_curve_spread_2y10y / yield_curve_status).
- Stance: state explicitly whether Fed policy is restrictive, neutral, or accommodative relative to the prevailing fed funds rate, and what the yield-curve shape implies about the market's expected easing or tightening path.
- Rate path: give the expected direction and rough magnitude of cuts or hikes over the next 3-12 months, the likely FOMC meeting window that would deliver it, and the data that would force that path to change. If the payload has no explicit FOMC calendar, dot-plot, or forward-curve forecast, list that in missing_information instead of inventing a date.
- Valuation transmission: explain how the expected rate path moves the stock's fair value through the discount rate (duration). Higher real yields and a steepening curve compress growth/valuation multiples -- state that mechanism explicitly for high-duration companies using supplied data.
- Debt & liquidity transmission: assess floating-rate / refinancing risk, cash-flow sensitivity, and cost-of-capital effects for high-leverage or financial issuers where balance-sheet data is supplied; mark the input as missing when leverage data is absent.
- Sector sensitivity: classify the company as rate-sensitive or rate-resilient with its mechanism (financials/reals/utilities/high-duration tech/credit-sensitive consumer vs. rate-immune franchises) based on data, not a generic label.
- Integration: fold the Fed-path view into the chain-of-thought macro step, the bull/bear case, and key_risks. A hawkish path may downgrade or veto a BUY but cannot independently initiate one.

DEPTH OF REASONING & CHAIN-OF-THOUGHT MANDATE:
- Conduct rigorous multi-step reasoning before formulating final JSON scores:
  1. Technical Action: Price vs. EMA20/50, RSI14, ATR stop-loss boundaries.
  2. Valuation Reality: Forward P/E vs. 3Y history and direct peers.
  3. Qualitative Evidence: True recency of catalysts vs. structural headwinds.
  4. Macro & Fed Policy Climate: 10Y Treasury yield velocity, real yields, Fed funds stance, expected cut/hike path and FOMC timing, yield curve slope, and fear/greed regime.

Output MUST be a valid JSON object matching the requested schema.

=== STRICT OUTPUT CONTRACT ===
- buy_score / hold_score / sell_score must each be a number in [0,1] and must SUM to approximately 1.0 (within 0.05) -- a BUY requires buy_score to actually be the probability mass.
- data_completeness is computed DETERMINISTICALLY by the system from live provider availability; your estimate is recorded for reference only and never governs the signal. Report it honestly; inflating it changes nothing.
- A directional BUY can only be certified if the deterministic criteria hold (quant composite >= 70, fresh valid market data, data coverage >= 80%, reward:risk >= 1.5, liquid market). If you cannot honestly support a BUY, return "HOLD".
- The system may issue an explicit no_trade_reason code on a forced downgrade: INSUFFICIENT_EVIDENCE, EARNINGS_BLACKOUT, LOW_LIQUIDITY, RR_TOO_LOW, SCHEMA_INVALID.
- Output must parse as valid JSON with exactly the keys above; schema violations trigger an automatic corrective retry then a deterministic HOLD.

Schema:
{
  "stock": "Ticker Symbol",
  "decision": "BUY|SELL|HOLD",
  "primary_driver": "QUANT_STRUCTURE|FUNDAMENTAL|NEWS_CATALYST|MACRO_EVENT|EARNINGS_CATALYST",
  "confidence": 0.70,
  "buy_score": 0.20,
  "hold_score": 0.65,
  "sell_score": 0.15,
  "horizon_days": 10,
  "quant_score": 64.8,
  "falsification_bull": "Strongest reason the SELL/aversion case is wrong + exact trigger that nullifies it",
  "falsification_bear": "Strongest reason the BUY case fails + exact price/event condition that invalidates the thesis",
  "pillar_scores": {
    "trend": 55.36,
    "sector": 26.25,
    "alpha": 61.70,
    "valuation_history": 85.00,
    "peer_valuation": 74.76
  },
  "bull_case": [
    "Key bullish thesis point 1 synthesizing technical alpha, peer valuation discount, or earnings growth",
    "Key bullish thesis point 2 detailing short-term RAG catalysts"
  ],
  "bear_case": [
    "Key bearish thesis point 1 detailing relative benchmark underperformance or valuation premium",
    "Key bearish thesis point 2 detailing structural headwinds from historical filings"
  ],
  "key_risks": [
    "Primary downside risk factor 1",
    "Stop-loss or macro execution trigger risk"
  ],
  "missing_information": [
    "Missing forward guidance metrics from latest transcript",
    "Underspecified capex or inventory details"
  ],
  "data_completeness": 0.87,
  "catalyst_analysis": [
    {
      "rank": 1,
      "catalyst": "Event and expected timing",
      "direction": "UPSIDE|DOWNSIDE|TWO_SIDED",
      "market_attention": "LOW|MEDIUM|HIGH with evidence",
      "why_it_matters": "Mechanism and expected impact",
      "confirmation": "Observable evidence that validates the impact"
    }
  ],
  "biggest_surprise_catalyst": "Catalyst and why consensus may underestimate it",
  "thesis_assumptions": {
    "bull": ["Assumption and supporting/required evidence"],
    "bear": ["Assumption and supporting/required evidence"]
  },
  "investor_questions": [
    "Question 1 an investor must answer before taking a position",
    "Question 2",
    "Question 3"
  ],
  "valuation_assessment": {
    "facts": ["Sourced valuation and operating facts"],
    "implied_expectations": ["Performance investors appear to be pricing in"],
    "undervalued_if": ["Conditional future outcome"],
    "fairly_valued_if": ["Conditional future outcome"],
    "overvalued_if": ["Conditional future outcome"]
  },
  "price_level_map": {
    "daily_trend": "UP|DOWN|SIDEWAYS|UNAVAILABLE",
    "weekly_trend": "UP|DOWN|SIDEWAYS|UNAVAILABLE",
    "support_levels": ["Level with rationale or UNAVAILABLE"],
    "resistance_levels": ["Level with rationale or UNAVAILABLE"],
    "bullish_scenario": {"confirmation": "Condition", "invalidation": "Condition", "key_levels": ["Level"]},
    "neutral_scenario": {"confirmation": "Condition", "invalidation": "Condition", "key_levels": ["Level"]},
    "bearish_scenario": {"confirmation": "Condition", "invalidation": "Condition", "key_levels": ["Level"]}
  }
}
"""

    LOCAL_QWEN_SYSTEM_INSTRUCTION = """
You are a senior swing trader and quantitative risk manager. Analyze only the supplied 1-10 day trading payload. Never invent missing market, portfolio, macro, or source data.

DECISION RULES:
- Treat the deterministic five-pillar composite score, setup geometry, and reward/risk as authoritative.
- BUY requires composite >= 70, data completeness >= 0.80, reward/risk >= 1.5, non-extreme volatility, and no earnings blackout. Otherwise return HOLD unless the evidence clearly supports SELL.
- SELL requires a weak quantitative structure or a high-reliability adverse catalyst confirmed by the supplied data.
- If days_to_earnings <= 3, return HOLD and identify the binary event risk.
- Treat news and macro as confirmation, vetoes, or risk flags; they cannot independently initiate a directional trade.
- Prefer SEC and company IR evidence over media and social commentary. Verify publication dates and never call stale evidence current.
- Explain divergences between trend, valuation, analyst targets, insider activity, and macro conditions.
- Provide concise, falsifiable bull/bear cases. State assumptions and missing information explicitly.
- Do not expose chain-of-thought. Return conclusions and concise evidence only.

STRICT OUTPUT CONTRACT:
Return exactly one JSON object with these keys and no markdown or commentary:
{
  "stock": "Ticker",
  "decision": "BUY|SELL|HOLD",
  "primary_driver": "QUANT_STRUCTURE|FUNDAMENTAL|NEWS_CATALYST|MACRO_EVENT|EARNINGS_CATALYST",
  "confidence": 0.0,
  "buy_score": 0.0,
  "hold_score": 0.0,
  "sell_score": 0.0,
  "horizon_days": 10,
  "quant_score": 0.0,
  "pillar_scores": {
    "trend": 0.0,
    "sector": 0.0,
    "alpha": 0.0,
    "valuation_history": 0.0,
    "peer_valuation": 0.0
  },
  "falsification_bull": "Condition that invalidates the SELL/HOLD case",
  "falsification_bear": "Condition that invalidates the BUY case",
  "bull_case": ["Concise bullish point"],
  "bear_case": ["Concise bearish point"],
  "key_risks": ["Concise risk"],
  "missing_information": ["Missing input or None"],
  "data_completeness": 0.0
}
buy_score, hold_score, and sell_score must each be in [0,1] and sum to approximately 1.0. pillar_scores and quant_score must be in [0,100]. confidence and data_completeness must be in [0,1].
"""

    LOCAL_PORTFOLIO_SYSTEM_INSTRUCTION = """
You are a risk-focused portfolio manager. Use only supplied holdings, allocations, classifications, and market data. Mark unavailable inputs explicitly and do not infer exposures that are not supplied.

Assess concentration, sector/geographic exposure, duplicated economic bets, correlation, growth/value and interest-rate sensitivity. Stress-test a 10% correction, 20% bear market, recession, higher rates, and volatility spike. Explain transmission channels, most exposed holdings, assumptions, limitations, diversification gaps, and conditional resilience options. Treat Fed policy as a first-class risk factor, but never invent dates, correlations, or exposures.

Return exactly one JSON object and no markdown:
{
  "portfolio_summary": "Concise overall risk assessment",
  "concentration_risks": [{"risk": "Description", "holdings": ["Ticker"], "severity": "LOW|MEDIUM|HIGH", "evidence": "Supplied evidence"}],
  "exposure_map": {"sector": [], "geographic": [], "growth_value": [], "rate_sensitivity": [], "correlated_clusters": []},
  "stress_tests": [{"scenario": "10% correction|20% bear market|recession|higher rates|volatility spike", "likely_impact": "Conditional analysis", "most_exposed": ["Ticker or cluster"], "assumptions_and_limits": ["Limitation"]}],
  "diversification_gaps": ["Gap"],
  "resilience_options": [{"possible_change": "Conditional change", "risk_reduced": "Risk", "trade_off": "Trade-off"}],
  "missing_information": ["Missing input"]
}
"""

    PORTFOLIO_SYSTEM_INSTRUCTION = """
You are a risk-focused portfolio manager. Review the supplied portfolio as an investment-committee risk memo, not as individualized investment advice. Use only supplied holdings, allocations, classifications, and market data. Mark unavailable inputs explicitly; do not infer allocations, correlations, geographic revenue exposure, duration, or factor exposures.

Evaluate concentration risk, sector and geographic exposure, duplicated economic bets, correlated positions, interest-rate sensitivity, growth/value factor exposure, and positions likely to move together during a drawdown. Distinguish a legal issuer diversification count from true economic diversification.

Stress-test the portfolio qualitatively against: a 10% broad-market correction, a 20% bear market, recession, higher interest rates, and a volatility spike. For each scenario, describe likely transmission channels, most exposed holdings or clusters, assumptions, and limitations. Do not invent precise loss estimates without position-level beta, correlation, and scenario data.

FEDERAL RESERVE POLICY ASSESSMENT MANDATE:
- Make rate policy a first-class dimension of the memo, not a footnote. Use any supplied fed funds rate, real/nominal yields, yield-curve level, and rate-velocity data to state the prevailing policy stance (restrictive, neutral, or accommodative) and the expected path of cuts or hikes over the next 3-12 months.
- Flag the likely FOMC meeting windows that create binary event risk over the analysis horizon, and identify which holdings or clusters are most exposed to a hawkish versus dovish surprise.
- For each rate-sensitive cluster, describe the mechanism with data, not a generic label: discount-rate/duration pressure on high-growth names, funding-cost and net-interest-margin effects on financials, refinancing/duration risk on credit-sensitive or high-leverage issuers, and yield-proxy flows into defensive dividend payers.
- Distinguish rate facts from rate assumptions, and list any missing rate inputs (e.g. dot plot, forward curve, position-level duration/beta) under missing_information rather than inferring them.

Identify the holdings and clusters that create the greatest risk, explain where diversification may be weaker than it appears, and offer possible resilience improvements as conditional trade-offs rather than directives. Return valid JSON only, matching the requested schema.

Schema:
{
  "portfolio_summary": "Concise overall risk assessment",
  "concentration_risks": [{"risk": "Description", "holdings": ["Ticker"], "severity": "LOW|MEDIUM|HIGH", "evidence": "Source-backed rationale"}],
  "exposure_map": {"sector": [], "geographic": [], "growth_value": [], "rate_sensitivity": [], "correlated_clusters": []},
  "stress_tests": [{"scenario": "10% correction|20% bear market|recession|higher rates|volatility spike", "likely_impact": "Conditional analysis", "most_exposed": ["Ticker or cluster"], "assumptions_and_limits": ["Limitation"]}],
  "diversification_gaps": ["Where diversification is weaker than it appears"],
  "resilience_options": [{"possible_change": "Conditional option", "risk_reduced": "Risk addressed", "trade_off": "Cost or lost exposure"}],
  "missing_information": ["Data required for a more reliable assessment"]
}
"""

    def __init__(self, top_k: int = 8):
        self.top_k = top_k
        print("[Metadata-Hybrid RAG] Loading FastEmbed BAAI/bge-small-en-v1.5 384-dim dense embedding model...")
        self.embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=450,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def convert_gloomberb_to_documents(self, gloomberb_payload: Dict, technical_data: Dict = None, institutional_data: Dict = None) -> List[Document]:
        """
        Converts qualitative textual data feeds strictly into LangChain Documents.
        Numerical indicators (RSI, EMA, ATR, Price, P/E, Yields, Options IV) are EXCLUDED from vector embedding
        and passed directly as structured data to the LLM.
        Applies Explicit Source Reliability Hierarchy:
        - SEC Filings: 1.00
        - Company IR & Press Releases: 0.95
        - Reuters / Bloomberg / Transcripts: 0.90
        - Analyst Research: 0.75
        - General Financial Media: 0.70
        - Social Media: 0.30
        """
        docs = []
        symbol = gloomberb_payload.get("symbol", "N/A")
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")

        # 1. Gloomberb News Catalysts -> Horizon: CURRENT (if <=14d) or HISTORICAL (if older)
        for item in gloomberb_payload.get("news", []):
            src = item.get('source', 'Gloomberb News')
            rel_score = 0.90 if any(w in src.lower() for w in ["reuters", "bloomberg", "wsj", "gloomberb"]) else 0.70
            pub_date = str(item.get('published_at') or item.get('publishedAt') or today_str)[:10]
            try:
                days_diff = (datetime.datetime.now() - datetime.datetime.strptime(pub_date, "%Y-%m-%d")).days
                horizon = "CURRENT" if days_diff <= 14 else "HISTORICAL"
            except Exception:
                horizon = "CURRENT"

            text = f"[{src} Catalyst (Published: {pub_date})] {item.get('title', '')}. Details: {item.get('summary', '')}"
            docs.append(Document(page_content=text, metadata={
                "ticker": symbol,
                "source": src,
                "document_type": "AggregatedNews",
                "published_at": pub_date,
                "effective_date": pub_date,
                "reliability": rel_score,
                "importance": 0.65,
                "time_horizon": horizon
            }))

        # 2. SEC EDGAR Filings (10-K = HISTORICAL, 10-Q / 8-K = CURRENT)
        for filing in gloomberb_payload.get("filings", []):
            form_type = str(filing.get('form', '')).upper()
            horizon = "HISTORICAL" if "10-K" in form_type else "CURRENT"
            text = f"[SEC EDGAR Form {filing.get('form')} ({filing.get('date')})] {filing.get('summary')}"
            docs.append(Document(page_content=text, metadata={
                "ticker": symbol,
                "source": "SEC EDGAR",
                "document_type": f"Form-{filing.get('form', 'Filing')}",
                "published_at": filing.get('date', today_str),
                "effective_date": filing.get('date', today_str),
                "reliability": 1.00,
                "importance": 1.00,
                "time_horizon": horizon
            }))

        # 3. Form 4 Insider Transactions & 13F Institutional Holdings -> Horizon: HISTORICAL
        insider = gloomberb_payload.get("insider_institutional", {})
        if insider:
            inst_pct = insider.get("institutional_ownership_pct", "N/A")
            txs = insider.get("insider_transactions", [])
            tx_text = "; ".join(f"{t.get('insider')}: {t.get('transaction')} ({t.get('shares')} shs)" for t in txs) if txs else "Form 4 filings logged."
            insider_text = f"[Form 4 Insiders & 13F Holdings] Institutional Ownership: {inst_pct}. Recent Insider Activity: {tx_text}."
            docs.append(Document(page_content=insider_text, metadata={
                "ticker": symbol,
                "source": "SEC EDGAR (Form 4/13F)",
                "document_type": "Form-4",
                "published_at": today_str,
                "effective_date": today_str,
                "reliability": 1.00,
                "importance": 0.85,
                "time_horizon": "HISTORICAL"
            }))

        # 4. Company Profile & Business Architecture Description -> Horizon: HISTORICAL
        profile = gloomberb_payload.get("profile", {})
        if profile and profile.get("description") != "N/A":
            profile_text = f"[Company Profile & Business Description] Sector: {profile.get('sector')}. Description: {profile.get('description')}"
            docs.append(Document(page_content=profile_text, metadata={
                "ticker": symbol,
                "source": "Gloomberb Profile",
                "document_type": "CompanyProfile",
                "published_at": today_str,
                "effective_date": today_str,
                "reliability": 0.85,
                "importance": 0.75,
                "time_horizon": "HISTORICAL"
            }))

        # 5. Wall Street Analyst Research, Ratings & Price Targets -> Horizon: CURRENT
        analyst = gloomberb_payload.get("analyst_ratings", {})
        if analyst and (analyst.get("recent_major_bank_actions") or analyst.get("mean_target_price") != "N/A"):
            actions_text = "; ".join(analyst.get("recent_major_bank_actions", [])[:6])
            analyst_text = (
                f"[Wall Street Analyst Research & Consensus] Mean Target Price: ${analyst.get('mean_target_price')}, "
                f"Median Target: ${analyst.get('median_target_price')}, High: ${analyst.get('high_target_price')}, Low: ${analyst.get('low_target_price')}. "
                f"Recent Major Bank Ratings: {actions_text if actions_text else 'Consensus ratings active.'}"
            )
            docs.append(Document(page_content=analyst_text, metadata={
                "ticker": symbol,
                "source": "Gloomberb Analyst Research",
                "document_type": "AnalystRatings",
                "published_at": today_str,
                "effective_date": today_str,
                "reliability": 0.85,
                "importance": 0.85,
                "time_horizon": "CURRENT"
            }))

        # 6. Upcoming Earnings Guidance, Estimates & Revision Momentum -> Horizon: CURRENT
        earnings = gloomberb_payload.get("earnings", {})
        if earnings and earnings.get("earnings_date") != "N/A":
            rev7 = earnings.get("eps_revisions_7d", {})
            rev30 = earnings.get("eps_revisions_30d", {})
            earnings_text = (
                f"[Earnings Calendar & Revision Momentum] Upcoming Earnings: {earnings.get('earnings_date')} ({earnings.get('timing')}). "
                f"EPS Consensus: ${earnings.get('eps_estimate')} (YoY Growth: {earnings.get('eps_growth_yoy')}), "
                f"Revenue Estimate: {earnings.get('revenue_estimate')}. "
                f"Analyst EPS Revisions: Last 7 Days (Up: {rev7.get('up', 0)}, Down: {rev7.get('down', 0)}), "
                f"Last 30 Days (Up: {rev30.get('up', 0)}, Down: {rev30.get('down', 0)})."
            )
            docs.append(Document(page_content=earnings_text, metadata={
                "ticker": symbol,
                "source": "Gloomberb Earnings Calendar",
                "document_type": "EarningsGuidance",
                "published_at": today_str,
                "effective_date": today_str,
                "reliability": 0.95,
                "importance": 0.90,
                "time_horizon": "CURRENT"
            }))

        # 7. Historical Earnings Surprises & Corporate Events -> Horizon: HISTORICAL
        events = gloomberb_payload.get("events", {})
        if events and events.get("historical_earnings_surprises"):
            surprises = events.get("historical_earnings_surprises", [])
            s_text = "; ".join(f"[{s.get('date')}: Actual ${s.get('actual')} vs Est ${s.get('estimate')} ({s.get('surprise_pct')})]" for s in surprises[:4])
            events_text = f"[Historical Earnings Surprises & Execution History] Past Quarters: {s_text}."
            docs.append(Document(page_content=events_text, metadata={
                "ticker": symbol,
                "source": "Gloomberb Corporate Events",
                "document_type": "EarningsSurprises",
                "published_at": today_str,
                "effective_date": today_str,
                "reliability": 1.00,
                "importance": 0.85,
                "time_horizon": "HISTORICAL"
            }))

        # 8. Major Benchmark Market Indices & Market Breadth -> Horizon: CURRENT
        indices = gloomberb_payload.get("market_indices", [])
        movers = gloomberb_payload.get("market_movers", [])
        if indices or movers:
            idx_str = ", ".join(f"{i.get('name', i.get('symbol'))} {i.get('change_pct')}" for i in indices)
            mover_str = ", ".join(f"{m.get('symbol')} {m.get('change_pct')}" for m in movers[:4]) if movers else "Normal breadth"
            market_text = f"[Major Benchmark Indices & Market Breadth] Index Performance: {idx_str}. Top Active Movers: {mover_str}."
            docs.append(Document(page_content=market_text, metadata={
                "ticker": symbol,
                "source": "Gloomberb Benchmark Indices",
                "document_type": "MarketBreadth",
                "published_at": today_str,
                "effective_date": today_str,
                "reliability": 0.95,
                "importance": 0.80,
                "time_horizon": "CURRENT"
            }))

        # 9. Institutional Multi-Source Qualitative Text Documents
        if institutional_data:
            # 5a. Company Investor-Relations (IR) Website Releases -> Horizon: CURRENT
            for ir in institutional_data.get("investor_relations", []):
                text = f"[Company Investor Relations ({ir.get('source')})] {ir.get('title')}. Summary: {ir.get('summary')}"
                docs.append(Document(page_content=text, metadata={
                    "ticker": symbol,
                    "source": ir.get('source', 'Investor Relations'),
                    "document_type": ir.get('document_type', 'IRRelease'),
                    "published_at": ir.get('published_at', today_str),
                    "effective_date": ir.get('published_at', today_str),
                    "reliability": ir.get('reliability', 0.95),
                    "importance": ir.get('importance', 0.90),
                    "time_horizon": "CURRENT"
                }))

            # 5b. SEC EDGAR Direct Filings -> Horizon: CURRENT (8-K / 10-Q) or HISTORICAL (10-K)
            for sec in institutional_data.get("sec_edgar_direct", []):
                form_type = str(sec.get('form', '')).upper()
                horizon = "HISTORICAL" if "10-K" in form_type else "CURRENT"
                text = f"[SEC EDGAR Direct Filing Form {sec.get('form')} ({sec.get('date')})] {sec.get('summary')}"
                docs.append(Document(page_content=text, metadata={
                    "ticker": symbol,
                    "source": "SEC EDGAR Direct",
                    "document_type": sec.get('document_type', f"Form-{sec.get('form')}"),
                    "published_at": sec.get('published_at', today_str),
                    "effective_date": sec.get('published_at', today_str),
                    "reliability": sec.get('reliability', 1.00),
                    "importance": sec.get('importance', 1.00),
                    "time_horizon": horizon
                }))

            # 5c. Quarterly Earnings Call Transcript Highlights -> Horizon: CURRENT
            for call in institutional_data.get("earnings_calls", []):
                text = f"[Quarterly Earnings Call Transcript Takeaways] {call.get('summary')}"
                docs.append(Document(page_content=text, metadata={
                    "ticker": symbol,
                    "source": "Earnings Call Feed",
                    "document_type": call.get('document_type', 'EarningsTranscript'),
                    "published_at": call.get('published_at', today_str),
                    "effective_date": call.get('published_at', today_str),
                    "reliability": call.get('reliability', 0.90),
                    "importance": call.get('importance', 1.00),
                    "time_horizon": "CURRENT"
                }))

            # 5d. Official Corporate Press Releases -> Horizon: CURRENT
            for pr in institutional_data.get("official_press_releases", []):
                text = f"[Official Corporate Press Release ({pr.get('source')})] {pr.get('title')}. Summary: {pr.get('summary')}"
                docs.append(Document(page_content=text, metadata={
                    "ticker": symbol,
                    "source": pr.get('source', 'Press Release'),
                    "document_type": pr.get('document_type', 'PressRelease'),
                    "published_at": pr.get('published_at', today_str),
                    "effective_date": pr.get('published_at', today_str),
                    "reliability": pr.get('reliability', 0.95),
                    "importance": pr.get('importance', 0.75),
                    "time_horizon": "CURRENT"
                }))

            # 5e. Reputable Financial Media News Coverage & Institutional Commentary -> Horizon: CURRENT
            for news_item in institutional_data.get("reputable_news", []):
                text = f"[Reputable Financial News ({news_item.get('source')})] {news_item.get('title')}. Details: {news_item.get('summary')}"
                docs.append(Document(page_content=text, metadata={
                    "ticker": symbol,
                    "source": news_item.get('source', 'Reputable News'),
                    "document_type": news_item.get('document_type', 'ReputableNews'),
                    "published_at": news_item.get('published_at', today_str),
                    "effective_date": news_item.get('published_at', today_str),
                    "reliability": news_item.get('reliability', 0.70),
                    "importance": news_item.get('importance', 0.70),
                    "time_horizon": "CURRENT"
                }))

        return docs

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generates 384-dimensional dense vector embeddings using FastEmbed."""
        embeddings_generator = self.embed_model.embed(texts)
        embeddings_list = [vec for vec in embeddings_generator]
        return np.array(embeddings_list, dtype=np.float32)

    def get_analytical_questions(self, symbol: str) -> Dict[str, Dict]:
        """
        Returns targeted sub-queries partitioned by temporal horizon:
        CURRENT CONTEXT (Last 24h / 7d / Latest Earnings) vs LONGER-TERM HISTORICAL CONTEXT.
        """
        return {
            # CURRENT CONTEXT (Last 24h / 7d / Latest Earnings)
            "Recent Material Events (Last 24h / 7d)": {
                "horizon": "CURRENT",
                "query": f"What recent material events, SEC 8-K/10-Q filings, press releases, or news catalysts changed for {symbol} in the last 7 days?"
            },
            "Short-Term Bullish & Bearish Drivers": {
                "horizon": "CURRENT",
                "query": f"What are the primary short-term bullish growth drivers and bearish downside risks for {symbol}?"
            },
            "Updated Guidance & Earnings Takeaways": {
                "horizon": "CURRENT",
                "query": f"What updated management guidance, executive outlook, or capex plans were issued in the latest earnings release for {symbol}?"
            },
            # LONGER-TERM HISTORICAL CONTEXT
            "Longer-Term Historical Execution Pattern": {
                "horizon": "HISTORICAL",
                "query": f"What is the longer-term historical business trend, product architecture, and multi-year execution pattern for {symbol}?"
            },
            "Persistent Structural & Valuation Headwinds": {
                "horizon": "HISTORICAL",
                "query": f"What historical valuation concerns, balance sheet liabilities, or structural headwinds have persisted over time for {symbol}?"
            },
            "Historical Macro & Industry Cycles": {
                "horizon": "HISTORICAL",
                "query": f"How do historical macro interest rate cycles, yield curve trends, and industry competitive dynamics impact {symbol}?"
            }
        }

    def retrieve_knowledge_by_questions(self, gloomberb_payload: Dict, technical_data: Dict = None, institutional_data: Dict = None, max_per_question: int = 2) -> Dict[str, List[str]]:
        """
        Executes Multi-Query Dual-Horizon Hybrid RAG Retrieval:
        1. Ingests all document chunks with time_horizon metadata ('CURRENT' vs 'HISTORICAL').
        2. Generates document vector embeddings.
        3. Queries vector space across 6 analytical questions partitioned by temporal horizon.
        4. Calculates Multiplicative Hybrid Score (Similarity x Reliability x Importance x Recency) per question.
        5. Returns top unique passages per question category.
        """
        symbol = gloomberb_payload.get("symbol", "N/A")
        print(f"[Multi-Query Hybrid RAG] Ingesting Gloomberb & Institutional Data for {symbol}...")

        raw_docs = self.convert_gloomberb_to_documents(gloomberb_payload, technical_data, institutional_data)
        doc_chunks = self.text_splitter.split_documents(raw_docs)
        chunk_texts = [d.page_content for d in doc_chunks]

        if not chunk_texts:
            return {"General": ["No document chunks available."]}

        print(f"[Multi-Query Hybrid RAG] Generating FastEmbed dense vectors for {len(chunk_texts)} document chunks...")
        doc_embeddings = self.generate_embeddings(chunk_texts)
        norm_docs = np.linalg.norm(doc_embeddings, axis=1)
        norm_docs[norm_docs == 0] = 1e-9

        questions_dict = self.get_analytical_questions(symbol)
        categorized_results = {}
        global_used_indices = set()

        print(f"[Multi-Query Hybrid RAG] Executing Dual-Horizon Multi-Query Search across 6 Analytical Categories:")

        for cat_name, q_info in questions_dict.items():
            target_horizon = q_info["horizon"]
            q_text = q_info["query"]

            q_emb = self.generate_embeddings([q_text])[0]
            norm_q = np.linalg.norm(q_emb)
            if norm_q == 0:
                norm_q = 1e-9

            cosine_sims = np.dot(doc_embeddings, q_emb) / (norm_docs * norm_q)

            scored_chunks = []
            for idx, doc in enumerate(doc_chunks):
                meta = doc.metadata or {}
                doc_horizon = meta.get("time_horizon", "CURRENT")

                # Filter by temporal horizon preference
                if doc_horizon != target_horizon:
                    horizon_penalty = 0.50  # Soft penalty if cross-horizon match
                else:
                    horizon_penalty = 1.00

                sim = float(cosine_sims[idx])
                rel = float(meta.get("reliability", 0.75))
                imp = float(meta.get("importance", 0.75))
                pub = meta.get("published_at", "N/A")
                rec = compute_recency_score(pub, doc_horizon)
                hybrid_score = compute_hybrid_score(sim, rel, imp, rec) * horizon_penalty

                scored_chunks.append({
                    "idx": idx,
                    "chunk": doc,
                    "hybrid_score": hybrid_score,
                    "sim": sim,
                    "metadata": meta
                })

            # Rank by Hybrid Score descending
            scored_chunks.sort(key=lambda x: x["hybrid_score"], reverse=True)

            category_passages = []
            count = 0
            for item in scored_chunks:
                if count >= max_per_question:
                    break
                idx = item["idx"]
                if idx in global_used_indices:
                    continue  # Skip duplicate chunks across questions

                global_used_indices.add(idx)
                count += 1

                m = item["metadata"]
                passage_text = item["chunk"].page_content
                formatted_passage = f"[Metadata: Ticker={m.get('ticker')} | Horizon={m.get('time_horizon')} | Source={m.get('source')} | Type={m.get('document_type')} | Published={m.get('published_at')} | Reliability={m.get('reliability')} | Importance={m.get('importance')}]\nContent: {passage_text}"
                category_passages.append(formatted_passage)

            print(f"  • [{target_horizon}] {cat_name}: Retrieved {len(category_passages)} top hybrid-ranked passages.")
            categorized_results[cat_name] = category_passages

        return categorized_results

    def retrieve_knowledge(self, gloomberb_payload: Dict, technical_data: Dict = None, institutional_data: Dict = None, query: str = None) -> List[str]:
        """Legacy single-query interface fallback, wrapper around retrieve_knowledge_by_questions."""
        cat_map = self.retrieve_knowledge_by_questions(gloomberb_payload, technical_data, institutional_data)
        flat_list = []
        for passages in cat_map.values():
            flat_list.extend(passages)
        return flat_list

    def get_nemotron_payload(
        self,
        gloomberb_payload: Dict,
        technical_data: Dict,
        institutional_data: Dict = None,
        prompt_profile: str = "full",
    ) -> Dict:
        """Build a full cloud prompt or a compact 8K local-model prompt."""
        if prompt_profile not in {"full", "compact"}:
            raise RAGPromptProfileError("prompt_profile must be 'full' or 'compact'")

        compact = prompt_profile == "compact"
        symbol = gloomberb_payload.get("symbol", "N/A")
        categorized_rag = self.retrieve_knowledge_by_questions(
            gloomberb_payload,
            technical_data,
            institutional_data,
            max_per_question=1 if compact else 2,
        )

        profile = gloomberb_payload.get("profile", {})
        peer_val = gloomberb_payload.get("peer_valuation", {})
        fin_ratios = gloomberb_payload.get("financials", {}).get("key_ratios", {})
        macro_econ = gloomberb_payload.get("macro_econ", {})
        sector_bench = gloomberb_payload.get("sector_benchmark", {})
        options_flow = gloomberb_payload.get("options", {})

        # Compute deterministic 5-pillar quantitative scores (0-100)
        quant_service = QuantitativeScoringService()
        quant_scores = quant_service.compute_5pillar_scores(technical_data, gloomberb_payload, sector_bench)

        # Channel-anchored Reward:Risk setup geometry -- varies with where price
        # sits inside its 20-day range (conservative stop/target), not fixed 2.5/1.5.
        struct_rr_info = QuantitativeScoringService.compute_channel_reward_risk(
            technical_data.get("current_price"),
            technical_data.get("atr"),
            technical_data.get("high_20d"),
            technical_data.get("low_20d"),
            technical_data.get("suggested_stop_loss"),
            technical_data.get("suggested_target_price")
        )
        # Wall Street mean target as a 12-month directional sanity check, NOT a 10-day reward.
        analyst_rr_info = QuantitativeScoringService.compute_reward_risk(
            technical_data.get("current_price"),
            technical_data.get("suggested_stop_loss"),
            technical_data.get("analyst_target_price")
        )

        # Aggregated source availability so the LLM can distinguish "no data"
        # from genuinely sourced material instead of fabricated/assumed content.
        source_status = dict(gloomberb_payload.get("data_source_status", {}))
        if institutional_data and isinstance(institutional_data, dict):
            source_status.update(institutional_data.get("source_status", {}))

        market_benchmark_summary = {
            "symbol": symbol,
            "data_source_status": source_status,
            "sector": profile.get("sector", "N/A"),
            "deterministic_5pillar_scores": quant_scores,
            "setup_geometry": {
                "entry_price": technical_data.get("current_price"),
                "stop_loss": technical_data.get("suggested_stop_loss"),
                "swing_target_10d": technical_data.get("suggested_target_price"),
                "structural_stop": struct_rr_info.get("structural_stop"),
                "structural_target": struct_rr_info.get("structural_target"),
                "distance_to_resistance_atr": struct_rr_info.get("distance_to_resistance_atr"),
                "distance_to_support_atr": struct_rr_info.get("distance_to_support_atr"),
                "reward_risk_ratio": struct_rr_info.get("reward_risk_ratio"),
                "breakeven_win_rate": struct_rr_info.get("breakeven_win_rate"),
                "wall_street_mean_target_12m": technical_data.get("analyst_target_price"),
                "wall_street_target_rr": analyst_rr_info.get("reward_risk_ratio")
            },
            "current_price": technical_data.get("current_price"),
            "change_5d_pct": technical_data.get("change_5d_pct"),
            "market_spy_5d_pct": technical_data.get("market_spy_5d_pct"),
            "relative_alpha_5d": technical_data.get("relative_alpha_5d"),
            "sector_etf_benchmark": sector_bench,
            "return_1y": profile.get("return_1y", "N/A"),
            "return_3y": profile.get("return_3y", "N/A"),
            "valuation_multiples": {
                "forward_pe": fin_ratios.get("forward_pe", "N/A"),
                "trailing_pe": fin_ratios.get("trailing_pe", "N/A"),
                "price_to_sales": peer_val.get("price_to_sales", "N/A"),
                "ev_to_ebitda": peer_val.get("ev_to_ebitda", "N/A"),
                "price_to_free_cash_flow": peer_val.get("price_to_free_cash_flow", "N/A")
            },
            "direct_peer_benchmarks": peer_val.get("direct_peer_benchmarks", []),
            "macro_market_sentiment": macro_econ,
            "benchmark_market_indices": gloomberb_payload.get("market_indices", []),
            "market_spy_correlation": gloomberb_payload.get("market_spy_correlation", {}),
            "realtime_quote": gloomberb_payload.get("quote", {}),
            "wall_street_analyst_coverage": {
                "mean_target_price": gloomberb_payload.get("analyst_ratings", {}).get("mean_target_price"),
                "median_target_price": gloomberb_payload.get("analyst_ratings", {}).get("median_target_price"),
                "high_target_price": gloomberb_payload.get("analyst_ratings", {}).get("high_target_price"),
                "low_target_price": gloomberb_payload.get("analyst_ratings", {}).get("low_target_price"),
                "recommendation_rating": gloomberb_payload.get("analyst_ratings", {}).get("recommendation_rating"),
                "recommendations_breakdown": gloomberb_payload.get("analyst_ratings", {}).get("recommendations_breakdown", {}),
                "recent_major_bank_actions": gloomberb_payload.get("analyst_ratings", {}).get("recent_major_bank_actions", [])[:5]
            },
            "upcoming_earnings_and_revisions": gloomberb_payload.get("earnings", {}),
            "historical_earnings_surprises": gloomberb_payload.get("events", {}).get("historical_earnings_surprises", []),
            "top_institutional_holders": gloomberb_payload.get("insider_institutional", {}).get("top_institutional_holders", []),
            "granular_options_flow": {
                "put_call_ratio": options_flow.get("put_call_ratio"),
                "call_volume": options_flow.get("call_volume"),
                "put_volume": options_flow.get("put_volume"),
                "call_open_interest": options_flow.get("call_open_interest"),
                "put_open_interest": options_flow.get("put_open_interest"),
                "implied_volatility": options_flow.get("implied_volatility"),
                "unusual_activity": options_flow.get("unusual_activity")
            },
            "growth_and_margins": {
                "revenue_growth": fin_ratios.get("revenue_growth", "N/A"),
                "earnings_growth": fin_ratios.get("earnings_growth", "N/A"),
                "profit_margins": fin_ratios.get("profit_margins", "N/A")
            },
            "technical_indicators": {
                "rsi14": technical_data.get("rsi14"),
                "ema20": technical_data.get("ema20"),
                "ema50": technical_data.get("ema50"),
                "atr": technical_data.get("atr"),
                "rvol_20d": technical_data.get("rvol_20d"),
                "vol_20d_mean": technical_data.get("vol_20d_mean"),
                "high_20d": technical_data.get("high_20d"),
                "low_20d": technical_data.get("low_20d"),
                "suggested_stop_loss": technical_data.get("suggested_stop_loss"),
                "suggested_target_price": technical_data.get("suggested_target_price"),
                "days_to_earnings": technical_data.get("days_to_earnings")
            }
        }

        analyst = gloomberb_payload.get("analyst_ratings", {})
        earnings = gloomberb_payload.get("earnings", {})
        insider_institutional = gloomberb_payload.get("insider_institutional", {})
        interest_rate_outlook = macro_econ.get("interest_rate_outlook", macro_econ)

        if compact:
            market_benchmark_summary = _bounded_json_value(
                {
                    "symbol": symbol,
                    "sector": profile.get("sector", "N/A"),
                    "source_status": source_status,
                    "deterministic_5pillar_scores": quant_scores,
                    "setup_geometry": market_benchmark_summary["setup_geometry"],
                    "price_action": {
                        "current_price": technical_data.get("current_price"),
                        "change_5d_pct": technical_data.get("change_5d_pct"),
                        "market_spy_5d_pct": technical_data.get("market_spy_5d_pct"),
                        "relative_alpha_5d": technical_data.get("relative_alpha_5d"),
                        "sector_etf_benchmark": sector_bench,
                        "return_1y": profile.get("return_1y", "N/A"),
                    },
                    "technicals": market_benchmark_summary["technical_indicators"],
                    "valuation": market_benchmark_summary["valuation_multiples"],
                    "direct_peers": peer_val.get("direct_peer_benchmarks", []),
                    "growth_and_margins": market_benchmark_summary["growth_and_margins"],
                    "analyst": {
                        "mean_target_price": analyst.get("mean_target_price"),
                        "recommendation_rating": analyst.get("recommendation_rating"),
                        "recent_major_bank_actions": analyst.get("recent_major_bank_actions", [])[:3],
                    },
                    "earnings": {
                        "earnings_date": earnings.get("earnings_date"),
                        "timing": earnings.get("timing"),
                        "eps_estimate": earnings.get("eps_estimate"),
                        "revenue_estimate": earnings.get("revenue_estimate"),
                        "days_to_earnings": technical_data.get("days_to_earnings"),
                    },
                    "options": market_benchmark_summary["granular_options_flow"],
                    "institutional": {
                        "top_holders": insider_institutional.get("top_institutional_holders", [])[:2],
                        "insider_transactions": insider_institutional.get("insider_transactions", [])[:3],
                    },
                    "interest_rate_outlook": interest_rate_outlook,
                },
                max_depth=5,
                max_items=8,
                max_string_chars=400,
            )
            user_prompt = f"""
================ COMPACT LOCAL EVIDENCE PACKAGE ================
Target: {symbol}
Unavailable source channels are absent evidence, not bullish or bearish facts. Do not fabricate missing values.

STRUCTURED DATA:
{json.dumps(market_benchmark_summary, separators=(",", ":"), ensure_ascii=False)}

CURRENT CONTEXT (latest 24h/7d and earnings):
"""
        else:
            user_prompt = f"""
================ 0. DATA SOURCE AVAILABILITY ================
Channels marked "source_unavailable" in data_source_status above produced NO real document. Treat them as absent: do NOT invent, assume, or "aggregate" placeholder content for them, and do NOT cite them as evidence. Factor their absence into data_completeness and missing_information.
Also, CURRENT CONTEXT passages (Section 2) only contain real, sourced documents (or an explicit "no passages" marker when none matched). If a category is empty, there is no supporting text — never fabricate one.

================ 1. STRUCTURED MARKET & QUANTITATIVE DATA ================
Target Ticker: {symbol}

{json.dumps(market_benchmark_summary, indent=2)}

================ 2. CURRENT CONTEXT (Last 24h / 7d / Latest Earnings) ================
"""

        current_cats = ["Recent Material Events (Last 24h / 7d)", "Short-Term Bullish & Bearish Drivers", "Updated Guidance & Earnings Takeaways"]
        for cat_name in current_cats:
            passages = categorized_rag.get(cat_name, [])
            user_prompt += f"\n>>> SUB-QUESTION: {cat_name} <<<\n"
            if not passages:
                user_prompt += "No specific current passages matched this category.\n"
            for p_idx, passage in enumerate(passages, 1):
                rendered_passage = _clip_prompt_text(str(passage), 700) if compact else str(passage)
                user_prompt += f"  [{p_idx}] {rendered_passage}\n\n"

        user_prompt += """
================ 3. LONGER-TERM HISTORICAL CONTEXT (Prior Filings & Multi-Year Patterns) ================
"""
        historical_cats = ["Longer-Term Historical Execution Pattern", "Persistent Structural & Valuation Headwinds", "Historical Macro & Industry Cycles"]
        for cat_name in historical_cats:
            passages = categorized_rag.get(cat_name, [])
            user_prompt += f"\n>>> SUB-QUESTION: {cat_name} <<<\n"
            if not passages:
                user_prompt += "No specific historical passages matched this category.\n"
            for p_idx, passage in enumerate(passages, 1):
                rendered_passage = _clip_prompt_text(str(passage), 700) if compact else str(passage)
                user_prompt += f"  [{p_idx}] {rendered_passage}\n\n"

        if compact:
            user_prompt += """
================ END PAYLOAD ================
Verify publication dates and source reliability. Treat earnings within 3 days, extreme implied volatility, RR below 1.5, incomplete data, or unresolved macro transmission as reasons to reduce conviction. Reconcile technical, valuation, analyst, insider, and macro evidence. Keep the final rationale concise and return only the required JSON object.
"""
        else:
            user_prompt += """
================ END PAYLOAD ================

Synthesize the Structured Market Data, Deterministic 5-Pillar Scores, Current Context (Last 24h/7d), and Longer-Term Historical Context.
Execute multi-step analytical reasoning:
1. Chronological & Catalyst Verification: Verify the publication dates of all passages. Do NOT treat prior-quarter results or stale headlines as immediate 24h/7d catalysts.
2. Event-Risk Assessment: If earnings are within 3 days (days_to_earnings <= 3) or implied volatility is high, evaluate the options market's binary gap risk. Do not rely solely on backward-looking momentum.
3. Divergence Resolution: Reconcile any divergence between technical momentum and peer valuation multiples or insider transactions.
4. Anchor your final decision and probabilities around the Composite Quantitative Score.
5. Fold in setup_geometry: a BUY requires reward_risk_ratio >= 1.5 as computed from the channel-anchored structural stop and target (capped by the 20-day range); if distance_to_resistance_atr < 1.0, the entry is likely a chase. If Wall Street's 12-month mean target implies wall_street_target_rr < 1.0, treat it as a directional contradiction to any BUY.
6. Fed Rate-Policy Integration: from the interest_rate_outlook block, state the implied Fed stance (restrictive/neutral/accommodative), the expected cut/hike path and FOMC timing window, and how that path transmits to this company's valuation, debt/financing costs, and sector before finalizing the decision. If fed funds rate, real yields, or a rate-path signal is missing, say so in missing_information instead of assuming neutral policy.
Return a valid JSON object matching the required schema.
"""

        system_instruction = (
            self.LOCAL_QWEN_SYSTEM_INSTRUCTION.strip()
            if compact
            else self.NEMOTRON_SYSTEM_INSTRUCTION.strip()
        )
        return {
            "system_instruction": system_instruction,
            "user_prompt": user_prompt.strip(),
            "technical_summary": market_benchmark_summary,
            "prompt_profile": prompt_profile,
        }

    def get_portfolio_analysis_payload(
        self,
        portfolio: Dict,
        analysis_horizon: str = "next 90 days",
        prompt_profile: str = "full",
    ) -> Dict:
        """Build the portfolio-risk prompt without fabricating unavailable data."""
        if prompt_profile not in {"full", "compact"}:
            raise RAGPromptProfileError("prompt_profile must be 'full' or 'compact'")
        if not isinstance(portfolio, dict):
            portfolio = {}

        compact = prompt_profile == "compact"
        if compact:
            prompt_portfolio = _bounded_json_value(
                portfolio,
                max_depth=6,
                max_items=24,
                max_string_chars=500,
            )
            user_prompt = f"""
================ COMPACT PORTFOLIO RISK REVIEW ================
Analysis horizon: {analysis_horizon}

PORTFOLIO DATA:
{json.dumps(prompt_portfolio, separators=(",", ":"), ensure_ascii=False)}

Review concentration, duplicated bets, factor/rate sensitivity, and the five required stress scenarios. Use only supplied fields, state uncertainty, and return only the required JSON object.
"""
            system_instruction = self.LOCAL_PORTFOLIO_SYSTEM_INSTRUCTION.strip()
        else:
            user_prompt = f"""
================ PORTFOLIO RISK REVIEW ================
Analysis horizon: {analysis_horizon}

Portfolio data:
{json.dumps(portfolio, indent=2)}

Required review:
1. Map concentration, sector, geographic, duplicated-bet, correlation, interest-rate, and growth/value risks.
2. Stress-test a 10% correction, 20% bear market, recession, higher rates, and volatility spike.
3. Identify the holdings or clusters creating the most portfolio risk and any diversification that is only apparent.
4. Offer conditional resilience options, each with their trade-off.

Return only the JSON object defined in the system instruction. If allocations, classifications, beta/correlation data, or geographic exposure are absent, list them in missing_information and explain the resulting limitation instead of guessing.
"""
            system_instruction = self.PORTFOLIO_SYSTEM_INSTRUCTION.strip()

        return {
            "system_instruction": system_instruction,
            "user_prompt": user_prompt.strip(),
            "prompt_profile": prompt_profile,
        }
