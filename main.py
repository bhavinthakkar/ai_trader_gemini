import sys
import argparse
import os
import json
import time
import requests
from dotenv import load_dotenv

from llm_service import query_llm, get_model_label, extract_json
from db import init_db, save_results
from market_agent import MarketAgent
from institutional_agent import InstitutionalDataAgent
from macro_agent import MacroDataAgent
from news_agent import NewsAgent
from risk_agent import RiskAgent
from analyst_agent import AnalystAgent

from gloomberb_service import GloomberbService
from institutional_data_service import InstitutionalDataService
from rag_service import RAGService

WATCHLIST = ["000660.KS"]

CHAT_ID = "969601315"

load_dotenv()
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


def normalize_master_trader_json(data: dict, symbol: str, m_data: dict = None) -> dict:
    if not isinstance(data, dict):
        data = {}
    if not isinstance(m_data, dict):
        m_data = {}

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

    # Normalize Confidence
    raw_conf = data.get("confidence") or buy_score or 0.70
    confidence = _to_float(raw_conf, 0.70)
    if confidence > 1.0:
        confidence = round(confidence / 100.0, 2) if confidence <= 100 else 0.70

    try:
        horizon_days = int(data.get("horizon_days") or 10)
    except (ValueError, TypeError):
        horizon_days = 10

    # Quantitative 5-Pillar Scores
    deterministic_scores = m_data.get("deterministic_5pillar_scores", {})
    quant_score = _to_float(data.get("quant_score") or deterministic_scores.get("composite_score"), 65.0)

    raw_pillars = data.get("pillar_scores") or {}
    if not isinstance(raw_pillars, dict):
        raw_pillars = {}

    pillar_scores = {
        "trend": _to_float(raw_pillars.get("trend") or deterministic_scores.get("trend_score"), 50.0),
        "sector": _to_float(raw_pillars.get("sector") or deterministic_scores.get("sector_score"), 50.0),
        "alpha": _to_float(raw_pillars.get("alpha") or deterministic_scores.get("market_alpha_score"), 50.0),
        "valuation_history": _to_float(raw_pillars.get("valuation_history") or deterministic_scores.get("valuation_history_score"), 50.0),
        "peer_valuation": _to_float(raw_pillars.get("peer_valuation") or deterministic_scores.get("peer_valuation_score"), 50.0)
    }

    # Array Fields
    bull_case = _to_list_of_strings(data.get("bull_case") or data.get("reason"), "Bullish trade setup and vector RAG catalysts evaluated.")
    bear_case = _to_list_of_strings(data.get("bear_case") or data.get("risk_assessment"), "Bearish trade setup and downside risks evaluated.")
    key_risks = _to_list_of_strings(data.get("key_risks") or data.get("key_risk"), "Volatile market conditions and stop-loss boundaries.")
    missing_info = _to_list_of_strings(data.get("missing_information"), "None")

    data_completeness = _to_float(data.get("data_completeness"), 0.87)

    return {
        "stock": _ensure_str(stock),
        "decision": decision,
        "confidence": confidence,
        "buy_score": buy_score,
        "hold_score": hold_score,
        "sell_score": sell_score,
        "horizon_days": horizon_days,
        "quant_score": quant_score,
        "pillar_scores": pillar_scores,
        "bull_case": bull_case,
        "bear_case": bear_case,
        "key_risks": key_risks,
        "missing_information": missing_info,
        "data_completeness": data_completeness,
        # Backward compatibility for SQLite DB string storage
        "reason": "; ".join(bull_case),
        "risk_assessment": "; ".join(key_risks)
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
        lines.append(f"• *Quant Score:* `{quant_score}/100` | *Data Coverage:* `{int(completeness * 100)}%`")
        lines.append(f"• *Pillars:* Trend: {pillars.get('trend', 0)} | Sector: {pillars.get('sector', 0)} | Alpha: {pillars.get('alpha', 0)} | ValHist: {pillars.get('valuation_history', 0)} | PeerVal: {pillars.get('peer_valuation', 0)}")
        lines.append(f"• *Probabilities:* Buy: {buy_score} | Hold: {hold_score} | Sell: {sell_score}")

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
        usage="python main.py {nemotron|gemini|twostage|gemma|qwen} [ticker]"
    )
    parser.add_argument(
        "model_arg",
        nargs="?",
        default=None,
        help="Required model short name: 'nemotron' (Nemotron-3 Super 120B), 'gemini' (Gemini 3.1 Pro), 'twostage' (Qwen2.5 14B + DeepSeek-R1 14B), 'gemma' (gemma4:12b), or 'qwen' (qwen2.5:14b)"
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
        help="Model short name: 'nemotron', 'gemini', 'twostage', 'gemma', or 'qwen'"
    )
    parser.add_argument(
        "--ticker", "-t",
        dest="ticker_opt",
        default=None,
        help="Stock ticker symbol (e.g. 000660.KS, NVDA) or comma-separated list of tickers"
    )
    args = parser.parse_args()

    raw_model = args.model_opt or args.model_arg
    if not raw_model:
        print("\n❌ ERROR: Model argument is required!")
        print("Usage: python main.py {nemotron|gemini|twostage|gemma|qwen} [ticker]")
        print("  - nemotron : Cloud Nemotron-3 Super 120B (NVIDIA)")
        print("  - gemini   : Cloud Gemini 3.1 Pro")
        print("  - twostage : 2-Stage Local (qwen2.5:14b extraction + qwen3:30b-a3b-instruct-2507-q4_K_M reasoning)")
        print("  - gemma    : Local Ollama gemma4:12b")
        print("  - qwen     : Local Ollama qwen2.5:14b\n")
        sys.exit(1)

    model_choice = str(raw_model).strip().lower()
    valid_models = ["nemotron", "nvidia", "gemini", "twostage", "gemma", "qwen", "local"]
    if model_choice not in valid_models:
        print(f"\n❌ ERROR: Invalid model choice '{raw_model}'!")
        print("Supported choices are: 'nemotron', 'gemini', 'twostage', 'gemma', 'qwen'\n")
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
    gloomberb_service = GloomberbService()
    institutional_service = InstitutionalDataService()
    rag_service = RAGService()

    all_results = []

    for idx, symbol in enumerate(watchlist, 1):
        print(f"\n--- [{idx}/{len(watchlist)}] Processing {symbol} ---")

        # 1. Technical Data Collection (RSI / EMA / ATR)
        m_data = market_agent.analyze(symbol)
        if not m_data:
            print(f"Skipping {symbol}: Insufficient price data.")
            continue

        # 2. Gloomberb Data Source Stream (News, Filings, Financials, Options, Insiders, Peer Valuation)
        gloomberb_payload = gloomberb_service.get_all_gloomberb_data(symbol)

        # 3. Institutional Multi-Source Data Stream (IR, SEC direct, Earnings calls, Press releases, Reputable news)
        institutional_payload = institutional_service.get_all_institutional_data(symbol)

        # 4. Dense Vector Embedding RAG & Prompt Payload Assembly for Nemotron-3 Super
        context = rag_service.get_nemotron_payload(gloomberb_payload, technical_data=m_data, institutional_data=institutional_payload)

        # 5. Nemotron 3 Super Reasoning Core & Market Analysis
        print(f"[Nemotron 3 Super] Executing market analysis for {symbol} via {model_label}...")

        try:
            res_content = query_llm(
                system_instruction=context["system_instruction"],
                user_prompt=context["user_prompt"],
                model_choice=model_choice
            )
            parsed = extract_json(res_content)

            res_obj = None
            if isinstance(parsed, dict):
                res_obj = parsed
            elif isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
                res_obj = parsed[0]

            if res_obj:
                normalized_obj = normalize_master_trader_json(res_obj, symbol, m_data)
                all_results.append(normalized_obj)
                print(f"[Nemotron 3 Super] Final Decision for {symbol}: {normalized_obj.get('decision')} (Conf: {normalized_obj.get('confidence')})")
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

    # Send Telegram alerts
    if all_results and telegram_token:
        digest_message = format_telegram_digest(all_results, model_label=model_label)
        send_telegram_digest(telegram_token, CHAT_ID, digest_message)


if __name__ == "__main__":
    main()
