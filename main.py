import sys
import argparse
import os
import json
import time
import requests
from dotenv import load_dotenv

from llm_service import query_llm
from db import init_db, save_results
from market_agent import MarketAgent
from news_agent import NewsAgent
from risk_agent import RiskAgent
from analyst_agent import AnalystAgent

WATCHLIST = [
    "FLKR",
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "TSLA",
    "AMD",
    "NFLX",
    "PLTR"
]

CHAT_ID = "969601315"

load_dotenv()
telegram_token = os.getenv("TELEGRAM_CHANNEL_API_TOKEN")

MASTER_TRADER_INSTRUCTION = """
You are a senior Master Trader & Portfolio Manager.

You receive short-term intelligence collected by four specialized sub-agents:
1. Market Agent: Price action, 5-day velocity, RSI, EMA20/50, ATR, Forward P/E, profit margins.
2. News Agent: Today's real-time headlines, news summary, sentiment, and short-term impact.
3. Risk Agent: Volatility ATR%, technical trend breakdown risk, RSI extreme risk, and stop-loss boundaries.
4. Analyst Agent: Wall Street consensus, mean target price, and recent rating actions from major investment banks (BofA, Barclays, Credit Suisse, DB, Evercore, Goldman, JPM, Morgan Stanley, UBS).

Your Task:
Synthesize the insights from all four agents to deliver a definitive short-term swing trade decision (1 to 10 trading days horizon).

Rules & Guidelines:
1. Ignore stale long-term macro narratives that are already priced in.
2. Synthesize News Agent impact + Market Agent momentum + Risk Agent factors + Analyst Agent bank ratings.
3. Return a valid JSON object matching the requested schema.

Schema:
{
  "stock": "Ticker Symbol",
  "decision": "BUY|SELL|HOLD",
  "confidence": 0.85,
  "reason": "Short-term swing setup rationale synthesizing Market, News, Risk & Analyst Agent insights",
  "news": "Today's key news summary & catalyst impact from News Agent",
  "investment_bank_coverage": "Recent rating actions & consensus from Analyst Agent",
  "risk_assessment": "Risk level & stop-loss boundary from Risk Agent",
  "push_notification": "TRUE|FALSE",
  "PE_and_PEG": "Forward PE ratio value"
}
"""


def format_telegram_digest(results, model_label="Gemini 3.6 Flash"):
    lines = [f"⚡ *4-Agent Short-Term Swing Digest ({model_label})* ⚡\n"]
    emoji_map = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}

    for item in results:
        stock = item.get("stock", item.get("symbol", "N/A"))
        decision = str(item.get("decision", "HOLD")).upper()
        confidence = item.get("confidence", 0.0)
        reason = item.get("reason", "")
        news = item.get("news", "")
        bank_coverage = item.get("investment_bank_coverage", "")
        risk_info = item.get("risk_assessment", "")
        pe_peg = item.get("PE_and_PEG", "")
        emoji = emoji_map.get(decision, "⚪")

        lines.append(f"{emoji} *{stock}* | *{decision}* (Conf: {confidence})")
        if pe_peg:
            lines.append(f"• *Fwd Valuation:* _{pe_peg}_")
        if bank_coverage:
            lines.append(f"• *Bank Coverage:* _{bank_coverage}_")
        if risk_info:
            lines.append(f"• *Risk Profile:* _{risk_info}_")
        if reason:
            lines.append(f"• *Swing Setup:* _{reason}_")
        if news:
            lines.append(f"• *Today's News & Impact:* _{news}_")
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
    parser = argparse.ArgumentParser(description="4-Agent Stock Swing Trading Analysis Pipeline")
    parser.add_argument("model_arg", nargs="?", default=None, help="Model choice: 'local' for gemma4:12b, otherwise uses Gemini 3.6 Flash")
    parser.add_argument("--model", "-m", dest="model_opt", default=None, help="Model choice: 'local' for gemma4:12b, otherwise uses Gemini 3.6 Flash")
    args = parser.parse_args()

    model_choice = args.model_opt or args.model_arg or "gemini"
    is_local = (str(model_choice).strip().lower() == "local")
    model_label = "gemma4:12b (Ollama)" if is_local else "Gemini 3.6 Flash"

    print(f"=== Initializing 4-Agent Stock Analysis Pipeline (Model: {model_label}) ===")
    market_agent = MarketAgent()
    news_agent = NewsAgent(model_choice=model_choice)
    risk_agent = RiskAgent()
    analyst_agent = AnalystAgent()

    all_results = []

    for idx, symbol in enumerate(WATCHLIST, 1):
        print(f"\n--- [{idx}/{len(WATCHLIST)}] Processing {symbol} ---")

        # 1. Market Agent Data Collection
        m_data = market_agent.analyze(symbol)
        if not m_data:
            print(f"Skipping {symbol}: Insufficient price data.")
            continue

        # 2. News Agent Data Collection & Sentiment Impact
        n_data = news_agent.analyze(symbol)

        # 3. Risk Agent Data Collection & Volatility Evaluation
        r_data = risk_agent.analyze(m_data)

        # 4. Analyst Agent Investment Bank Rating Extraction
        a_data = analyst_agent.analyze(symbol)

        # 5. Master Trader Synthesis
        payload = {
            "stock": symbol,
            "market_agent_data": m_data,
            "news_agent_data": n_data,
            "risk_agent_data": r_data,
            "analyst_agent_data": a_data
        }

        print(f"[MasterTrader] Synthesizing 4-agent insights for {symbol} via {model_label}...")
        prompt = f"4-Agent Payload for {symbol}:\n{json.dumps(payload, indent=2)}"

        try:
            res_content = query_llm(
                system_instruction=MASTER_TRADER_INSTRUCTION,
                user_prompt=prompt,
                model_choice=model_choice
            )
            parsed = json.loads(res_content)

            res_obj = None
            if isinstance(parsed, dict):
                res_obj = parsed
            elif isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
                res_obj = parsed[0]

            if res_obj:
                if "stock" not in res_obj:
                    res_obj["stock"] = symbol
                all_results.append(res_obj)
                print(f"[MasterTrader] Decision for {symbol}: {res_obj.get('decision')} (Conf: {res_obj.get('confidence')})")
            else:
                print(f"Warning: Master Trader returned empty output for {symbol}.")

        except Exception as e:
            print(f"Master Trader synthesis error for {symbol}: {e}")

    if not all_results:
        print("\nNo analysis results generated.")
        return

    print("\n================ Aggregated Analysis Output ================")
    print(json.dumps(all_results, indent=2))

    # Auto-save results to SQLite DB
    save_results(all_results, model_used=model_label)

    if all_results and telegram_token:
        digest_message = format_telegram_digest(all_results, model_label=model_label)
        send_telegram_digest(telegram_token, CHAT_ID, digest_message)


if __name__ == "__main__":
    main()
