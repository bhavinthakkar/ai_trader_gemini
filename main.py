import os
import json
import time
import pandas as pd
import requests
import ta
import yfinance as yf
from dotenv import load_dotenv
from google import genai
from google.genai import types

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
api_key = os.getenv("GEMINI_API_KEY")
telegram_token = os.getenv("TELEGRAM_CHANNEL_API_TOKEN")

client = genai.Client(api_key=api_key)

SYSTEM_INSTRUCTION = """
You are a senior short-term swing trader and equity research analyst covering major Wall Street investment bank perspectives.

Your analysis MUST FOCUS STRICTLY ON SHORT-TERM SWING TRADING (1 to 10 trading days horizon).

Rules & Analytical Guidelines:
1. SHORT-TERM SWING FOCUS (IGNORE PRICED-IN LONG-TERM NOISE):
   - Ignore old, stale historical news, macro narratives, and long-term brand stories that are ALREADY PRICED IN.
   - Focus strictly on immediate short-term price catalysts, Forward P/E ratio, current quarter profit margins, and guidance for next quarter.
   - Evaluate short-term price velocity (5-day return %), RSI momentum, and EMA20 vs EMA50 alignment.

2. INVESTMENT BANK COVERAGE & RATING ACTIONS:
   - Factor in recent Wall Street equity research, price target revisions, and rating actions from major investment banks:
     1) Bank of America (BofA)
     2) Barclays
     3) Credit Suisse
     4) Deutsche Bank (DB)
     5) Evercore ISI
     6) Goldman Sachs
     7) JPMorgan
     8) Morgan Stanley
     9) UBS
   - Highlight recent upgrades, downgrades, price target raises/cuts, or conviction list changes from these firms.

3. REQUIRED METRICS & OUTPUT SCHEMA:
   - Forward P/E Ratio (valuation relative to immediate near-term earnings potential).
   - Current Quarter Profit Margins & Quarterly Earnings/Revenue Growth.
   - Next Quarter Guidance & Recent Guidance Revisions.
   - Major Investment Bank Stance & Wall Street Consensus.

Schema:
[
  {
    "stock": "Ticker Symbol",
    "decision": "BUY|SELL|HOLD",
    "confidence": 0.85,
    "reason": "Short-term swing setup analysis incorporating Forward PE, Q profit margins, next Q guidance, and 5-day momentum",
    "news": "Fresh short-term catalysts & next Q guidance updates",
    "investment_bank_coverage": "Recent rating actions, price targets & consensus from major banks (BofA, Barclays, Credit Suisse, DB, Evercore, Goldman, JPM, Morgan Stanley, UBS)",
    "push_notification": "TRUE|FALSE",
    "PE_and_PEG": "Forward PE / PEG ratio values"
  }
]
"""

config = types.GenerateContentConfig(
    system_instruction=SYSTEM_INSTRUCTION,
    response_mime_type="application/json"
)


def collect_market_data():
    market_data = []
    print("Collecting short-term market metrics & forward fundamentals...")
    for Symbol in WATCHLIST:
        try:
            ticker = yf.Ticker(Symbol)
            df = ticker.history(period="6mo", interval="1d")

            if df.empty:
                continue

            df = df.dropna(subset=["Close"])
            if len(df) < 20:
                continue

            df["RSI"] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()
            df["EMA20"] = ta.trend.EMAIndicator(close=df["Close"], window=20).ema_indicator()
            df["EMA50"] = ta.trend.EMAIndicator(close=df["Close"], window=50).ema_indicator()
            atr = ta.volatility.AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14)
            df["ATR"] = atr.average_true_range()

            latest = df.iloc[-1]
            price = latest["Close"]
            rsi = latest["RSI"]
            ema20 = latest["EMA20"]
            ema50 = latest["EMA50"]
            atr_val = latest["ATR"]

            if pd.isna(price):
                continue

            # Short-term 5-day price return %
            price_5d_ago = df["Close"].iloc[-6] if len(df) >= 6 else df["Close"].iloc[0]
            change_5d_pct = ((price - price_5d_ago) / price_5d_ago) * 100

            # Forward fundamentals & margins
            try:
                info = ticker.info or {}
            except Exception:
                info = {}

            forward_pe = info.get("forwardPE")
            profit_margins = info.get("profitMargins")
            earnings_growth = info.get("earningsGrowth")
            rev_growth = info.get("revenueGrowth")

            market_data.append({
                "symbol": Symbol,
                "price": round(float(price), 2),
                "change_5d_pct": f"{change_5d_pct:+.2f}%",
                "rsi14": round(float(rsi), 2) if not pd.isna(rsi) else None,
                "ema20": round(float(ema20), 2) if not pd.isna(ema20) else None,
                "ema50": round(float(ema50), 2) if not pd.isna(ema50) else None,
                "atr": round(float(atr_val), 2) if not pd.isna(atr_val) else None,
                "forward_pe": round(float(forward_pe), 2) if forward_pe else "N/A",
                "profit_margins_this_q": f"{profit_margins * 100:.1f}%" if profit_margins else "N/A",
                "earnings_growth_yoy": f"{earnings_growth * 100:.1f}%" if earnings_growth else "N/A",
                "revenue_growth_yoy": f"{rev_growth * 100:.1f}%" if rev_growth else "N/A"
            })
        except Exception as e:
            print(f"Error gathering data for {Symbol}: {e}")

    return market_data


def format_telegram_digest(results):
    lines = ["⚡ *Short-Term Swing Market Digest* ⚡\n"]
    emoji_map = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}

    for item in results:
        stock = item.get("stock", item.get("symbol", "N/A"))
        decision = str(item.get("decision", "HOLD")).upper()
        confidence = item.get("confidence", 0.0)
        reason = item.get("reason", "")
        news = item.get("news", "")
        bank_coverage = item.get("investment_bank_coverage", item.get("bank_ratings", ""))
        pe_peg = item.get("PE_and_PEG", "")
        emoji = emoji_map.get(decision, "⚪")

        lines.append(f"{emoji} *{stock}* | *{decision}* (Conf: {confidence:.2f})")
        if pe_peg:
            lines.append(f"• *Fwd Valuation:* _{pe_peg}_")
        if bank_coverage:
            lines.append(f"• *Bank Coverage:* _{bank_coverage}_")
        if reason:
            lines.append(f"• *Swing Setup:* _{reason}_")
        if news:
            lines.append(f"• *Next Q / Short-Term Catalysts:* _{news}_")
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
    stock_data = collect_market_data()
    if not stock_data:
        print("No stock data collected.")
        return

    print(f"Sending short-term swing analysis batch ({len(stock_data)} stocks) to Gemini API...")
    batch_prompt = f"Analyze the following short-term stock market data & forward metrics:\n{json.dumps(stock_data, indent=2)}"

    MODELS_ORDER = [
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-1.5-flash"
    ]

    response = None
    for model_name in MODELS_ORDER:
        try:
            print(f"Attempting short-term analysis with model '{model_name}'...")
            response = client.models.generate_content(
                model=model_name,
                contents=batch_prompt,
                config=config
            )
            print(f"Successfully generated analysis using '{model_name}'.")
            break
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err or "404" in err or "NOT_FOUND" in err or "INVALID_ARGUMENT" in err:
                print(f"Model '{model_name}' unavailable/rate-limited. Switching to next model...")
                continue
            else:
                print(f"Error with '{model_name}': {e}. Trying next model...")
                continue

    if not response:
        print("All models in the fallback sequence failed.")
        return

    print("\n--- Gemini Short-Term Analysis Output ---")
    print(response.text)

    try:
        results = json.loads(response.text)
    except Exception as e:
        print(f"Error parsing JSON output: {e}")
        results = []

    if results and telegram_token:
        digest_message = format_telegram_digest(results)
        send_telegram_digest(telegram_token, CHAT_ID, digest_message)


if __name__ == "__main__":
    main()





