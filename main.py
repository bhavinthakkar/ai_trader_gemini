WATCHLIST = [
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


import yfinance as yf
import requests
import ta
from dotenv import load_dotenv
import os
from google import genai


CHAT_ID = "969601315"


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY");
telegram_token = os.getenv("TELEGRAM_CHANNEL_API_TOKEN");

client = genai.Client(api_key=api_key)




for Symbol in WATCHLIST:
    print(Symbol)
    df = yf.download(
        Symbol,
        period="6mo",
        interval="1d"
    )

    df.columns = df.columns.get_level_values(0)

    df["RSI"] = ta.momentum.RSIIndicator(
        close=df["Close"],
        window=14
    ).rsi()

    df["EMA20"] = ta.trend.EMAIndicator(
        close=df["Close"],
        window=20
    ).ema_indicator()



    df["EMA50"] = ta.trend.EMAIndicator(
        close=df["Close"],
        window=50
    ).ema_indicator()

    df["SMA20"] = ta.trend.SMAIndicator(
        close=df["Close"],
        window=20
    ).sma_indicator()

    macd = ta.trend.MACD(
        close=df["Close"]
    )

    df["MACD"] = macd.macd()
    df["MACD_SIGNAL"] = macd.macd_signal()
    df["MACD_HIST"] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(
        close=df["Close"],
        window=20,
        window_dev=2
    )

    df["BB_UPPER"] = bb.bollinger_hband()
    df["BB_MIDDLE"] = bb.bollinger_mavg()
    df["BB_LOWER"] = bb.bollinger_lband()
    
    atr = ta.volatility.AverageTrueRange(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=14
    )

    df["ATR"] = atr.average_true_range()
    df["VOL_MA20"] = (
        df["Volume"]
        .rolling(20)
        .mean()
    )

    mfi = ta.volume.MFIIndicator(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        volume=df["Volume"]
    )

    df["MFI"] = mfi.money_flow_index()

    latest = df.iloc[-1]
    price = latest["Close"]
    rsi = latest["RSI"]
    ema20 = latest["EMA20"]
    ema50 = latest["EMA50"]
    atr = latest["ATR"]
 
    prompt = f"""
    You are a professional swing trader.

    Analyze the following stock:

    Symbol: {Symbol}

    Current Price: {price:.2f}
    RSI(14): {rsi:.2f}
    EMA20: {ema20:.2f}
    EMA50: {ema50:.2f}
    ATR: {atr:.2f}

    Rules:
    - Consider trend and momentum.
    - Do not invent information.
    - Fetch latest headlines.
    - Read today's financial news.
    - Check earnings announcements.
    - Monitor social media.
    - Findout today's stock price.
    - Findout current market sentiment.
    - Findout stock specific news / Upgrades / downgrades
    - If the price moves more than 5% within a day, update "push_notification" to "true" in JSON schema
    - Provide PE and PEG values in the responce JSON
    - Return JSON only.

    Schema:

    {{
      "stock": "stock name"
      "decision": "BUY|SELL|HOLD",
      "confidence": 0.0,
      "reason": "short explanation"
      "news": "stock specific news"
      "push_notification": "TRUE|FALSE"
      "PE and PEG": "Values"
    }}
    """

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt
    )


    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"

    requests.post(
        url,
        json={
            "chat_id": CHAT_ID,
            "text": response.text
        }
    )

    print(response.text)
