import pandas as pd
import ta
import yfinance as yf

class MarketAgent:
    """
    Market Agent: Collects price action history, 5-day velocity, 
    technical indicators (RSI14, EMA20, EMA50, ATR), and forward fundamental metrics.
    """
    
    def analyze(self, symbol: str) -> dict:
        print(f"[MarketAgent] Fetching technicals and fundamentals for {symbol}...")
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="6mo", interval="1d")

            if df.empty:
                return {}

            df = df.dropna(subset=["Close"])
            if len(df) < 20:
                return {}

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
                return {}

            # Short-term 5-day price return %
            price_5d_ago = df["Close"].iloc[-6] if len(df) >= 6 else df["Close"].iloc[0]
            change_5d_pct = ((price - price_5d_ago) / price_5d_ago) * 100

            # Forward fundamentals & analyst consensus
            try:
                info = ticker.info or {}
            except Exception:
                info = {}

            forward_pe = info.get("forwardPE")
            profit_margins = info.get("profitMargins")
            earnings_growth = info.get("earningsGrowth")
            rev_growth = info.get("revenueGrowth")
            target_mean_price = info.get("targetMeanPrice")
            recommendation_key = info.get("recommendationKey")

            return {
                "symbol": symbol,
                "current_price": round(float(price), 2),
                "change_5d_pct": f"{change_5d_pct:+.2f}%",
                "rsi14": round(float(rsi), 2) if not pd.isna(rsi) else None,
                "ema20": round(float(ema20), 2) if not pd.isna(ema20) else None,
                "ema50": round(float(ema50), 2) if not pd.isna(ema50) else None,
                "atr": round(float(atr_val), 2) if not pd.isna(atr_val) else None,
                "forward_pe": round(float(forward_pe), 2) if forward_pe else "N/A",
                "profit_margins": f"{profit_margins * 100:.1f}%" if profit_margins else "N/A",
                "earnings_growth_yoy": f"{earnings_growth * 100:.1f}%" if earnings_growth else "N/A",
                "revenue_growth_yoy": f"{rev_growth * 100:.1f}%" if rev_growth else "N/A",
                "wall_street_consensus": recommendation_key.upper() if recommendation_key else "N/A",
                "target_price": round(float(target_mean_price), 2) if target_mean_price else "N/A"
            }
        except Exception as e:
            print(f"[MarketAgent] Error analyzing {symbol}: {e}")
            return {}
