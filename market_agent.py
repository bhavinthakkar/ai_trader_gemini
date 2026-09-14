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

            # Volume & Liquidity dynamics (RVOL & 20-day volume average)
            volume = int(latest["Volume"]) if "Volume" in df and not pd.isna(latest["Volume"]) else 0
            vol_20d_mean = float(df["Volume"].tail(20).mean()) if "Volume" in df and len(df) >= 20 else None
            rvol_20d = round(volume / vol_20d_mean, 2) if (vol_20d_mean and vol_20d_mean > 0) else 1.0
            avg_dollar_vol_20d = round(float((df["Close"] * df["Volume"]).tail(20).mean()), 0) if "Volume" in df and len(df) >= 20 else None

            # 20-day Channel Extremes (Breakout / Support / Resistance levels)
            high_20d = round(float(df["High"].tail(20).max()), 2) if len(df) >= 20 else round(float(price), 2)
            low_20d = round(float(df["Low"].tail(20).min()), 2) if len(df) >= 20 else round(float(price), 2)

            # Quantitative stop-loss and swing target price levels
            suggested_stop_loss = round(float(price - (1.5 * atr_val)), 2) if atr_val else round(float(price * 0.95), 2)
            suggested_target_price = round(float(price + (2.5 * atr_val)), 2) if atr_val else round(float(price * 1.08), 2)

            # Forward fundamentals & analyst consensus
            try:
                info = ticker.info or {}
            except Exception:
                info = {}

            # Days until next quarterly earnings announcement
            days_to_earnings = None
            try:
                import datetime
                cal = getattr(ticker, "calendar", None)
                if cal is not None:
                    # calendar can be dict or DataFrame
                    e_date = None
                    if isinstance(cal, dict) and "Earnings Date" in cal:
                        raw_dates = cal["Earnings Date"]
                        if isinstance(raw_dates, list) and len(raw_dates) > 0:
                            e_date = raw_dates[0]
                        elif raw_dates:
                            e_date = raw_dates
                    elif isinstance(cal, pd.DataFrame) and not cal.empty:
                        if "Earnings Date" in cal.index:
                            e_date = cal.loc["Earnings Date"].iloc[0]
                    
                    if e_date is not None:
                        if isinstance(e_date, (datetime.date, datetime.datetime)):
                            target_dt = e_date.date() if isinstance(e_date, datetime.datetime) else e_date
                        else:
                            target_dt = pd.to_datetime(str(e_date)).date()
                        now_dt = datetime.date.today()
                        diff = (target_dt - now_dt).days
                        if diff >= 0:
                            days_to_earnings = int(diff)
            except Exception:
                pass

            # S&P 500 Market Benchmark (SPY) 5-day relative return comparison
            market_5d_pct = "N/A"
            relative_alpha_5d = "N/A"
            try:
                spy_df = yf.Ticker("SPY").history(period="1mo", interval="1d")
                if not spy_df.empty and len(spy_df) >= 6:
                    spy_latest = spy_df["Close"].iloc[-1]
                    spy_5d_ago = spy_df["Close"].iloc[-6]
                    spy_change = ((spy_latest - spy_5d_ago) / spy_5d_ago) * 100
                    market_5d_pct = f"{spy_change:+.2f}%"
                    rel_diff = change_5d_pct - spy_change
                    relative_alpha_5d = f"{rel_diff:+.2f}%"
            except Exception:
                pass

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
                "market_spy_5d_pct": market_5d_pct,
                "relative_alpha_5d": relative_alpha_5d,
                "rsi14": round(float(rsi), 2) if not pd.isna(rsi) else None,
                "ema20": round(float(ema20), 2) if not pd.isna(ema20) else None,
                "ema50": round(float(ema50), 2) if not pd.isna(ema50) else None,
                "atr": round(float(atr_val), 2) if not pd.isna(atr_val) else None,
                "volume": volume,
                "vol_20d_mean": int(vol_20d_mean) if vol_20d_mean else None,
                "rvol_20d": rvol_20d,
                "avg_dollar_vol_20d": avg_dollar_vol_20d,
                "high_20d": high_20d,
                "low_20d": low_20d,
                "suggested_stop_loss": suggested_stop_loss,
                "suggested_target_price": suggested_target_price,
                "days_to_earnings": days_to_earnings,
                "forward_pe": round(float(forward_pe), 2) if forward_pe else "N/A",
                "profit_margins": f"{profit_margins * 100:.1f}%" if profit_margins else "N/A",
                "earnings_growth_yoy": f"{earnings_growth * 100:.1f}%" if earnings_growth else "N/A",
                "revenue_growth_yoy": f"{rev_growth * 100:.1f}%" if rev_growth else "N/A",
                "wall_street_consensus": recommendation_key.upper() if recommendation_key else "N/A",
                "target_price": round(float(target_mean_price), 2) if target_mean_price else suggested_target_price,
                "analyst_target_price": round(float(target_mean_price), 2) if target_mean_price else None
            }
        except Exception as e:
            print(f"[MarketAgent] Error analyzing {symbol}: {e}")
            return {}
