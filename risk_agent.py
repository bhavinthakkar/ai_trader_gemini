class RiskAgent:
    """
    Risk Agent: Evaluates volatility risk (ATR%), RSI overbought/oversold risk,
    trend breakdown, valuation risk, and calculates suggested stop-loss boundaries.
    """

    def analyze(self, market_data: dict) -> dict:
        symbol = market_data.get("symbol", "UNKNOWN")
        print(f"[RiskAgent] Calculating risk profile for {symbol}...")
        
        if not market_data:
            return {
                "risk_level": "HIGH",
                "risk_score": "8/10",
                "risk_factors": ["Insufficient market data available"],
                "suggested_stop_loss": "N/A"
            }

        risk_factors = []
        risk_score = 0

        price = market_data.get("current_price", 0)
        rsi = market_data.get("rsi14")
        ema20 = market_data.get("ema20")
        ema50 = market_data.get("ema50")
        atr = market_data.get("atr")
        forward_pe = market_data.get("forward_pe")

        # 1. Volatility Risk (ATR relative to price)
        if price > 0 and atr and (atr / price) > 0.04:
            vol_pct = (atr / price) * 100
            risk_factors.append(f"High daily volatility (ATR: ${atr}, {vol_pct:.1f}% of stock price)")
            risk_score += 2

        # 2. Momentum & Extreme RSI Risk
        if rsi is not None:
            if rsi > 70:
                risk_factors.append(f"Overbought RSI ({rsi}) - Elevated risk of mean-reversion pullback")
                risk_score += 3
            elif rsi < 30:
                risk_factors.append(f"Oversold RSI ({rsi}) - High risk of ongoing selling pressure or capitulation")
                risk_score += 2

        # 3. Technical Trend Alignment Risk
        if price and ema20 and ema50:
            if price < ema20 < ema50:
                risk_factors.append(f"Price (${price}) trading below EMA20 (${ema20}) & EMA50 (${ema50}) - Downtrend alignment")
                risk_score += 3
            elif price > ema20 > ema50:
                # Bullish trend alignment lowers risk
                pass

        # 4. Forward Valuation Risk
        if isinstance(forward_pe, (int, float)) and forward_pe > 50:
            risk_factors.append(f"Elevated Forward P/E ({forward_pe}) - High valuation multiple compression risk")
            risk_score += 2

        # Determine Risk Level Tier
        if risk_score >= 6:
            risk_level = "HIGH"
        elif risk_score >= 3:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # Calculate suggested ATR-based stop loss level
        if price and atr:
            suggested_stop_loss = round(price - (1.5 * atr), 2)
        elif price:
            suggested_stop_loss = round(price * 0.95, 2)
        else:
            suggested_stop_loss = "N/A"

        return {
            "risk_level": risk_level,
            "risk_score": f"{risk_score}/10",
            "risk_factors": risk_factors if risk_factors else ["Low technical and valuation risk"],
            "suggested_stop_loss": suggested_stop_loss
        }
