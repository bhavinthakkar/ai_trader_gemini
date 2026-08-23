from typing import Dict

class QuantitativeScoringService:
    """
    Deterministic Quantitative 5-Pillar Scoring Engine:
    Calculates explicit 0–100 scores for:
    1. Historical Trend (25%)
    2. Sector Performance (20%)
    3. Market Alpha vs SPY (20%)
    4. Valuation vs History (15%)
    5. Peer Valuation Multiples (20%)
    
    Produces a weighted Composite Quantitative Score (0 to 100).
    """

    @staticmethod
    def clamp(value: float, min_val: float = 0.0, max_val: float = 100.0) -> float:
        return max(min_val, min(max_val, value))

    @staticmethod
    def _clean_float(val, default: float = 0.0) -> float:
        """Safely parses string percentages (e.g. '-4.64%') or numeric types to float."""
        if val is None:
            return default
        if isinstance(val, (int, float)):
            return float(val)
        try:
            s = str(val).replace("%", "").replace("+", "").strip()
            return float(s)
        except Exception:
            return default

    def calculate_trend_score(self, technical_data: Dict) -> float:
        """
        Calculates Historical Trend Score (0–100):
        - Base: 50.0
        - Price > EMA20: +15.0, Price < EMA20: -15.0
        - Price > EMA50: +15.0, Price < EMA50: -15.0
        - RSI14:
            - 45 - 65 (Healthy bullish momentum): +10.0
            - 30 - 44 (Oversold bounce area): +5.0
            - > 70 (Overbought risk): -10.0
            - < 30 (Extreme downtrend): -10.0
        - 5-day Velocity: +1.0 point per +1.0% change (capped at +/- 10)
        """
        if not technical_data:
            return 50.0

        price = self._clean_float(technical_data.get("current_price"), 0.0)
        ema20 = self._clean_float(technical_data.get("ema20"), price)
        ema50 = self._clean_float(technical_data.get("ema50"), price)
        rsi14 = self._clean_float(technical_data.get("rsi14"), 50.0)
        change_5d = self._clean_float(technical_data.get("change_5d_pct"), 0.0)

        score = 50.0

        # EMA alignment
        if price > ema20:
            score += 15.0
        else:
            score -= 15.0

        if price > ema50:
            score += 15.0
        else:
            score -= 15.0

        # RSI14 check
        if 45.0 <= rsi14 <= 65.0:
            score += 10.0
        elif 30.0 <= rsi14 < 45.0:
            score += 5.0
        elif rsi14 > 70.0:
            score -= 10.0
        elif rsi14 < 30.0:
            score -= 10.0

        # 5d Velocity adjustment
        vel_adj = max(-10.0, min(10.0, change_5d))
        score += vel_adj

        return self.clamp(score)

    def calculate_sector_score(self, technical_data: Dict, sector_bench: Dict) -> float:
        """
        Calculates Sector Relative Performance Score (0–100):
        - Base: 50.0
        - Computes stock 5d velocity vs Sector ETF 5d velocity.
        - +5.0 points per +1% outperformance vs sector.
        """
        if not technical_data or not sector_bench:
            return 50.0

        stock_5d = self._clean_float(technical_data.get("change_5d_pct"), 0.0)
        sec_5d = self._clean_float(sector_bench.get("sector_change_pct"), 0.0)

        diff = stock_5d - sec_5d
        score = 50.0 + (diff * 5.0)

        return self.clamp(score)

    def calculate_market_alpha_score(self, technical_data: Dict) -> float:
        """
        Calculates Market Alpha Score vs S&P 500 SPY (0–100):
        - Base: 50.0
        - Uses relative_alpha_5d vs SPY.
        - +5.0 points per +1% relative alpha.
        """
        if not technical_data:
            return 50.0

        alpha_5d = self._clean_float(technical_data.get("relative_alpha_5d"), 0.0)
        score = 50.0 + (alpha_5d * 5.0)

        return self.clamp(score)

    def calculate_valuation_history_score(self, gloomberb_payload: Dict) -> float:
        """
        Calculates Valuation vs History Score (0–100):
        - Base: 50.0
        - Forward P/E vs Trailing P/E (earnings expansion discount):
            - If Fwd P/E < Trailing P/E: +20.0 (earnings acceleration)
            - If Fwd P/E > Trailing P/E: -10.0 (earnings contraction)
        - Revenue Growth YoY: +15.0 if > 15%, +5.0 if > 5%, -10.0 if negative
        """
        fin = gloomberb_payload.get("financials", {})
        ratios = fin.get("key_ratios", {})

        fwd_pe = self._clean_float(ratios.get("forward_pe"), None)
        trl_pe = self._clean_float(ratios.get("trailing_pe"), None)
        rev_growth = self._clean_float(ratios.get("revenue_growth"), None)

        score = 50.0

        if fwd_pe is not None and trl_pe is not None and trl_pe > 0:
            if fwd_pe < trl_pe:
                score += 20.0
            else:
                score -= 10.0

        if rev_growth is not None:
            if rev_growth > 0.15:
                score += 15.0
            elif rev_growth > 0.05:
                score += 5.0
            elif rev_growth < 0:
                score -= 10.0

        return self.clamp(score)

    def calculate_peer_valuation_score(self, gloomberb_payload: Dict) -> float:
        """
        Calculates Valuation vs Direct Peers Score (0–100):
        - Base: 50.0
        - Compares target stock Forward P/E & EV/EBITDA against peer group averages.
        """
        peer_data = gloomberb_payload.get("peer_valuation", {})
        ratios = gloomberb_payload.get("financials", {}).get("key_ratios", {})
        benchmarks = peer_data.get("direct_peer_benchmarks", [])

        if not benchmarks:
            return 50.0

        stock_fwd_pe = self._clean_float(ratios.get("forward_pe"), None)
        stock_ev_ebitda = self._clean_float(peer_data.get("ev_to_ebitda"), None)

        peer_pes = [self._clean_float(p.get("forward_pe"), None) for p in benchmarks if self._clean_float(p.get("forward_pe"), None) is not None]
        peer_evs = [self._clean_float(p.get("ev_to_ebitda"), None) for p in benchmarks if self._clean_float(p.get("ev_to_ebitda"), None) is not None]

        score = 50.0

        # Forward P/E comparison
        if stock_fwd_pe is not None and peer_pes:
            avg_peer_pe = sum(peer_pes) / len(peer_pes)
            if avg_peer_pe > 0:
                pe_diff_pct = (avg_peer_pe - stock_fwd_pe) / avg_peer_pe
                # If stock_fwd_pe < avg_peer_pe, discount = positive score boost
                score += (pe_diff_pct * 30.0)

        # EV/EBITDA comparison
        if stock_ev_ebitda is not None and peer_evs:
            avg_peer_ev = sum(peer_evs) / len(peer_evs)
            if avg_peer_ev > 0:
                ev_diff_pct = (avg_peer_ev - stock_ev_ebitda) / avg_peer_ev
                score += (ev_diff_pct * 30.0)

        return self.clamp(score)

    def compute_5pillar_scores(self, technical_data: Dict, gloomberb_payload: Dict, sector_bench: Dict = None) -> Dict:
        """
        Computes explicit 0-100 scores across all 5 quantitative pillars and weighted composite score.
        Weights:
        - Historical Trend: 25%
        - Sector Performance: 20%
        - Market Alpha: 20%
        - Valuation History: 15%
        - Peer Valuation: 20%
        """
        s_trend = self.calculate_trend_score(technical_data)
        s_sector = self.calculate_sector_score(technical_data, sector_bench or gloomberb_payload.get("sector_benchmark", {}))
        s_alpha = self.calculate_market_alpha_score(technical_data)
        s_val_hist = self.calculate_valuation_history_score(gloomberb_payload)
        s_peer_val = self.calculate_peer_valuation_score(gloomberb_payload)

        composite = (0.25 * s_trend) + (0.20 * s_sector) + (0.20 * s_alpha) + (0.15 * s_val_hist) + (0.20 * s_peer_val)

        if composite >= 70.0:
            signal = "Strong Bullish Quant Signal (BUY candidate if fundamental RAG confirms)"
        elif composite >= 45.0:
            signal = "Neutral / Mixed Quant Signal (HOLD candidate; BUY requires exceptional fundamental catalyst)"
        else:
            signal = "Bearish / Overvalued Quant Signal (SELL / Caution candidate)"

        return {
            "trend_score": round(s_trend, 2),
            "sector_relative_score": round(s_sector, 2),
            "market_alpha_score": round(s_alpha, 2),
            "valuation_history_score": round(s_val_hist, 2),
            "peer_valuation_score": round(s_peer_val, 2),
            "composite_quantitative_score": round(composite, 2),
            "quant_signal": signal
        }
