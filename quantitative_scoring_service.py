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

    Produces a weighted Composite Quantitative Score (0 to 100), dampened
    by a volatility factor so high-ATR names are mechanically pushed toward
    HOLD/SELL even when momentum is strong.
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

    @staticmethod
    def compute_vol_factor(atr, price) -> float:
        """
        Returns a deterministic volatility dampening factor for the composite score.
        Mirrors the RiskAgent's own ATR% threshold (risk_agent.py: ATR/price > 4% is
        flagged HIGH) but as a graduated multiplier instead of a high/low cliff.

        ATR% (daily, as % of price):
          <= 1.5%  -> 1.00  (calm)
          <= 2.5%  -> 0.95  (normal)
          <= 4.0%  -> 0.85  (elevated)
          >  4.0%  -> 0.70  (extreme -- mechanically pushed out of BUY range)
        """
        if atr is None or not price or price <= 0:
            return 1.0
        try:
            atr_pct = (float(atr) / float(price)) * 100.0
        except (TypeError, ValueError):
            return 1.0

        if atr_pct <= 1.5:
            return 1.0
        elif atr_pct <= 2.5:
            return 0.95
        elif atr_pct <= 4.0:
            return 0.85
        return 0.70

    @staticmethod
    def compute_reward_risk(entry_price, stop_loss, target_price) -> Dict:
        """
        Computes the setup reward:risk ratio and breakeven win rate for a long trade.

        reward_risk_ratio = (target - entry) / (entry - stop)
        breakeven_win_rate = 1 / (1 + reward_risk_ratio)   -- the minimum win rate
        needed for the trade to be profitable on average.

        Returns a dict with both values (None when the geometry is unavailable
        or degenerate, e.g. stop >= entry). Always horizon-consistent: compute it
        against the trade's own swing target and stop.
        """
        entry = QuantitativeScoringService._clean_float(entry_price, 0.0)
        stop = QuantitativeScoringService._clean_float(stop_loss, None)
        target = QuantitativeScoringService._clean_float(target_price, None)

        if entry <= 0.0 or stop is None or target is None:
            return {"reward_risk_ratio": None, "breakeven_win_rate": None}

        downside = entry - stop
        upside = target - entry
        if downside <= 0.0:
            return {"reward_risk_ratio": None, "breakeven_win_rate": None}

        rr = round(upside / downside, 2)
        breakeven = round(1.0 / (1.0 + rr), 3) if rr > 0.0 else None
        return {"reward_risk_ratio": rr, "breakeven_win_rate": breakeven}

    @staticmethod
    def compute_channel_reward_risk(entry_price, atr, high_20d, low_20d, suggested_stop_loss=None, suggested_target_price=None) -> Dict:
        """
        Computes a channel-anchored reward:risk so the ratio actually varies with
        where price sits inside its 20-day trading range, instead of collapsing to
        the fixed 2.5/1.5 symmetric-ATR ratio.

        Conservative geometry (never overstates reward):
          structural_stop   = min(price - 1.5*ATR, low_20d  - 0.5*ATR)  -- larger honest downside
          structural_target = min(price + 2.5*ATR, high_20d + 0.5*ATR)  -- capped at real resistance

        Returns a dict with structural_stop, structural_target, reward_risk_ratio,
        breakeven_win_rate, distance_to_resistance_atr, distance_to_support_atr.
        """
        entry = QuantitativeScoringService._clean_float(entry_price, 0.0)
        atr_v = QuantitativeScoringService._clean_float(atr, None)
        high = QuantitativeScoringService._clean_float(high_20d, None)
        low = QuantitativeScoringService._clean_float(low_20d, None)
        vol_stop = QuantitativeScoringService._clean_float(suggested_stop_loss, None)
        vol_tgt = QuantitativeScoringService._clean_float(suggested_target_price, None)

        empty = {
            "structural_stop": None,
            "structural_target": None,
            "reward_risk_ratio": None,
            "breakeven_win_rate": None,
            "distance_to_resistance_atr": None,
            "distance_to_support_atr": None
        }
        if entry <= 0.0:
            return empty

        # Conservative stop: the lower bound (bigger downside = never overstated RR).
        stop_candidates = []
        if vol_stop is not None:
            stop_candidates.append(vol_stop)
        if low is not None and atr_v is not None:
            stop_candidates.append(low - 0.5 * atr_v)
        elif low is not None:
            stop_candidates.append(low)
        valid_stops = [s for s in stop_candidates if s < entry]
        structural_stop = min(valid_stops) if valid_stops else None

        # Conservative target: the higher bound of the capped value (min of candidates)
        tgt_candidates = []
        if vol_tgt is not None:
            tgt_candidates.append(vol_tgt)
        if high is not None and atr_v is not None:
            tgt_candidates.append(high + 0.5 * atr_v)
        elif high is not None:
            tgt_candidates.append(high)
        valid_tgts = [t for t in tgt_candidates if t > entry]
        structural_target = min(valid_tgts) if valid_tgts else None

        rr_info = QuantitativeScoringService.compute_reward_risk(entry, structural_stop, structural_target)

        dist_res, dist_sup = None, None
        if atr_v is not None and atr_v > 0:
            if high is not None:
                dist_res = round((high - entry) / atr_v, 2)
            if low is not None:
                dist_sup = round((entry - low) / atr_v, 2)

        return {
            "structural_stop": structural_stop,
            "structural_target": structural_target,
            "reward_risk_ratio": rr_info.get("reward_risk_ratio"),
            "breakeven_win_rate": rr_info.get("breakeven_win_rate"),
            "distance_to_resistance_atr": dist_res,
            "distance_to_support_atr": dist_sup
        }

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

        raw_composite = (0.25 * s_trend) + (0.20 * s_sector) + (0.20 * s_alpha) + (0.15 * s_val_hist) + (0.20 * s_peer_val)

        # Volatility dampening: the composite must reflect risk-adjusted quality, not raw momentum.
        vol_factor = QuantitativeScoringService.compute_vol_factor(
            technical_data.get("atr") if technical_data else None,
            technical_data.get("current_price") if technical_data else None
        )
        composite = round(raw_composite * vol_factor, 2)

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
            "raw_composite": round(raw_composite, 2),
            "vol_factor": vol_factor,
            "composite_quantitative_score": composite,
            "quant_signal": signal
        }

    @staticmethod
    def _looks_real(val) -> bool:
        """True only when a value is a genuine number (not 'N/A', '-', empty, 0 sentinel)."""
        if val is None:
            return False
        if isinstance(val, bool):
            return False
        try:
            f = float(val)
        except (TypeError, ValueError):
            return False
        return f != 0.0

    @staticmethod
    def _source_said(source_status, channel: str, counts) -> bool:
        """
        Check a gloomberb `data_source_status` style dict for a channel availability.
        `counts` is one of source_status's keys containing fetched items.
        """
        ch = (source_status or {}).get(channel) or {}
        if ch.get("available") is False:
            return False
        if ch.get("status") == "unavailable":
            return False
        if ch.get("source_unavailable") is True:
            return False
        if counts and ch.get("fetch_count", 0) == 0:
            return False
        return True

    @staticmethod
    def _items_are_fresh(items, max_days: int, date_key: str) -> bool:
        if not items:
            return False
        from datetime import datetime
        now = datetime.utcnow()
        for item in items:
            stamp = (item or {}).get(date_key)
            if not stamp:
                continue
            try:
                ts = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=None)
                ref = now.replace(tzinfo=None)
                age = (ref - ts.replace(tzinfo=None)).days
                if 0 <= age <= max_days:
                    return True
            except (TypeError, ValueError):
                continue
        return False

    @staticmethod
    def compute_data_coverage(m_data=None, macro_data=None, gloomberb_payload=None, institutional_payload=None) -> float:
        """
        Deterministic data coverage (0..1) that the system computes itself, once, from
        live provider availability -- never echoed by the model.

        Each provider earns its full weight only when it actually returned a real,
        (where relevant) timely payload:
          technical (price/ATR/stop/target fresh)  0.25
          macro snapshot present                    0.10
          news channel available + fresh            0.10
          SEC filings channel available              0.10
          earnings-call/schedule available           0.05
          fundamentals financials real               0.10
          analyst consensus present                  0.10
          institutional ownership sources (weighted) 0.20
                                                     1.00

        Everything that is missing is simply dropped from the numerator, so missing
        providers lower coverage instead of inflating it.
        """
        m_data = m_data or {}
        macro_data = macro_data or {}
        gloomberb_payload = gloomberb_payload or {}
        institutional_payload = institutional_payload or {}

        try:
            price = float(m_data.get("current_price") or 0.0)
            atr = float(m_data.get("atr") or 0.0)
            stop = float(m_data.get("suggested_stop_loss") or 0.0)
            target = float(m_data.get("suggested_target_price") or m_data.get("target_price") or 0.0)
            tech_ok = price > 0.0 and atr > 0.0 and stop > 0.0 and target > 0.0 \
                and (m_data.get("rsi14") is not None or m_data.get("rvol_20d") is not None)
        except (TypeError, ValueError):
            tech_ok = False

        macro_ok = bool(macro_data)

        src_status = gloomberb_payload.get("data_source_status") or {}
        news_ok = QuantitativeScoringService._source_said(
            src_status, "news", ["fetch_count"])
        if news_ok:
            news_ok = QuantitativeScoringService._items_are_fresh(
                gloomberb_payload.get("news") or [], max_days=7, date_key="published_at")

        filings_ok = QuantitativeScoringService._source_said(
            src_status, "filings", ["fetch_count"])

        earnings_ok = False
        earnings_dt = (gloomberb_payload.get("earnings") or {}).get("earnings_date")
        if earnings_dt is not None and str(earnings_dt).strip().upper() not in ("N/A", "-", ""):
            earnings_ok = True

        fin_ok = False
        fin_pillars = (gloomberb_payload.get("financials") or {}).get("key_ratios") or {}
        if isinstance(fin_pillars, dict) and any(
            QuantitativeScoringService._looks_real(v) for v in fin_pillars.values()):
            fin_ok = True

        analyst_ok = QuantitativeScoringService._looks_real(
            (gloomberb_payload.get("analyst_ratings") or {}).get("mean_target_price"))

        # Institutional ownership via weighted source_status channels.
        inst_status = institutional_payload.get("source_status") or {}
        inst_channels = {
            "sec_edgar_direct": 0.40,
            "reputable_news": 0.30,
            "investor_relations": 0.15,
            "official_press_releases": 0.10,
            "earnings_calls": 0.05,
        }
        inst_ok = 0.0
        for channel, weight in inst_channels.items():
            ch = inst_status.get(channel) or {}
            available = ch.get("available", False) and ch.get("source_unavailable", False) is False
            if available:
                inst_ok += weight

        contributions = [
            (0.25, tech_ok),
            (0.10, macro_ok),
            (0.10, news_ok),
            (0.10, filings_ok),
            (0.05, earnings_ok),
            (0.10, fin_ok),
            (0.10, analyst_ok),
            (0.20, inst_ok),
        ]
        numerator = sum(w * float(ok) for w, ok in contributions)
        return round(min(1.0, numerator), 3)
