from typing import Dict, Any, Optional, List

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

    @staticmethod
    def compute_short_reward_risk(entry_price, stop_loss, target_price) -> Dict:
        """
        Computes the setup reward:risk ratio and breakeven win rate for a short trade.

        reward_risk_ratio = (entry - target) / (stop - entry)
        breakeven_win_rate = 1 / (1 + reward_risk_ratio)

        For a short trade: target < entry < stop.
        """
        entry = QuantitativeScoringService._clean_float(entry_price, 0.0)
        stop = QuantitativeScoringService._clean_float(stop_loss, None)
        target = QuantitativeScoringService._clean_float(target_price, None)

        if entry <= 0.0 or stop is None or target is None:
            return {"reward_risk_ratio": None, "breakeven_win_rate": None}

        risk = stop - entry
        reward = entry - target
        if risk <= 0.0 or reward <= 0.0:
            return {"reward_risk_ratio": None, "breakeven_win_rate": None}

        rr = round(reward / risk, 2)
        breakeven = round(1.0 / (1.0 + rr), 3) if rr > 0.0 else None
        return {"reward_risk_ratio": rr, "breakeven_win_rate": breakeven}

    @staticmethod
    def compute_channel_short_reward_risk(entry_price, atr, high_20d, low_20d, suggested_stop_loss=None, suggested_target_price=None) -> Dict:
        """
        Computes channel-anchored reward:risk for a short trade.
        Conservative geometry (never overstates reward):
          structural_stop   = max(price + 1.5*ATR, high_20d + 0.5*ATR)  -- honest upside risk
          structural_target = max(price - 2.5*ATR, low_20d  - 0.5*ATR)  -- capped at real support
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

        # Conservative stop for short: higher bound (larger upside risk = never understated risk)
        stop_candidates = []
        if vol_stop is not None and vol_stop > entry:
            stop_candidates.append(vol_stop)
        elif atr_v is not None and atr_v > 0:
            stop_candidates.append(entry + 1.5 * atr_v)
        if high is not None and atr_v is not None:
            stop_candidates.append(high + 0.5 * atr_v)
        elif high is not None and high > entry:
            stop_candidates.append(high)
        valid_stops = [s for s in stop_candidates if s > entry]
        structural_stop = max(valid_stops) if valid_stops else (entry + 1.5 * atr_v if atr_v else None)

        # Conservative target for short: higher floor (closest support, conservative drop)
        tgt_candidates = []
        if vol_tgt is not None and vol_tgt < entry:
            tgt_candidates.append(vol_tgt)
        elif atr_v is not None and atr_v > 0:
            tgt_candidates.append(entry - 2.5 * atr_v)
        if low is not None and atr_v is not None:
            tgt_candidates.append(low - 0.5 * atr_v)
        elif low is not None and low < entry:
            tgt_candidates.append(low)
        valid_tgts = [t for t in tgt_candidates if t < entry]
        structural_target = max(valid_tgts) if valid_tgts else (entry - 2.5 * atr_v if atr_v else None)

        rr_info = QuantitativeScoringService.compute_short_reward_risk(entry, structural_stop, structural_target)

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

    @staticmethod
    def evaluate_short_expected_move_vs_stop(entry_price, short_stop_loss, expected_move) -> Dict:
        """
        Evaluates short trade risk: does option-implied expected move exceed the overhead stop distance?
        For a short trade: short_stop_distance = short_stop_loss - entry_price.
        """
        entry = QuantitativeScoringService._clean_float(entry_price, 0.0)
        stop = QuantitativeScoringService._clean_float(short_stop_loss, None)
        em = QuantitativeScoringService._clean_float(expected_move, None)

        if entry <= 0.0 or stop is None or stop <= entry or em is None or em <= 0.0:
            return {
                "stop_distance": None,
                "stop_distance_pct": None,
                "expected_move": em,
                "expected_move_pct": None,
                "ratio": None,
                "stop_inside_expected_move": False,
                "status": "UNKNOWN"
            }

        stop_dist = round(stop - entry, 2)
        stop_dist_pct = round((stop_dist / entry) * 100.0, 2)
        em_pct = round((em / entry) * 100.0, 2)
        ratio = round(em / stop_dist, 2) if stop_dist > 0.0 else 999.0
        stop_inside = em > stop_dist

        if stop_inside:
            status = "VIOLATION"
        elif ratio >= 0.8:
            status = "COMPRESSED"
        else:
            status = "ADEQUATE"

        return {
            "stop_distance": stop_dist,
            "stop_distance_pct": stop_dist_pct,
            "expected_move": em,
            "expected_move_pct": em_pct,
            "ratio": ratio,
            "stop_inside_expected_move": stop_inside,
            "status": status
        }

    @staticmethod
    def compute_expected_move(price, iv, dte: float = 10.0, atm_straddle: float = None) -> Dict:
        """
        Computes option-implied expected move (in $ and %) over a given horizon/DTE.

        If ATM straddle price is available:
            expected_move = 0.85 * (Call_ATM + Put_ATM)
        Otherwise, from implied volatility (IV) and Days To Expiration (DTE):
            expected_move = Price * IV * sqrt(DTE / 365)
        """
        p = QuantitativeScoringService._clean_float(price, 0.0)
        straddle = QuantitativeScoringService._clean_float(atm_straddle, None)
        iv_v = QuantitativeScoringService._clean_float(iv, None)
        dte_v = max(QuantitativeScoringService._clean_float(dte, 10.0), 1.0)

        if straddle is not None and straddle > 0.0:
            em = round(0.85 * straddle, 2)
            em_pct = round((em / p) * 100.0, 2) if p > 0.0 else 0.0
            return {
                "expected_move": em,
                "expected_move_pct": em_pct,
                "dte": int(dte_v),
                "calculation_method": "ATM_STRADDLE"
            }

        if p > 0.0 and iv_v is not None and iv_v > 0.0:
            iv_dec = iv_v / 100.0 if iv_v > 1.0 else iv_v
            em = round(p * iv_dec * ((dte_v / 365.0) ** 0.5), 2)
            em_pct = round((em / p) * 100.0, 2)
            return {
                "expected_move": em,
                "expected_move_pct": em_pct,
                "dte": int(dte_v),
                "calculation_method": "IV_FORMULA"
            }

        return {
            "expected_move": None,
            "expected_move_pct": None,
            "dte": int(dte_v),
            "calculation_method": "NONE"
        }

    @staticmethod
    def evaluate_expected_move_vs_stop(entry_price, stop_loss, expected_move) -> Dict:
        """
        Evaluates setup risk: does the option-implied expected move exceed the stop distance?

        When expected_move > stop_distance, the stop loss sits inside the 1-standard-deviation
        normal options volatility noise band, creating a high-probability stop-out on market noise.
        """
        entry = QuantitativeScoringService._clean_float(entry_price, 0.0)
        stop = QuantitativeScoringService._clean_float(stop_loss, None)
        em = QuantitativeScoringService._clean_float(expected_move, None)

        if entry <= 0.0 or stop is None or stop >= entry or em is None or em <= 0.0:
            return {
                "stop_distance": None,
                "stop_distance_pct": None,
                "expected_move": em,
                "expected_move_pct": None,
                "ratio": None,
                "stop_inside_expected_move": False,
                "status": "UNKNOWN"
            }

        stop_dist = round(entry - stop, 2)
        stop_dist_pct = round((stop_dist / entry) * 100.0, 2)
        em_pct = round((em / entry) * 100.0, 2)
        ratio = round(em / stop_dist, 2) if stop_dist > 0.0 else 999.0
        stop_inside = em > stop_dist

        if stop_inside:
            status = "VIOLATION"
        elif ratio >= 0.8:
            status = "COMPRESSED"
        else:
            status = "ADEQUATE"

        return {
            "stop_distance": stop_dist,
            "stop_distance_pct": stop_dist_pct,
            "expected_move": em,
            "expected_move_pct": em_pct,
            "ratio": ratio,
            "stop_inside_expected_move": stop_inside,
            "status": status
        }

    @staticmethod
    def evaluate_volatility_regime(atr, price, implied_volatility=None, historical_volatility=None) -> Dict:
        """
        Assesses the multi-source volatility regime combining ATR, Implied Volatility (IV),
        and the IV/HV spread.
        """
        p = QuantitativeScoringService._clean_float(price, 0.0)
        atr_v = QuantitativeScoringService._clean_float(atr, 0.0)
        vol_factor = QuantitativeScoringService.compute_vol_factor(atr_v, p)
        atr_pct = round((atr_v / p) * 100.0, 2) if p > 0.0 else 0.0

        iv_pct = QuantitativeScoringService._clean_float(implied_volatility, None)
        if iv_pct is not None and iv_pct <= 2.0:
            iv_pct = round(iv_pct * 100.0, 2)
        hv_pct = QuantitativeScoringService._clean_float(historical_volatility, None)
        if hv_pct is not None and hv_pct <= 2.0:
            hv_pct = round(hv_pct * 100.0, 2)

        iv_hv_ratio = round(iv_pct / hv_pct, 2) if (iv_pct is not None and hv_pct is not None and hv_pct > 0.0) else None

        extreme_atr = vol_factor < 0.85
        extreme_iv = iv_pct is not None and iv_pct > 65.0
        extreme_iv_hv_spread = iv_hv_ratio is not None and iv_hv_ratio > 2.0

        is_extreme = extreme_atr or extreme_iv or extreme_iv_hv_spread

        reasons = []
        if extreme_atr:
            reasons.append(f"ATR {atr_pct}% of price (vol factor {vol_factor:.2f}) exceeds normal threshold")
        if extreme_iv:
            reasons.append(f"implied volatility {iv_pct:.1f}% exceeds 65% crisis ceiling")
        if extreme_iv_hv_spread:
            reasons.append(f"IV/HV ratio {iv_hv_ratio:.2f}x (IV {iv_pct:.1f}% vs HV {hv_pct:.1f}%) signals severe tail-risk expansion")

        if is_extreme:
            regime = "EXTREME"
        elif (iv_pct is not None and iv_pct > 40.0) or vol_factor < 0.95 or (iv_hv_ratio is not None and iv_hv_ratio >= 1.5):
            regime = "ELEVATED"
            if iv_pct is not None and iv_pct > 40.0:
                reasons.append(f"elevated IV ({iv_pct:.1f}%)")
            if vol_factor < 0.95:
                reasons.append(f"elevated ATR ({atr_pct:.1f}%)")
            if iv_hv_ratio is not None and iv_hv_ratio >= 1.5:
                reasons.append(f"heightened IV/HV ratio ({iv_hv_ratio:.2f}x)")
        elif (iv_pct is not None and iv_pct < 20.0) and vol_factor == 1.0:
            regime = "CALM"
        else:
            regime = "NORMAL"

        return {
            "vol_regime": regime,
            "is_extreme": is_extreme,
            "reasons": reasons,
            "vol_factor": vol_factor,
            "atr_pct": atr_pct,
            "iv_pct": iv_pct,
            "hv_pct": hv_pct,
            "iv_hv_ratio": iv_hv_ratio
        }

    @staticmethod
    def evaluate_event_risk(days_to_earnings=None, days_to_next_event=None, next_event_type=None, next_event_date=None) -> Dict:
        """
        Evaluates scheduled company and market event risks beyond earnings.
        """
        dte = None
        if days_to_earnings is not None:
            try:
                dte = int(days_to_earnings)
            except (ValueError, TypeError):
                dte = None

        if dte is not None and dte <= 3:
            return {
                "is_blackout": True,
                "reason_code": "EARNINGS_BLACKOUT",
                "description": f"earnings report in {dte} day(s) is a binary gap-risk event",
                "event_type": "EARNINGS",
                "days_to_event": dte,
                "event_date": None
            }

        dev = None
        if days_to_next_event is not None:
            try:
                dev = int(days_to_next_event)
            except (ValueError, TypeError):
                dev = None

        evt_type = str(next_event_type or "COMPANY_EVENT").upper()

        if dev is not None and dev <= 2:
            desc = (
                f"scheduled company event '{evt_type}' in {dev} day(s) "
                f"({next_event_date or 'imminent'}) poses binary event gap-risk"
            )
            return {
                "is_blackout": True,
                "reason_code": "EVENT_RISK",
                "description": desc,
                "event_type": evt_type,
                "days_to_event": dev,
                "event_date": next_event_date
            }

        if dev is not None and dev <= 7:
            return {
                "is_blackout": False,
                "reason_code": None,
                "description": f"upcoming scheduled company event '{evt_type}' in {dev} day(s)",
                "event_type": evt_type,
                "days_to_event": dev,
                "event_date": next_event_date
            }

        return {
            "is_blackout": False,
            "reason_code": None,
            "description": None,
            "event_type": None,
            "days_to_event": None,
            "event_date": None
        }

    @staticmethod
    def _track_input(name: str, value: Any, is_valid: bool = None, max_age_days: int = None, timestamp: str = None) -> Dict:
        """
        Tracks availability, freshness, and value of an individual quantitative scoring input.
        """
        if is_valid is None:
            is_valid = QuantitativeScoringService._looks_real(value)
        freshness = "unavailable"
        if is_valid:
            freshness = "fresh"
            if timestamp and max_age_days:
                try:
                    import datetime
                    ts = datetime.datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=None)
                    ref = datetime.datetime.utcnow().replace(tzinfo=None)
                    age = (ref - ts.replace(tzinfo=None)).days
                    if age > max_age_days:
                        freshness = "stale"
                except Exception:
                    pass
        return {
            "name": name,
            "value": value if is_valid else None,
            "available": bool(is_valid),
            "freshness": freshness,
            "timestamp": timestamp
        }

    def evaluate_trend_pillar(self, technical_data: Dict) -> Dict:
        """
        Evaluates Historical Trend Pillar with visible missingness tracking:
        - Inputs: current_price, ema20, ema50, rsi14, change_5d_pct
        - Status:
            - 'measured': all 5 inputs available
            - 'partial': price and >= 1 indicator available
            - 'unavailable': price missing or no indicators available (score = None)
        """
        technical_data = technical_data or {}
        p_raw = technical_data.get("current_price")
        price = self._clean_float(p_raw, 0.0)
        has_price = price > 0.0

        ema20_raw = technical_data.get("ema20")
        ema20_v = self._clean_float(ema20_raw, None)
        has_ema20 = ema20_v is not None and ema20_v > 0.0

        ema50_raw = technical_data.get("ema50")
        ema50_v = self._clean_float(ema50_raw, None)
        has_ema50 = ema50_v is not None and ema50_v > 0.0

        rsi_raw = technical_data.get("rsi14")
        rsi_v = self._clean_float(rsi_raw, None)
        has_rsi = rsi_v is not None and rsi_v >= 0.0

        chg_raw = technical_data.get("change_5d_pct")
        chg_v = self._clean_float(chg_raw, None)
        has_chg = chg_v is not None

        tracking = {
            "current_price": self._track_input("current_price", price if has_price else None, is_valid=has_price),
            "ema20": self._track_input("ema20", ema20_v, is_valid=has_ema20),
            "ema50": self._track_input("ema50", ema50_v, is_valid=has_ema50),
            "rsi14": self._track_input("rsi14", rsi_v, is_valid=has_rsi),
            "change_5d_pct": self._track_input("change_5d_pct", chg_v, is_valid=has_chg)
        }

        indicators_present = sum([has_ema20, has_ema50, has_rsi, has_chg])

        if not has_price or indicators_present == 0:
            return {
                "score": None,
                "status": "unavailable",
                "inputs": tracking
            }

        status = "measured" if indicators_present == 4 else "partial"

        score = 50.0
        if has_ema20:
            score += 15.0 if price > ema20_v else -15.0
        if has_ema50:
            score += 15.0 if price > ema50_v else -15.0

        if has_rsi:
            if 45.0 <= rsi_v <= 65.0:
                score += 10.0
            elif 30.0 <= rsi_v < 45.0:
                score += 5.0
            elif rsi_v > 70.0:
                score -= 10.0
            elif rsi_v < 30.0:
                score -= 10.0

        if has_chg:
            vel_adj = max(-10.0, min(10.0, chg_v))
            score += vel_adj

        return {
            "score": round(self.clamp(score), 2),
            "status": status,
            "inputs": tracking
        }

    def evaluate_sector_pillar(self, technical_data: Dict, sector_bench: Dict) -> Dict:
        """
        Evaluates Sector Relative Performance Pillar:
        - Inputs: stock_5d (change_5d_pct), sector_5d (sector_change_pct)
        - Status:
            - 'measured': both stock 5d and sector benchmark 5d available
            - 'partial': sector 5d available but stock 5d missing
            - 'unavailable': sector benchmark missing or lacks sector_change_pct (score = None)
        """
        technical_data = technical_data or {}
        sector_bench = sector_bench or {}

        stock_5d_raw = technical_data.get("change_5d_pct")
        stock_5d = self._clean_float(stock_5d_raw, None)
        has_stock_5d = stock_5d is not None

        sec_5d_raw = sector_bench.get("sector_change_pct")
        sec_5d = self._clean_float(sec_5d_raw, None)
        has_sec_5d = sec_5d is not None

        tracking = {
            "stock_5d": self._track_input("stock_5d", stock_5d, is_valid=has_stock_5d),
            "sector_5d": self._track_input("sector_5d", sec_5d, is_valid=has_sec_5d),
            "sector_symbol": self._track_input("sector_symbol", sector_bench.get("sector_symbol"), is_valid=bool(sector_bench.get("sector_symbol")))
        }

        if not has_sec_5d:
            # Without sector benchmark data, relative sector performance cannot be measured
            return {
                "score": None,
                "status": "unavailable",
                "inputs": tracking
            }

        if not has_stock_5d:
            return {
                "score": 50.0,
                "status": "partial",
                "inputs": tracking
            }

        diff = stock_5d - sec_5d
        score = 50.0 + (diff * 5.0)

        return {
            "score": round(self.clamp(score), 2),
            "status": "measured",
            "inputs": tracking
        }

    def evaluate_market_alpha_pillar(self, technical_data: Dict) -> Dict:
        """
        Evaluates Market Alpha Pillar vs S&P 500:
        - Inputs: relative_alpha_5d
        - Status:
            - 'measured': relative_alpha_5d available
            - 'unavailable': missing relative_alpha_5d (score = None)
        """
        technical_data = technical_data or {}
        alpha_raw = technical_data.get("relative_alpha_5d")
        alpha_5d = self._clean_float(alpha_raw, None)
        has_alpha = alpha_5d is not None

        tracking = {
            "relative_alpha_5d": self._track_input("relative_alpha_5d", alpha_5d, is_valid=has_alpha)
        }

        if not has_alpha:
            return {
                "score": None,
                "status": "unavailable",
                "inputs": tracking
            }

        score = 50.0 + (alpha_5d * 5.0)
        return {
            "score": round(self.clamp(score), 2),
            "status": "measured",
            "inputs": tracking
        }

    def evaluate_valuation_history_pillar(self, gloomberb_payload: Dict) -> Dict:
        """
        Evaluates Valuation vs History Pillar:
        - Inputs: forward_pe, trailing_pe, revenue_growth
        - Status:
            - 'measured': both P/E expansion/contraction pair and revenue_growth available
            - 'partial': either P/E pair or revenue_growth available
            - 'unavailable': neither available (score = None)
        """
        gloomberb_payload = gloomberb_payload or {}
        fin = gloomberb_payload.get("financials", {})
        ratios = fin.get("key_ratios", {})

        fwd_pe = self._clean_float(ratios.get("forward_pe"), None)
        trl_pe = self._clean_float(ratios.get("trailing_pe"), None)
        has_pe_pair = (fwd_pe is not None and trl_pe is not None and trl_pe > 0.0)

        rev_growth = self._clean_float(ratios.get("revenue_growth"), None)
        has_rev_growth = rev_growth is not None

        tracking = {
            "forward_pe": self._track_input("forward_pe", fwd_pe, is_valid=fwd_pe is not None),
            "trailing_pe": self._track_input("trailing_pe", trl_pe, is_valid=trl_pe is not None),
            "revenue_growth": self._track_input("revenue_growth", rev_growth, is_valid=has_rev_growth)
        }

        if not has_pe_pair and not has_rev_growth:
            return {
                "score": None,
                "status": "unavailable",
                "inputs": tracking
            }

        status = "measured" if (has_pe_pair and has_rev_growth) else "partial"
        score = 50.0

        if has_pe_pair:
            if fwd_pe < trl_pe:
                score += 20.0
            else:
                score -= 10.0

        if has_rev_growth:
            if rev_growth > 0.15:
                score += 15.0
            elif rev_growth > 0.05:
                score += 5.0
            elif rev_growth < 0.0:
                score -= 10.0

        return {
            "score": round(self.clamp(score), 2),
            "status": status,
            "inputs": tracking
        }

    def evaluate_peer_valuation_pillar(self, gloomberb_payload: Dict) -> Dict:
        """
        Evaluates Valuation vs Direct Peers Pillar:
        - Inputs: stock forward_pe, stock ev_to_ebitda, peer benchmarks
        - Status:
            - 'measured': both forward P/E and EV/EBITDA comparisons available
            - 'partial': either forward P/E or EV/EBITDA comparison available
            - 'unavailable': no peers or no comparable metrics (score = None)
        """
        gloomberb_payload = gloomberb_payload or {}
        peer_data = gloomberb_payload.get("peer_valuation", {})
        ratios = gloomberb_payload.get("financials", {}).get("key_ratios", {})
        benchmarks = peer_data.get("direct_peer_benchmarks", [])

        stock_fwd_pe = self._clean_float(ratios.get("forward_pe"), None)
        stock_ev_ebitda = self._clean_float(peer_data.get("ev_to_ebitda"), None)

        peer_pes = [self._clean_float(p.get("forward_pe"), None) for p in benchmarks if self._clean_float(p.get("forward_pe"), None) is not None]
        peer_evs = [self._clean_float(p.get("ev_to_ebitda"), None) for p in benchmarks if self._clean_float(p.get("ev_to_ebitda"), None) is not None]

        pe_active = stock_fwd_pe is not None and len(peer_pes) > 0 and (sum(peer_pes) / len(peer_pes)) > 0
        ev_active = stock_ev_ebitda is not None and len(peer_evs) > 0 and (sum(peer_evs) / len(peer_evs)) > 0

        tracking = {
            "stock_forward_pe": self._track_input("stock_forward_pe", stock_fwd_pe, is_valid=stock_fwd_pe is not None),
            "stock_ev_ebitda": self._track_input("stock_ev_ebitda", stock_ev_ebitda, is_valid=stock_ev_ebitda is not None),
            "peer_count": self._track_input("peer_count", len(benchmarks), is_valid=len(benchmarks) > 0),
            "peer_avg_pe": self._track_input("peer_avg_pe", round(sum(peer_pes)/len(peer_pes), 2) if peer_pes else None, is_valid=bool(peer_pes)),
            "peer_avg_ev": self._track_input("peer_avg_ev", round(sum(peer_evs)/len(peer_evs), 2) if peer_evs else None, is_valid=bool(peer_evs))
        }

        if not pe_active and not ev_active:
            return {
                "score": None,
                "status": "unavailable",
                "inputs": tracking
            }

        status = "measured" if (pe_active and ev_active) else "partial"
        score = 50.0

        if pe_active:
            avg_peer_pe = sum(peer_pes) / len(peer_pes)
            pe_diff_pct = (avg_peer_pe - stock_fwd_pe) / avg_peer_pe
            score += (pe_diff_pct * 30.0)

        if ev_active:
            avg_peer_ev = sum(peer_evs) / len(peer_evs)
            ev_diff_pct = (avg_peer_ev - stock_ev_ebitda) / avg_peer_ev
            score += (ev_diff_pct * 30.0)

        return {
            "score": round(self.clamp(score), 2),
            "status": status,
            "inputs": tracking
        }

    def evaluate_financial_quality(self, gloomberb_payload: Dict) -> Dict:
        """
        Evaluates comprehensive financial quality and balance-sheet solvency:
        - Free Cash Flow (FCF) Margin and Trend
        - Operating Margin and Pricing Power Trend
        - Net Debt / EBITDA Solvency and Debt Burden
        - Interest Coverage Ratio
        - Earnings Quality Ratio (Operating Cash Flow / Net Income accrual test)
        - Share Dilution Rate
        - Industry Peer-relative Balance Sheet Comparison
        - Deterministic Value-Trap Detection
        """
        gloomberb_payload = gloomberb_payload or {}
        fin = gloomberb_payload.get("financials", {})
        fq = fin.get("financial_quality") or {}
        ratios = fin.get("key_ratios", {})
        peer_data = gloomberb_payload.get("peer_valuation", {})
        benchmarks = peer_data.get("direct_peer_benchmarks", [])

        # Extract core financial quality inputs
        fcf_margin = self._clean_float(fq.get("fcf_margin") if fq.get("fcf_margin") is not None else ratios.get("fcf_margin"), None)
        op_margin = self._clean_float(fq.get("operating_margin") if fq.get("operating_margin") is not None else ratios.get("operating_margins"), None)
        net_debt_ebitda = self._clean_float(fq.get("net_debt_to_ebitda") if fq.get("net_debt_to_ebitda") is not None else ratios.get("net_debt_to_ebitda"), None)
        interest_cov = self._clean_float(fq.get("interest_coverage") if fq.get("interest_coverage") is not None else ratios.get("interest_coverage"), None)
        earnings_quality = self._clean_float(fq.get("earnings_quality_ratio") if fq.get("earnings_quality_ratio") is not None else ratios.get("earnings_quality_ratio"), None)
        dilution_rate = self._clean_float(fq.get("share_dilution_rate"), None)

        net_debt = self._clean_float(fq.get("net_debt"), None)

        # Peer benchmarks for relative comparison
        peer_fcfs = [self._clean_float(p.get("fcf_margin"), None) for p in benchmarks if self._clean_float(p.get("fcf_margin"), None) is not None]
        peer_debts = [self._clean_float(p.get("net_debt_to_ebitda"), None) for p in benchmarks if self._clean_float(p.get("net_debt_to_ebitda"), None) is not None]
        avg_peer_fcf = (sum(peer_fcfs) / len(peer_fcfs)) if peer_fcfs else None
        avg_peer_debt = (sum(peer_debts) / len(peer_debts)) if peer_debts else None

        has_fcf = fcf_margin is not None
        has_op = op_margin is not None
        has_debt = net_debt_ebitda is not None or (net_debt is not None and net_debt <= 0.0)
        has_cov = interest_cov is not None or (net_debt is not None and net_debt <= 0.0)
        has_eq = earnings_quality is not None

        tracking = {
            "fcf_margin": self._track_input("fcf_margin", fcf_margin, is_valid=has_fcf),
            "operating_margin": self._track_input("operating_margin", op_margin, is_valid=has_op),
            "net_debt_to_ebitda": self._track_input("net_debt_to_ebitda", net_debt_ebitda, is_valid=has_debt),
            "interest_coverage": self._track_input("interest_coverage", interest_cov, is_valid=has_cov),
            "earnings_quality_ratio": self._track_input("earnings_quality_ratio", earnings_quality, is_valid=has_eq),
            "share_dilution_rate": self._track_input("share_dilution_rate", dilution_rate, is_valid=dilution_rate is not None)
        }

        valid_count = sum([has_fcf, has_op, has_debt, has_cov, has_eq])
        if valid_count == 0:
            return {
                "score": None,
                "status": "unavailable",
                "metrics": fq,
                "flags": [],
                "is_value_trap": False,
                "value_trap_reasons": [],
                "inputs": tracking
            }

        status = "measured" if valid_count >= 4 else "partial"
        score = 50.0
        flags = []

        # 1. FCF Margin (Cash generation power)
        if has_fcf:
            if fcf_margin >= 20.0:
                score += 15.0
            elif fcf_margin >= 10.0:
                score += 8.0
            elif fcf_margin >= 0.0:
                score += 3.0
            else:
                score -= 15.0
                flags.append(f"negative FCF margin ({fcf_margin:.1f}%) signals cash burn")

        # 2. Operating Margin (Pricing power & profitability)
        if has_op:
            if op_margin >= 25.0:
                score += 10.0
            elif op_margin >= 12.0:
                score += 5.0
            elif op_margin < 0.0:
                score -= 10.0
                flags.append(f"negative operating margin ({op_margin:.1f}%) reflects operational unprofitability")

        # 3. Net Debt / EBITDA (Solvency & Capital Structure)
        is_net_cash = (net_debt is not None and net_debt <= 0.0) or (net_debt_ebitda is not None and net_debt_ebitda <= 0.0)
        if is_net_cash:
            score += 15.0  # Fortress balance sheet
        elif net_debt_ebitda is not None:
            if net_debt_ebitda <= 1.5:
                score += 8.0
            elif net_debt_ebitda <= 3.0:
                score += 0.0
            elif net_debt_ebitda <= 4.5:
                score -= 10.0
                flags.append(f"elevated leverage (Net Debt/EBITDA {net_debt_ebitda:.1f}x)")
            else:
                score -= 20.0
                flags.append(f"distressed leverage (Net Debt/EBITDA {net_debt_ebitda:.1f}x exceeds 4.5x ceiling)")

        # 4. Interest Coverage Ratio (Debt servicing safety)
        if is_net_cash:
            score += 10.0  # Zero interest burden
        elif interest_cov is not None:
            if interest_cov >= 8.0:
                score += 10.0
            elif interest_cov >= 3.5:
                score += 5.0
            elif interest_cov >= 1.5:
                score -= 5.0
                flags.append(f"tight interest coverage ({interest_cov:.1f}x)")
            else:
                score -= 15.0
                flags.append(f"severe interest coverage strain ({interest_cov:.1f}x < 1.5x)")

        # 5. Earnings Quality (Operating Cash Flow / Net Income)
        if has_eq:
            if earnings_quality >= 1.1:
                score += 10.0
            elif earnings_quality >= 0.8:
                score += 4.0
            elif earnings_quality >= 0.4:
                score -= 8.0
                flags.append(f"weak cash conversion (OCF/Net Income {earnings_quality:.2f}x)")
            else:
                score -= 15.0
                flags.append(f"severe accrual divergence (OCF/Net Income {earnings_quality:.2f}x signals poor earnings quality)")

        # 6. Share Dilution Rate
        if dilution_rate is not None:
            if dilution_rate > 5.0:
                score -= 8.0
                flags.append(f"high annual share dilution ({dilution_rate:.1f}%)")
            elif dilution_rate <= 0.0:
                score += 5.0  # Net buybacks

        # 7. Peer-Relative Comparison Boost
        if has_fcf and avg_peer_fcf is not None:
            if fcf_margin > avg_peer_fcf + 5.0:
                score += 5.0
            elif fcf_margin < avg_peer_fcf - 5.0:
                score -= 5.0

        if not is_net_cash and net_debt_ebitda is not None and avg_peer_debt is not None:
            if net_debt_ebitda < avg_peer_debt - 1.0:
                score += 5.0
            elif net_debt_ebitda > avg_peer_debt + 1.5:
                score -= 5.0

        final_fq_score = round(self.clamp(score), 2)

        # 8. Deterministic Value-Trap Detection
        fwd_pe = self._clean_float(ratios.get("forward_pe"), None)
        peer_pes = [self._clean_float(p.get("forward_pe"), None) for p in benchmarks if self._clean_float(p.get("forward_pe"), None) is not None]
        avg_pe = (sum(peer_pes) / len(peer_pes)) if peer_pes else None

        pe_discount = (avg_pe - fwd_pe) / avg_pe if (fwd_pe is not None and avg_pe is not None and avg_pe > 0) else 0.0
        is_cheap = (fwd_pe is not None and fwd_pe < 15.0) or (pe_discount >= 0.20)

        is_distressed_debt = (net_debt_ebitda is not None and net_debt_ebitda > 3.5)
        is_burning_cash = (fcf_margin is not None and fcf_margin < 0.0)
        is_insolvent = (interest_cov is not None and interest_cov < 2.0 and not is_net_cash)
        is_low_quality = final_fq_score < 40.0

        is_value_trap = False
        value_trap_reasons = []

        if is_cheap and (is_distressed_debt or is_burning_cash or is_insolvent or is_low_quality):
            is_value_trap = True
            pe_str = f"low multiple (forward P/E {fwd_pe:.1f}x)" if fwd_pe else "cheap valuation"
            reasons = []
            if is_distressed_debt:
                reasons.append(f"heavy leverage (Net Debt/EBITDA {net_debt_ebitda:.1f}x)")
            if is_burning_cash:
                reasons.append(f"negative cash flow (FCF margin {fcf_margin:.1f}%)")
            if is_insolvent:
                reasons.append(f"strained debt servicing (interest coverage {interest_cov:.1f}x)")
            if is_low_quality and not (is_distressed_debt or is_burning_cash or is_insolvent):
                reasons.append(f"substandard financial quality score ({final_fq_score}/100)")
            value_trap_reasons.append(f"Value trap: {pe_str} masks structural risk: {', '.join(reasons)}")

        metrics_dict = dict(fq)
        metrics_dict.update({
            "fcf_margin": fcf_margin,
            "operating_margin": op_margin,
            "net_debt_to_ebitda": net_debt_ebitda,
            "interest_coverage": interest_cov,
            "earnings_quality_ratio": earnings_quality,
            "share_dilution_rate": dilution_rate,
            "peer_avg_fcf_margin": round(avg_peer_fcf, 2) if avg_peer_fcf is not None else None,
            "peer_avg_net_debt_to_ebitda": round(avg_peer_debt, 2) if avg_peer_debt is not None else None,
        })

        return {
            "score": final_fq_score,
            "status": status,
            "metrics": metrics_dict,
            "flags": flags,
            "is_value_trap": is_value_trap,
            "value_trap_reasons": value_trap_reasons,
            "inputs": tracking
        }

    def calculate_trend_score(self, technical_data: Dict) -> float:
        """Calculates Historical Trend Score (0–100) with backward compatibility."""
        res = self.evaluate_trend_pillar(technical_data)
        return res["score"] if res["score"] is not None else 50.0

    def calculate_sector_score(self, technical_data: Dict, sector_bench: Dict) -> float:
        """Calculates Sector Relative Performance Score (0–100) with backward compatibility."""
        res = self.evaluate_sector_pillar(technical_data, sector_bench)
        return res["score"] if res["score"] is not None else 50.0

    def calculate_market_alpha_score(self, technical_data: Dict) -> float:
        """Calculates Market Alpha Score vs S&P 500 (0–100) with backward compatibility."""
        res = self.evaluate_market_alpha_pillar(technical_data)
        return res["score"] if res["score"] is not None else 50.0

    def calculate_valuation_history_score(self, gloomberb_payload: Dict) -> float:
        """Calculates Valuation vs History Score (0–100) with backward compatibility."""
        res = self.evaluate_valuation_history_pillar(gloomberb_payload)
        return res["score"] if res["score"] is not None else 50.0

    def calculate_peer_valuation_score(self, gloomberb_payload: Dict) -> float:
        """Calculates Valuation vs Direct Peers Score (0–100) with backward compatibility."""
        res = self.evaluate_peer_valuation_pillar(gloomberb_payload)
        return res["score"] if res["score"] is not None else 50.0

    def compute_5pillar_scores(self, technical_data: Dict, gloomberb_payload: Dict, sector_bench: Dict = None) -> Dict:
        """
        Computes explicit scores across all 5 quantitative pillars and weighted composite score,
        with visible missingness tracking, dynamic weight re-normalization, and financial quality integration.

        Base Pillar Weights:
        - Historical Trend: 25%
        - Sector Performance: 20%
        - Market Alpha: 20%
        - Valuation History: 15%
        - Peer Valuation: 20%
        """
        p_trend = self.evaluate_trend_pillar(technical_data)
        p_sector = self.evaluate_sector_pillar(technical_data, sector_bench or (gloomberb_payload or {}).get("sector_benchmark", {}))
        p_alpha = self.evaluate_market_alpha_pillar(technical_data)
        p_val_hist = self.evaluate_valuation_history_pillar(gloomberb_payload)
        p_peer_val = self.evaluate_peer_valuation_pillar(gloomberb_payload)
        p_fq = self.evaluate_financial_quality(gloomberb_payload)

        pillar_evals = {
            "trend": p_trend,
            "sector": p_sector,
            "alpha": p_alpha,
            "valuation_history": p_val_hist,
            "peer_valuation": p_peer_val
        }

        weights = {
            "trend": 0.25,
            "sector": 0.20,
            "alpha": 0.20,
            "valuation_history": 0.15,
            "peer_valuation": 0.20
        }

        # Available pillars (score is not None)
        avail_pillars = {k: ev for k, ev in pillar_evals.items() if ev.get("score") is not None}
        avail_weight_sum = sum(weights[k] for k in avail_pillars)

        if avail_weight_sum > 0.0:
            # Re-normalize weights strictly across measured/partial available pillars
            raw_composite = sum(weights[k] * avail_pillars[k]["score"] for k in avail_pillars) / avail_weight_sum

            # Financial Quality Adjustment & Value-Trap Guard
            if p_fq.get("is_value_trap"):
                raw_composite = max(0.0, raw_composite - 15.0)
            elif p_fq.get("score") is not None and p_fq.get("score") >= 80.0:
                raw_composite = min(100.0, raw_composite + 3.0)

            # Volatility dampening
            vol_factor = QuantitativeScoringService.compute_vol_factor(
                technical_data.get("atr") if technical_data else None,
                technical_data.get("current_price") if technical_data else None
            )
            composite = round(raw_composite * vol_factor, 2)
            raw_composite_out = round(raw_composite, 2)

            if composite >= 70.0:
                signal = "Strong Bullish Quant Signal (BUY candidate if fundamental RAG confirms)"
            elif composite >= 45.0:
                signal = "Neutral / Mixed Quant Signal (HOLD candidate; BUY requires exceptional fundamental catalyst)"
            else:
                signal = "Bearish / Overvalued Quant Signal (SELL / Caution candidate)"
        else:
            raw_composite_out = None
            composite = None
            vol_factor = 1.0
            signal = "Unavailable Quant Signal (Insufficient pillar data)"

        return {
            "composite_quantitative_score": composite,
            "raw_composite": raw_composite_out,
            "trend_score": p_trend["score"],
            "sector_relative_score": p_sector["score"],
            "market_alpha_score": p_alpha["score"],
            "valuation_history_score": p_val_hist["score"],
            "peer_valuation_score": p_peer_val["score"],
            "vol_factor": vol_factor,
            "quant_signal": signal,
            "financial_quality_score": p_fq.get("score"),
            "is_value_trap": p_fq.get("is_value_trap", False),
            "pillar_status": {
                "trend": p_trend["status"],
                "sector": p_sector["status"],
                "alpha": p_alpha["status"],
                "valuation_history": p_val_hist["status"],
                "peer_valuation": p_peer_val["status"],
            },
            "pillar_input_tracking": {
                "trend": p_trend["inputs"],
                "sector": p_sector["inputs"],
                "alpha": p_alpha["inputs"],
                "valuation_history": p_val_hist["inputs"],
                "peer_valuation": p_peer_val["inputs"],
            },
            "available_pillars_count": len(avail_pillars),
            "total_pillars_count": 5,
            "financial_quality": p_fq,
            "value_trap_reasons": p_fq.get("value_trap_reasons", [])
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
    def _channel_available(status_entry, counts=None) -> bool:
        """
        Normalize a source-status entry that may be either a dict (`{available: bool,
        status, source_unavailable, fetch_count}`) or a string (`"available"` /
        `"source_unavailable"`). Anything else, including a missing entry, means the
        source is NOT available -- never assume availability from silence.
        """
        if isinstance(status_entry, str):
            return status_entry.strip().lower() in ("available", "ok", "ready", "true", "1")
        if status_entry is None:
            return False
        if not isinstance(status_entry, dict):
            return False
        if status_entry.get("available") is False:
            return False
        if status_entry.get("status") == "unavailable":
            return False
        if status_entry.get("source_unavailable") is True:
            return False
        if counts and status_entry.get("fetch_count", 0) == 0:
            return False
        return True

    @staticmethod
    def _source_said(source_status, channel: str, counts) -> bool:
        """
        Check a `data_source_status` / `source_status` payload (strings or dicts) for a
        channel's availability. A malformed/non-dict payload yields unavailable.
        """
        src = source_status or {}
        if not isinstance(src, dict):
            return False
        return QuantitativeScoringService._channel_available(src.get(channel), counts)

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
        inst_status = institutional_payload.get("source_status")
        inst_status = inst_status if isinstance(inst_status, dict) else {}
        inst_channels = {
            "sec_edgar_direct": 0.40,
            "reputable_news": 0.30,
            "investor_relations": 0.15,
            "official_press_releases": 0.10,
            "earnings_calls": 0.05,
        }
        inst_ok = 0.0
        for channel, weight in inst_channels.items():
            if QuantitativeScoringService._channel_available(inst_status.get(channel)):
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
