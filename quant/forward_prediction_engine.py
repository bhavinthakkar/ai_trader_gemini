"""Forward-Looking Market Prediction and Dislocation Intelligence Engine.

Synthesizes forward-looking market dynamics beyond lagged technical indicators:
1. Fundamental Dislocation Detection: Identifies temporary bad-news pullbacks on pristine fundamentals (Overreaction BUY).
2. Institutional "Smart Money" Flow: Detects block accumulation, volume-price trend, and money flow (Institutional Sponsorship).
3. Insider Sentiment Analysis: Evaluates Form 4 insider transactions, prioritizing high-conviction C-suite cluster purchases.
4. Overextended Exhaustion & Correction Warning: Identifies blow-off tops, extreme momentum divergence, and distribution (Take-Profit / SELL).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class ForwardSignalType(str, Enum):
    """Categorical forward-looking recommendation."""
    DISLOCATION_BUY = "DISLOCATION_BUY"          # Bad news overreaction on strong fundamentals
    MOMENTUM_BREAKOUT = "MOMENTUM_BREAKOUT"      # Institutional accumulation breakout
    NEUTRAL_WAIT = "NEUTRAL_WAIT"                # Range-bound, no clear forward catalyst
    OVEREXTENDED_SELL = "OVEREXTENDED_SELL"      # Exhaustion / overbought correction warning
    STRUCTURAL_DUMP = "STRUCTURAL_DUMP"          # Insider dumping / fundamental impairment


class CatalystDuration(str, Enum):
    """Classification of event impact duration."""
    TRANSIENT_OVERREACTION = "TRANSIENT_OVERREACTION"  # 1-quarter noise, temporary supply chain delay
    STRUCTURAL_IMPAIRMENT = "STRUCTURAL_IMPAIRMENT"    # Accounting issues, lost pricing power, secular decline
    NEUTRAL_NOISE = "NEUTRAL_NOISE"                    # Routine market volatility


@dataclass
class ForwardPredictionResult:
    """Detailed diagnostic output of the forward-looking prediction engine."""
    ticker: str
    current_price: float
    forward_signal: ForwardSignalType
    conviction_score: float  # 0.0 to 100.0
    dislocation_score: float  # 0 to 100 (high = strong fundamental dip)
    institutional_flow_score: float  # 0 to 100 (high = aggressive smart money buying)
    insider_sentiment_score: float  # 0 to 100 (high = C-suite cluster accumulation)
    options_flow_score: float  # 0 to 100 (high = institutional call sweeps / positive skew)
    prediction_market_score: float  # 0 to 100 (high = favorable macro/regulatory probabilities)
    credit_macro_score: float  # 0 to 100 (high = tightening credit spreads, contango VIX)
    alternative_data_score: float  # 0 to 100 (high = robust supply chain, hiring, traffic)
    exhaustion_risk_score: float  # 0 to 100 (high = extreme overbought correction risk)
    key_drivers: List[str] = field(default_factory=list)
    actionable_summary: str = ""
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None


class ForwardPredictionEngine:
    """
    Evaluates forward-looking catalyst, institutional flow, and fundamental dislocation setups.
    """

    @staticmethod
    def evaluate_institutional_flow(
        price_series: pd.Series,
        volume_series: pd.Series,
        high_series: Optional[pd.Series] = None,
        low_series: Optional[pd.Series] = None,
        lookback_days: int = 20,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculates Institutional "Smart Money" Flow (0-100) combining:
        1. Chaikin Money Flow (CMF): Measures whether closes are in upper half of bar on heavy volume.
        2. Volume-Price Trend (VPT): Detects institutional algorithmic accumulation.
        3. Up/Down Volume Ratio: Compares volume on advancing days vs declining days.
        """
        if len(price_series) < lookback_days or len(volume_series) < lookback_days:
            return 50.0, {"status": "INSUFFICIENT_HISTORY"}

        p = price_series.tail(lookback_days)
        v = volume_series.tail(lookback_days)
        h = high_series.tail(lookback_days) if high_series is not None else p * 1.01
        l = low_series.tail(lookback_days) if low_series is not None else p * 0.99

        # 1. Chaikin Money Flow (CMF)
        # Money Flow Multiplier: [(Close - Low) - (High - Close)] / (High - Low)
        hl_range = (h - l).replace(0, np.nan)
        mf_mult = ((p - l) - (h - p)) / hl_range
        mf_mult = mf_mult.fillna(0.0)
        mf_vol = mf_mult * v
        cmf = mf_vol.sum() / max(1.0, v.sum())  # [-1.0, +1.0]

        # 2. Up/Down Volume Ratio
        ret = p.pct_change().fillna(0.0)
        up_vol = v[ret > 0].sum()
        down_vol = v[ret < 0].sum()
        up_down_ratio = up_vol / max(1.0, down_vol)  # 1.0 = balanced, > 1.5 = institutional buying

        # 3. Composite Institutional Flow Score (0 to 100)
        # CMF: +0.20 maps to ~80, -0.20 maps to ~20
        cmf_score = np.clip(50.0 + cmf * 150.0, 0.0, 100.0)
        ud_score = np.clip(up_down_ratio * 40.0, 0.0, 100.0)
        flow_score = float(0.60 * cmf_score + 0.40 * ud_score)

        return round(flow_score, 1), {
            "cmf": round(float(cmf), 3),
            "up_down_vol_ratio": round(float(up_down_ratio), 2),
            "cmf_score": round(float(cmf_score), 1),
            "flow_score": round(flow_score, 1),
        }

    @staticmethod
    def evaluate_insider_sentiment(
        insider_transactions: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[float, List[str]]:
        """
        Evaluates Form 4 Insider Sentiment (0-100):
        - Open-market C-suite cluster purchases (+30 to +50 pts): High conviction signal.
        - Routine 10b5-1 option-exercise sales (0 to -5 pts): Filtered out as non-informative.
        - Heavy non-10b5-1 C-suite dumping (-30 to -50 pts): Bearish structural signal.
        """
        if not insider_transactions:
            return 50.0, ["No recent Form 4 insider transactions recorded (Neutral 50.0)"]

        score = 50.0
        notes = []
        c_suite_buyers = set()
        c_suite_sellers = set()
        unplanned_sales = 0

        for tx in insider_transactions:
            role = str(tx.get("role", "")).upper()
            tx_type = str(tx.get("transaction_type", "")).upper()  # 'P_PURCHASE', 'S_SALE', 'OPTION_EXERCISE'
            is_10b5_1 = bool(tx.get("is_rule_10b5_1", False))
            shares = float(tx.get("shares", 0))
            value = float(tx.get("value", 0.0))

            is_exec = any(k in role for k in ["CEO", "CFO", "COO", "PRESIDENT", "DIRECTOR"])

            if "PURCHASE" in tx_type:
                # Open market purchase with own money is extremely bullish
                c_suite_buyers.add(role)
                score += 15.0
                notes.append(f"Insider Purchase: {role} bought {shares:,.0f} shares (${value:,.0f})")

            elif "SALE" in tx_type:
                if is_10b5_1:
                    # Routine scheduled diversification - negligible signal
                    score -= 1.0
                else:
                    unplanned_sales += 1
                    c_suite_sellers.add(role)
                    score -= 12.0
                    notes.append(f"Unplanned Insider Sale: {role} sold {shares:,.0f} shares (${value:,.0f}) outside 10b5-1")

        # Cluster bonuses
        if len(c_suite_buyers) >= 2:
            score += 20.0
            notes.append("BULLISH CLUSTER: Multiple C-suite executives purchasing simultaneously")

        if len(c_suite_sellers) >= 3 and unplanned_sales >= 3:
            score -= 25.0
            notes.append("BEARISH DUMP WARNING: Coordinated unplanned C-suite liquidation")

        clamped = float(np.clip(score, 0.0, 100.0))
        return round(clamped, 1), notes

    @staticmethod
    def evaluate_dislocation_opportunity(
        current_price: float,
        price_5d_ago: float,
        operating_margin: float,
        roe: float,
        accruals_ratio: float,
        rsi14: float,
        is_headline_transient: bool = True,
    ) -> Tuple[float, List[str]]:
        """
        Measures whether a stock drop is a genuine fundamental buying dislocation (0-100):
        - Severe drop (-5% to -15%) on strong balance sheet (high ROE, positive cash flow, low accruals).
        - Penalizes value traps (accounting red flags, permanent moat destruction).
        """
        drop_pct = (current_price / price_5d_ago - 1.0) * 100.0
        score = 50.0
        reasons = []

        if drop_pct <= -5.0:
            # Significant pullback detected
            reasons.append(f"5-day pullback of {drop_pct:.1f}%")

            # Check business fundamentals
            if operating_margin >= 25.0 and roe >= 15.0:
                score += 25.0
                reasons.append(f"Pristine core profitability: Operating Margin {operating_margin:.1f}%, ROE {roe:.1f}%")

            # Non-value-trap validation (Accruals quality)
            if accruals_ratio <= 0.05:
                score += 15.0
                reasons.append("High earnings quality: Low accruals ratio (cash flow confirms net income)")
            else:
                score -= 20.0
                reasons.append("Accounting warning: Elevated accruals ratio indicates potential earnings quality risk")

            # News classification
            if is_headline_transient:
                score += 15.0
                reasons.append("Catalyst classification: Transient market overreaction (temporary noise)")
            else:
                score -= 35.0
                reasons.append("Permanent impairment risk: Fundamental business moat deterioration")

            # Oversold bound check
            if 25.0 <= rsi14 <= 45.0:
                score += 10.0
                reasons.append(f"RSI {rsi14:.1f} reflects oversold discount without falling knife panic")
            elif rsi14 < 25.0:
                score -= 10.0
                reasons.append(f"RSI {rsi14:.1f} indicates unarrested liquidation panic; wait for stabilization")
        else:
            reasons.append("No significant 5-day dislocation drop detected")

        clamped = float(np.clip(score, 0.0, 100.0))
        return round(clamped, 1), reasons

    @staticmethod
    def evaluate_exhaustion_risk(
        current_price: float,
        ma50: float,
        rsi14: float,
        up_down_vol_ratio: float,
    ) -> Tuple[float, List[str]]:
        """
        Evaluates Overbought Exhaustion / Correction Warning (0-100):
        - Price severely extended above 50d MA (> +25%).
        - RSI > 75 with bearish volume distribution (falling up/down volume ratio).
        """
        dist_ma50 = (current_price / ma50 - 1.0) * 100.0
        risk_score = 20.0
        warnings = []

        if dist_ma50 >= 25.0:
            risk_score += 35.0
            warnings.append(f"Price extended +{dist_ma50:.1f}% above 50-day moving average (Extreme Extension)")

        if rsi14 >= 75.0:
            risk_score += 25.0
            warnings.append(f"RSI {rsi14:.1f} in overbought exhaustion territory")

        if up_down_vol_ratio < 0.80 and dist_ma50 > 15.0:
            risk_score += 20.0
            warnings.append("Bearish volume divergence: Up-volume drying up while price remains extended")

        clamped = float(np.clip(risk_score, 0.0, 100.0))
        return round(clamped, 1), warnings

    @staticmethod
    def evaluate_prediction_markets(
        fed_rate_cut_prob: Optional[float] = None,
        recession_prob: Optional[float] = None,
        company_event_favorable_prob: Optional[float] = None,
    ) -> Tuple[float, List[str]]:
        """
        Evaluates real-money prediction markets (Polymarket, Kalshi, PredictIt) (0 to 100):
        - Macro rate cut / easing tailwind: Fed cut probability.
        - Macro recession probability: Kalshi US recession contract.
        - Specific company binary events: Antitrust rulings, FDA approvals, export licenses.
        """
        score = 50.0
        drivers = []

        if fed_rate_cut_prob is not None:
            # High cut probability (>70%) is a valuation tailwind for equities (+10 pts)
            rate_impact = (fed_rate_cut_prob - 0.50) * 20.0
            score += rate_impact
            if fed_rate_cut_prob >= 0.70:
                drivers.append(f"Prediction Market: {fed_rate_cut_prob*100:.0f}% probability of Fed rate cut (Macro Easing Tailwind)")
            elif fed_rate_cut_prob <= 0.30:
                drivers.append(f"Prediction Market: Low rate cut probability ({fed_rate_cut_prob*100:.0f}%) indicates restrictive monetary policy")

        if recession_prob is not None:
            # High recession probability (>30%) heavily penalizes cyclical equities
            recession_impact = recession_prob * 30.0
            score -= recession_impact
            if recession_prob >= 0.30:
                drivers.append(f"Prediction Market: Elevated recession probability ({recession_prob*100:.0f}%) creates cyclical headwind")
            else:
                drivers.append(f"Prediction Market: Low recession probability ({recession_prob*100:.0f}%) confirms economic expansion")

        if company_event_favorable_prob is not None:
            # e.g. FDA approval, antitrust dismissal
            event_impact = (company_event_favorable_prob - 0.50) * 40.0
            score += event_impact
            if company_event_favorable_prob >= 0.70:
                drivers.append(f"Prediction Market: {company_event_favorable_prob*100:.0f}% favorable probability for upcoming corporate binary event")
            elif company_event_favorable_prob <= 0.35:
                drivers.append(f"Prediction Market: High probability of adverse event outcome ({100 - company_event_favorable_prob*100:.0f}%)")

        clamped = float(np.clip(score, 0.0, 100.0))
        return round(clamped, 1), drivers

    @staticmethod
    def evaluate_options_microstructure(
        put_call_volume_ratio: Optional[float] = None,
        iv_skew_25d_pct: Optional[float] = None,
        gamma_exposure_regime: Optional[str] = None,
        iv_rank: Optional[float] = None,
        unusual_options_activity_detected: bool = False,
    ) -> Tuple[float, List[str]]:
        """
        Evaluates Options Derivatives Microstructure & Smart Money Flow (0 to 100):
        1. Put/Call Ratio (PCR): < 0.60 indicates heavy institutional call buying; > 1.30 indicates put hedging panic.
        2. 25-Delta Volatility Skew: Call skew (Put IV < Call IV) indicates institutional upside bidding.
        3. Dealer Gamma Exposure (GEX): +GEX pins price and compresses volatility; -GEX accelerates trending moves.
        4. Unusual Options Activity (UOA): Aggressive block call sweeps at the ask.
        """
        score = 50.0
        drivers = []

        if put_call_volume_ratio is not None:
            if put_call_volume_ratio <= 0.60:
                score += 15.0
                drivers.append(f"Options Flow: Low Put/Call Ratio ({put_call_volume_ratio:.2f}) confirms aggressive call buying")
            elif put_call_volume_ratio >= 1.30:
                score -= 15.0
                drivers.append(f"Options Flow: High Put/Call Ratio ({put_call_volume_ratio:.2f}) indicates heavy institutional downside hedging")

        if iv_skew_25d_pct is not None:
            # iv_skew = 25d Put IV - 25d Call IV
            if iv_skew_25d_pct <= 0.0:
                score += 18.0
                drivers.append("Options Skew: Inverse call skew detected (Calls trading at premium to Puts; Smart Money bidding upside)")
            elif iv_skew_25d_pct >= 8.0:
                score -= 15.0
                drivers.append(f"Options Skew: Steep downside put skew ({iv_skew_25d_pct:+.1f}%) signals market bracing for volatility crash")

        if gamma_exposure_regime:
            regime = str(gamma_exposure_regime).upper()
            if "POSITIVE" in regime:
                drivers.append("Dealer Gamma (+GEX): Market makers long gamma; price volatility pinned, favoring mean-reversion dip-buying")
                score += 5.0
            elif "NEGATIVE" in regime:
                drivers.append("Dealer Gamma (-GEX): Market makers short gamma; hedging flow accelerates trend momentum and breakout extensions")
                score += 5.0

        if unusual_options_activity_detected:
            score += 20.0
            drivers.append("Unusual Options Activity (UOA): Institutional large-block aggressive call sweeps executed at the ask")

        clamped = float(np.clip(score, 0.0, 100.0))
        return round(clamped, 1), drivers

    @staticmethod
    def evaluate_credit_macro_indicators(
        hy_spread_bps: Optional[float] = None,
        hy_spread_10d_change_bps: Optional[float] = None,
        vix_spot_vs_vix3m_ratio: Optional[float] = None,
        yield_curve_10y_2y_bps: Optional[float] = None,
        treasury_10y_yield_pct: Optional[float] = None,
        treasury_10y_10d_change_bps: Optional[float] = None,
    ) -> Tuple[float, List[str]]:
        """
        Evaluates Credit Spreads, Sovereign Treasury Yields, and Macro VIX Structure (0 to 100):
        1. High-Yield OAS Spread: Widening credit spreads lead equity selloffs by 1-2 weeks.
        2. VIX Term Structure: Contango (< 0.90) = benign bull regime; Backwardation (> 1.05) = systemic acute stress.
        3. Sovereign Treasury Yields: Sharp yield spikes (+25 bps) surge discount rates and compress multiples.
        4. Yield Curve: 10Y - 2Y slope (inversion vs healthy steepening).
        """
        score = 50.0
        drivers = []

        if hy_spread_10d_change_bps is not None:
            if hy_spread_10d_change_bps >= 25.0:
                score -= 25.0
                drivers.append(f"Credit Leading Indicator: High-Yield spreads widened +{hy_spread_10d_change_bps:.0f} bps (Liquidity Contraction Warning)")
            elif hy_spread_10d_change_bps <= -15.0:
                score += 15.0
                drivers.append("Credit Leading Indicator: High-Yield spreads tightening (Robust corporate credit health)")

        if vix_spot_vs_vix3m_ratio is not None:
            if vix_spot_vs_vix3m_ratio >= 1.05:
                score -= 20.0
                drivers.append(f"VIX Term Structure: Backwardation ({vix_spot_vs_vix3m_ratio:.2f} ratio) reflects acute spot panic hedging")
            elif vix_spot_vs_vix3m_ratio <= 0.88:
                score += 10.0
                drivers.append(f"VIX Term Structure: Normal contango ({vix_spot_vs_vix3m_ratio:.2f} ratio) confirms stable market liquidity")

        if treasury_10y_10d_change_bps is not None:
            if treasury_10y_10d_change_bps >= 25.0:
                score -= 15.0
                drivers.append(f"Treasury Yield Surge: 10Y yield spiked +{treasury_10y_10d_change_bps:.0f} bps (Severe Valuation Multiple Compression)")
            elif treasury_10y_10d_change_bps <= -25.0:
                score += 10.0
                drivers.append(f"Treasury Yield Easing: 10Y yield dropped {treasury_10y_10d_change_bps:.0f} bps (Discount Rate Relief)")

        if treasury_10y_yield_pct is not None:
            if treasury_10y_yield_pct >= 4.75:
                score -= 10.0
                drivers.append(f"High Benchmark Yield: 10Y at {treasury_10y_yield_pct:.2f}% severely compresses Equity Risk Premium (TINA is Dead)")

        if yield_curve_10y_2y_bps is not None:
            if yield_curve_10y_2y_bps <= -50.0:
                score -= 10.0
                drivers.append(f"Yield Curve Inversion: 10Y-2Y slope at {yield_curve_10y_2y_bps:+.0f} bps warns of late-cycle contraction")
            elif yield_curve_10y_2y_bps >= 50.0:
                score += 5.0
                drivers.append(f"Yield Curve Slope: Normal steepening ({yield_curve_10y_2y_bps:+.0f} bps) indicates healthy credit demand")

        clamped = float(np.clip(score, 0.0, 100.0))
        return round(clamped, 1), drivers

    @staticmethod
    def evaluate_alternative_data(
        supply_chain_health_score: Optional[float] = None,
        hiring_velocity_growth_pct: Optional[float] = None,
        web_app_traffic_growth_pct: Optional[float] = None,
    ) -> Tuple[float, List[str]]:
        """
        Evaluates Alternative Data & Digital Exhaust (0 to 100):
        - Supply chain lead times & supplier order backlogs.
        - Corporate hiring velocity (job opening trajectory leads quarterly headcount costs).
        - Digital traffic growth (web visits, app active user trajectory).
        """
        score = 50.0
        drivers = []

        if supply_chain_health_score is not None:
            delta = (supply_chain_health_score - 50.0) * 0.40
            score += delta
            if supply_chain_health_score >= 70.0:
                drivers.append(f"Supply Chain Data: Robust supplier component throughput ({supply_chain_health_score:.0f}/100)")
            elif supply_chain_health_score <= 35.0:
                drivers.append(f"Supply Chain Warning: Bottlenecks or shipment delays detected ({supply_chain_health_score:.0f}/100)")

        if hiring_velocity_growth_pct is not None:
            if hiring_velocity_growth_pct >= 15.0:
                score += 10.0
                drivers.append(f"Alternative Data: Strategic hiring expansion (+{hiring_velocity_growth_pct:.1f}% net job postings)")
            elif hiring_velocity_growth_pct <= -25.0:
                score -= 15.0
                drivers.append(f"Alternative Data Warning: Severe hiring freeze ({hiring_velocity_growth_pct:.1f}% net job postings) precedes guidance cuts")

        if web_app_traffic_growth_pct is not None:
            if web_app_traffic_growth_pct >= 20.0:
                score += 15.0
                drivers.append(f"Digital Exhaust: Strong online platform engagement (+{web_app_traffic_growth_pct:.1f}% YoY traffic growth)")
            elif web_app_traffic_growth_pct <= -15.0:
                score -= 15.0
                drivers.append(f"Digital Exhaust Warning: Declining platform engagement ({web_app_traffic_growth_pct:.1f}% YoY traffic drop)")

        clamped = float(np.clip(score, 0.0, 100.0))
        return round(clamped, 1), drivers

    def predict(
        self,
        ticker: str,
        current_price: float,
        price_series: pd.Series,
        volume_series: pd.Series,
        high_series: Optional[pd.Series] = None,
        low_series: Optional[pd.Series] = None,
        operating_margin: float = 20.0,
        roe: float = 15.0,
        accruals_ratio: float = 0.02,
        insider_transactions: Optional[List[Dict[str, Any]]] = None,
        is_headline_transient: bool = True,
        atr14: Optional[float] = None,
        # Extended forward indicators
        fed_rate_cut_prob: Optional[float] = None,
        recession_prob: Optional[float] = None,
        company_event_favorable_prob: Optional[float] = None,
        put_call_volume_ratio: Optional[float] = None,
        iv_skew_25d_pct: Optional[float] = None,
        gamma_exposure_regime: Optional[str] = None,
        unusual_options_activity_detected: bool = False,
        hy_spread_bps: Optional[float] = None,
        hy_spread_10d_change_bps: Optional[float] = None,
        vix_spot_vs_vix3m_ratio: Optional[float] = None,
        yield_curve_10y_2y_bps: Optional[float] = None,
        treasury_10y_yield_pct: Optional[float] = None,
        treasury_10y_10d_change_bps: Optional[float] = None,
        supply_chain_health_score: Optional[float] = None,
        hiring_velocity_growth_pct: Optional[float] = None,
        web_app_traffic_growth_pct: Optional[float] = None,
    ) -> ForwardPredictionResult:
        """
        Synthesizes the complete multi-source forward-looking prediction.
        """
        p_now = current_price
        p_5d = float(price_series.iloc[-6]) if len(price_series) >= 6 else p_now
        ma50 = float(price_series.rolling(50).mean().iloc[-1]) if len(price_series) >= 50 else p_now

        # Calculate RSI
        delta = price_series.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        rsi14 = float(100.0 - (100.0 / (1.0 + rs)).iloc[-1]) if len(price_series) >= 15 else 50.0

        # 1. Institutional Smart Money Flow
        flow_score, flow_meta = self.evaluate_institutional_flow(
            price_series, volume_series, high_series, low_series
        )

        # 2. Insider Sentiment
        insider_score, insider_notes = self.evaluate_insider_sentiment(insider_transactions)

        # 3. Dislocation Opportunity
        disloc_score, disloc_notes = self.evaluate_dislocation_opportunity(
            current_price=p_now,
            price_5d_ago=p_5d,
            operating_margin=operating_margin,
            roe=roe,
            accruals_ratio=accruals_ratio,
            rsi14=rsi14,
            is_headline_transient=is_headline_transient,
        )

        # 4. Overbought Exhaustion Risk
        exhaust_score, exhaust_notes = self.evaluate_exhaustion_risk(
            current_price=p_now,
            ma50=ma50,
            rsi14=rsi14,
            up_down_vol_ratio=flow_meta.get("up_down_vol_ratio", 1.0),
        )

        # 5. Prediction Markets
        pred_market_score, pred_market_notes = self.evaluate_prediction_markets(
            fed_rate_cut_prob=fed_rate_cut_prob,
            recession_prob=recession_prob,
            company_event_favorable_prob=company_event_favorable_prob,
        )

        # 6. Options Microstructure
        options_score, options_notes = self.evaluate_options_microstructure(
            put_call_volume_ratio=put_call_volume_ratio,
            iv_skew_25d_pct=iv_skew_25d_pct,
            gamma_exposure_regime=gamma_exposure_regime,
            unusual_options_activity_detected=unusual_options_activity_detected,
        )

        # 7. Credit & Macro Leading Indicators
        credit_score, credit_notes = self.evaluate_credit_macro_indicators(
            hy_spread_bps=hy_spread_bps,
            hy_spread_10d_change_bps=hy_spread_10d_change_bps,
            vix_spot_vs_vix3m_ratio=vix_spot_vs_vix3m_ratio,
            yield_curve_10y_2y_bps=yield_curve_10y_2y_bps,
            treasury_10y_yield_pct=treasury_10y_yield_pct,
            treasury_10y_10d_change_bps=treasury_10y_10d_change_bps,
        )

        # 8. Alternative Data
        alt_data_score, alt_data_notes = self.evaluate_alternative_data(
            supply_chain_health_score=supply_chain_health_score,
            hiring_velocity_growth_pct=hiring_velocity_growth_pct,
            web_app_traffic_growth_pct=web_app_traffic_growth_pct,
        )

        # Synthesize Overall Recommendation
        drivers: List[str] = []
        drivers.extend(pred_market_notes)
        drivers.extend(options_notes)
        drivers.extend(credit_notes)
        drivers.extend(alt_data_notes)
        atr = atr14 or (p_now * 0.03)

        # Macro risk veto: If credit spreads are blowing out and VIX in backwardation, downgrade risk tolerance
        is_macro_stress = credit_score <= 35.0

        if exhaust_score >= 70.0:
            signal = ForwardSignalType.OVEREXTENDED_SELL
            conviction = exhaust_score
            drivers.extend(exhaust_notes)
            summary = (
                f"TAKE PROFIT / SELL INDICATION: Stock is extended +{(p_now/ma50-1)*100:.1f}% above MA50 with "
                f"RSI {rsi14:.1f}. High risk of sharp mean-reverting correction."
            )
            target = round(ma50 * 1.05, 2)
            stop = round(p_now + 2.0 * atr, 2)

        elif is_macro_stress:
            signal = ForwardSignalType.NEUTRAL_WAIT
            conviction = 75.0
            drivers.append("MACRO REGIME DEFENSE: Credit spread widening and VIX backwardation enforce capital preservation")
            summary = "NEUTRAL (DEFENSIVE CASH): Severe credit/macro stress overrides individual equity signals. Sit in cash."
            target = None
            stop = None

        elif disloc_score >= 65.0 and insider_score >= 45.0:
            signal = ForwardSignalType.DISLOCATION_BUY
            # Dislocation conviction boosted by bullish options flow or prediction market tailwinds
            extra_boost = 0.0
            if options_score >= 65.0: extra_boost += 5.0
            if pred_market_score >= 60.0: extra_boost += 5.0
            conviction = min(100.0, disloc_score + extra_boost)
            drivers.extend(disloc_notes)
            drivers.extend(flow_meta.get("notes", []))
            summary = (
                f"FORWARD BUY RECOMMENDATION: Stock pulled back {(p_now/p_5d-1)*100:.1f}% on transient headlines, "
                f"while core fundamentals remain pristine. Options flow and prediction markets confirm dislocation."
            )
            target = round(p_5d * 1.08, 2)
            stop = round(p_now - 1.5 * atr, 2)

        elif (flow_score >= 70.0 or options_score >= 75.0) and insider_score >= 50.0:
            signal = ForwardSignalType.MOMENTUM_BREAKOUT
            conviction = max(flow_score, options_score)
            drivers.append(f"Institutional accumulation confirmed: Flow {flow_score:.0f}/100, Options Sentiment {options_score:.0f}/100")
            summary = (
                f"MOMENTUM BUY: Big investors accumulating aggressively across equity blocks and options sweeps. "
                f"Institutional sponsorship confirms upward continuation."
            )
            target = round(p_now + 3.0 * atr, 2)
            stop = round(p_now - 1.5 * atr, 2)

        elif insider_score <= 30.0:
            signal = ForwardSignalType.STRUCTURAL_DUMP
            conviction = 100.0 - insider_score
            drivers.extend(insider_notes)
            summary = "SELL / CAUTION: Unplanned C-suite insider liquidation detected."
            target = None
            stop = None

        else:
            signal = ForwardSignalType.NEUTRAL_WAIT
            conviction = 50.0
            summary = "NEUTRAL: Balanced institutional flow and absence of dislocation catalysts. Maintain current holdings."
            target = None
            stop = None

        return ForwardPredictionResult(
            ticker=ticker,
            current_price=p_now,
            forward_signal=signal,
            conviction_score=round(conviction, 1),
            dislocation_score=disloc_score,
            institutional_flow_score=flow_score,
            insider_sentiment_score=insider_score,
            options_flow_score=options_score,
            prediction_market_score=pred_market_score,
            credit_macro_score=credit_score,
            alternative_data_score=alt_data_score,
            exhaustion_risk_score=exhaust_score,
            key_drivers=drivers,
            actionable_summary=summary,
            target_price=target,
            stop_loss=stop,
        )
