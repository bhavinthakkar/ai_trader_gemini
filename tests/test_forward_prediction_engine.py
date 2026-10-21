"""Unit tests for Forward-Looking Market Prediction and Dislocation Intelligence Engine."""

import pytest
import numpy as np
import pandas as pd

from quant.forward_prediction_engine import (
    ForwardPredictionEngine,
    ForwardSignalType,
)


@pytest.fixture
def base_price_history():
    """Generates 100 days of price and volume data."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    base = 100.0
    prices = [base]
    for _ in range(99):
        prices.append(prices[-1] * (1.0 + np.random.normal(0.001, 0.015)))
    volumes = np.random.uniform(1_000_000, 5_000_000, 100)
    return pd.Series(prices, index=dates), pd.Series(volumes, index=dates)


def test_fundamental_dislocation_buy_recommendation(base_price_history):
    """Tests that a 5-day drop on transient bad news with strong fundamentals generates DISLOCATION_BUY."""
    prices, volumes = base_price_history
    prices = prices.copy()
    # Force a -8% pullback over the last 5 days
    prices.iloc[-1] = prices.iloc[-6] * 0.92

    engine = ForwardPredictionEngine()
    result = engine.predict(
        ticker="QUALITY_CORP",
        current_price=float(prices.iloc[-1]),
        price_series=prices,
        volume_series=volumes,
        operating_margin=35.0,  # High margin
        roe=28.0,               # High ROE
        accruals_ratio=0.01,    # High earnings quality
        is_headline_transient=True,
    )

    assert result.forward_signal == ForwardSignalType.DISLOCATION_BUY
    assert result.dislocation_score >= 65.0
    assert result.target_price is not None
    assert result.stop_loss is not None
    assert "Pristine core profitability" in " ".join(result.key_drivers)


def test_overextended_sell_recommendation(base_price_history):
    """Tests that an overbought, parabolic stock generates an OVEREXTENDED_SELL recommendation."""
    prices, volumes = base_price_history
    prices = prices.copy()
    # Force a +35% surge above the 50d average
    ma50 = prices.rolling(50).mean().iloc[-1]
    prices.iloc[-1] = ma50 * 1.35

    engine = ForwardPredictionEngine()
    result = engine.predict(
        ticker="EXTENDED_TECH",
        current_price=float(prices.iloc[-1]),
        price_series=prices,
        volume_series=volumes,
    )

    assert result.forward_signal == ForwardSignalType.OVEREXTENDED_SELL
    assert result.exhaustion_risk_score >= 70.0
    assert "TAKE PROFIT / SELL" in result.actionable_summary


def test_insider_cluster_buying_elevates_sentiment():
    """Tests that C-suite open-market purchases elevate the insider sentiment score."""
    engine = ForwardPredictionEngine()
    txs = [
        {"role": "CEO", "transaction_type": "P_PURCHASE", "shares": 25000, "value": 2500000, "is_rule_10b5_1": False},
        {"role": "CFO", "transaction_type": "P_PURCHASE", "shares": 10000, "value": 1000000, "is_rule_10b5_1": False},
    ]
    score, notes = engine.evaluate_insider_sentiment(txs)
    assert score >= 80.0
    assert any("BULLISH CLUSTER" in n for n in notes)


def test_insider_10b5_1_routine_selling_filtered_out():
    """Tests that routine scheduled 10b5-1 sales do not trigger false-positive bear warnings."""
    engine = ForwardPredictionEngine()
    txs = [
        {"role": "CEO", "transaction_type": "S_SALE", "shares": 5000, "value": 500000, "is_rule_10b5_1": True},
    ]
    score, notes = engine.evaluate_insider_sentiment(txs)
    # Routine sale leaves score close to neutral (>= 45.0)
    assert score >= 45.0


def test_unplanned_c_suite_dump_triggers_bear_warning():
    """Tests that multiple non-10b5-1 C-suite sales trigger a dump warning."""
    engine = ForwardPredictionEngine()
    txs = [
        {"role": "CEO", "transaction_type": "S_SALE", "shares": 50000, "value": 5000000, "is_rule_10b5_1": False},
        {"role": "CFO", "transaction_type": "S_SALE", "shares": 30000, "value": 3000000, "is_rule_10b5_1": False},
        {"role": "COO", "transaction_type": "S_SALE", "shares": 25000, "value": 2500000, "is_rule_10b5_1": False},
    ]
    score, notes = engine.evaluate_insider_sentiment(txs)
    assert score <= 30.0
    assert any("BEARISH DUMP WARNING" in n for n in notes)


def test_prediction_markets_macro_and_binary_events():
    """Tests evaluation of macro rate cut probabilities and corporate binary event contracts."""
    engine = ForwardPredictionEngine()
    
    # Bullish scenario: High Fed rate cut prob, low recession, favorable event outcome
    score_bull, notes_bull = engine.evaluate_prediction_markets(
        fed_rate_cut_prob=0.85,
        recession_prob=0.10,
        company_event_favorable_prob=0.80,
    )
    assert score_bull >= 65.0
    assert any("Fed rate cut" in n for n in notes_bull)
    assert any("favorable probability" in n for n in notes_bull)

    # Bearish scenario: Low cut prob, high recession prob, low event prob
    score_bear, notes_bear = engine.evaluate_prediction_markets(
        fed_rate_cut_prob=0.15,
        recession_prob=0.55,
        company_event_favorable_prob=0.20,
    )
    assert score_bear <= 35.0
    assert any("Elevated recession probability" in n for n in notes_bear)


def test_options_microstructure_uoa_and_skew():
    """Tests options flow: Put/Call ratio, 25-delta skew, and Unusual Options Activity (UOA)."""
    engine = ForwardPredictionEngine()

    # Smart money call sweep & negative 25d skew (calls trading at a premium)
    score_bull, notes_bull = engine.evaluate_options_microstructure(
        put_call_volume_ratio=0.45,
        iv_skew_25d_pct=-2.5,
        gamma_exposure_regime="POSITIVE_GAMMA",
        unusual_options_activity_detected=True,
    )
    assert score_bull >= 85.0
    assert any("Low Put/Call Ratio" in n for n in notes_bull)
    assert any("Inverse call skew detected" in n for n in notes_bull)
    assert any("Unusual Options Activity" in n for n in notes_bull)

    # Downside panic put hedging & steep put skew
    score_bear, notes_bear = engine.evaluate_options_microstructure(
        put_call_volume_ratio=1.65,
        iv_skew_25d_pct=11.0,
        gamma_exposure_regime="NEGATIVE_GAMMA",
        unusual_options_activity_detected=False,
    )
    assert score_bear <= 30.0
    assert any("High Put/Call Ratio" in n for n in notes_bear)
    assert any("Steep downside put skew" in n for n in notes_bear)


def test_credit_macro_leading_indicator_and_veto(base_price_history):
    """Tests that high-yield spread widening and VIX backwardation trigger defensive capital preservation."""
    prices, volumes = base_price_history
    prices = prices.copy()
    # Force a -8% pullback on pristine fundamentals which would otherwise be DISLOCATION_BUY
    prices.iloc[-1] = prices.iloc[-6] * 0.92

    engine = ForwardPredictionEngine()
    result = engine.predict(
        ticker="MACRO_SENSITIVE",
        current_price=float(prices.iloc[-1]),
        price_series=prices,
        volume_series=volumes,
        operating_margin=30.0,
        roe=22.0,
        accruals_ratio=0.02,
        is_headline_transient=True,
        # Credit macro stress inputs:
        hy_spread_10d_change_bps=45.0,        # Rapid widening > +25 bps
        vix_spot_vs_vix3m_ratio=1.18,          # Severe VIX backwardation > 1.05
    )

    # Macro veto overrides buy signal to protect capital
    assert result.credit_macro_score <= 35.0
    assert result.forward_signal == ForwardSignalType.NEUTRAL_WAIT
    assert "MACRO REGIME DEFENSE" in " ".join(result.key_drivers)
    assert "NEUTRAL (DEFENSIVE CASH)" in result.actionable_summary


def test_options_uoa_triggers_momentum_breakout(base_price_history):
    """Tests that institutional block call sweeps trigger MOMENTUM_BREAKOUT."""
    prices, volumes = base_price_history
    engine = ForwardPredictionEngine()

    result = engine.predict(
        ticker="BREAKOUT_TICKER",
        current_price=float(prices.iloc[-1]),
        price_series=prices,
        volume_series=volumes,
        put_call_volume_ratio=0.38,
        iv_skew_25d_pct=-1.5,
        unusual_options_activity_detected=True,
    )

    assert result.forward_signal == ForwardSignalType.MOMENTUM_BREAKOUT
    assert result.options_flow_score >= 80.0
    assert "MOMENTUM BUY" in result.actionable_summary


def test_treasury_yield_spike_and_yield_curve_inversion(base_price_history):
    """Tests that a sharp 10Y Treasury yield spike and deep yield curve inversion penalize macro credit score."""
    prices, volumes = base_price_history
    engine = ForwardPredictionEngine()

    score, drivers = engine.evaluate_credit_macro_indicators(
        treasury_10y_yield_pct=4.85,
        treasury_10y_10d_change_bps=30.0,
        yield_curve_10y_2y_bps=-60.0,
    )

    # Base 50 - 15 (surge) - 10 (elevated yield) - 10 (curve inversion) = 15.0
    assert score <= 20.0
    assert any("Treasury Yield Surge" in d for d in drivers)
    assert any("High Benchmark Yield" in d for d in drivers)
    assert any("Yield Curve Inversion" in d for d in drivers)

    # In end-to-end predict, this enforces defensive cash
    result = engine.predict(
        ticker="RATE_SENSITIVE_TECH",
        current_price=float(prices.iloc[-1]),
        price_series=prices,
        volume_series=volumes,
        treasury_10y_yield_pct=4.85,
        treasury_10y_10d_change_bps=30.0,
        yield_curve_10y_2y_bps=-60.0,
    )
    assert result.credit_macro_score <= 35.0
    assert result.forward_signal == ForwardSignalType.NEUTRAL_WAIT
    assert "MACRO REGIME DEFENSE" in " ".join(result.key_drivers)


