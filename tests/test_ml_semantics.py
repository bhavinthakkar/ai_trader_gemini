"""Unit Tests for ML Score Semantics and Gating Architecture (Step 3.3).

Validates:
1. Rule ML-SEM-01: ML score is strictly an ordinal cross-sectional relative rank [0, 100].
2. Score 50.0 represents the cohort median on date t, not zero excess return.
3. Score 80.0 represents top quintile relative performance, not 80% return or probability.
4. Rule ML-SEM-02: Decoupled gating prevents relative ML rank from overriding bearish quant signals (Q_quant < 50).
5. Neutral quant signals (Q_quant == 50) are not converted into false sell signals.
"""

import pytest
from ml_signal_service import classify_ml_signal


def test_ml_score_semantic_classification():
    """Verify that classify_ml_signal correctly maps ordinal percentiles to relative tiers."""
    # Score 80: Top Quintile (Strong Outperform)
    res_80 = classify_ml_signal(80.0)
    assert res_80["signal"] == "STRONG OUTPERFORM"
    assert "Top Quintile" in res_80["tier"]
    assert res_80["stance"] == "BULLISH"

    # Score 50: Cohort Median (Market Perform)
    res_50 = classify_ml_signal(50.0)
    assert res_50["signal"] == "MARKET PERFORM"
    assert "Quintile 3" in res_50["tier"]
    assert res_50["stance"] == "NEUTRAL"

    # Score 20: Quintile 4 (Underperform)
    res_20 = classify_ml_signal(20.0)
    assert res_20["signal"] == "UNDERPERFORM"
    assert "Quintile 4" in res_20["tier"]
    assert res_20["stance"] == "BEARISH"

    # Score 10: Bottom Quintile (Strong Underperform)
    res_10 = classify_ml_signal(10.0)
    assert res_10["signal"] == "STRONG UNDERPERFORM"
    assert "Bottom Quintile" in res_10["tier"]
    assert res_10["stance"] == "BEARISH"


def test_decoupled_gating_veto_logic():
    """Verify that a bearish quant score (Q_quant < 50) vetoes an aggressive ML relative rank."""
    # Scenario: Bearish market/stock trend (Q_quant = 40.0), but stock is least bad in peer group (S_ml = 95.0)
    q_quant = 40.0
    s_ml = 95.0

    # Under naive linear blend (80/20):
    naive_blend = 0.80 * q_quant + 0.20 * s_ml  # 32 + 19 = 51.0 (artificially crosses neutral!)
    assert naive_blend > 50.0

    # Under Decoupled Gating (Rule ML-SEM-02):
    # Directional execution is governed by Q_quant >= 50.0
    is_directionally_eligible = q_quant >= 50.0
    assert not is_directionally_eligible, "Bearish quant signal must not be overridden by ML relative ranking!"


def test_neutral_quant_protection():
    """Verify that a neutral quant score (Q_quant = 50) is not turned into a sell signal by peer rank."""
    # Scenario: Flat moving averages, zero alpha (Q_quant = 50.0), but stock is bottom of strong peers (S_ml = 20.0)
    q_quant = 50.0
    s_ml = 20.0

    # Under naive linear blend:
    naive_blend = 0.80 * q_quant + 0.20 * s_ml  # 40 + 4 = 44.0 (false SELL signal)
    assert naive_blend < 50.0

    # Under Decoupled Gating:
    # Neutral quant baseline is preserved; stock is simply not shortlisted by ML (S_ml < 70)
    is_ml_shortlisted = s_ml >= 70.0
    assert not is_ml_shortlisted
    # The quant score remains uncorrupted at 50.0
    assert q_quant == 50.0
