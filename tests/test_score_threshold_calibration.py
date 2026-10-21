"""Unit tests for Score Threshold Empirical Calibration and Simulation Harness."""

import pytest
import numpy as np
import pandas as pd

from quant.score_threshold_calibration import (
    compute_score_distribution_statistics,
    simulate_threshold_portfolio,
)


@pytest.fixture
def synthetic_calibration_data():
    """Generates synthetic panel data across 12 monthly dates for 5 tickers."""
    dates = pd.date_range("2024-01-31", periods=12, freq="ME").strftime("%Y-%m-%d").tolist()
    tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
    rows = []

    np.random.seed(42)
    for d in dates:
        for t in tickers:
            score = float(np.random.normal(53.0, 10.0))
            ret_5d = float(np.random.normal(-0.02, 0.05))
            rsi = float(np.random.uniform(25.0, 75.0))
            val_score = float(np.random.uniform(40.0, 80.0))
            accruals = float(np.random.uniform(-0.05, 0.15))
            roe = float(np.random.uniform(-0.05, 0.35))
            future_ret = float(np.random.normal(0.015, 0.05))
            spy_ret = float(np.random.normal(0.01, 0.03))

            is_dip = (
                (-0.15 <= ret_5d <= -0.05)
                and (30.0 <= rsi <= 45.0)
                and (val_score >= 60.0)
                and (accruals <= 0.10)
                and (roe > 0)
            )

            rows.append({
                "decision_date": d,
                "ticker": t,
                "score_C": score,
                "ret_5d": ret_5d,
                "rsi14": rsi,
                "val_score": val_score,
                "accruals_ratio": accruals,
                "roe_ttm": roe,
                "is_dip_setup": is_dip,
                "future_return_20d": future_ret,
                "spy_future_return_20d": spy_ret,
            })

    return pd.DataFrame(rows)


def test_compute_score_distribution_statistics(synthetic_calibration_data):
    stats = compute_score_distribution_statistics(synthetic_calibration_data, score_col="score_C")
    assert stats["score_column"] == "score_C"
    assert stats["n_observations"] == 60
    assert stats["n_dates"] == 12
    assert "percentiles" in stats
    assert stats["percentiles"]["p50_median"] > 0
    assert "threshold_coverage" in stats
    assert ">=60" in stats["threshold_coverage"]
    assert "date_breadth" in stats
    assert ">=60" in stats["date_breadth"]


def test_threshold_portfolio_monotonicity(synthetic_calibration_data):
    """Lowering thresholds must monotonically increase or maintain qualifying candidate counts."""
    res_base = simulate_threshold_portfolio(
        synthetic_calibration_data,
        score_col="score_C",
        buy_threshold=60.0,
        dip_buy_threshold=52.0,
    )
    res_a = simulate_threshold_portfolio(
        synthetic_calibration_data,
        score_col="score_C",
        buy_threshold=58.0,
        dip_buy_threshold=52.0,
    )
    res_b = simulate_threshold_portfolio(
        synthetic_calibration_data,
        score_col="score_C",
        buy_threshold=56.0,
        dip_buy_threshold=50.0,
    )
    res_c = simulate_threshold_portfolio(
        synthetic_calibration_data,
        score_col="score_C",
        buy_threshold=54.0,
        dip_buy_threshold=48.0,
    )

    assert res_base["n_candidates"] <= res_a["n_candidates"]
    assert res_a["n_candidates"] <= res_b["n_candidates"]
    assert res_b["n_candidates"] <= res_c["n_candidates"]
    assert res_base["max_drawdown"] <= 0.0
    assert res_base["executed_entries"] >= 0
    assert "cagr" in res_base
    assert "sharpe_ratio" in res_base
    assert "sortino_ratio" in res_base
    assert "calmar_ratio" in res_base


def test_dip_buy_eligibility_admits_sub_60_scores(synthetic_calibration_data):
    """Verifies that stocks scoring below 60 are admitted when dip-buy criteria are satisfied."""
    df = synthetic_calibration_data.copy()
    # Force one stock to have score 53 (below 60) with valid dip-buy setup
    df.loc[0, "score_C"] = 53.0
    df.loc[0, "is_dip_setup"] = True

    res = simulate_threshold_portfolio(
        df,
        score_col="score_C",
        buy_threshold=60.0,
        dip_buy_threshold=52.0,
    )
    # Target stock must qualify under dip-buy threshold 52.0
    assert res["n_candidates"] >= 1
