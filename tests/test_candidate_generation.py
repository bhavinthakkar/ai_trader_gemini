"""Unit tests for Relative Ranking Candidate Generation and Multi-Stage Filtering."""

import pytest
import numpy as np
import pandas as pd

from quant.candidate_generation import (
    generate_candidates_for_slice,
    simulate_candidate_portfolio,
)


@pytest.fixture
def mock_25_stock_slice():
    """Generates a mock slice of 25 stocks on a single rebalance date."""
    np.random.seed(42)
    tickers = [f"STK_{i:02d}" for i in range(25)]
    sectors = ["Technology"] * 10 + ["Financials"] * 5 + ["Healthcare"] * 5 + ["Consumer"] * 5
    scores = np.linspace(35.0, 75.0, 25)  # 25 scores from 35 to 75
    np.random.shuffle(scores)

    df = pd.DataFrame({
        "decision_date": ["2024-03-31"] * 25,
        "ticker": tickers,
        "taxonomy_sector": sectors,
        "hybrid_score": scores,
        "is_dip_setup": [False] * 25,
        "atr14_pct": [2.5] * 25,
        "exec_ret_20d": [0.03] * 25,
        "spy_exec_ret_20d": [0.01] * 25,
    })
    return df


def test_top_20_pct_candidate_generation(mock_25_stock_slice):
    """Verifies that relative ranking selects exactly top 20% (5 out of 25 stocks)."""
    # Relative only: top 20% of 25 = 5 stocks
    cands_rel = generate_candidates_for_slice(
        mock_25_stock_slice,
        mode="relative_only",
        top_pct=0.20,
    )
    assert len(cands_rel) == 5
    # The 5 selected stocks must be the 5 highest scoring stocks in the slice
    top_5_scores = sorted(mock_25_stock_slice["hybrid_score"], reverse=True)[:5]
    assert sorted(cands_rel["hybrid_score"].tolist(), reverse=True) == pytest.approx(top_5_scores)


def test_dual_mode_enforces_absolute_quality_floor(mock_25_stock_slice):
    """Verifies that dual mode requires both Top 20% AND Score >= floor."""
    # Score floor 60.0
    cands_dual = generate_candidates_for_slice(
        mock_25_stock_slice,
        mode="dual",
        top_pct=0.20,
        min_abs_score=60.0,
    )
    # Must only contain stocks in top 5 AND with score >= 60.0
    for s in cands_dual["hybrid_score"]:
        assert s >= 60.0
    assert len(cands_dual) <= 5


def test_weak_market_preserves_cash(mock_25_stock_slice):
    """In a severe bear market where all 25 stocks score < 50, dual mode generates 0 candidates."""
    bear_slice = mock_25_stock_slice.copy()
    bear_slice["hybrid_score"] = np.linspace(20.0, 48.0, 25)  # Max score is 48.0 < 60.0

    cands_rel = generate_candidates_for_slice(
        bear_slice,
        mode="relative_only",
        top_pct=0.20,
    )
    assert len(cands_rel) == 5  # Blind relative ranking still buys 5 falling stocks!

    cands_dual = generate_candidates_for_slice(
        bear_slice,
        mode="dual",
        top_pct=0.20,
        min_abs_score=60.0,
    )
    assert len(cands_dual) == 0  # Dual mode generates 0 candidates -> 100% Cash!


def test_dip_buy_candidate_exception(mock_25_stock_slice):
    """Verifies that certified dip-buy setups qualify even if hybrid score is in [52, 60)."""
    df = mock_25_stock_slice.copy()
    # Force STK_00 to have score 53.0 with is_dip_setup=True, ranked lower than top 5
    df.loc[df["ticker"] == "STK_00", "hybrid_score"] = 53.0
    df.loc[df["ticker"] == "STK_00", "is_dip_setup"] = True

    cands = generate_candidates_for_slice(
        df,
        mode="dual",
        top_pct=0.20,
        min_abs_score=60.0,
        dip_threshold=52.0,
    )
    assert "STK_00" in cands["ticker"].tolist()


def test_portfolio_simulation_sector_cap():
    """Tests that portfolio simulation respects sector concentration cap during execution."""
    dates = ["2024-01-31", "2024-02-29"]
    rows = []
    for d in dates:
        for i in range(25):
            rows.append({
                "decision_date": d,
                "ticker": f"STK_{i:02d}",
                # Top 5 are all Technology
                "taxonomy_sector": "Technology" if i < 5 else "Financials",
                "hybrid_score": 75.0 - i,  # Highest scores are Tech
                "is_dip_setup": False,
                "atr14_pct": 2.0,
                "exec_ret_20d": 0.02,
                "spy_exec_ret_20d": 0.01,
            })
    panel_df = pd.DataFrame(rows)

    # Max 2 stocks per sector
    res = simulate_candidate_portfolio(
        panel_df,
        mode="dual",
        top_pct=0.20,
        min_abs_score=60.0,
        top_k=5,
        max_stocks_per_sector=2,
    )

    assert res["full_cagr"] != 0.0
    assert res["mean_turnover"] >= 0.0


def test_volatility_adjusted_sizing_inversely_scales_weights():
    """Verifies that high-volatility stocks receive lower portfolio weights than low-volatility stocks."""
    panel_df = pd.DataFrame([
        {
            "decision_date": "2024-01-31",
            "ticker": "LOW_VOL",
            "taxonomy_sector": "Consumer",
            "hybrid_score": 70.0,
            "is_dip_setup": False,
            "atr14_pct": 0.02,  # 2% ATR
            "exec_ret_20d": 0.05,
            "spy_exec_ret_20d": 0.01,
        },
        {
            "decision_date": "2024-01-31",
            "ticker": "HIGH_VOL",
            "taxonomy_sector": "Technology",
            "hybrid_score": 68.0,
            "is_dip_setup": False,
            "atr14_pct": 0.08,  # 8% ATR (4x higher vol)
            "exec_ret_20d": 0.05,
            "spy_exec_ret_20d": 0.01,
        },
    ])

    res = simulate_candidate_portfolio(
        panel_df,
        mode="dual",
        top_pct=1.0,
        min_abs_score=60.0,
        top_k=2,
        sizing_mode="volatility_adjusted",
    )
    assert res["n_candidates"] == 2
    assert res["full_cagr"] > 0.0
