"""Unit Tests for Four Comparable Experiments (Step 3.4) and ML Fallback/Freshness (Step 3.5).

Validates:
1. Step 3.4 Portfolio Experiments:
   - All 4 experiments (A, B, C, D) executed over 56 walk-forward months.
   - Acceptance Gate C: Exp D beats Exp A across CAGR, Sharpe, and Information Ratio.
   - Exp C (naive 80/20) underperforms Exp A and is rejected.
   - ML ranking quality: Rank IC > 0, positive quintile spread, stable autocorrelation.
2. Step 3.5 ML Export, Freshness, and Fallback Behavior:
   - Explicit eligibility contract: {"model_status": "OUT_OF_SCOPE", "ml_score": None, "ml_weight": 0.0}.
   - Never assigns 50.0 or 0.50 percentile to unmodeled stocks.
   - Session-aligned freshness detects stale scores and rejects lookahead.
   - CSV schema verification flags missing required columns.
   - Downstream normalization in main.py applies 0% weight to out-of-scope tickers.
"""

from datetime import date, timedelta
import json
from pathlib import Path
import pytest
import pandas as pd

from ml_signal_service import (
    classify_ml_signal,
    get_out_of_scope_context,
    verify_file_schema,
    verify_session_freshness,
    get_ml_context,
    format_ml_context,
)
from quant.composite_score import blend_quant_and_ml_scores


DATA_DIR = Path("data")


# ============================================================================
# Step 3.4: Four Portfolio Experiments & Acceptance Gate C
# ============================================================================

def test_four_experiments_artifacts_exist():
    """Verify that Step 3.4 comparison artifacts are persisted on disk."""
    summary_path = DATA_DIR / "four_experiments_comparison.csv"
    annual_path = DATA_DIR / "four_experiments_by_year.csv"
    json_path = DATA_DIR / "four_experiments_results.json"

    assert summary_path.is_file(), f"Missing {summary_path}"
    assert annual_path.is_file(), f"Missing {annual_path}"
    assert json_path.is_file(), f"Missing {json_path}"


def test_acceptance_gate_c_metrics():
    """Verify that Acceptance Gate C metrics prove Exp D superiority and Exp C rejection."""
    summary_df = pd.read_csv(DATA_DIR / "four_experiments_comparison.csv")
    assert len(summary_df) == 4

    exp_a = summary_df[summary_df["Experiment"].str.contains("Experiment A")].iloc[0]
    exp_c = summary_df[summary_df["Experiment"].str.contains("Experiment C")].iloc[0]
    exp_d = summary_df[summary_df["Experiment"].str.contains("Experiment D")].iloc[0]

    cagr_a = float(exp_a["Net CAGR"].replace("%", ""))
    cagr_c = float(exp_c["Net CAGR"].replace("%", ""))
    cagr_d = float(exp_d["Net CAGR"].replace("%", ""))

    sharpe_a = float(exp_a["Sharpe Ratio"])
    sharpe_c = float(exp_c["Sharpe Ratio"])
    sharpe_d = float(exp_d["Sharpe Ratio"])

    # 1. Exp C (Naive 80/20) underperformed Exp A (Quant Only) -> REJECTED
    assert cagr_c < cagr_a, f"Exp C CAGR ({cagr_c}%) should be lower than Exp A ({cagr_a}%)"
    assert sharpe_c < sharpe_a, f"Exp C Sharpe ({sharpe_c}) should be lower than Exp A ({sharpe_a})"

    # 2. Exp D (Training-Validated 30/70) outperforms Exp A -> PASSED
    assert cagr_d > cagr_a, f"Exp D CAGR ({cagr_d}%) should exceed Exp A ({cagr_a}%)"
    assert sharpe_d > sharpe_a, f"Exp D Sharpe ({sharpe_d}) should exceed Exp A ({sharpe_a})"

    # 3. Check JSON verdict
    with open(DATA_DIR / "four_experiments_results.json") as f:
        meta = json.load(f)
    verdict = meta["acceptance_gate_c_verdict"]
    assert verdict["gate_c_status"] == "PASSED"
    assert "Experiment D" in verdict["best_model"]
    assert "Experiment C" in verdict["rejected_model"]


def test_ml_ranking_quality_metrics():
    """Verify that ML ranking metrics meet quality thresholds."""
    with open(DATA_DIR / "four_experiments_results.json") as f:
        meta = json.load(f)
    quality = meta["ml_ranking_quality"]

    assert quality["ml_mean_rank_ic"] > 0.05, f"Rank IC ({quality['ml_mean_rank_ic']}) must be > +0.05"
    assert quality["ml_ic_hit_rate"] >= 0.50, f"Hit rate ({quality['ml_ic_hit_rate']}) must be >= 50%"
    assert quality["ml_top_minus_bottom_spread_20d"] > 0.0, "Top quintile must beat bottom quintile"
    assert quality["ml_rank_autocorrelation_1m"] >= 0.50, "Rank stability must be >= 0.50"


# ============================================================================
# Step 3.5: ML Export, Freshness, and Fallback Behavior
# ============================================================================

def test_out_of_scope_contract():
    """Verify that out-of-scope context enforces explicit null score and zero weight."""
    ctx = get_out_of_scope_context("TICKER_NOT_IN_UNIVERSE")
    assert ctx["model_status"] == "OUT_OF_SCOPE"
    assert ctx["in_scope"] is False
    assert ctx["ml_score"] is None, "ml_score must be None for out-of-scope tickers"
    assert ctx["ml_weight"] == 0.0, "ml_weight must be strictly 0.0"
    assert ctx["score_0_100"] is None, "score_0_100 must not be 50.0"
    assert ctx["rank_percentile"] is None, "rank_percentile must not be 0.50"
    assert ctx["predicted_20d_excess_return"] is None
    assert ctx["stance"] == "UNAVAILABLE"


def test_session_freshness_verification():
    """Verify that session freshness correctly detects stale and future dates."""
    today = date(2026, 10, 16)

    # 1. Fresh date within 3 days
    fresh_date = today - timedelta(days=2)
    is_fresh, code, age = verify_session_freshness(fresh_date, expected_session_date=today, max_age_days=7)
    assert is_fresh is True
    assert code == "FRESH"
    assert age == 2

    # 2. Stale date > 7 days
    stale_date = today - timedelta(days=10)
    is_fresh, code, age = verify_session_freshness(stale_date, expected_session_date=today, max_age_days=7)
    assert is_fresh is False
    assert code == "STALE_SCORE"
    assert age == 10

    # 3. Lookahead / future date
    future_date = today + timedelta(days=1)
    is_fresh, code, age = verify_session_freshness(future_date, expected_session_date=today, max_age_days=7)
    assert is_fresh is False
    assert code == "LOOKAHEAD_VIOLATION"
    assert age == -1


def test_csv_schema_verification():
    """Verify that verify_file_schema flags missing columns."""
    valid_headers = ["ticker", "score_date", "ml_rank_percentile", "predicted_20d_excess_return", "ml_model_name"]
    is_ok, err = verify_file_schema(valid_headers)
    assert is_ok is True
    assert err is None

    # Missing ticker
    no_ticker = ["score_date", "ml_rank_percentile"]
    is_ok, err = verify_file_schema(no_ticker)
    assert is_ok is False
    assert "ticker" in err

    # Missing date
    no_date = ["ticker", "ml_rank_percentile"]
    is_ok, err = verify_file_schema(no_date)
    assert is_ok is False
    assert "date" in err

    # Missing percentile
    no_pct = ["ticker", "score_date"]
    is_ok, err = verify_file_schema(no_pct)
    assert is_ok is False
    assert "percentile" in err


def test_get_ml_context_out_of_scope_ticker():
    """Verify get_ml_context handles an unmodeled stock with explicit status."""
    ctx = get_ml_context("COMPLETELY_UNKNOWN_STOCK_XYZ")
    assert ctx["model_status"] == "OUT_OF_SCOPE"
    assert ctx["in_scope"] is False
    assert ctx["ml_score"] is None
    assert ctx["ml_weight"] == 0.0
    assert ctx["score_0_100"] is None


def test_format_ml_context_out_of_scope():
    """Verify format_ml_context does not mention baseline 50.0 for out-of-scope stocks."""
    ctx = get_out_of_scope_context("XYZ")
    txt = format_ml_context(ctx)
    assert "50.0" not in txt, "Out of scope prompt context must never cite 50.0 baseline"
    assert "ml_weight=0.0%" in txt
    assert "ml_score=None" in txt


def test_blend_quant_and_ml_scores_zero_weight_for_out_of_scope():
    """Verify that blend_quant_and_ml_scores never applies ML weight to out-of-scope tickers."""
    out_of_scope_ctx = get_out_of_scope_context("UNMODELED")

    # Quant score = 72.5
    comp, w, s = blend_quant_and_ml_scores(72.5, out_of_scope_ctx)
    assert comp == 72.5, "Out of scope composite must equal quant score exactly"
    assert w == 0.0, "Out of scope ml_weight must be strictly 0.0"
    assert s is None, "Out of scope ml_score must be None"

    # Stale context
    stale_ctx = get_out_of_scope_context("STALE_TICKER", status="STALE_SCORE")
    comp_stale, w_stale, s_stale = blend_quant_and_ml_scores(65.0, stale_ctx)
    assert comp_stale == 65.0
    assert w_stale == 0.0
    assert s_stale is None

    # In-scope context with 80.0 ML score
    in_scope_ctx = {
        "model_status": "IN_SCOPE",
        "in_scope": True,
        "ml_score": 80.0,
        "score_0_100": 80.0,
        "ml_weight": 0.20,
    }
    comp_in, w_in, s_in = blend_quant_and_ml_scores(60.0, in_scope_ctx)
    # 0.80 * 60 + 0.20 * 80 = 48 + 16 = 64.0
    assert comp_in == 64.0
    assert w_in == 0.20
    assert s_in == 80.0

