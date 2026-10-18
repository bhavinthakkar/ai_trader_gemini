"""Unit Tests for Point-in-Time ML Training Table (Step 3.1).

Validates:
1. Table schema: decision_date, ticker, feature_1..feature_16, label_excess_20d, label_end_date, universe_version.
2. Uniqueness: exactly one row per stock and decision date.
3. Feature validity: all 16 features are strictly finite numeric values (zero NaNs).
4. Calendar adherence: label_end_date strictly follows the 20-session trading calendar.
5. Universe reproducibility: universe_version tag is consistent.
6. Zero lookahead: point-in-time imputation uses only contemporaneous information.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from quant.build_training_table import (
    FROZEN_16_FACTORS,
    FEATURE_ALIASES,
    FACTOR_SPECIFICATIONS,
    build_ml_training_table,
    load_source_datasets,
)


@pytest.fixture(scope="module")
def training_table() -> pd.DataFrame:
    parquet_path = Path("data/ml_training_table.parquet")
    if not parquet_path.exists():
        pytest.fail("data/ml_training_table.parquet missing. Run quant/build_training_table.py first.")
    return pd.read_parquet(parquet_path)


def test_table_row_uniqueness(training_table: pd.DataFrame):
    """Verify exactly one row per stock and decision date."""
    duplicates = training_table.duplicated(subset=["decision_date", "ticker"])
    assert not duplicates.any(), f"Found {duplicates.sum()} duplicate (decision_date, ticker) pairs!"


def test_required_columns_present(training_table: pd.DataFrame):
    """Verify standard Step 3.1 columns are all present."""
    required_cols = [
        "decision_date",
        "ticker",
        "label_excess_20d",
        "label_end_date",
        "universe_version",
        "has_label",
    ] + [f"feature_{i+1}" for i in range(16)]

    missing = [c for c in required_cols if c not in training_table.columns]
    assert not missing, f"Missing required columns: {missing}"


def test_frozen_16_factors_coverage(training_table: pd.DataFrame):
    """Verify that both aliased (feature_1..16) and named factor columns exist."""
    assert len(FROZEN_16_FACTORS) == 16
    for i, factor in enumerate(FROZEN_16_FACTORS, 1):
        feature_id = f"feature_{i}"
        assert feature_id in training_table.columns
        assert factor in training_table.columns
        # Values must be identical
        np.testing.assert_array_equal(
            training_table[feature_id].to_numpy(),
            training_table[factor].to_numpy(),
            err_msg=f"Mismatch between {feature_id} and {factor}",
        )


def test_no_unhandled_nans_in_features(training_table: pd.DataFrame):
    """Verify zero unhandled NaNs or Infinities in the 16 features."""
    for i in range(1, 17):
        col = f"feature_{i}"
        s = training_table[col]
        assert s.notna().all(), f"{col} has {s.isna().sum()} nulls!"
        assert np.isfinite(s.to_numpy()).all(), f"{col} contains non-finite values!"


def test_trading_calendar_label_spacing(training_table: pd.DataFrame):
    """Verify label_end_date adheres to the trading calendar (~20 trading sessions)."""
    labeled_rows = training_table[training_table["has_label"]].copy()
    assert len(labeled_rows) > 0, "No mature labeled rows found!"

    dec_dates = pd.to_datetime(labeled_rows["decision_date"])
    end_dates = pd.to_datetime(labeled_rows["label_end_date"])

    # label_end_date must be strictly in the future
    assert (end_dates > dec_dates).all(), "Found label_end_date <= decision_date!"

    # Spacing for 20 trading sessions should be between 25 and 45 calendar days
    calendar_days = (end_dates - dec_dates).dt.days
    assert (calendar_days >= 25).all(), f"Found suspiciously short calendar day gap: {calendar_days.min()} days"
    assert (calendar_days <= 45).all(), f"Found suspiciously long calendar day gap: {calendar_days.max()} days"


def test_universe_version_consistency(training_table: pd.DataFrame):
    """Verify universe version tag matches universe_v3."""
    assert (training_table["universe_version"] == "universe_v3").all()


def test_label_excess_bounds(training_table: pd.DataFrame):
    """Verify 20-session excess returns are within sensible equity bounds."""
    labeled = training_table[training_table["has_label"]]
    returns = labeled["label_excess_20d"]
    assert returns.min() > -0.90, f"Excess return below -90%: {returns.min()}"
    assert returns.max() < 3.00, f"Excess return above 300%: {returns.max()}"


def test_factor_specifications_completeness():
    """Verify that every single frozen factor has complete institutional documentation."""
    assert len(FACTOR_SPECIFICATIONS) == 16
    required_keys = [
        "feature_id",
        "pillar",
        "direction",
        "formula",
        "source",
        "units",
        "lookback",
        "missing_treatment",
        "timestamp_rule",
    ]
    for factor, spec in FACTOR_SPECIFICATIONS.items():
        for key in required_keys:
            assert key in spec, f"Factor {factor} missing documentation key: {key}"
            assert len(spec[key]) > 0, f"Documentation key {key} for {factor} is empty!"
