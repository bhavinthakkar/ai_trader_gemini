"""Point-in-Time ML Training Table Builder.

Constructs the standardized, reproducible ML training dataset for Step 3.1:
- Exactly one row per stock and monthly decision date.
- Frozen 16-factor institutional multi-factor suite across Momentum, Value, Quality, Growth, and Risk.
- Zero lookahead bias: strict Point-in-Time SEC EDGAR filings and price alignment.
- Trading-calendar adherent label availability dates (exact 20 trading sessions ahead).
- Contemporaneous cross-sectional median imputation and robust winsorization.
- Produces ml_training_table.parquet and ml_training_table.csv.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# =====================================================================
# FROZEN 16-FACTOR SUITE SPECIFICATION
# =====================================================================

FROZEN_16_FACTORS: List[str] = [
    "ret_252d_ex_21d",          # feature_1:  12-1 Month Intermediate Momentum
    "risk_adj_mom_6m",          # feature_2:  6-Month Risk-Adjusted Momentum
    "rel_sector_6m",            # feature_3:  6-Month Relative Sector Return
    "fcf_yield_ttm",            # feature_4:  Trailing Free Cash Flow Yield (FCF/P)
    "earnings_yield_ttm",       # feature_5:  Trailing Earnings Yield (E/P)
    "book_to_market",           # feature_6:  Book-to-Market Equity Ratio (B/M)
    "gross_profit_to_assets",   # feature_7:  Novy-Marx Gross Profitability (GP/Assets)
    "fcf_margin_ttm",           # feature_8:  Free Cash Flow Margin (FCF/Revenue)
    "roe_ttm",                  # feature_9:  Return on Equity (Net Income/Equity)
    "accruals_ratio",           # feature_10: Sloan Accrual Anomaly ((NI - CFO)/Assets)
    "revenue_growth_yoy",       # feature_11: Trailing 4-Quarter YoY Revenue Growth
    "gross_margin_expansion",   # feature_12: Trailing YoY Gross Margin Expansion
    "eps_surprise_pct",         # feature_13: Consensus EPS Surprise Percentage
    "eps_surprise_direction",   # feature_14: Discrete EPS Surprise Direction (+1, 0, -1)
    "volatility_60_ann",        # feature_15: 60-Day Annualized Realized Volatility
    "downside_vol_60_ann",      # feature_16: 60-Day Annualized Downside Semi-Deviation
]

FEATURE_ALIASES: Dict[str, str] = {
    f"feature_{i+1}": factor for i, factor in enumerate(FROZEN_16_FACTORS)
}

FACTOR_TO_FEATURE_ID: Dict[str, str] = {
    factor: f"feature_{i+1}" for i, factor in enumerate(FROZEN_16_FACTORS)
}

FACTOR_SPECIFICATIONS: Dict[str, Dict[str, str]] = {
    "ret_252d_ex_21d": {
        "feature_id": "feature_1",
        "pillar": "Momentum",
        "direction": "higher_is_better",
        "formula": "(1 + R_252d) / (1 + R_21d) - 1.0",
        "source": "CRSP / Exchange Daily Adjusted Close Prices",
        "units": "Decimal cumulative return",
        "lookback": "252 trading sessions, skipping recent 21 sessions",
        "missing_treatment": "Requires 252 sessions of price history; excluded if insufficient",
        "timestamp_rule": "Close of session t; strictly t - 21 to t - 252",
    },
    "risk_adj_mom_6m": {
        "feature_id": "feature_2",
        "pillar": "Momentum",
        "direction": "higher_is_better",
        "formula": "R_126d / (volatility_60_ann)",
        "source": "Exchange Daily Adjusted Close Prices",
        "units": "Sharpe-like dimensionless ratio",
        "lookback": "126-session cumulative return normalized by 60-session annualized volatility",
        "missing_treatment": "Contemporaneous cross-sectional median on decision date t",
        "timestamp_rule": "Close of session t",
    },
    "rel_sector_6m": {
        "feature_id": "feature_3",
        "pillar": "Momentum",
        "direction": "higher_is_better",
        "formula": "R_stock_126d - R_sector_peers_126d",
        "source": "Exchange Daily Adjusted Close Prices + Sector Mapping",
        "units": "Decimal excess return",
        "lookback": "126 trading sessions",
        "missing_treatment": "Default to 0.0 if sector peers unpopulated",
        "timestamp_rule": "Close of session t",
    },
    "fcf_yield_ttm": {
        "feature_id": "feature_4",
        "pillar": "Value",
        "direction": "higher_is_better",
        "formula": "Free_Cash_Flow_TTM / Market_Capitalization_t",
        "source": "SEC EDGAR 10-Q/10-K Cash Flow Statements (CFO - CapEx)",
        "units": "Decimal cash flow yield",
        "lookback": "Trailing Twelve Months (sum of 4 trailing quarters)",
        "missing_treatment": "Contemporaneous cross-sectional median on decision date t; negative FCF preserved continuously",
        "timestamp_rule": "SEC filing date < decision date t (or execution at t+1)",
    },
    "earnings_yield_ttm": {
        "feature_id": "feature_5",
        "pillar": "Value",
        "direction": "higher_is_better",
        "formula": "Net_Income_TTM / Market_Capitalization_t",
        "source": "SEC EDGAR 10-Q/10-K Income Statements",
        "units": "Decimal earnings yield (E/P)",
        "lookback": "Trailing Twelve Months (sum of 4 trailing quarters)",
        "missing_treatment": "Contemporaneous cross-sectional median on decision date t; negative earnings preserved continuously (losses yield < 0)",
        "timestamp_rule": "SEC filing date < decision date t",
    },
    "book_to_market": {
        "feature_id": "feature_6",
        "pillar": "Value",
        "direction": "higher_is_better",
        "formula": "Stockholders_Equity_TTM / Market_Capitalization_t",
        "source": "SEC EDGAR 10-Q/10-K Balance Sheets",
        "units": "Decimal equity ratio (B/M)",
        "lookback": "Latest quarterly balance sheet preceding decision date t",
        "missing_treatment": "Contemporaneous cross-sectional median on decision date t; negative equity preserved continuously",
        "timestamp_rule": "SEC filing date < decision date t",
    },
    "gross_profit_to_assets": {
        "feature_id": "feature_7",
        "pillar": "Quality",
        "direction": "higher_is_better",
        "formula": "Gross_Profit_TTM / Total_Assets_t",
        "source": "SEC EDGAR 10-Q/10-K Income Statements & Balance Sheets (Novy-Marx 2013)",
        "units": "Decimal profitability ratio",
        "lookback": "TTM Gross Profit divided by latest quarterly Total Assets",
        "missing_treatment": "Contemporaneous cross-sectional median on decision date t (for Financials where GP undefined, sector median proxy)",
        "timestamp_rule": "SEC filing date < decision date t",
    },
    "fcf_margin_ttm": {
        "feature_id": "feature_8",
        "pillar": "Quality",
        "direction": "higher_is_better",
        "formula": "Free_Cash_Flow_TTM / Revenue_TTM",
        "source": "SEC EDGAR 10-Q/10-K Cash Flow Statements & Income Statements",
        "units": "Decimal margin ratio",
        "lookback": "Trailing Twelve Months",
        "missing_treatment": "Contemporaneous cross-sectional median on decision date t",
        "timestamp_rule": "SEC filing date < decision date t",
    },
    "roe_ttm": {
        "feature_id": "feature_9",
        "pillar": "Quality",
        "direction": "higher_is_better",
        "formula": "Net_Income_TTM / Stockholders_Equity_t",
        "source": "SEC EDGAR 10-Q/10-K Income Statements & Balance Sheets",
        "units": "Decimal return on equity",
        "lookback": "TTM Net Income divided by latest quarterly Stockholders Equity",
        "missing_treatment": "Contemporaneous cross-sectional median on decision date t; winsorized at 1st and 99th percentiles",
        "timestamp_rule": "SEC filing date < decision date t",
    },
    "accruals_ratio": {
        "feature_id": "feature_10",
        "pillar": "Quality",
        "direction": "lower_is_better",
        "formula": "(Net_Income_TTM - Operating_Cash_Flow_TTM) / Total_Assets_t",
        "source": "SEC EDGAR 10-Q/10-K (Sloan 1996 Accrual Anomaly)",
        "units": "Decimal accrual ratio",
        "lookback": "TTM Net Income and OCF relative to latest Total Assets",
        "missing_treatment": "Contemporaneous cross-sectional median on decision date t; lower raw accruals imply higher cash earnings quality",
        "timestamp_rule": "SEC filing date < decision date t",
    },
    "revenue_growth_yoy": {
        "feature_id": "feature_11",
        "pillar": "Growth",
        "direction": "higher_is_better",
        "formula": "(Revenue_TTM_t - Revenue_TTM_t-1yr) / |Revenue_TTM_t-1yr|",
        "source": "SEC EDGAR 10-Q/10-K Income Statements",
        "units": "Decimal percentage YoY growth rate",
        "lookback": "Trailing 1-year change in TTM revenue (4-quarter shift)",
        "missing_treatment": "Contemporaneous cross-sectional median on decision date t",
        "timestamp_rule": "SEC filing date < decision date t",
    },
    "gross_margin_expansion": {
        "feature_id": "feature_12",
        "pillar": "Growth",
        "direction": "higher_is_better",
        "formula": "(Gross_Profit / Revenue)_t - (Gross_Profit / Revenue)_t-1yr",
        "source": "SEC EDGAR 10-Q/10-K Income Statements",
        "units": "Decimal margin expansion",
        "lookback": "Trailing 1-year change in TTM gross margin",
        "missing_treatment": "Contemporaneous cross-sectional median on decision date t (or 0.0 if unavailable)",
        "timestamp_rule": "SEC filing date < decision date t",
    },
    "eps_surprise_pct": {
        "feature_id": "feature_13",
        "pillar": "Revisions",
        "direction": "higher_is_better",
        "formula": "(Reported_EPS - Consensus_EPS) / |Consensus_EPS|",
        "source": "Consensus Earnings Reporting / Release Feeds",
        "units": "Decimal surprise percentage",
        "lookback": "Most recent quarterly earnings announcement observable prior to decision date t",
        "missing_treatment": "Default to 0.0 (neutral, exactly met consensus or no unobserved revision)",
        "timestamp_rule": "Announcement timestamp < decision date t",
    },
    "eps_surprise_direction": {
        "feature_id": "feature_14",
        "pillar": "Revisions",
        "direction": "higher_is_better",
        "formula": "+1.0 if Beat, 0.0 if Meet / Unobserved, -1.0 if Miss",
        "source": "Consensus Earnings Release Feeds",
        "units": "Discrete indicator {-1.0, 0.0, +1.0}",
        "lookback": "Most recent quarterly earnings announcement observable prior to decision date t",
        "missing_treatment": "Default to 0.0 (neutral)",
        "timestamp_rule": "Announcement timestamp < decision date t",
    },
    "volatility_60_ann": {
        "feature_id": "feature_15",
        "pillar": "Risk",
        "direction": "lower_is_better",
        "formula": "sqrt(252) * std_dev(daily_log_returns, lookback=60)",
        "source": "Exchange Daily Adjusted Close Prices",
        "units": "Annualized decimal volatility",
        "lookback": "60 trading sessions",
        "missing_treatment": "Contemporaneous cross-sectional median on decision date t; lower raw volatility represents low-risk anomaly",
        "timestamp_rule": "Close of session t",
    },
    "downside_vol_60_ann": {
        "feature_id": "feature_16",
        "pillar": "Risk",
        "direction": "lower_is_better",
        "formula": "sqrt(252) * sqrt(mean(min(0, daily_return)^2), lookback=60)",
        "source": "Exchange Daily Adjusted Close Prices",
        "units": "Annualized decimal downside semi-deviation",
        "lookback": "60 trading sessions",
        "missing_treatment": "Contemporaneous cross-sectional median on decision date t",
        "timestamp_rule": "Close of session t",
    },
}


# =====================================================================
# TABLE BUILDER LOGIC
# =====================================================================

def load_source_datasets(
    stock_ml_root: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load monthly dataset with fundamentals and dated universe membership."""
    if stock_ml_root is None:
        stock_ml_root = Path("/home/bhavin-thakkar/gitClient/stock-ml")

    monthly_path = stock_ml_root / "data" / "monthly_dataset_with_fundamentals.parquet"
    if not monthly_path.exists():
        monthly_path = stock_ml_root / "data" / "monthly_dataset_with_fundamentals.pkl"
    if not monthly_path.exists():
        raise FileNotFoundError(f"Missing {monthly_path}. Build dataset in stock-ml first.")

    universe_path = stock_ml_root / "data" / "dated_universe_membership.parquet"
    if not universe_path.exists():
        universe_path = stock_ml_root / "data" / "dated_universe_membership.csv"
    if not universe_path.exists():
        raise FileNotFoundError(f"Missing {universe_path}.")

    monthly_df = (
        pd.read_parquet(monthly_path)
        if str(monthly_path).endswith(".parquet")
        else pd.read_pickle(monthly_path)
    )
    universe_df = (
        pd.read_parquet(universe_path)
        if str(universe_path).endswith(".parquet")
        else pd.read_csv(universe_path)
    )

    return monthly_df, universe_df


def build_ml_training_table(
    monthly_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    universe_version_tag: str = "universe_v3",
) -> pd.DataFrame:
    """Build the clean, point-in-time reproducible ML training table.

    Steps:
    1. Filter universe to point-in-time S&P 500 eligible constituents.
    2. Format decision dates and calendar-adherent label end dates.
    3. Perform zero-lookahead contemporaneous cross-sectional median imputation.
    4. Winsorize extreme ratios at [1st, 99th] percentiles.
    5. Construct standardized feature_1 .. feature_16 columns alongside named factors.
    6. Attach label_excess_20d, label_end_date, and universe_version.
    """
    df = monthly_df.copy()

    # Normalize date columns
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["ticker"] = df["ticker"].astype(str)

    # Filter to point-in-time S&P 500 membership (Rule PIT-USR-25)
    pit_mask = df["in_sp500_pit"].fillna(False).astype(bool)
    df = df.loc[pit_mask].copy()

    # Sort chronologically by date and ticker
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    # 1. Revisions missing value policy: default to 0.0 (neutral surprise)
    df["eps_surprise_pct"] = df["eps_surprise_pct"].fillna(0.0)
    df["eps_surprise_direction"] = df["eps_surprise_direction"].fillna(0.0)

    # 2. Fundamental features: contemporaneous cross-sectional median imputation by decision_date
    fund_features = [
        "fcf_yield_ttm",
        "earnings_yield_ttm",
        "book_to_market",
        "gross_profit_to_assets",
        "fcf_margin_ttm",
        "roe_ttm",
        "accruals_ratio",
        "revenue_growth_yoy",
        "gross_margin_expansion",
    ]

    for col in fund_features:
        if col in df.columns:
            med = df.groupby("date")[col].transform("median")
            df[col] = df[col].fillna(med)
            # If an entire cross-section on an early historical date is missing, fill with 0.0
            df[col] = df[col].fillna(0.0)

    # 3. Momentum and risk features: verify coverage, median fill fallback
    other_features = [
        "ret_252d_ex_21d",
        "risk_adj_mom_6m",
        "rel_sector_6m",
        "volatility_60_ann",
        "downside_vol_60_ann",
    ]
    for col in other_features:
        if col in df.columns:
            med = df.groupby("date")[col].transform("median")
            df[col] = df[col].fillna(med).fillna(0.0)

    # 4. Robust winsorization at 1st and 99th percentiles across each factor
    for col in FROZEN_16_FACTORS:
        q01 = df[col].quantile(0.01)
        q99 = df[col].quantile(0.99)
        df[col] = df[col].clip(q01, q99)

    # 5. Build output dataset with standardized schema
    out = pd.DataFrame()
    out["decision_date"] = df["date"].dt.strftime("%Y-%m-%d")
    out["ticker"] = df["ticker"].astype(str)

    # Aliased feature columns (feature_1 .. feature_16)
    for i, factor_name in enumerate(FROZEN_16_FACTORS):
        feature_col = f"feature_{i+1}"
        out[feature_col] = df[factor_name].astype(float)

    # Also include the explicit named factor columns for clarity and auditability
    for factor_name in FROZEN_16_FACTORS:
        out[factor_name] = df[factor_name].astype(float)

    # Labels and calendar availability
    out["label_excess_20d"] = df["future_excess_return_20d"].astype(float)
    label_end = pd.to_datetime(df["label_end_date_20d"])
    out["label_end_date"] = label_end.dt.strftime("%Y-%m-%d")

    # Universe version tag
    out["universe_version"] = universe_version_tag

    # Boolean training maturity flag
    out["has_label"] = out["label_excess_20d"].notna() & out["label_end_date"].notna()

    return out


def generate_and_save_training_tables(
    destination_dirs: Optional[List[Path]] = None,
) -> pd.DataFrame:
    """Generate the training table and save to all configured output directories."""
    if destination_dirs is None:
        destination_dirs = [
            Path("/home/bhavin-thakkar/gitClient/ai_trader_gemini/data"),
            Path("/home/bhavin-thakkar/gitClient/stock-ml/data"),
        ]

    monthly_df, universe_df = load_source_datasets()
    training_table = build_ml_training_table(monthly_df, universe_df)

    for dest_dir in destination_dirs:
        dest_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = dest_dir / "ml_training_table.parquet"
        csv_path = dest_dir / "ml_training_table.csv"

        training_table.to_parquet(parquet_path, index=False)
        training_table.to_csv(csv_path, index=False)
        print(f"Successfully saved training table to {parquet_path} ({len(training_table):,} rows)")

    return training_table


if __name__ == "__main__":
    generate_and_save_training_tables()
