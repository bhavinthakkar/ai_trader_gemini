"""Score Threshold Empirical Calibration and Walk-Forward Backtesting Harness.

Empirically calibrates BUY and Dip-Buy thresholds against actual hybrid score distributions
across historical walk-forward dates (2022 to 2026).

Compares:
1. Current baseline: BUY 60.0, Dip-buy 52.0
2. Test A:           BUY 58.0, Dip-buy 52.0
3. Test B:           BUY 56.0, Dip-buy 50.0
4. Test C:           BUY 54.0, Dip-buy 48.0

Evaluates across:
- Full Walk-Forward Horizon (2022 to 2026, 56 monthly rebalances)
- Development / Tuning Window (2022 to 2024, 36 monthly rebalances)
- Untuned Final Holdout Window (2025 to 2026, 20 monthly rebalances)
- Annual test windows (2022, 2023, 2024, 2025, 2026)

Tracks:
- Number of BUY candidates (cross-sectional qualifying count)
- Executed simulated entries (portfolio additions)
- Net return (CAGR, cumulative return, excess return vs SPY)
- Maximum drawdown (MDD)
- Sharpe, Sortino, Calmar ratios
- Turnover and transaction costs (10 bps one-way / 20 bps round-trip)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from quant.experiment_harness import load_and_prepare_experiment_data


DATA_DIR = Path("data")


def get_default_monthly_path() -> Path:
    """Finds available monthly dataset path with fundamentals."""
    candidates = [
        Path("/home/bhavin/Desktop/git/stock_ml/data/monthly_dataset_with_fundamentals.parquet"),
        Path("/home/bhavin-thakkar/gitClient/stock-ml/data/monthly_dataset_with_fundamentals.parquet"),
        DATA_DIR / "monthly_dataset_with_fundamentals.parquet",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return candidates[0]


def load_calibration_dataset(
    monthly_path: Optional[Path] = None,
    predictions_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Loads walk-forward predictions, calculates quant and hybrid scores, and attaches dip-buy indicators."""
    m_path = monthly_path or get_default_monthly_path()
    df = load_and_prepare_experiment_data(predictions_path=predictions_path, monthly_path=m_path)

    df_m = pd.read_parquet(m_path)
    df_m["decision_date"] = pd.to_datetime(df_m["date"]).dt.strftime("%Y-%m-%d")
    df_m["ticker"] = df_m["ticker"].astype(str)

    # Attach ret_5d for structural dip-buy drawdown evaluation
    df = pd.merge(df, df_m[["decision_date", "ticker", "ret_5d"]], on=["decision_date", "ticker"], how="left")

    # Structural Dip-Buy Eligibility Conditions:
    # 1. 5-day price drop between 5% and 15% (ret_5d in [-0.15, -0.05])
    # 2. RSI_14 in [30.0, 45.0] (oversold, not collapsed)
    # 3. Valuation history score >= 60.0 (historically undervalued)
    # 4. Solvency / Non-value-trap check (accruals <= 0.10 and roe > 0)
    drop_ok = (-0.15 <= df["ret_5d"]) & (df["ret_5d"] <= -0.05)
    rsi_ok = (30.0 <= df["rsi14"]) & (df["rsi14"] <= 45.0)
    val_ok = df["val_score"] >= 60.0
    trap_free = (df["accruals_ratio"] <= 0.10) & (df["roe_ttm"] > 0)
    df["is_dip_setup"] = drop_ok & rsi_ok & val_ok & trap_free

    return df


def compute_score_distribution_statistics(df: pd.DataFrame, score_col: str = "score_C") -> Dict[str, Any]:
    """Measures the distribution of the final hybrid score across historical dates."""
    s = df[score_col].dropna()
    percentiles = {
        "p01": round(float(s.quantile(0.01)), 2),
        "p05": round(float(s.quantile(0.05)), 2),
        "p10": round(float(s.quantile(0.10)), 2),
        "p25": round(float(s.quantile(0.25)), 2),
        "p50_median": round(float(s.median()), 2),
        "p75": round(float(s.quantile(0.75)), 2),
        "p90": round(float(s.quantile(0.90)), 2),
        "p95": round(float(s.quantile(0.95)), 2),
        "p99": round(float(s.quantile(0.99)), 2),
    }

    threshold_bins = [65.0, 60.0, 58.0, 56.0, 54.0, 52.0, 50.0, 48.0, 45.0]
    threshold_coverage = {}
    for th in threshold_bins:
        cnt = int((s >= th).sum())
        threshold_coverage[f">={th:.0f}"] = {
            "count": cnt,
            "pct": round(cnt / len(s) * 100.0, 2),
        }

    # Breadth per date (how many stocks >= threshold on each rebalance date)
    date_breadth = {}
    total_dates = int(df["decision_date"].nunique())
    for th in [60.0, 58.0, 56.0, 54.0, 52.0]:
        counts_per_date = df[df[score_col] >= th].groupby("decision_date")["ticker"].count()
        dates_with_0 = total_dates - len(counts_per_date)
        dates_ge1 = int((counts_per_date >= 1).sum())
        dates_ge2 = int((counts_per_date >= 2).sum())
        dates_ge3 = int((counts_per_date >= 3).sum())
        date_breadth[f">={th:.0f}"] = {
            "dates_with_zero_candidates": dates_with_0,
            "pct_dates_with_zero": round(dates_with_0 / total_dates * 100.0, 1),
            "dates_with_ge_1": dates_ge1,
            "dates_with_ge_2": dates_ge2,
            "dates_with_ge_3": dates_ge3,
        }

    return {
        "score_column": score_col,
        "n_observations": int(len(s)),
        "n_dates": total_dates,
        "mean": round(float(s.mean()), 2),
        "std": round(float(s.std()), 2),
        "min": round(float(s.min()), 2),
        "max": round(float(s.max()), 2),
        "skewness": round(float(s.skew()), 4),
        "kurtosis": round(float(s.kurt()), 4),
        "percentiles": percentiles,
        "threshold_coverage": threshold_coverage,
        "date_breadth": date_breadth,
    }


def simulate_threshold_portfolio(
    df: pd.DataFrame,
    score_col: str,
    buy_threshold: float,
    dip_buy_threshold: float,
    top_k: int = 2,
    one_way_cost: float = 0.0010,
    allocation_mode: str = "fixed_slot",  # 'fixed_slot' (50% per slot, unfilled is cash) or 'dynamic' (100% if 1 qualifies)
) -> Dict[str, Any]:
    """
    Simulates walk-forward portfolio rebalancing under specific BUY and Dip-Buy thresholds.
    """
    is_buy = (df[score_col] >= buy_threshold) | (df["is_dip_setup"] & (df[score_col] >= dip_buy_threshold))
    total_candidates = int(is_buy.sum())

    dates = sorted(df["decision_date"].unique())
    prev_holdings: set = set()
    records: List[Dict[str, Any]] = []
    total_new_entries = 0

    for d in dates:
        grp = df[df["decision_date"] == d]
        qual = grp[is_buy.loc[grp.index]].sort_values(score_col, ascending=False)
        top = qual.head(top_k)
        new_holdings = set(top["ticker"])

        # Count new executed simulated entries
        new_buys = len(new_holdings - prev_holdings)
        total_new_entries += new_buys

        # Turnover and friction
        if not prev_holdings and not new_holdings:
            turnover = 0.0
            cost = 0.0
        elif not prev_holdings:
            turnover = len(new_holdings) / top_k
            cost = turnover * one_way_cost
        else:
            changed = len(prev_holdings - new_holdings) + len(new_holdings - prev_holdings)
            turnover = changed / (2.0 * top_k)
            cost = turnover * (2.0 * one_way_cost)

        # Gross return calculation
        if len(top) == 0:
            gross_ret = 0.0
        elif len(top) == 1:
            if allocation_mode == "dynamic":
                gross_ret = float(top["future_return_20d"].iloc[0])
            else:
                # 50% in stock, 50% cash
                gross_ret = 0.5 * float(top["future_return_20d"].iloc[0])
        else:
            gross_ret = float(top["future_return_20d"].mean())

        spy_ret = float(grp["spy_future_return_20d"].iloc[0])
        net_ret = gross_ret - cost
        excess_ret = net_ret - spy_ret

        records.append({
            "decision_date": d,
            "year": int(d[:4]),
            "n_qualifying": len(qual),
            "n_held": len(new_holdings),
            "new_entries": new_buys,
            "gross_return": gross_ret,
            "cost": cost,
            "net_return": net_ret,
            "spy_return": spy_ret,
            "excess_return": excess_ret,
            "turnover": turnover,
        })
        prev_holdings = new_holdings

    res_df = pd.DataFrame(records)
    n_years = len(res_df) / 12.0
    cum_prod = (1.0 + res_df["net_return"]).cumprod()
    spy_cum = (1.0 + res_df["spy_return"]).cumprod()

    cagr = float(cum_prod.iloc[-1] ** (1.0 / n_years) - 1.0) if n_years > 0 else 0.0
    cum_net = float(cum_prod.iloc[-1] - 1.0)
    spy_cagr = float(spy_cum.iloc[-1] ** (1.0 / n_years) - 1.0) if n_years > 0 else 0.0

    drawdowns = cum_prod / cum_prod.cummax() - 1.0
    mdd = float(drawdowns.min()) if len(drawdowns) > 0 else 0.0

    ann_vol = float(res_df["net_return"].std() * np.sqrt(12)) if len(res_df) > 1 else 0.0
    sharpe = float((res_df["net_return"].mean() * 12.0) / (ann_vol + 1e-8))

    downside = res_df["net_return"][res_df["net_return"] < 0]
    downside_vol = float(downside.std() * np.sqrt(12)) if len(downside) > 1 else ann_vol
    sortino = float((res_df["net_return"].mean() * 12.0) / (downside_vol + 1e-8))

    calmar = float(cagr / abs(mdd)) if mdd < 0 else 0.0
    mean_turnover = float(res_df["turnover"].mean())
    ann_excess = float(res_df["excess_return"].mean() * 12.0)

    # Window breakdown
    window_stats = {}
    for yr, ydf in res_df.groupby("year"):
        y_cum = (1.0 + ydf["net_return"]).prod() - 1.0
        y_spy = (1.0 + ydf["spy_return"]).prod() - 1.0
        window_stats[str(yr)] = {
            "net_return": round(float(y_cum) * 100.0, 2),
            "spy_return": round(float(y_spy) * 100.0, 2),
            "excess_return": round(float(y_cum - y_spy) * 100.0, 2),
            "avg_qualifying_candidates": round(float(ydf["n_qualifying"].mean()), 1),
            "executed_entries": int(ydf["new_entries"].sum()),
            "turnover": round(float(ydf["turnover"].mean()) * 100.0, 1),
        }

    return {
        "buy_threshold": buy_threshold,
        "dip_buy_threshold": dip_buy_threshold,
        "allocation_mode": allocation_mode,
        "n_candidates": total_candidates,
        "executed_entries": total_new_entries,
        "cagr": round(cagr * 100.0, 2),
        "cum_net_return": round(cum_net * 100.0, 2),
        "spy_cagr": round(spy_cagr * 100.0, 2),
        "annualized_excess": round(ann_excess * 100.0, 2),
        "max_drawdown": round(mdd * 100.0, 2),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "calmar_ratio": round(calmar, 2),
        "mean_turnover": round(mean_turnover * 100.0, 1),
        "window_breakdown": window_stats,
        "history_df": res_df,
    }


def run_full_calibration_suite(
    score_col: str = "score_C",
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Executes the full experimental calibration across all 4 configurations and windows:
    - Current Baseline (BUY 60, Dip 52)
    - Test A (BUY 58, Dip 52)
    - Test B (BUY 56, Dip 50)
    - Test C (BUY 54, Dip 48)
    """
    out_dir = output_dir or DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_calibration_dataset()
    dist_stats = compute_score_distribution_statistics(df, score_col=score_col)

    configs = [
        ("Current baseline", 60.0, 52.0),
        ("Test A", 58.0, 52.0),
        ("Test B", 56.0, 50.0),
        ("Test C", 54.0, 48.0),
    ]

    # Partition into Development (2022-2024) and Holdout (2025-2026)
    dev_df = df[df["decision_date"] < "2025-01-01"].copy()
    holdout_df = df[df["decision_date"] >= "2025-01-01"].copy()

    summary_records = []
    window_records = []
    detailed_results = {
        "distribution_statistics": dist_stats,
        "full_horizon": {},
        "development_window_2022_2024": {},
        "untuned_final_holdout_2025_2026": {},
    }

    for name, buy_th, dip_th in configs:
        # 1. Full walk-forward
        full_res = simulate_threshold_portfolio(df, score_col, buy_th, dip_th)
        # 2. Development window (2022-2024, 36 months)
        dev_res = simulate_threshold_portfolio(dev_df, score_col, buy_th, dip_th)
        # 3. Final untuned holdout (2025-2026, 20 months)
        holdout_res = simulate_threshold_portfolio(holdout_df, score_col, buy_th, dip_th)

        # Store detailed results
        detailed_results["full_horizon"][name] = {k: v for k, v in full_res.items() if k != "history_df"}
        detailed_results["development_window_2022_2024"][name] = {k: v for k, v in dev_res.items() if k != "history_df"}
        detailed_results["untuned_final_holdout_2025_2026"][name] = {k: v for k, v in holdout_res.items() if k != "history_df"}

        summary_records.append({
            "Configuration": name,
            "BUY Threshold": buy_th,
            "Dip-Buy Threshold": dip_th,
            "Full Candidates": full_res["n_candidates"],
            "Full Entries": full_res["executed_entries"],
            "Full Net CAGR (%)": full_res["cagr"],
            "Full MDD (%)": full_res["max_drawdown"],
            "Full Sharpe": full_res["sharpe_ratio"],
            "Full Turnover (%)": full_res["mean_turnover"],
            "Dev Net CAGR (%) [2022-24]": dev_res["cagr"],
            "Dev MDD (%) [2022-24]": dev_res["max_drawdown"],
            "Dev Sharpe [2022-24]": dev_res["sharpe_ratio"],
            "Holdout Net CAGR (%) [2025-26]": holdout_res["cagr"],
            "Holdout MDD (%) [2025-26]": holdout_res["max_drawdown"],
            "Holdout Sharpe [2025-26]": holdout_res["sharpe_ratio"],
        })

        for yr_str, y_data in full_res["window_breakdown"].items():
            window_records.append({
                "Configuration": name,
                "Year": yr_str,
                "Net Return (%)": y_data["net_return"],
                "SPY Return (%)": y_data["spy_return"],
                "Excess Return (%)": y_data["excess_return"],
                "Avg Candidates": y_data["avg_qualifying_candidates"],
                "Executed Entries": y_data["executed_entries"],
                "Turnover (%)": y_data["turnover"],
            })

    # Save reports
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(out_dir / "score_threshold_calibration_summary.csv", index=False)

    window_df = pd.DataFrame(window_records)
    window_df.to_csv(out_dir / "score_threshold_calibration_by_window.csv", index=False)

    with open(out_dir / "score_threshold_calibration_distribution.json", "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2)

    return detailed_results


if __name__ == "__main__":
    res = run_full_calibration_suite()
    print("Calibration Suite successfully generated.")
    print("Summary:")
    summary_df = pd.read_csv(DATA_DIR / "score_threshold_calibration_summary.csv")
    print(summary_df.to_string(index=False))
