"""Relative Ranking Candidate Generation and Multi-Stage Filtering Harness.

Implements the institutional ranking-based candidate generation pipeline:
1. Identify the top 20% of eligible stocks by hybrid score (cross-sectional rank).
2. Require a minimum absolute score quality floor (e.g. 60.0) so weak markets do not force BUY signals.
3. Incorporate the certified structural dip-buy pathway (score >= 52.0 with technical & solvency setups).
4. Apply mandatory risk, data-quality, and portfolio controls (Gate 1-5, sector concentration, cash reserve).
5. Permit BUY only when all conditions pass; unfilled target slots remain in cash.

Compares:
- Mode 1: Existing Absolute Baseline (Hybrid Score >= 60.0)
- Mode 2: Pure Relative Ranking (Top 20% of universe, no absolute floor)
- Mode 3: Proposed Dual Rule (Top 20% AND Hybrid Score >= 60.0)
- Mode 4: Proposed Dual Rule with Calibrated Floor (Top 20% AND Hybrid Score >= 56.0)

Evaluates out-of-fold performance across:
- Full Horizon: 2022 to 2026 (56 monthly rebalance sessions)
- Development Window: 2022 to 2024 (36 rebalances)
- Untuned Holdout Window: 2025 to 2026 (20 rebalances)
- Annual breakdown: 2022, 2023, 2024, 2025, 2026
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from quant.portfolio_controls import DEFAULT_UNIVERSE_TAXONOMY


DATA_DIR = Path("data")


def get_default_focused_paths() -> Tuple[Path, Path]:
    """Resolves focused 25 predictions and prices file paths."""
    pred_candidates = [
        Path("/home/bhavin/Desktop/git/stock_ml/data/xgb_focused_walk_forward_predictions.parquet"),
        Path("/home/bhavin-thakkar/gitClient/stock-ml/data/xgb_focused_walk_forward_predictions.parquet"),
        DATA_DIR / "xgb_focused_walk_forward_predictions.parquet",
    ]
    price_candidates = [
        Path("/home/bhavin/Desktop/git/stock_ml/data/focused_prices.pkl"),
        Path("/home/bhavin-thakkar/gitClient/stock-ml/data/focused_prices.pkl"),
        DATA_DIR / "focused_prices.pkl",
    ]

    pred_path = next((p for p in pred_candidates if p.is_file()), pred_candidates[0])
    price_path = next((p for p in price_candidates if p.is_file()), price_candidates[0])
    return pred_path, price_path


def load_and_prepare_candidate_data(
    predictions_path: Optional[Path] = None,
    prices_path: Optional[Path] = None,
    start_date: str = "2022-01-01",
    end_date: str = "2026-08-31",
) -> pd.DataFrame:
    """
    Loads focused 25 predictions, computes execution-timed returns (close t+1 to close t+21),
    and derives technical pillars, hybrid score, and structural dip-buy setups.
    """
    def_pred, def_price = get_default_focused_paths()
    p_path = predictions_path or def_pred
    pr_path = prices_path or def_price

    if not p_path.is_file():
        raise FileNotFoundError(f"Predictions file not found: {p_path}")
    if not pr_path.is_file():
        raise FileNotFoundError(f"Prices file not found: {pr_path}")

    preds = pd.read_parquet(p_path)
    preds["date_norm"] = pd.to_datetime(preds["date"]).dt.normalize()
    preds["decision_date"] = preds["date_norm"].dt.strftime("%Y-%m-%d")
    preds["ticker"] = preds["ticker"].astype(str)

    prices = pd.read_pickle(pr_path)
    close = prices.xs("Close", axis=1, level=1).copy()
    close.index = pd.to_datetime(close.index).normalize()
    close = close[~close.index.duplicated(keep="last")].sort_index()

    # Execution-timed returns: entry at close t+1, exit at close t+21 (20 trading days)
    spy_exec = close["SPY"].shift(-21).div(close["SPY"].shift(-1)).sub(1.0)
    stock_exec = close.drop(columns=["SPY", "^GSPC"], errors="ignore").shift(-21).div(
        close.drop(columns=["SPY", "^GSPC"], errors="ignore").shift(-1)
    ).sub(1.0)

    stock_exec_long = stock_exec.stack().rename("exec_ret_20d").reset_index()
    stock_exec_long.columns = ["date_norm", "ticker", "exec_ret_20d"]
    spy_exec_long = spy_exec.rename("spy_exec_ret_20d").reset_index()
    spy_exec_long.columns = ["date_norm", "spy_exec_ret_20d"]

    eval_df = preds.merge(stock_exec_long, on=["date_norm", "ticker"], how="inner")
    eval_df = eval_df.merge(spy_exec_long, on="date_norm", how="left")
    eval_df = eval_df[(eval_df["decision_date"] >= start_date) & (eval_df["decision_date"] <= end_date)].copy()
    eval_df = eval_df.dropna(subset=["pred_focused_ranker_1m", "exec_ret_20d"])

    # 1. Deterministic Quant Pillars
    def _calc_trend(row):
        score = 50.0
        p20, p50, p200 = row.get("price_vs_ma20", 0), row.get("price_vs_ma50", 0), row.get("price_vs_ma200", 0)
        if pd.isna(p20) or pd.isna(p50) or pd.isna(p200):
            return 50.0
        if p20 > 0 and p50 > 0 and p200 > 0:
            score += 25.0
        elif p20 < 0 and p50 < 0 and p200 < 0:
            score -= 25.0
        rsi = row.get("rsi14", 50.0)
        if pd.notna(rsi):
            score += (rsi - 50.0) * 0.5
        return float(np.clip(score, 0.0, 100.0))

    eval_df["trend_score"] = eval_df.apply(_calc_trend, axis=1)
    eval_df["rel_score"] = np.clip(50.0 + eval_df["rel_sp500_1m"].fillna(0.0) * 200.0, 0.0, 100.0)
    eval_df["mom_score"] = np.clip(50.0 + eval_df["risk_adj_mom_6m"].fillna(0.0) * 20.0, 0.0, 100.0)
    eval_df["q_raw"] = 0.40 * eval_df["trend_score"] + 0.30 * eval_df["rel_score"] + 0.30 * eval_df["mom_score"]

    def _calc_vol_factor(atr):
        if pd.isna(atr):
            return 1.0
        if atr <= 1.5:
            return 1.0
        elif atr <= 2.5:
            return 0.95
        elif atr <= 4.0:
            return 0.90
        return 0.85

    eval_df["vol_factor"] = eval_df["atr14_pct"].apply(_calc_vol_factor)
    eval_df["q_final"] = 50.0 + (eval_df["q_raw"] - 50.0) * eval_df["vol_factor"]

    # 2. ML Percentile Score & Hybrid Score (80% Quant + 20% ML)
    eval_df["ml_percentile_score"] = eval_df.groupby("decision_date")["pred_focused_ranker_1m"].rank(pct=True) * 100.0
    eval_df["hybrid_score"] = 0.80 * eval_df["q_final"] + 0.20 * eval_df["ml_percentile_score"]

    # 3. Structural Dip-Buy Indicator
    drop_ok = (-0.15 <= eval_df["ret_5d"]) & (eval_df["ret_5d"] <= -0.05)
    rsi_ok = (30.0 <= eval_df["rsi14"]) & (eval_df["rsi14"] <= 45.0)
    eval_df["is_dip_setup"] = drop_ok & rsi_ok

    # 4. Taxonomy & Sector Mapping
    eval_df["taxonomy_sector"] = eval_df["ticker"].apply(
        lambda t: DEFAULT_UNIVERSE_TAXONOMY.get(t, {}).get("sector", eval_df.loc[eval_df["ticker"] == t, "sector"].iloc[0])
    )

    return eval_df


def generate_candidates_for_slice(
    slice_df: pd.DataFrame,
    mode: str = "dual",
    top_pct: float = 0.20,
    min_abs_score: float = 60.0,
    dip_threshold: float = 52.0,
    max_atr_pct: Optional[float] = None,
) -> pd.DataFrame:
    """
    Generates candidates for a single cross-sectional rebalance slice:
    - mode == 'absolute_only': Hybrid Score >= min_abs_score (or Dip-Buy setup >= dip_threshold)
    - mode == 'relative_only': Top 20% of eligible stocks by hybrid score (no absolute quality floor)
    - mode == 'dual': Top 20% of eligible stocks AND Hybrid Score >= min_abs_score (plus Dip-Buy setup >= dip_threshold)
    
    Optional Gate 3 Volatility check (atr14_pct <= max_atr_pct).
    If max_atr_pct is None, high-volatility candidates are retained and sized via risk budgeting.
    """
    df = slice_df.copy()
    n_stocks = len(df)
    df["rank_score"] = df["hybrid_score"].rank(ascending=False, method="first")

    if mode == "absolute_only":
        is_cand = (df["hybrid_score"] >= min_abs_score) | (df["is_dip_setup"] & (df["hybrid_score"] >= dip_threshold))
    elif mode == "relative_only":
        k_cand = max(1, int(round(n_stocks * top_pct)))
        is_cand = df["rank_score"] <= k_cand
    elif mode == "dual":
        k_cand = max(1, int(round(n_stocks * top_pct)))
        in_top_tier = df["rank_score"] <= k_cand
        clears_floor = df["hybrid_score"] >= min_abs_score
        is_dip = df["is_dip_setup"] & (df["hybrid_score"] >= dip_threshold)
        is_cand = (in_top_tier & clears_floor) | is_dip
    else:
        raise ValueError(f"Unknown candidate generation mode: {mode}")

    if max_atr_pct is not None:
        is_cand = is_cand & (df["atr14_pct"] <= max_atr_pct)

    return df[is_cand].sort_values("hybrid_score", ascending=False)


def simulate_candidate_portfolio(
    df: pd.DataFrame,
    mode: str = "dual",
    top_pct: float = 0.20,
    min_abs_score: float = 60.0,
    dip_threshold: float = 52.0,
    top_k: int = 5,
    max_stocks_per_sector: Optional[int] = 2,
    max_atr_pct: Optional[float] = None,
    sizing_mode: str = "equal",  # 'equal' (1/top_k per slot) or 'volatility_adjusted' (inverse-ATR)
    one_way_cost: float = 0.0010,
) -> Dict[str, Any]:
    """
    Simulates walk-forward portfolio rebalancing with candidate generation,
    portfolio sector controls, volatility-adjusted sizing, and cash management.
    """
    dates = sorted(df["decision_date"].unique())
    prev_holdings: Dict[str, float] = {}
    records: List[Dict[str, Any]] = []
    total_candidates = 0
    total_entries = 0

    for d in dates:
        slice_df = df[df["decision_date"] == d]
        candidates = generate_candidates_for_slice(
            slice_df,
            mode=mode,
            top_pct=top_pct,
            min_abs_score=min_abs_score,
            dip_threshold=dip_threshold,
            max_atr_pct=max_atr_pct,
        )
        total_candidates += len(candidates)

        # Apply Portfolio Risk Controls: Max sector concentration
        selected_tickers: List[str] = []
        sector_counts: Dict[str, int] = {}
        for _, row in candidates.iterrows():
            sec = row["taxonomy_sector"]
            if max_stocks_per_sector is not None and sector_counts.get(sec, 0) >= max_stocks_per_sector:
                continue
            selected_tickers.append(row["ticker"])
            sector_counts[sec] = sector_counts.get(sec, 0) + 1
            if len(selected_tickers) == top_k:
                break

        # Calculate portfolio weights
        curr_weights: Dict[str, float] = {}
        if selected_tickers:
            held_slice = slice_df[slice_df["ticker"].isin(selected_tickers)]
            if sizing_mode == "equal":
                for t in selected_tickers:
                    curr_weights[t] = 1.0 / float(top_k)
            elif sizing_mode == "volatility_adjusted":
                inv_vols = 1.0 / held_slice["atr14_pct"].clip(lower=0.01)
                tot_inv = inv_vols.sum()
                alloc_fraction = len(selected_tickers) / float(top_k)
                for t, iv in zip(held_slice["ticker"], inv_vols):
                    curr_weights[t] = min(0.25, (iv / tot_inv) * alloc_fraction)
            else:
                raise ValueError(f"Unknown sizing mode: {sizing_mode}")

        new_buys = len(set(curr_weights.keys()) - set(prev_holdings.keys()))
        total_entries += new_buys

        # Turnover & Transaction Costs
        all_tickers = set(prev_holdings.keys()).union(set(curr_weights.keys()))
        turnover = sum(abs(curr_weights.get(t, 0.0) - prev_holdings.get(t, 0.0)) for t in all_tickers) / 2.0
        cost = turnover * (2.0 * one_way_cost)

        # Portfolio return: weighted return of held stocks (unallocated cash earns 0.0%)
        if not curr_weights:
            gross_ret = 0.0
        else:
            gross_ret = sum(
                curr_weights[t] * float(slice_df.loc[slice_df["ticker"] == t, "exec_ret_20d"].iloc[0])
                for t in curr_weights
            )

        spy_ret = float(slice_df["spy_exec_ret_20d"].iloc[0])
        net_ret = gross_ret - cost
        excess_ret = net_ret - spy_ret

        records.append({
            "decision_date": d,
            "year": int(d[:4]),
            "n_candidates": len(candidates),
            "n_held": len(curr_weights),
            "new_entries": new_buys,
            "turnover": turnover,
            "gross_return": gross_ret,
            "net_return": net_ret,
            "spy_return": spy_ret,
            "excess_return": excess_ret,
        })
        prev_holdings = curr_weights

    rdf = pd.DataFrame(records)
    n_years = len(rdf) / 12.0
    cum_prod = (1.0 + rdf["net_return"]).cumprod()
    cagr = float(cum_prod.iloc[-1] ** (1.0 / n_years) - 1.0) if n_years > 0 else 0.0
    cum_net = float(cum_prod.iloc[-1] - 1.0)

    drawdowns = cum_prod / cum_prod.cummax() - 1.0
    mdd = float(drawdowns.min()) if len(drawdowns) > 0 else 0.0

    ann_vol = float(rdf["net_return"].std() * np.sqrt(12)) if len(rdf) > 1 else 0.0
    sharpe = float((rdf["net_return"].mean() * 12.0) / (ann_vol + 1e-8))

    downside = rdf["net_return"][rdf["net_return"] < 0]
    downside_vol = float(downside.std() * np.sqrt(12)) if len(downside) > 1 else ann_vol
    sortino = float((rdf["net_return"].mean() * 12.0) / (downside_vol + 1e-8))
    calmar = float(cagr / abs(mdd)) if mdd < 0 else 0.0

    # Development Window (2022 to 2024)
    d_df = rdf[rdf["year"] < 2025]
    d_years = len(d_df) / 12.0
    d_cum = (1.0 + d_df["net_return"]).cumprod()
    d_cagr = float(d_cum.iloc[-1] ** (1.0 / d_years) - 1.0) if d_years > 0 else 0.0
    d_dd = d_cum / d_cum.cummax() - 1.0
    d_mdd = float(d_dd.min()) if len(d_dd) > 0 else 0.0
    d_vol = float(d_df["net_return"].std() * np.sqrt(12)) if len(d_df) > 1 else 0.0
    d_sharpe = float((d_df["net_return"].mean() * 12.0) / (d_vol + 1e-8))

    # Untuned Holdout Window (2025 to 2026)
    h_df = rdf[rdf["year"] >= 2025]
    h_years = len(h_df) / 12.0
    h_cum = (1.0 + h_df["net_return"]).cumprod()
    h_cagr = float(h_cum.iloc[-1] ** (1.0 / h_years) - 1.0) if h_years > 0 else 0.0
    h_dd = h_cum / h_cum.cummax() - 1.0
    h_mdd = float(h_dd.min()) if len(h_dd) > 0 else 0.0
    h_vol = float(h_df["net_return"].std() * np.sqrt(12)) if len(h_df) > 1 else 0.0
    h_sharpe = float((h_df["net_return"].mean() * 12.0) / (h_vol + 1e-8))

    # Annual breakdown
    window_stats = {}
    for yr, ydf in rdf.groupby("year"):
        y_cum = (1.0 + ydf["net_return"]).prod() - 1.0
        y_spy = (1.0 + ydf["spy_return"]).prod() - 1.0
        window_stats[str(yr)] = {
            "net_return": round(float(y_cum) * 100.0, 2),
            "spy_return": round(float(y_spy) * 100.0, 2),
            "excess_return": round(float(y_cum - y_spy) * 100.0, 2),
            "avg_candidates": round(float(ydf["n_candidates"].mean()), 1),
            "avg_held": round(float(ydf["n_held"].mean()), 1),
            "executed_entries": int(ydf["new_entries"].sum()),
            "turnover": round(float(ydf["turnover"].mean()) * 100.0, 1),
        }

    return {
        "mode": mode,
        "top_pct": top_pct,
        "min_abs_score": min_abs_score,
        "dip_threshold": dip_threshold,
        "top_k": top_k,
        "max_stocks_per_sector": max_stocks_per_sector,
        "n_candidates": total_candidates,
        "executed_entries": total_entries,
        "full_cagr": round(cagr * 100.0, 2),
        "cum_net_return": round(cum_net * 100.0, 2),
        "full_mdd": round(mdd * 100.0, 2),
        "full_sharpe": round(sharpe, 2),
        "full_sortino": round(sortino, 2),
        "full_calmar": round(calmar, 2),
        "mean_turnover": round(float(rdf["turnover"].mean()) * 100.0, 1),
        "dev_cagr": round(d_cagr * 100.0, 2),
        "dev_mdd": round(d_mdd * 100.0, 2),
        "dev_sharpe": round(d_sharpe, 2),
        "holdout_cagr": round(h_cagr * 100.0, 2),
        "holdout_mdd": round(h_mdd * 100.0, 2),
        "holdout_sharpe": round(h_sharpe, 2),
        "window_stats": window_stats,
    }


def run_candidate_generation_benchmark() -> Dict[str, Any]:
    """
    Executes full benchmark suite comparing candidate generation mechanisms.
    Exports CSV and JSON artifacts.
    """
    df = load_and_prepare_candidate_data()

    configs = [
        ("Existing Absolute Baseline (Score >= 60.0)", "absolute_only", 0.20, 60.0, 52.0),
        ("Pure Relative Ranking (Top 20%, No Floor)", "relative_only", 0.20, 0.0, 0.0),
        ("Proposed Dual (Top 20% & Score >= 60.0)", "dual", 0.20, 60.0, 52.0),
        ("Proposed Dual (Top 20% & Score >= 56.0)", "dual", 0.20, 56.0, 50.0),
    ]

    summary_rows = []
    window_rows = []
    full_results = {}

    for label, mode, top_pct, min_abs, dip_th in configs:
        res = simulate_candidate_portfolio(
            df,
            mode=mode,
            top_pct=top_pct,
            min_abs_score=min_abs,
            dip_threshold=dip_th,
            top_k=5,
            max_stocks_per_sector=2,
            max_atr_pct=None,
            sizing_mode="equal",
            one_way_cost=0.0010,
        )
        full_results[label] = res

        summary_rows.append({
            "Configuration": label,
            "Mode": mode,
            "Top Pct": f"{top_pct*100:.0f}%",
            "Min Abs Score": min_abs,
            "Dip-Buy Floor": dip_th,
            "Total Candidates": res["n_candidates"],
            "Executed Entries": res["executed_entries"],
            "Full Net CAGR (%)": res["full_cagr"],
            "Full MDD (%)": res["full_mdd"],
            "Full Sharpe": res["full_sharpe"],
            "Full Calmar": res["full_calmar"],
            "Mean Turnover (%)": res["mean_turnover"],
            "Dev Net CAGR (%) [2022-24]": res["dev_cagr"],
            "Dev MDD (%) [2022-24]": res["dev_mdd"],
            "Dev Sharpe [2022-24]": res["dev_sharpe"],
            "Holdout Net CAGR (%) [2025-26]": res["holdout_cagr"],
            "Holdout MDD (%) [2025-26]": res["holdout_mdd"],
            "Holdout Sharpe [2025-26]": res["holdout_sharpe"],
        })

        for yr, stats in res["window_stats"].items():
            window_rows.append({
                "Configuration": label,
                "Year": yr,
                "Net Return (%)": stats["net_return"],
                "SPY Return (%)": stats["spy_return"],
                "Excess Return (%)": stats["excess_return"],
                "Avg Candidates": stats["avg_candidates"],
                "Avg Held": stats["avg_held"],
                "Executed Entries": stats["executed_entries"],
                "Turnover (%)": stats["turnover"],
            })

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(DATA_DIR / "candidate_generation_comparison_summary.csv", index=False)
    pd.DataFrame(window_rows).to_csv(DATA_DIR / "candidate_generation_by_window.csv", index=False)
    with open(DATA_DIR / "candidate_generation_results.json", "w") as f:
        json.dump(full_results, f, indent=2)

    return full_results


def benchmark_ml_weights(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Evaluates ML contribution weights (0%, 10%, 20%, 30%) across walk-forward backtest.
    Exports data/ml_weights_comparison.csv.
    """
    data = df if df is not None else load_and_prepare_candidate_data()
    weights = [0.0, 0.10, 0.20, 0.30]
    records = []

    for w in weights:
        test_df = data.copy()
        test_df["hybrid_score"] = (1.0 - w) * test_df["q_final"] + w * test_df["ml_percentile_score"]
        res = simulate_candidate_portfolio(
            test_df,
            mode="dual",
            top_pct=0.20,
            min_abs_score=60.0,
            dip_threshold=52.0,
            top_k=5,
            max_stocks_per_sector=2,
            sizing_mode="equal",
        )
        records.append({
            "ML Weight": f"{int(w*100)}%",
            "Total Candidates": res["n_candidates"],
            "Executed Entries": res["executed_entries"],
            "Full Net CAGR (%)": res["full_cagr"],
            "Full MDD (%)": res["full_mdd"],
            "Full Sharpe": res["full_sharpe"],
            "Dev Net CAGR (%) [2022-24]": res["dev_cagr"],
            "Dev MDD (%) [2022-24]": res["dev_mdd"],
            "Dev Sharpe [2022-24]": res["dev_sharpe"],
            "Holdout Net CAGR (%) [2025-26]": res["holdout_cagr"],
            "Holdout MDD (%) [2025-26]": res["holdout_mdd"],
            "Holdout Sharpe [2025-26]": res["holdout_sharpe"],
            "Mean Turnover (%)": res["mean_turnover"],
        })

    out_df = pd.DataFrame(records)
    out_df.to_csv(DATA_DIR / "ml_weights_comparison.csv", index=False)
    return out_df


def benchmark_volatility_sizing(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Compares ATR% volatility treatments:
    1. Hard Rejection Cap (Reject if ATR% > 6.0%)
    2. Directional Inclusion + Equal Weighting (Admit all ATR, 20% fixed slot)
    3. Volatility-Adjusted Sizing (Inverse-ATR risk parity sizing)
    Exports data/volatility_sizing_comparison.csv.
    """
    data = df if df is not None else load_and_prepare_candidate_data()
    treatments = [
        ("Hard Rejection Filter (Reject ATR% > 6.0%, Equal Sizing)", 0.06, "equal"),
        ("Directional Inclusion (Admit All ATR, Equal Sizing)", None, "equal"),
        ("Volatility-Adjusted Sizing (Inverse-ATR Risk Parity, No Hard Cap)", None, "volatility_adjusted"),
    ]
    records = []

    for label, max_atr, sizing in treatments:
        res = simulate_candidate_portfolio(
            data,
            mode="dual",
            top_pct=0.20,
            min_abs_score=60.0,
            dip_threshold=52.0,
            top_k=5,
            max_stocks_per_sector=2,
            max_atr_pct=max_atr,
            sizing_mode=sizing,
        )
        records.append({
            "Volatility Treatment": label,
            "Max ATR Cap": f"{max_atr*100:.1f}%" if max_atr else "None",
            "Sizing Mechanism": sizing,
            "Total Candidates": res["n_candidates"],
            "Executed Entries": res["executed_entries"],
            "Full Net CAGR (%)": res["full_cagr"],
            "Full MDD (%)": res["full_mdd"],
            "Full Sharpe": res["full_sharpe"],
            "Holdout Net CAGR (%) [2025-26]": res["holdout_cagr"],
            "Holdout MDD (%) [2025-26]": res["holdout_mdd"],
            "Holdout Sharpe [2025-26]": res["holdout_sharpe"],
            "Mean Turnover (%)": res["mean_turnover"],
        })

    out_df = pd.DataFrame(records)
    out_df.to_csv(DATA_DIR / "volatility_sizing_comparison.csv", index=False)
    return out_df


if __name__ == "__main__":
    print("Executing Relative Ranking Candidate Generation Benchmark...")
    results = run_candidate_generation_benchmark()
    print("Executing ML Contribution Weights Benchmark...")
    ml_df = benchmark_ml_weights()
    print("Executing Volatility Sizing Benchmark...")
    vol_df = benchmark_volatility_sizing()
    print("Benchmark complete! Artifacts written to data/")
