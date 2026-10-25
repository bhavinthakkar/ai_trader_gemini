"""Exploratory walk-forward evaluation of Timestral-209M on daily market data.

Example:
    ./venv/bin/python timestral_walk_forward.py --tickers AAPL,MSFT,NVDA,SPY

Each forecast uses only the trailing close-to-close log returns available at
the forecast date. Benchmarks are zero-return and trailing-20-day momentum.
This is a shadow evaluation and its output is not consumed by trading decisions.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from timestral_service import DEFAULT_MODEL_ID, forecast_series


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "data" / "timestral_walk_forward"
DEFAULT_ML_ARTIFACT = ROOT / "data" / "xgb_lambdamart_walk_forward_predictions.parquet"


def download_close_prices(tickers: list[str], start: str) -> pd.DataFrame:
    raw = yf.download(
        tickers=tickers,
        start=start,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no historical prices")

    if isinstance(raw.columns, pd.MultiIndex):
        level_zero = raw.columns.get_level_values(0)
        if "Close" in level_zero:
            close = raw["Close"]
        else:
            close = raw.xs("Close", axis=1, level=1)
    elif "Close" in raw.columns:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    else:
        raise RuntimeError("Yahoo Finance response has no Close column")

    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    close = close.reindex(columns=tickers).dropna(how="all")
    # All variates must be observed at each time step for this initial test.
    return close.dropna(how="any").astype(float)


def evaluate(
    prices: pd.DataFrame,
    *,
    context_length: int = 512,
    horizon: int = 20,
    stride: int = 21,
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "cpu",
) -> tuple[pd.DataFrame, dict]:
    if len(prices) < context_length + horizon:
        raise ValueError(
            f"Need at least {context_length + horizon} aligned price rows; got {len(prices)}"
        )
    if (prices <= 0).any().any():
        raise ValueError("prices must be positive")
    log_returns = np.log(prices / prices.shift(1)).dropna()
    rows = []
    q10, q50, q90 = 0.10, 0.50, 0.90

    # End each context at a decision close; score only the next full horizon.
    first_cut = context_length
    final_cut = len(log_returns) - horizon
    for cut in range(first_cut, final_cut + 1, stride):
        context = log_returns.iloc[cut - context_length:cut].to_numpy(dtype=np.float32).T[None, :, :]
        forecast = forecast_series(
            context,
            horizon,
            model_id=model_id,
            device=device,
        )
        levels = forecast["quantile_levels"]
        cube = forecast["quantiles"]  # (batch, variate, horizon, quantile)
        actual = log_returns.iloc[cut:cut + horizon].to_numpy(dtype=float).T
        momentum_baseline = np.expm1(
            log_returns.iloc[cut - horizon:cut].sum(axis=0).to_numpy(dtype=float)
        )

        for variate, ticker in enumerate(prices.columns):
            actual_path = np.cumsum(actual[variate])
            median_path = np.cumsum(cube[0, variate, :, levels.index(q50)])
            for step in range(horizon):
                rows.append({
                    "forecast_date": log_returns.index[cut - 1].isoformat(),
                    "target_date": log_returns.index[cut + step].isoformat(),
                    "ticker": ticker,
                    "step": step + 1,
                    "actual_log_return": float(actual[variate, step]),
                    "actual_cumulative_return": float(math.expm1(actual_path[step])),
                    "timestral_median_log_return": float(cube[0, variate, step, levels.index(q50)]),
                    "timestral_q10_log_return": float(cube[0, variate, step, levels.index(q10)]),
                    "timestral_q90_log_return": float(cube[0, variate, step, levels.index(q90)]),
                    "timestral_cumulative_return_median": float(math.expm1(median_path[step])),
                    "baseline_cumulative_return": 0.0,
                    "momentum_baseline_cumulative_return": float(momentum_baseline[variate]),
                    "model_id": model_id,
                })

    results = pd.DataFrame(rows)
    summary = summarize(results, model_id=model_id, horizon=horizon, stride=stride)
    return results, summary


def summarize(results: pd.DataFrame, *, model_id: str, horizon: int, stride: int) -> dict:
    if results.empty:
        raise ValueError("No walk-forward forecasts were produced")
    at_horizon = results[results["step"] == horizon].copy()

    def score(group: pd.DataFrame) -> dict:
        final = group[group["step"] == horizon]
        actual = final["actual_cumulative_return"].to_numpy()
        predicted = final["timestral_cumulative_return_median"].to_numpy()
        zero = final["baseline_cumulative_return"].to_numpy()
        momentum = final["momentum_baseline_cumulative_return"].to_numpy()
        daily_actual = group["actual_log_return"].to_numpy()
        daily_q10 = group["timestral_q10_log_return"].to_numpy()
        daily_q90 = group["timestral_q90_log_return"].to_numpy()
        return {
            "forecast_count": int(len(final)),
            "timestral_20d_return_mae": float(np.mean(np.abs(predicted - actual))),
            "zero_return_baseline_20d_mae": float(np.mean(np.abs(zero - actual))),
            "trailing_20d_momentum_baseline_mae": float(np.mean(np.abs(momentum - actual))),
            "timestral_directional_accuracy": float(np.mean(np.sign(predicted) == np.sign(actual))),
            "trailing_20d_momentum_directional_accuracy": float(np.mean(np.sign(momentum) == np.sign(actual))),
            "timestral_daily_q10_q90_coverage": float(
                np.mean((daily_actual >= daily_q10) & (daily_actual <= daily_q90))
            ),
        }

    return {
        "model_id": model_id,
        "forecast_horizon_trading_days": horizon,
        "stride_trading_days": stride,
        "tickers": sorted(at_horizon["ticker"].unique().tolist()),
        **score(results),
        "per_ticker": {
            ticker: score(group)
            for ticker, group in results.groupby("ticker", sort=True)
        },
        "evaluation_start": at_horizon["forecast_date"].min(),
        "evaluation_end": at_horizon["target_date"].max(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": "Yahoo Finance auto-adjusted daily Close",
        "data_caveat": (
            "Exploratory historical evaluation: current Yahoo Finance adjusted history is not a "
            "point-in-time vendor snapshot; this report is not evidence of investable performance."
        ),
    }


def evaluate_existing_ml_dates(
    prices: pd.DataFrame,
    baseline_rows: pd.DataFrame,
    *,
    context_length: int = 512,
    horizon: int = 20,
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "cpu",
) -> tuple[pd.DataFrame, dict]:
    """Score Timestral against the existing point-in-time ML ranking artifact.

    Timestral's relative-return forecast is the predicted simple return of each
    stock less the predicted SPY return. Dates and labels come from the ML
    artifact; only price history through each decision date enters the model.
    """
    if "SPY" not in prices.columns:
        raise ValueError("prices must include SPY for relative-return comparison")
    log_returns = np.log(prices / prices.shift(1)).dropna()
    required = {"decision_date", "ticker", "label_excess_20d", "ml_percentile_score"}
    missing = required - set(baseline_rows.columns)
    if missing:
        raise ValueError(f"ML artifact is missing columns: {sorted(missing)}")

    rows = baseline_rows[baseline_rows["ticker"].isin(prices.columns)].copy()
    rows["decision_date"] = pd.to_datetime(rows["decision_date"]).dt.normalize()
    rows = rows.dropna(subset=["label_excess_20d", "ml_percentile_score"])
    output = []

    for decision_date, group in rows.groupby("decision_date", sort=True):
        cut = int(log_returns.index.searchsorted(decision_date, side="right"))
        if cut < context_length or cut + horizon > len(log_returns):
            continue
        if log_returns.index[cut - 1].normalize() != decision_date:
            continue

        context = log_returns.iloc[cut - context_length:cut].to_numpy(dtype=np.float32).T[None, :, :]
        forecast = forecast_series(
            context,
            horizon,
            model_id=model_id,
            device=device,
        )
        levels = forecast["quantile_levels"]
        medians = forecast["quantiles"][0, :, :, levels.index(0.50)]
        total_returns = np.expm1(medians.sum(axis=1))
        spy_return = float(total_returns[prices.columns.get_loc("SPY")])

        for item in group.itertuples(index=False):
            ticker_index = prices.columns.get_loc(item.ticker)
            output.append({
                "decision_date": decision_date.date().isoformat(),
                "ticker": item.ticker,
                "label_end_date": str(getattr(item, "label_end_date", "")),
                "timestral_predicted_stock_return": float(total_returns[ticker_index]),
                "timestral_predicted_spy_return": spy_return,
                "timestral_predicted_excess_return": float(total_returns[ticker_index] - spy_return),
                "realized_excess_return": float(item.label_excess_20d),
                "ml_percentile_score": float(item.ml_percentile_score),
                "model_id": model_id,
            })

    result = pd.DataFrame(output)
    if result.empty:
        raise ValueError("No dates/tickers from the ML artifact could be aligned to price history")
    summary = summarize_ml_comparison(result, horizon=horizon, model_id=model_id)
    return result, summary


def summarize_ml_comparison(results: pd.DataFrame, *, horizon: int, model_id: str) -> dict:
    actual = results["realized_excess_return"].to_numpy()
    predicted = results["timestral_predicted_excess_return"].to_numpy()
    rows = []
    for name, values in (
        ("timestral", results["timestral_predicted_excess_return"]),
        ("existing_ml", results["ml_percentile_score"]),
    ):
        candidate = results[["decision_date", "ticker", "realized_excess_return"]].copy()
        candidate["score"] = values.to_numpy()
        candidate["quintile"] = candidate.groupby("decision_date")["score"].rank(
            method="first", pct=True
        )
        top = candidate[candidate["quintile"] > 0.8]["realized_excess_return"].mean()
        bottom = candidate[candidate["quintile"] <= 0.2]["realized_excess_return"].mean()
        rows.append((name, float(top - bottom)))
    return {
        "model_id": model_id,
        "forecast_horizon_trading_days": horizon,
        "matched_forecasts": int(len(results)),
        "decision_dates": int(results["decision_date"].nunique()),
        "tickers": sorted(results["ticker"].unique().tolist()),
        "timestral_relative_return_mae": float(np.mean(np.abs(predicted - actual))),
        "zero_excess_return_baseline_mae": float(np.mean(np.abs(actual))),
        "timestral_relative_return_directional_accuracy": float(
            np.mean(np.sign(predicted) == np.sign(actual))
        ),
        "timestral_vs_realized_excess_spearman": float(
            results["timestral_predicted_excess_return"].corr(
                results["realized_excess_return"], method="spearman"
            )
        ),
        "existing_ml_vs_realized_excess_spearman": float(
            results["ml_percentile_score"].corr(
                results["realized_excess_return"], method="spearman"
            )
        ),
        "timestral_top_minus_bottom_quintile_excess_return": dict(rows)["timestral"],
        "existing_ml_top_minus_bottom_quintile_excess_return": dict(rows)["existing_ml"],
        "evaluation_start": results["decision_date"].min(),
        "evaluation_end": results["decision_date"].max(),
        "comparison_caveat": (
            "The ML artifact is point-in-time, but Timestral inputs use current Yahoo Finance "
            "auto-adjusted history; historical corporate-action adjustments may differ from "
            "the prices available at each decision date. Shadow evaluation only."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", default="AAPL,MSFT,NVDA,SPY")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--stride", type=int, default=21)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--compare-existing-ml",
        action="store_true",
        help="Also align forecasts to the existing point-in-time LambdaMART walk-forward artifact",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    tickers = [item.strip().upper() for item in args.tickers.split(",") if item.strip()]
    if not tickers:
        parser.error("--tickers must contain at least one symbol")

    baseline_rows = None
    if args.compare_existing_ml:
        baseline_rows = pd.read_parquet(DEFAULT_ML_ARTIFACT)
        tickers = sorted(set(tickers) | set(baseline_rows["ticker"].unique()) | {"SPY"})

    print(f"Fetching {len(tickers)} ticker histories from {args.start}...")
    prices = download_close_prices(tickers, args.start)
    print(f"Aligned history: {len(prices)} bars ({prices.index[0].date()} to {prices.index[-1].date()})")
    predictions, summary = evaluate(
        prices,
        context_length=args.context,
        horizon=args.horizon,
        stride=args.stride,
        model_id=args.model,
        device=args.device,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model.rsplit('/', 1)[-1].lower()}_{args.horizon}d"
    predictions_path = args.output_dir / f"{stem}_predictions.csv"
    summary_path = args.output_dir / f"{stem}_summary.json"
    predictions.to_csv(predictions_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Predictions: {predictions_path}")
    print(f"Summary: {summary_path}")

    if baseline_rows is not None:
        matched, matched_summary = evaluate_existing_ml_dates(
            prices,
            baseline_rows,
            context_length=args.context,
            horizon=args.horizon,
            model_id=args.model,
            device=args.device,
        )
        matched_path = args.output_dir / f"{stem}_ml_comparison.csv"
        matched_summary_path = args.output_dir / f"{stem}_ml_comparison.json"
        matched.to_csv(matched_path, index=False)
        matched_summary_path.write_text(
            json.dumps(matched_summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(matched_summary, indent=2))
        print(f"ML comparison predictions: {matched_path}")
        print(f"ML comparison summary: {matched_summary_path}")


if __name__ == "__main__":
    main()
