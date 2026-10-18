"""Walk-Forward Validation and Production Training for LambdaMART (Step 3.2).

Executes:
1. Purged walk-forward validation across test years (2022 to 2026).
2. Per-date Rank IC calculation, Information Coefficient Information Ratio (ICIR), and hit rate.
3. Top vs Bottom quintile spread evaluation.
4. Production model fitting on all mature labels with zero lookahead.
5. Export of out-of-sample prediction panels and evaluation reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from quant.build_training_table import FROZEN_16_FACTORS
from quant.lambdamart_ranker import LambdaMARTRanker


DATA_DIR = Path("data")
MODELS_DIR = Path("models")
FEATURES_16 = [f"feature_{i+1}" for i in range(16)]
TEST_YEARS = [2022, 2023, 2024, 2025, 2026]


def run_purged_walk_forward_evaluation(
    data_path: Optional[Path] = None,
    test_years: Optional[List[int]] = None,
    relevance_method: str = "rank",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Run institutional walk-forward evaluation using LambdaMART ranker."""
    if data_path is None:
        data_path = DATA_DIR / "ml_training_table.parquet"

    df = pd.read_parquet(data_path)
    labeled_df = df[df["has_label"]].copy()

    test_years = test_years or TEST_YEARS
    predictions_list = []
    year_metrics = []

    print(f"=== Starting LambdaMART Purged Walk-Forward Evaluation ({min(test_years)} - {max(test_years)}) ===")

    for year in test_years:
        cutoff = f"{year}-01-01"
        test_start = f"{year}-01-01"
        test_end = f"{year}-12-31"

        # 1. Filter purged training set: strictly decision_date < cutoff AND label_end_date <= cutoff
        train_mask = (labeled_df["decision_date"] < cutoff) & (labeled_df["label_end_date"] <= cutoff)
        test_mask = (labeled_df["decision_date"] >= test_start) & (labeled_df["decision_date"] <= test_end)

        train_data = labeled_df.loc[train_mask].sort_values(["decision_date", "ticker"]).reset_index(drop=True)
        test_data = labeled_df.loc[test_mask].sort_values(["decision_date", "ticker"]).reset_index(drop=True)

        if train_data.empty or test_data.empty:
            print(f"Skipping year {year}: insufficient train ({len(train_data)}) or test ({len(test_data)}) rows.")
            continue

        # 2. Instantiate and fit LambdaMART ranker
        ranker = LambdaMARTRanker(relevance_method=relevance_method)
        audit_info = ranker.train_purged(
            training_table=train_data,
            cutoff_date=cutoff,
            feature_cols=FEATURES_16,
            target_col="label_excess_20d",
            label_end_col="label_end_date",
            group_col="decision_date",
        )

        # 3. Predict on unseen test data using date-contemporaneous percentile normalization
        raw_scores = ranker.predict_raw(test_data[FEATURES_16])
        test_data["ml_raw_score"] = raw_scores
        test_data["ml_percentile_score"] = ranker.predict_cross_sectional_percentiles(
            test_data,
            feature_cols=FEATURES_16,
            group_col="decision_date",
            centering=True,
        )
        test_data["walk_forward_year"] = year

        # 4. Compute monthly Rank IC
        monthly_ics = []
        for d, grp in test_data.groupby("decision_date"):
            if len(grp) >= 3:
                ic, _ = spearmanr(grp["ml_percentile_score"], grp["label_excess_20d"])
                if np.isfinite(ic):
                    monthly_ics.append(ic)

        mean_ic = float(np.mean(monthly_ics)) if monthly_ics else 0.0
        std_ic = float(np.std(monthly_ics)) if monthly_ics else 0.0
        icir = mean_ic / (std_ic + 1e-8)
        hit_rate = float(np.mean([ic > 0 for ic in monthly_ics])) if monthly_ics else 0.0

        print(
            f"Year {year}: {len(train_data):4d} train rows (latest label {audit_info['max_label_matured']}) | "
            f"{len(test_data):3d} test rows | Mean Rank IC: {mean_ic:+.4f} | ICIR: {icir:.2f} | Hit Rate: {hit_rate*100:.1f}%"
        )

        year_metrics.append({
            "year": year,
            "train_samples": int(len(train_data)),
            "test_samples": int(len(test_data)),
            "test_months": int(len(monthly_ics)),
            "mean_rank_ic": round(mean_ic, 4),
            "std_rank_ic": round(std_ic, 4),
            "icir": round(icir, 2),
            "ic_hit_rate": round(hit_rate, 4),
        })

        predictions_list.append(test_data)

    all_preds_df = pd.concat(predictions_list, ignore_index=True)
    metrics_df = pd.DataFrame(year_metrics)

    # Compute overall walk-forward stats
    all_monthly_ics = []
    for d, grp in all_preds_df.groupby("decision_date"):
        if len(grp) >= 3:
            ic, _ = spearmanr(grp["ml_percentile_score"], grp["label_excess_20d"])
            if np.isfinite(ic):
                all_monthly_ics.append(ic)

    overall_mean_ic = float(np.mean(all_monthly_ics))
    overall_std_ic = float(np.std(all_monthly_ics))
    overall_icir = overall_mean_ic / (overall_std_ic + 1e-8)
    overall_hit_rate = float(np.mean([ic > 0 for ic in all_monthly_ics]))

    # Top quintile vs bottom quintile forward return spread
    all_preds_df["pred_quintile"] = all_preds_df.groupby("decision_date")["ml_percentile_score"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False)
    )
    quintile_returns = all_preds_df.groupby("pred_quintile")["label_excess_20d"].mean()
    top_minus_bottom_spread = float(quintile_returns.iloc[-1] - quintile_returns.iloc[0])

    summary_stats = {
        "overall_mean_rank_ic": round(overall_mean_ic, 4),
        "overall_std_rank_ic": round(overall_std_ic, 4),
        "overall_icir": round(overall_icir, 2),
        "overall_hit_rate": round(overall_hit_rate, 4),
        "total_walk_forward_months": len(all_monthly_ics),
        "total_predictions": len(all_preds_df),
        "top_minus_bottom_excess_spread": round(top_minus_bottom_spread, 4),
        "quintile_returns": {f"Q{k+1}": round(float(v), 4) for k, v in quintile_returns.items()},
    }

    print("\n=== Overall Walk-Forward Summary ===")
    print(f"Total Months: {summary_stats['total_walk_forward_months']}")
    print(f"Overall Mean Rank IC: {summary_stats['overall_mean_rank_ic']:+.4f}")
    print(f"Overall ICIR: {summary_stats['overall_icir']:.2f}")
    print(f"Positive IC Hit Rate: {summary_stats['overall_hit_rate']*100:.1f}%")
    print(f"Top - Bottom Quintile Spread: {summary_stats['top_minus_bottom_excess_spread']*100:+.2f}%")
    print("Quintile Forward Excess Returns:", summary_stats["quintile_returns"])

    return all_preds_df, metrics_df, summary_stats


def train_production_ranker(
    data_path: Optional[Path] = None,
    output_model_path: Optional[Path] = None,
    relevance_method: str = "rank",
) -> Tuple[LambdaMARTRanker, Dict[str, Union[int, str, float]]]:
    """Train production LambdaMART model on all mature historical data."""
    if data_path is None:
        data_path = DATA_DIR / "ml_training_table.parquet"
    if output_model_path is None:
        output_model_path = MODELS_DIR / "lambdamart_production_ranker"

    df = pd.read_parquet(data_path)
    labeled_df = df[df["has_label"]].copy()

    # Sort contiguously
    sorted_df = labeled_df.sort_values(["decision_date", "ticker"]).reset_index(drop=True)
    latest_decision_date = str(sorted_df["decision_date"].max())
    latest_matured_date = str(sorted_df["label_end_date"].max())

    ranker = LambdaMARTRanker(relevance_method=relevance_method)
    X, y, groups, sorted_train = ranker.prepare_grouped_data(
        sorted_df,
        feature_cols=FEATURES_16,
        target_col="label_excess_20d",
        group_col="decision_date",
        relevance_method=relevance_method,
    )

    ranker.fit(X, y, groups, feature_names=FEATURES_16)
    ranker.save_model(output_model_path)

    importances = ranker.get_feature_importances()
    prod_meta = {
        "model_type": "LambdaMARTRanker_XGBRanker",
        "objective": ranker.params["objective"],
        "relevance_method": relevance_method,
        "total_train_samples": len(sorted_train),
        "total_decision_dates": len(groups),
        "min_decision_date": str(sorted_train["decision_date"].min()),
        "max_decision_date": latest_decision_date,
        "latest_label_maturity": latest_matured_date,
        "feature_importances": {k: round(float(v), 4) for k, v in importances.items()},
    }

    with open(output_model_path.with_suffix(".meta.json"), "w") as f:
        json.dump(prod_meta, f, indent=2)

    print(f"\nProduction LambdaMART ranker saved to {output_model_path} ({len(sorted_train)} samples)")
    return ranker, prod_meta


def execute_full_pipeline():
    """Run walk-forward evaluation, export tables, and train production model."""
    preds_df, metrics_df, summary_stats = run_purged_walk_forward_evaluation()

    # Save artifacts in both repositories
    dest_dirs = [
        DATA_DIR,
        Path("/home/bhavin-thakkar/gitClient/stock-ml/data"),
    ]

    for d in dest_dirs:
        d.mkdir(parents=True, exist_ok=True)
        preds_df.to_parquet(d / "xgb_lambdamart_walk_forward_predictions.parquet", index=False)
        preds_df.to_csv(d / "xgb_lambdamart_walk_forward_predictions.csv", index=False)
        metrics_df.to_csv(d / "xgb_lambdamart_metrics.csv", index=False)
        with open(d / "xgb_lambdamart_summary.json", "w") as f:
            json.dump(summary_stats, f, indent=2)

    # Train and save production model
    train_production_ranker()


if __name__ == "__main__":
    execute_full_pipeline()
