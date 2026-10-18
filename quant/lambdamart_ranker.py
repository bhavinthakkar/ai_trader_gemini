"""Institutional LambdaMART Ranker with Purged Walk-Forward Cutoffs.

Implements Step 3.2:
- Treats each decision date cross-section as an independent ranking group.
- Maps realized forward excess returns into discrete relevance grades per decision date.
- Employs XGBRanker with Normalized Discounted Cumulative Gain (rank:ndcg) objective.
- Enforces strict zero-lookahead purged training cutoffs (label_end_date <= cutoff).
- Transforms out-of-sample predictions into contemporaneous cross-sectional percentiles [0, 100]
  without fitting on future outcomes or holdout labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from xgboost import XGBRanker


# Default institutional hyperparameters for monthly multi-factor ranking
DEFAULT_LAMBDAMART_PARAMS = {
    "objective": "rank:ndcg",
    "eval_metric": "ndcg@5",
    "n_estimators": 120,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "random_state": 42,
    "n_jobs": -1,
}


class LambdaMARTRanker:
    """Institutional LambdaMART ranking engine for cross-sectional equity selection."""

    def __init__(
        self,
        params: Optional[Dict[str, Union[int, float, str]]] = None,
        relevance_method: str = "rank",
    ):
        """Initialize ranker with hyperparameters and relevance assignment strategy.

        Args:
            params: Dictionary of XGBRanker hyperparameters.
            relevance_method: 'rank' for discrete 0..N-1 integer ranks (recommended for 10-50 stocks),
                             or 'quintile' for 0..4 relevance grades.
        """
        self.params = {**DEFAULT_LAMBDAMART_PARAMS, **(params or {})}
        self.relevance_method = relevance_method
        self.model: Optional[XGBRanker] = None
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    @staticmethod
    def compute_relevance_grades(
        df: pd.DataFrame,
        target_col: str = "label_excess_20d",
        group_col: str = "decision_date",
        method: str = "rank",
    ) -> pd.Series:
        """Assign discrete non-negative integer relevance grades per group.

        For each decision date cross-section:
        - If method == 'rank': Stocks are ordered by target_col and assigned
          integer ranks 0, 1, ..., N_t - 1, where N_t - 1 is the best performer.
        - If method == 'quintile': Stocks are bucketed into quintiles 0, 1, 2, 3, 4.

        Returns:
            pd.Series of integer relevance grades with identical index to df.
        """
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in DataFrame.")
        if group_col not in df.columns:
            raise KeyError(f"Group column '{group_col}' not found in DataFrame.")

        def _assign_group_relevance(s: pd.Series) -> pd.Series:
            n = len(s)
            if n <= 1:
                return pd.Series(0, index=s.index, dtype=int)

            if method == "rank":
                # Method 'first' guarantees unique ranks 0 .. n-1
                return (s.rank(method="first", ascending=True).astype(int) - 1)
            elif method == "quintile":
                # Divide into up to 5 bins, handling small groups gracefully
                num_bins = min(5, n)
                return pd.qcut(
                    s.rank(method="first"),
                    q=num_bins,
                    labels=False,
                ).astype(int)
            else:
                raise ValueError(f"Unknown relevance method: '{method}'. Must be 'rank' or 'quintile'.")

        return df.groupby(group_col, group_keys=False)[target_col].apply(_assign_group_relevance)

    @staticmethod
    def prepare_grouped_data(
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: Optional[str] = None,
        group_col: str = "decision_date",
        relevance_method: str = "rank",
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray], np.ndarray, pd.DataFrame]:
        """Sort data contiguously by group and extract X, y_relevance, and group counts.

        XGBRanker requires that observations belonging to the same query group
        appear contiguously in the dataset.

        Returns:
            Tuple of (X_df, y_relevance_array, groups_array, sorted_df)
        """
        missing_feats = [col for col in feature_cols if col not in df.columns]
        if missing_feats:
            raise KeyError(f"Features missing from DataFrame: {missing_feats}")

        sort_cols = [group_col]
        if "ticker" in df.columns:
            sort_cols.append("ticker")

        sorted_df = df.sort_values(sort_cols).reset_index(drop=True)
        groups = sorted_df.groupby(group_col, sort=False).size().to_numpy()

        if len(sorted_df) != groups.sum():
            raise ValueError(f"Group sum ({groups.sum()}) does not match sorted rows ({len(sorted_df)})")

        X = sorted_df[feature_cols].copy()

        y = None
        if target_col is not None and target_col in sorted_df.columns:
            y_series = LambdaMARTRanker.compute_relevance_grades(
                sorted_df, target_col=target_col, group_col=group_col, method=relevance_method
            )
            y = y_series.to_numpy()

        return X, y, groups, sorted_df

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        group: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "LambdaMARTRanker":
        """Fit the underlying XGBRanker on grouped data.

        Args:
            X: Feature matrix of shape (N, num_features).
            y: Integer relevance grades of shape (N,).
            group: Array of group sizes summing to N.
            feature_names: Optional list of feature names.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_mat = X.to_numpy()
        else:
            self.feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]
            X_mat = np.asarray(X)

        y_arr = np.asarray(y, dtype=int)
        group_arr = np.asarray(group, dtype=int)

        if len(X_mat) != group_arr.sum():
            raise ValueError(f"Group sizes sum to {group_arr.sum()}, but X has {len(X_mat)} rows.")
        if len(X_mat) != len(y_arr):
            raise ValueError(f"X has {len(X_mat)} rows, but y has {len(y_arr)} rows.")

        self.model = XGBRanker(**self.params)
        self.model.fit(X_mat, y_arr, group=group_arr)
        self.is_fitted = True
        return self

    def train_purged(
        self,
        training_table: pd.DataFrame,
        cutoff_date: str,
        feature_cols: List[str],
        target_col: str = "label_excess_20d",
        label_end_col: str = "label_end_date",
        group_col: str = "decision_date",
    ) -> Dict[str, Union[int, str]]:
        """Train LambdaMART with strict zero-lookahead purged cutoff.

        Only includes historical decision dates whose labels matured on or before cutoff_date:
        `decision_date < cutoff_date` AND `label_end_date <= cutoff_date`.

        Returns:
            Dictionary with training audit metadata.
        """
        df = training_table.copy()
        if "has_label" in df.columns:
            df = df.loc[df["has_label"]].copy()

        cutoff_ts = pd.Timestamp(cutoff_date).normalize()
        dec_dates = pd.to_datetime(df[group_col]).dt.normalize()
        end_dates = pd.to_datetime(df[label_end_col]).dt.normalize()

        # Strict Purged Condition
        train_mask = (dec_dates < cutoff_ts) & (end_dates <= cutoff_ts)
        purged_df = df.loc[train_mask].copy()

        if purged_df.empty:
            raise ValueError(f"No labeled observations available on or before cutoff {cutoff_date}.")

        X, y, groups, sorted_train = self.prepare_grouped_data(
            purged_df,
            feature_cols=feature_cols,
            target_col=target_col,
            group_col=group_col,
            relevance_method=self.relevance_method,
        )

        self.fit(X, y, groups, feature_names=feature_cols)

        return {
            "cutoff_date": cutoff_date,
            "train_samples": int(len(sorted_train)),
            "train_decision_dates": int(len(groups)),
            "min_decision_date": str(sorted_train[group_col].min()),
            "max_decision_date": str(sorted_train[group_col].max()),
            "max_label_matured": str(sorted_train[label_end_col].max()),
        }

    def predict_raw(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Generate raw continuous ranking scores from LambdaMART."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() or train_purged() first.")
        if isinstance(X, pd.DataFrame):
            X_mat = X[self.feature_names].to_numpy()
        else:
            X_mat = np.asarray(X)
        return self.model.predict(X_mat)

    def predict_cross_sectional_percentiles(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        group_col: str = "decision_date",
        centering: bool = True,
    ) -> pd.Series:
        """Generate cross-sectional percentile scores [0.0, 100.0] independently per decision date.

        Crucial Rule:
        - Uses ONLY the model predictions of stocks active on date t.
        - Does NOT fit or use any distribution on future returns or holdout labels.
        - Guaranteed zero holdout leakage.

        Formula:
            Percentile = ((rank - 0.5) / N_t) * 100.0 if centering else (rank / N_t) * 100.0

        Returns:
            pd.Series of percentile scores in [0.0, 100.0] indexed identically to df.
        """
        feats = feature_cols or self.feature_names
        raw_preds = self.predict_raw(df[feats])

        temp_df = pd.DataFrame({
            "group": df[group_col].to_numpy(),
            "raw": raw_preds,
            "_orig_idx": np.arange(len(df)),
        }, index=df.index)

        def _calc_pct(s: pd.Series) -> pd.Series:
            n = len(s)
            if n <= 1:
                return pd.Series(50.0, index=s.index)
            ranks = s.rank(method="average", ascending=True)
            if centering:
                pct = ((ranks - 0.5) / n) * 100.0
            else:
                pct = (ranks / n) * 100.0
            return pct.clip(0.0, 100.0)

        pct_series = temp_df.groupby("group", group_keys=False)["raw"].apply(_calc_pct)
        return pct_series

    def get_feature_importances(self) -> pd.Series:
        """Return feature importances sorted descending by gain."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model is not fitted.")
        return pd.Series(
            self.model.feature_importances_,
            index=self.feature_names,
        ).sort_values(ascending=False)

    def save_model(self, file_path: Union[str, Path]) -> None:
        """Serialize trained model and configuration."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model is not fitted. Cannot save.")
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Save XGBoost booster model
        booster_path = path.with_suffix(".json")
        self.model.save_model(str(booster_path))
        # Save metadata wrapper
        meta = {
            "params": self.params,
            "relevance_method": self.relevance_method,
            "feature_names": self.feature_names,
            "booster_file": booster_path.name,
        }
        meta_path = path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def load_model(self, file_path: Union[str, Path]) -> "LambdaMARTRanker":
        """Load serialized model and configuration."""
        path = Path(file_path)
        meta_path = path.with_suffix(".meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing metadata file {meta_path}")
        with open(meta_path, "r") as f:
            meta = json.load(f)

        self.params = meta["params"]
        self.relevance_method = meta["relevance_method"]
        self.feature_names = meta["feature_names"]

        booster_path = path.parent / meta["booster_file"]
        self.model = XGBRanker(**self.params)
        self.model.load_model(str(booster_path))
        self.is_fitted = True
        return self
