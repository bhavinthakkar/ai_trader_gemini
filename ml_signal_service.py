"""Load fresh 20-session ML scores produced by the sibling stock_ml repository."""

from __future__ import annotations

import csv
import os
from datetime import date
from pathlib import Path

DEFAULT_SCORES_PATH = Path(__file__).resolve().parent.parent / "stock_ml" / "data" / "xgb_current_scores_1m.csv"


def get_ml_context(symbol: str, scores_path: str | os.PathLike | None = None, max_age_days: int = 7) -> dict | None:
    """Return a fresh point-in-time 20-session score, or None when unavailable."""
    path = Path(scores_path or os.getenv("STOCK_ML_SCORES_PATH") or DEFAULT_SCORES_PATH)
    if not path.is_file():
        return None
    wanted = str(symbol or "").strip().upper()
    if not wanted or "." in wanted:
        return None
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        row = next((item for item in rows if item.get("ticker", "").strip().upper() == wanted), None)
        if row is None:
            return None
        score_date = date.fromisoformat(row["score_date"][:10])
        age_days = (date.today() - score_date).days
        if age_days < 0 or age_days > max_age_days or int(row.get("target_horizon_sessions", 0)) != 20:
            return None
        percentile = float(row["ml_rank_percentile"])
        predicted = float(row["predicted_20d_excess_return"])
        if not 0.0 <= percentile <= 1.0:
            return None
        return {
            "ticker": wanted,
            "score_date": score_date.isoformat(),
            "age_days": age_days,
            "rank": int(row["ml_rank"]),
            "universe_size": len(rows),
            "rank_percentile": percentile,
            "score_0_100": round(percentile * 100.0, 2),
            "predicted_20d_excess_return": predicted,
            "model_name": row["ml_model_name"],
            "model_as_of": row["model_as_of"],
            "target_horizon_sessions": 20,
        }
    except (OSError, ValueError, KeyError, TypeError):
        return None


def format_ml_context(context: dict | None) -> str:
    if not context:
        return (
            "\n\n================ 20-SESSION ML SCORE ================\n"
            "No fresh one-month point-in-time ML score is available. Do not invent one; "
            "the hybrid composite will use the deterministic market score alone.\n"
        )
    return (
        "\n\n================ 20-SESSION ML SCORE ================\n"
        f"ML score: {context['score_0_100']:.1f}/100 (rank #{context['rank']} of "
        f"{context['universe_size']}, percentile {context['rank_percentile']:.3f}); "
        f"predicted 20-session excess return {context['predicted_20d_excess_return'] * 100:+.2f}%. "
        f"Score date {context['score_date']}; model {context['model_name']} "
        f"(trained as of {context['model_as_of']}).\n"
        "This is a material 20% component of the deterministic hybrid quantitative score. "
        "Compare this ML signal with your independent 20-session relative forecast, report the "
        "outlook in forecast_20d, and classify agreement in ml_signal_assessment. Keep the forecast "
        "separate from the trade action. Do not treat the prediction "
        "as a probability or a guaranteed return.\n"
    )
