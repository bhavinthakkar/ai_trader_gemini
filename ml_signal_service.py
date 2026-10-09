"""Load fresh long-horizon stock ranks produced by the stock_ml repository."""

from __future__ import annotations

import csv
import os
from datetime import date
from pathlib import Path

DEFAULT_SCORES_PATH = Path(__file__).resolve().parent.parent / "stock_ml" / "data" / "xgb_current_scores.csv"


def get_ml_context(symbol: str, scores_path: str | os.PathLike | None = None, max_age_days: int = 7) -> dict | None:
    """Return a fresh 12-month model score for a ticker, or None when unavailable."""
    path = Path(scores_path or os.getenv("STOCK_ML_SCORES_PATH") or DEFAULT_SCORES_PATH)
    if not path.is_file():
        return None
    wanted = str(symbol or "").strip().upper()
    if not wanted or "." in wanted:
        return None  # The model covers US point-in-time S&P 500 symbols only.
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            row = next((item for item in csv.DictReader(handle)
                        if item.get("ticker", "").strip().upper() == wanted), None)
        if row is None:
            return None
        score_date = date.fromisoformat(row["score_date"][:10])
        age_days = (date.today() - score_date).days
        if age_days < 0 or age_days > max_age_days:
            return None
        with path.open(newline="", encoding="utf-8") as handle:
            universe_size = sum(1 for _ in csv.DictReader(handle))
        return {
            "ticker": wanted,
            "score_date": score_date.isoformat(),
            "age_days": age_days,
            "rank": int(row["ml_rank"]),
            "universe_size": universe_size,
            "predicted_12m_excess_return": float(row["predicted_12m_excess_return"]),
            "model_name": row.get("model_name", ""),
            "model_as_of": row.get("model_as_of", ""),
            "target_horizon_days": 252,
        }
    except (OSError, ValueError, KeyError, TypeError):
        return None


def format_ml_context(context: dict | None) -> str:
    if not context:
        return (
            "\n\n================ LONG-HORIZON ML CONTEXT ================\n"
            "No fresh point-in-time S&P 500 ML score is available for this ticker. "
            "Do not infer or fabricate one.\n"
        )
    expected = context["predicted_12m_excess_return"] * 100.0
    return (
        "\n\n================ LONG-HORIZON ML CONTEXT ================\n"
        f"Separate research signal: this ticker ranks #{context['rank']} of "
        f"{context['universe_size']} in the point-in-time US S&P 500 universe "
        f"(score date {context['score_date']}); the model estimates {expected:+.2f}% "
        f"12-month excess return. Model: {context['model_name']} (trained as of "
        f"{context['model_as_of']}).\n"
        "This is a 252-session expected excess return, not a probability or a "
        "1–10 day trade signal. Treat it as longer-term context; assess the "
        "short-term setup independently and explain material disagreement.\n"
    )
