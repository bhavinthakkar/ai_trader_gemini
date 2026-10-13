"""Load fresh 20-session ML scores produced by the sibling stock_ml repository."""

from __future__ import annotations

import csv
import os
from datetime import date
from pathlib import Path

DEFAULT_SCORES_PATH = Path(__file__).resolve().parent.parent / "stock_ml" / "data" / "xgb_current_scores_1m.csv"


def classify_ml_signal(score_0_100: float | None, rank_percentile: float | None = None) -> dict:
    """
    Classify a 0-100 ML percentile score into a standardized standalone indicator.
    Tiers:
      >= 80: STRONG OUTPERFORM (Top Quintile / High Conviction Alpha)
      >= 60: OUTPERFORM (Quintile 2 / Favorable Alpha)
      >= 40: MARKET PERFORM (Quintile 3 / Neutral Benchmark Performance)
      >= 20: UNDERPERFORM (Quintile 4 / Unfavorable Alpha)
      <  20: STRONG UNDERPERFORM (Bottom Quintile / High Conviction Short)
    """
    if score_0_100 is None and rank_percentile is not None:
        score_0_100 = round(rank_percentile * 100.0, 2)
    if score_0_100 is None:
        return {
            "signal": "UNAVAILABLE",
            "badge": "⚪ N/A",
            "tier": "Unranked",
            "color": "#64748b",
            "stance": "UNAVAILABLE",
        }
    val = float(score_0_100)
    if val >= 80.0:
        return {
            "signal": "STRONG OUTPERFORM",
            "badge": "🟢 STRONG OUTPERFORM",
            "tier": "Top Quintile (Top 20% Alpha)",
            "color": "#3dd68c",
            "stance": "BULLISH",
        }
    if val >= 60.0:
        return {
            "signal": "OUTPERFORM",
            "badge": "🟢 OUTPERFORM",
            "tier": "Quintile 2 (60-80% Alpha)",
            "color": "#22c55e",
            "stance": "BULLISH",
        }
    if val >= 40.0:
        return {
            "signal": "MARKET PERFORM",
            "badge": "⚪ NEUTRAL",
            "tier": "Quintile 3 (40-60% Alpha)",
            "color": "#94a3b8",
            "stance": "NEUTRAL",
        }
    if val >= 20.0:
        return {
            "signal": "UNDERPERFORM",
            "badge": "🔴 UNDERPERFORM",
            "tier": "Quintile 4 (20-40% Alpha)",
            "color": "#f87171",
            "stance": "BEARISH",
        }
    return {
        "signal": "STRONG UNDERPERFORM",
        "badge": "🔴 STRONG UNDERPERFORM",
        "tier": "Bottom Quintile (Bottom 20% Alpha)",
        "color": "#ef4444",
        "stance": "BEARISH",
    }


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
        score_0_100 = round(percentile * 100.0, 2)
        cls = classify_ml_signal(score_0_100)
        return {
            "ticker": wanted,
            "score_date": score_date.isoformat(),
            "age_days": age_days,
            "rank": int(row["ml_rank"]),
            "universe_size": len(rows),
            "rank_percentile": percentile,
            "score_0_100": score_0_100,
            "predicted_20d_excess_return": predicted,
            "model_name": row["ml_model_name"],
            "model_as_of": row["model_as_of"],
            "target_horizon_sessions": 20,
            "signal": cls["signal"],
            "badge": cls["badge"],
            "tier": cls["tier"],
            "stance": cls["stance"],
            "sector": row.get("sector", ""),
            "in_top_10_portfolio": str(row.get("in_top_10_portfolio", "")).strip().lower() in ("true", "1", "yes"),
            "rank_fundamentals": int(row["rank_pit_fundamentals_rank_1m"]) if row.get("rank_pit_fundamentals_rank_1m") and str(row["rank_pit_fundamentals_rank_1m"]).strip().isdigit() else None,
            "rank_technical": int(row["rank_pit_technical_rank_1m"]) if row.get("rank_pit_technical_rank_1m") and str(row["rank_pit_technical_rank_1m"]).strip().isdigit() else None,
        }
    except (OSError, ValueError, KeyError, TypeError):
        return None


def get_all_ml_scores(scores_path: str | os.PathLike | None = None, max_age_days: int = 7) -> list[dict]:
    """Return all scored universe stocks ranked by point-in-time ML model."""
    path = Path(scores_path or os.getenv("STOCK_ML_SCORES_PATH") or DEFAULT_SCORES_PATH)
    if not path.is_file():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return []
        results = []
        for row in rows:
            ticker = row.get("ticker", "").strip().upper()
            if not ticker:
                continue
            try:
                score_date = date.fromisoformat(row["score_date"][:10])
                age_days = (date.today() - score_date).days
                if age_days < 0 or age_days > max_age_days or int(row.get("target_horizon_sessions", 0)) != 20:
                    continue
                percentile = float(row["ml_rank_percentile"])
                predicted = float(row["predicted_20d_excess_return"])
                score_0_100 = round(percentile * 100.0, 2)
                cls = classify_ml_signal(score_0_100)
                results.append({
                    "ticker": ticker,
                    "rank": int(row["ml_rank"]),
                    "score_0_100": score_0_100,
                    "rank_percentile": percentile,
                    "predicted_20d_excess_return": predicted,
                    "signal": cls["signal"],
                    "badge": cls["badge"],
                    "tier": cls["tier"],
                    "stance": cls["stance"],
                    "sector": row.get("sector", ""),
                    "in_top_10_portfolio": str(row.get("in_top_10_portfolio", "")).strip().lower() in ("true", "1", "yes"),
                    "rank_fundamentals": int(row["rank_pit_fundamentals_rank_1m"]) if row.get("rank_pit_fundamentals_rank_1m") and str(row["rank_pit_fundamentals_rank_1m"]).strip().isdigit() else None,
                    "rank_technical": int(row["rank_pit_technical_rank_1m"]) if row.get("rank_pit_technical_rank_1m") and str(row["rank_pit_technical_rank_1m"]).strip().isdigit() else None,
                    "score_date": score_date.isoformat(),
                    "age_days": age_days,
                    "model_name": row["ml_model_name"],
                    "universe_size": len(rows),
                })
            except (ValueError, KeyError, TypeError):
                continue
        results.sort(key=lambda x: x["rank"])
        return results
    except OSError:
        return []


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
