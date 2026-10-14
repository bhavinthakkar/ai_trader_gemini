"""Load fresh 20-session ML scores produced by the sibling stock_ml repository."""

from __future__ import annotations

import csv
import os
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STOCK_ML_DATA = ROOT.parent / "stock_ml" / "data"

FOCUSED_SCORES_PATH = STOCK_ML_DATA / "xgb_focused_current_scores.csv"
BROAD_SCORES_PATH = STOCK_ML_DATA / "xgb_current_scores_1m.csv"

# Prefer the new focused 25-stock ETF leaders model, falling back to broad S&P 500
DEFAULT_SCORES_PATH = FOCUSED_SCORES_PATH if FOCUSED_SCORES_PATH.exists() else BROAD_SCORES_PATH


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


def get_out_of_scope_context(symbol: str) -> dict:
    """Return a sensible, structured out-of-scope context for unranked or non-universe tickers."""
    wanted = str(symbol or "").strip().upper()
    return {
        "ticker": wanted,
        "in_scope": False,
        "status": "OUT_OF_SCOPE",
        "score_date": date.today().isoformat(),
        "age_days": 0,
        "rank": None,
        "universe_size": 25,
        "rank_percentile": 0.50,
        "score_0_100": 50.0,
        "raw_score_0_100": None,
        "predicted_20d_excess_return": 0.0,
        "model_name": "xgb_focused_ranker_latest",
        "model_as_of": date.today().isoformat(),
        "target_horizon_sessions": 20,
        "signal": "OUT_OF_SCOPE",
        "badge": "⚪ OUT OF SCOPE",
        "tier": "Outside Focused Universe",
        "stance": "NEUTRAL",
        "sector": "Other",
        "in_top_5_portfolio": False,
        "in_top_10_portfolio": False,
        "rank_fundamentals": None,
        "rank_technical": None,
        "is_focused_universe": False,
        "message": f"'{wanted}' is outside the focused 25-stock ETF leaders ML model universe.",
    }


def _read_scores_file(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def get_ml_context(symbol: str, scores_path: str | os.PathLike | None = None, max_age_days: int = 7) -> dict | None:
    """Return a fresh point-in-time 20-session score, or a sensible out-of-scope context when outside universe."""
    wanted = str(symbol or "").strip().upper()
    if not wanted:
        return None

    # Strip exchange suffix for matching if present (e.g. NVDA.DE -> NVDA)
    base_sym = wanted.split(".")[0]

    custom_path = Path(scores_path) if scores_path else (Path(os.getenv("STOCK_ML_SCORES_PATH")) if os.getenv("STOCK_ML_SCORES_PATH") else None)

    # Search candidates in priority order: custom path -> focused 25 model -> broad S&P 500 model
    search_paths = [custom_path] if custom_path else [FOCUSED_SCORES_PATH, BROAD_SCORES_PATH]

    for p in search_paths:
        if not p or not p.is_file():
            continue
        rows = _read_scores_file(p)
        if not rows:
            continue

        row = next((item for item in rows if item.get("ticker", "").strip().upper() in (wanted, base_sym)), None)
        if row is None:
            continue

        try:
            score_date = date.fromisoformat(row["score_date"][:10])
            age_days = (date.today() - score_date).days
            if age_days < 0 or age_days > max_age_days or int(row.get("target_horizon_sessions", 0)) != 20:
                continue

            percentile = float(row["ml_rank_percentile"])
            predicted = float(row["predicted_20d_excess_return"])
            if not 0.0 <= percentile <= 1.0:
                continue

            score_0_100 = round(percentile * 100.0, 2)
            cls = classify_ml_signal(score_0_100)
            is_focused = "focused" in str(p.name).lower() or len(rows) <= 50

            in_top10 = (
                str(row.get("in_top_10_portfolio", "")).strip().lower() in ("true", "1", "yes")
                or str(row.get("in_top_5_portfolio", "")).strip().lower() in ("true", "1", "yes")
            )

            return {
                "ticker": wanted,
                "in_scope": True,
                "status": "IN_SCOPE",
                "score_date": score_date.isoformat(),
                "age_days": age_days,
                "rank": int(row["ml_rank"]),
                "universe_size": len(rows),
                "rank_percentile": percentile,
                "score_0_100": score_0_100,
                "raw_score_0_100": score_0_100,
                "predicted_20d_excess_return": predicted,
                "model_name": row["ml_model_name"],
                "model_as_of": row["model_as_of"],
                "target_horizon_sessions": 20,
                "signal": cls["signal"],
                "badge": cls["badge"],
                "tier": cls["tier"],
                "stance": cls["stance"],
                "sector": row.get("sector", ""),
                "in_top_5_portfolio": str(row.get("in_top_5_portfolio", "")).strip().lower() in ("true", "1", "yes"),
                "in_top_10_portfolio": in_top10,
                "is_focused_universe": is_focused,
                "rank_fundamentals": int(row["rank_pit_fundamentals_rank_1m"]) if row.get("rank_pit_fundamentals_rank_1m") and str(row["rank_pit_fundamentals_rank_1m"]).strip().isdigit() else None,
                "rank_technical": int(row["rank_pit_technical_rank_1m"]) if row.get("rank_pit_technical_rank_1m") and str(row["rank_pit_technical_rank_1m"]).strip().isdigit() else None,
            }
        except (ValueError, KeyError, TypeError):
            continue

    # Ticker was not found in any active model universe -> return sensible out-of-scope context
    return get_out_of_scope_context(wanted)


def get_all_ml_scores(scores_path: str | os.PathLike | None = None, max_age_days: int = 7) -> list[dict]:
    """Return all scored universe stocks ranked by point-in-time ML model."""
    path = Path(scores_path or os.getenv("STOCK_ML_SCORES_PATH") or DEFAULT_SCORES_PATH)
    if not path.is_file():
        return []
    try:
        rows = _read_scores_file(path)
        if not rows:
            return []
        results = []
        is_focused = "focused" in str(path.name).lower() or len(rows) <= 50
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
                in_top10 = (
                    str(row.get("in_top_10_portfolio", "")).strip().lower() in ("true", "1", "yes")
                    or str(row.get("in_top_5_portfolio", "")).strip().lower() in ("true", "1", "yes")
                )
                results.append({
                    "ticker": ticker,
                    "in_scope": True,
                    "rank": int(row["ml_rank"]),
                    "score_0_100": score_0_100,
                    "rank_percentile": percentile,
                    "predicted_20d_excess_return": predicted,
                    "signal": cls["signal"],
                    "badge": cls["badge"],
                    "tier": cls["tier"],
                    "stance": cls["stance"],
                    "sector": row.get("sector", ""),
                    "in_top_5_portfolio": str(row.get("in_top_5_portfolio", "")).strip().lower() in ("true", "1", "yes"),
                    "in_top_10_portfolio": in_top10,
                    "is_focused_universe": is_focused,
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
    if not context.get("in_scope", True):
        sym = context.get("ticker", "This stock")
        return (
            "\n\n================ 20-SESSION ML SCORE ================\n"
            f"ML Status: OUT OF SCOPE for {sym}.\n"
            f"This ticker is outside the focused 25-stock ETF leaders quantitative model universe.\n"
            "Assigned baseline: 50.0/100, predicted 20-session excess return 0.00% (Market Perform / Neutral).\n"
            "The quantitative decision composite will rely 100% on deterministic 5-pillar market metrics.\n"
            "In ml_signal_assessment, report disposition as 'NEUTRAL' with reason "
            f"'{sym} is outside the focused ML model universe'.\n"
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
