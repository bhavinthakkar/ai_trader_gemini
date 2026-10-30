"""Load fresh 20-session ML scores produced by the sibling stock-ml repository.

Implements Step 3.5 ML Export, Freshness, and Fallback Behavior:
- Source path resolution with priority order (custom -> focused 25 -> broad S&P 500 -> LambdaMART production).
- Strict CSV schema and model candidate verification.
- Session-aligned freshness checking against prediction date and maximum staleness.
- Explicit model eligibility contract:
    - In-scope tickers: {"model_status": "IN_SCOPE", "ml_score": score, "ml_weight": weight, ...}
    - Out-of-scope tickers: {"model_status": "OUT_OF_SCOPE", "ml_score": None, "ml_weight": 0.0, ...}
    - Stale / invalid scores: {"model_status": "STALE_SCORE", "ml_score": None, "ml_weight": 0.0, ...}
- Never injects an artificial 50.0 score for out-of-scope or stale instruments.
"""

from __future__ import annotations

import csv
import os
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

ROOT = Path(__file__).resolve().parent

# Check potential locations for stock-ml data (hyphenated and underscore variants)
STOCK_ML_DATA_CANDIDATES = [
    ROOT.parent / "stock-ml" / "data",
    ROOT.parent / "stock_ml" / "data",
    ROOT / "data",
]
STOCK_ML_DATA = next((p for p in STOCK_ML_DATA_CANDIDATES if p.exists()), STOCK_ML_DATA_CANDIDATES[0])

FOCUSED_SCORES_PATH = STOCK_ML_DATA / "xgb_focused_current_scores.csv"
BROAD_SCORES_PATH = STOCK_ML_DATA / "xgb_current_scores_1m.csv"

# Prefer the new focused 25-stock ETF leaders model, falling back to broad S&P 500
DEFAULT_SCORES_PATH = FOCUSED_SCORES_PATH if FOCUSED_SCORES_PATH.exists() else BROAD_SCORES_PATH


def classify_ml_signal(score_0_100: float | None, rank_percentile: float | None = None) -> dict:
    """
    Classify a 0-100 ML percentile score into a standardized standalone indicator.

    Semantic Contract (Rule ML-SEM-01):
      - score_0_100 is strictly an ordinal cross-sectional percentile ranking score
        within the active peer group on decision date t.
      - A score of 80 indicates the top 20th percentile of expected relative outperformance
        among peers. It does NOT represent an expected return of +80%, nor an 80% probability of rising.
      - A score of 50 indicates the peer cohort median. It does NOT imply zero forward excess return.
      - Directional trade execution is governed by the deterministic quantitative engine (Q_quant >= 50.0).

    Tiers:
      >= 80: STRONG OUTPERFORM (Top Quintile / High Conviction Alpha)
      >= 60: OUTPERFORM (Quintile 2 / Favorable Alpha)
      >= 40: MARKET PERFORM (Quintile 3 / Cohort Median)
      >= 20: UNDERPERFORM (Quintile 4 / Unfavorable Alpha)
      <  20: STRONG UNDERPERFORM (Bottom Quintile / Relative Lag)
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


def get_out_of_scope_context(
    symbol: str,
    status: str = "OUT_OF_SCOPE",
    message: str | None = None,
    universe_size: int = 0,
) -> dict:
    """Return explicit out-of-scope / fallback model eligibility context.

    Enforces the explicit contract:
    {
        "model_status": "OUT_OF_SCOPE",  # or "STALE_SCORE", "SCHEMA_MISMATCH", "FILE_NOT_FOUND"
        "ml_score": None,
        "ml_weight": 0.0,
    }

    Never assigns 50.0 to an out-of-scope ticker. A null score and zero weight
    prevent accidental ML contributions to unmodeled stocks.
    """
    wanted = str(symbol or "").strip().upper()
    default_msg = (
        f"'{wanted}' is outside the active ML model universe."
        if status == "OUT_OF_SCOPE"
        else f"ML score unavailable for '{wanted}' ({status})."
    )
    msg = message or default_msg

    badge_map = {
        "OUT_OF_SCOPE": "⚪ OUT OF SCOPE",
        "STALE_SCORE": "⚪ STALE SCORE",
        "SCHEMA_MISMATCH": "⚪ SCHEMA MISMATCH",
        "FILE_NOT_FOUND": "⚪ NO MODEL DATA",
        "LOOKAHEAD_VIOLATION": "⚪ INVALID SESSION",
    }

    tier_map = {
        "OUT_OF_SCOPE": "Outside Model Universe",
        "STALE_SCORE": "ML Score Stale",
        "SCHEMA_MISMATCH": "ML Schema Mismatch",
        "FILE_NOT_FOUND": "ML Data Absent",
        "LOOKAHEAD_VIOLATION": "Session Lookahead",
    }

    return {
        "ticker": wanted,
        "model_status": status,
        "status": status,
        "in_scope": False,
        "ml_score": None,
        "ml_weight": 0.0,
        "score_0_100": None,
        "rank_percentile": None,
        "raw_score_0_100": None,
        "predicted_20d_excess_return": None,
        "rank": None,
        "universe_size": universe_size,
        "score_date": None,
        "age_days": None,
        "model_name": None,
        "model_as_of": None,
        "target_horizon_sessions": 20,
        "signal": "OUT_OF_SCOPE" if status == "OUT_OF_SCOPE" else "UNAVAILABLE",
        "badge": badge_map.get(status, "⚪ UNAVAILABLE"),
        "tier": tier_map.get(status, f"ML {status}"),
        "stance": "UNAVAILABLE",
        "sector": "Other",
        "in_top_5_portfolio": False,
        "in_top_10_portfolio": False,
        "rank_fundamentals": None,
        "rank_technical": None,
        "is_focused_universe": False,
        "message": msg,
    }


def verify_file_schema(headers: list[str]) -> Tuple[bool, Optional[str]]:
    """Verify that a candidate score CSV file contains all required schema fields."""
    hdr_set = {h.strip() for h in headers if h}
    if "ticker" not in hdr_set:
        return False, "Missing 'ticker' column in header"

    date_col = next((c for c in ("score_date", "decision_date", "date") if c in hdr_set), None)
    if not date_col:
        return False, "Missing date column (expected score_date, decision_date, or date)"

    pct_col = next(
        (c for c in ("ml_rank_percentile", "rank_percentile", "ml_percentile_score", "score_0_100") if c in hdr_set),
        None,
    )
    if not pct_col:
        return False, "Missing ranking percentile column"

    return True, None


def verify_session_freshness(
    score_date: date,
    expected_session_date: Union[date, str, None] = None,
    max_age_days: int = 7,
) -> Tuple[bool, str, int]:
    """Verify score freshness relative to the target prediction session.

    Returns:
        (is_fresh, status_code, age_days)
    """
    if expected_session_date is None:
        ref_date = date.today()
    elif isinstance(expected_session_date, str):
        ref_date = date.fromisoformat(expected_session_date[:10])
    else:
        ref_date = expected_session_date

    age_days = (ref_date - score_date).days

    if age_days < 0:
        return False, "LOOKAHEAD_VIOLATION", age_days
    if age_days > max_age_days:
        return False, "STALE_SCORE", age_days
    return True, "FRESH", age_days


def _resolve_score_file_paths(custom_path: Union[str, os.PathLike, None] = None) -> List[Path]:
    """Resolve candidate score files in strict priority order."""
    candidates: List[Path] = []

    # Priority 1: User-specified custom path
    if custom_path:
        candidates.append(Path(custom_path))

    # Priority 2: Environment variable path
    env_path = os.getenv("STOCK_ML_SCORES_PATH")
    if env_path:
        candidates.append(Path(env_path))

    # Priority 3: Preferred focused 25-stock ETF leaders model
    candidates.extend([
        ROOT.parent / "stock-ml" / "data" / "xgb_focused_current_scores.csv",
        ROOT.parent / "stock_ml" / "data" / "xgb_focused_current_scores.csv",
        ROOT / "data" / "xgb_focused_current_scores.csv",
    ])

    # Priority 4: Fallback broad S&P 500 model
    candidates.extend([
        ROOT.parent / "stock-ml" / "data" / "xgb_current_scores_1m.csv",
        ROOT.parent / "stock_ml" / "data" / "xgb_current_scores_1m.csv",
        ROOT / "data" / "xgb_current_scores_1m.csv",
    ])

    # Priority 5: Production LambdaMART walk-forward output
    candidates.extend([
        ROOT / "data" / "xgb_lambdamart_walk_forward_predictions.csv",
        ROOT.parent / "stock-ml" / "data" / "xgb_lambdamart_walk_forward_predictions.csv",
    ])

    # Remove duplicates preserving priority order
    seen = set()
    result = []
    for c in candidates:
        norm = str(c.resolve()) if c.exists() else str(c)
        if norm not in seen:
            seen.add(norm)
            result.append(c)
    return result


def _read_scores_file(path: Path) -> List[dict]:
    """Read CSV rows safely from disk."""
    if not path.is_file():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def get_ml_context(
    symbol: str,
    scores_path: Union[str, os.PathLike, None] = None,
    expected_session_date: Union[date, str, None] = None,
    max_age_days: int = int(os.getenv("STOCK_ML_MAX_AGE_DAYS", "30")),
    hybrid_weight: float = 0.20,
) -> dict:
    """Return a fresh point-in-time 20-session score, or explicit out-of-scope context when outside universe.

    Guarantees:
      - Validates ticker presence and finite score.
      - Verifies file schema and 20-session horizon.
      - Enforces freshness against expected_session_date (or today).
      - Returns {"model_status": "OUT_OF_SCOPE", "ml_score": None, "ml_weight": 0.0} when not in universe.
      - Never assigns neutral 50.0 to out-of-scope instruments.
    """
    wanted = str(symbol or "").strip().upper()
    if not wanted:
        return get_out_of_scope_context("", status="OUT_OF_SCOPE", message="Empty symbol provided.")

    base_sym = wanted.split(".")[0]
    search_paths = _resolve_score_file_paths(scores_path)
    existing_paths = [p for p in search_paths if p.is_file()]

    if not existing_paths:
        return get_out_of_scope_context(
            wanted,
            status="FILE_NOT_FOUND",
            message="No active ML score files found in configured repository paths.",
        )

    last_status = "OUT_OF_SCOPE"
    last_message = f"'{wanted}' is outside the active ML model universe."
    universe_size_candidate = 0

    for p in existing_paths:
        rows = _read_scores_file(p)
        if not rows:
            continue

        headers = list(rows[0].keys())
        is_schema_ok, schema_err = verify_file_schema(headers)
        if not is_schema_ok:
            last_status = "SCHEMA_MISMATCH"
            last_message = f"Schema verification failed for {p.name}: {schema_err}"
            continue

        universe_size_candidate = len(rows)
        row = next(
            (item for item in rows if item.get("ticker", "").strip().upper() in (wanted, base_sym)),
            None,
        )
        if row is None:
            # Ticker not in this candidate file, try next fallback in priority order
            continue

        # Ticker found in file: parse and validate row attributes
        try:
            date_raw = row.get("score_date") or row.get("decision_date") or row.get("date")
            if not date_raw:
                last_status = "SCHEMA_MISMATCH"
                last_message = f"Row for {wanted} in {p.name} lacks valid date column."
                continue
            score_date = date.fromisoformat(str(date_raw)[:10])

            is_fresh, freshness_status, age_days = verify_session_freshness(
                score_date,
                expected_session_date=expected_session_date,
                max_age_days=max_age_days,
            )
            if not is_fresh:
                return get_out_of_scope_context(
                    wanted,
                    status=freshness_status,
                    message=(
                        f"ML score for {wanted} in {p.name} as of {score_date.isoformat()} is invalid "
                        f"({freshness_status}: age {age_days}d vs max {max_age_days}d)."
                    ),
                    universe_size=len(rows),
                )

            # Target horizon verification (must be 20 sessions)
            horizon = int(row.get("target_horizon_sessions", 20) or 20)
            if horizon != 20:
                return get_out_of_scope_context(
                    wanted,
                    status="SCHEMA_MISMATCH",
                    message=f"Model target horizon {horizon} sessions does not match required 20 sessions.",
                    universe_size=len(rows),
                )

            # Percentile extraction and validation
            pct_val = (
                row.get("ml_rank_percentile")
                or row.get("rank_percentile")
                or row.get("ml_percentile_score")
            )
            if pct_val is None and row.get("score_0_100") is not None:
                percentile = float(row["score_0_100"]) / 100.0
            elif pct_val is not None:
                percentile = float(pct_val)
            else:
                return get_out_of_scope_context(
                    wanted,
                    status="SCHEMA_MISMATCH",
                    message=f"Missing percentile score for {wanted} in {p.name}.",
                    universe_size=len(rows),
                )

            if not 0.0 <= percentile <= 1.0:
                return get_out_of_scope_context(
                    wanted,
                    status="SCHEMA_MISMATCH",
                    message=f"Percentile {percentile} outside valid [0, 1] range.",
                    universe_size=len(rows),
                )

            score_0_100 = round(percentile * 100.0, 2)
            cls = classify_ml_signal(score_0_100)

            # Predicted return extraction
            pred_raw = (
                row.get("predicted_20d_excess_return")
                or row.get("pred_pit_core_rank_1m")
                or row.get("predicted_relevance")
                or row.get("label_excess_20d")
            )
            predicted = float(pred_raw) if pred_raw is not None else 0.0

            rank_raw = row.get("ml_rank") or row.get("rank")
            rank = int(float(rank_raw)) if rank_raw is not None else 1

            is_focused = "focused" in str(p.name).lower() or len(rows) <= 50

            in_top10 = (
                str(row.get("in_top_10_portfolio", "")).strip().lower() in ("true", "1", "yes")
                or str(row.get("in_top_5_portfolio", "")).strip().lower() in ("true", "1", "yes")
            )

            # SUCCESS: Fully verified in-scope prediction
            return {
                "ticker": wanted,
                "model_status": "IN_SCOPE",
                "in_scope": True,
                "ml_score": score_0_100,
                "ml_weight": hybrid_weight,
                "score_0_100": score_0_100,
                "raw_score_0_100": score_0_100,
                "rank_percentile": percentile,
                "predicted_20d_excess_return": predicted,
                "rank": rank,
                "universe_size": len(rows),
                "score_date": score_date.isoformat(),
                "age_days": age_days,
                "model_name": row.get("ml_model_name", "xgb_lambdamart_production"),
                "model_as_of": row.get("model_as_of", score_date.isoformat()),
                "target_horizon_sessions": 20,
                "signal": cls["signal"],
                "badge": cls["badge"],
                "tier": cls["tier"],
                "stance": cls["stance"],
                "sector": row.get("sector", "Other"),
                "in_top_5_portfolio": str(row.get("in_top_5_portfolio", "")).strip().lower() in ("true", "1", "yes"),
                "in_top_10_portfolio": in_top10,
                "is_focused_universe": is_focused,
                "rank_fundamentals": (
                    int(row["rank_pit_fundamentals_rank_1m"])
                    if row.get("rank_pit_fundamentals_rank_1m")
                    and str(row["rank_pit_fundamentals_rank_1m"]).strip().isdigit()
                    else None
                ),
                "rank_technical": (
                    int(row["rank_pit_technical_rank_1m"])
                    if row.get("rank_pit_technical_rank_1m")
                    and str(row["rank_pit_technical_rank_1m"]).strip().isdigit()
                    else None
                ),
                "message": f"Fresh 20-session score for {wanted}: rank #{rank} of {len(rows)}.",
            }
        except (ValueError, KeyError, TypeError) as exc:
            last_status = "SCHEMA_MISMATCH"
            last_message = f"Data parsing exception for {wanted} in {p.name}: {exc}"
            continue

    # Ticker was not found in any valid active model file
    return get_out_of_scope_context(
        wanted,
        status=last_status,
        message=last_message,
        universe_size=universe_size_candidate,
    )


def get_all_ml_scores(
    scores_path: Union[str, os.PathLike, None] = None,
    expected_session_date: Union[date, str, None] = None,
    max_age_days: int = int(os.getenv("STOCK_ML_MAX_AGE_DAYS", "30")),
) -> List[dict]:
    """Return all scored universe stocks ranked by point-in-time ML model."""
    search_paths = _resolve_score_file_paths(scores_path)
    existing_paths = [p for p in search_paths if p.is_file()]
    if not existing_paths:
        return []

    path = existing_paths[0]
    rows = _read_scores_file(path)
    if not rows:
        return []

    is_schema_ok, _ = verify_file_schema(list(rows[0].keys()))
    if not is_schema_ok:
        return []

    results = []
    is_focused = "focused" in str(path.name).lower() or len(rows) <= 50

    for row in rows:
        ticker = row.get("ticker", "").strip().upper()
        if not ticker:
            continue
        try:
            date_raw = row.get("score_date") or row.get("decision_date") or row.get("date")
            if not date_raw:
                continue
            score_date = date.fromisoformat(str(date_raw)[:10])

            is_fresh, _, age_days = verify_session_freshness(
                score_date,
                expected_session_date=expected_session_date,
                max_age_days=max_age_days,
            )
            if not is_fresh or int(row.get("target_horizon_sessions", 20) or 20) != 20:
                continue

            pct_val = (
                row.get("ml_rank_percentile")
                or row.get("rank_percentile")
                or row.get("ml_percentile_score")
            )
            if pct_val is None and row.get("score_0_100") is not None:
                percentile = float(row["score_0_100"]) / 100.0
            elif pct_val is not None:
                percentile = float(pct_val)
            else:
                continue

            if not 0.0 <= percentile <= 1.0:
                continue

            score_0_100 = round(percentile * 100.0, 2)
            cls = classify_ml_signal(score_0_100)
            pred_raw = (
                row.get("predicted_20d_excess_return")
                or row.get("pred_pit_core_rank_1m")
                or row.get("predicted_relevance")
                or row.get("label_excess_20d")
            )
            predicted = float(pred_raw) if pred_raw is not None else 0.0

            rank_raw = row.get("ml_rank") or row.get("rank")
            rank = int(float(rank_raw)) if rank_raw is not None else len(results) + 1

            in_top10 = (
                str(row.get("in_top_10_portfolio", "")).strip().lower() in ("true", "1", "yes")
                or str(row.get("in_top_5_portfolio", "")).strip().lower() in ("true", "1", "yes")
            )

            results.append({
                "ticker": ticker,
                "model_status": "IN_SCOPE",
                "in_scope": True,
                "rank": rank,
                "ml_score": score_0_100,
                "ml_weight": 0.20,
                "score_0_100": score_0_100,
                "rank_percentile": percentile,
                "predicted_20d_excess_return": predicted,
                "signal": cls["signal"],
                "badge": cls["badge"],
                "tier": cls["tier"],
                "stance": cls["stance"],
                "sector": row.get("sector", "Other"),
                "in_top_5_portfolio": str(row.get("in_top_5_portfolio", "")).strip().lower() in ("true", "1", "yes"),
                "in_top_10_portfolio": in_top10,
                "is_focused_universe": is_focused,
                "score_date": score_date.isoformat(),
                "age_days": age_days,
                "model_name": row.get("ml_model_name", "xgb_lambdamart_production"),
                "universe_size": len(rows),
            })
        except (ValueError, KeyError, TypeError):
            continue

    results.sort(key=lambda x: x["rank"])
    return results


def format_ml_context(context: dict | None) -> str:
    """Format ML signal context for prompt payload assembly."""
    if not context:
        return (
            "\n\n================ 20-SESSION ML SCORE ================\n"
            "No fresh one-month point-in-time ML score is available. Do not invent one; "
            "the hybrid composite will use the deterministic market score alone (ml_weight = 0.0%).\n"
        )

    status = context.get("model_status", "IN_SCOPE" if context.get("in_scope") else "OUT_OF_SCOPE")
    if status != "IN_SCOPE":
        sym = context.get("ticker", "This stock")
        msg = context.get("message", f"{sym} is outside the active ML model universe.")
        return (
            "\n\n================ 20-SESSION ML SCORE ================\n"
            f"ML Status: {status} for {sym}.\n"
            f"Reason: {msg}\n"
            f"Model eligibility: model_status={status}, ml_score=None, ml_weight=0.0%.\n"
            "The decision composite relies 100% on deterministic 5-pillar quantitative scores.\n"
            "In ml_signal_assessment, report disposition as 'OUT_OF_SCOPE' or 'UNAVAILABLE' "
            f"with reason '{msg}'. Do not invent an ML score.\n"
        )

    return (
        "\n\n================ 20-SESSION ML SCORE ================\n"
        f"ML score: {context['score_0_100']:.1f}/100 (rank #{context['rank']} of "
        f"{context['universe_size']}, percentile {context['rank_percentile']:.3f}); "
        f"predicted 20-session excess return {context['predicted_20d_excess_return'] * 100:+.2f}%. "
        f"Score date {context['score_date']}; model {context['model_name']} "
        f"(trained as of {context['model_as_of']}).\n"
        f"Model eligibility: model_status=IN_SCOPE, ml_score={context['score_0_100']:.1f}, "
        f"ml_weight={context.get('ml_weight', 0.20)*100:.0f}%.\n"
        "Compare this ML signal with your independent 20-session relative forecast, report the "
        "outlook in forecast_20d, and classify agreement in ml_signal_assessment. Keep the forecast "
        "separate from the trade action. Do not treat the prediction "
        "as a probability or a guaranteed return.\n"
    )
