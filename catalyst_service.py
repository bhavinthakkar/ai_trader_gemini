"""Deterministic catalyst-reversion and catalyst-extension certificates.

News is the trigger; fundamentals are the filter. A headline never creates a
trade by itself -- it must coincide with a measurable dislocation between price
and the slow-moving valuation pillars.

Two certificates are recognised:

- ``DIP_BUY``        adverse catalyst, price discounted, valuation intact.
- ``EXTENSION_SELL`` favourable catalyst, price extended, valuation rich.

Direction is inferred from price action rather than headline tone, because the
news payload carries no sentiment field and tone classification on headlines is
unreliable. Everything here is arithmetic on supplied numbers: no model output
is trusted, and a certificate either qualifies or explains why it did not.
"""

from __future__ import annotations

import datetime
import re
from typing import Final, Iterable, Sequence

CATALYST_MAX_AGE_DAYS: Final[int] = 3

DIP_MIN_DROP_PCT_5D: Final[float] = 5.0
DIP_MAX_DROP_PCT_5D: Final[float] = 15.0
DIP_MIN_VALUATION_HISTORY: Final[float] = 60.0
DIP_MIN_PEER_VALUATION: Final[float] = 55.0
DIP_MIN_RSI: Final[float] = 30.0
DIP_MIN_RVOL: Final[float] = 1.2
DIP_MIN_COMPOSITE: Final[float] = 45.0

EXTENSION_MIN_RUNUP_PCT_5D: Final[float] = 5.0
EXTENSION_MIN_RSI: Final[float] = 70.0
EXTENSION_MAX_PEER_VALUATION: Final[float] = 45.0
EXTENSION_MAX_VALUATION_HISTORY: Final[float] = 45.0
EXTENSION_MIN_RANGE_PROXIMITY: Final[float] = 0.85
EXTENSION_MAX_COMPOSITE: Final[float] = 65.0
SHORT_MAX_COMPOSITE: Final[float] = 40.0

CATALYST_CONFIDENCE_BOOST: Final[float] = 0.15
CATALYST_CONFIDENCE_CEILING: Final[float] = 0.85

DIP_BUY: Final[str] = "DIP_BUY"
EXTENSION_SELL: Final[str] = "EXTENSION_SELL"
NO_CATALYST: Final[str] = "NONE"


def _to_float(value: object, default: float | None = None) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if result != result:  # NaN
        return default
    return result


def _parse_date(value: object) -> datetime.date | None:
    if not value:
        return None
    try:
        return datetime.datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def build_relevance_needles(symbol: str, company_name: str | None = None) -> list[str]:
    """Lowercase tokens that mark a news item as being about this company.

    Vendors return generic market commentary alongside ticker-specific articles,
    so a bare substring test is not enough: ``GE`` must not match ``GENERAL``,
    and ``BRK`` must not match ``BRK.B``'s peers by accident.
    """
    needles: list[str] = []
    bare = re.sub(r"[.\-_].*$", "", str(symbol or "")).strip().lower()
    if len(bare) >= 2:
        needles.append(bare)
    if company_name:
        cleaned = re.sub(r"[^a-z0-9 ]", " ", str(company_name).lower()).strip()
        if len(cleaned) >= 3:
            needles.append(cleaned)
            head = cleaned.split()[0]
            if len(head) >= 4:
                needles.append(head)
    seen: list[str] = []
    for needle in needles:
        if needle not in seen:
            seen.append(needle)
    return seen


def _mentions(text: str, needles: Sequence[str]) -> bool:
    lowered = text.lower()
    for needle in needles:
        if re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", lowered):
            return True
    return False


def find_fresh_catalysts(
    news_items: Iterable[dict],
    needles: Sequence[str],
    max_age_days: int = CATALYST_MAX_AGE_DAYS,
    today: datetime.date | None = None,
) -> list[dict]:
    """Symbol-relevant news published within ``max_age_days``.

    Undated items are treated as stale rather than assumed current, so a feed
    that omits timestamps cannot manufacture a catalyst.
    """
    if not needles:
        return []
    reference = today or datetime.date.today()
    fresh: list[dict] = []
    for item in news_items or []:
        if not isinstance(item, dict):
            continue
        published = _parse_date(item.get("published_at"))
        if published is None:
            continue
        age_days = (reference - published).days
        if age_days < 0 or age_days > max_age_days:
            continue
        title = str(item.get("title") or "")
        summary = str(item.get("summary") or "")
        if not _mentions(f"{title} {summary}", needles):
            continue
        fresh.append({**item, "age_days": age_days})
    fresh.sort(key=lambda entry: entry["age_days"])
    return fresh


def _range_proximity(technical: dict) -> float | None:
    price = _to_float(technical.get("current_price"))
    high = _to_float(technical.get("high_20d"))
    low = _to_float(technical.get("low_20d"))
    if price is None or high is None or low is None or high <= low:
        return None
    return (price - low) / (high - low)


def _dip_reasons(metrics: dict) -> list[str]:
    failed: list[str] = []
    if (metrics["change_5d_pct"] or 0.0) > -DIP_MIN_DROP_PCT_5D:
        failed.append(
            f"5-day move {metrics['change_5d_pct']}% is not a dip of at least "
            f"-{DIP_MIN_DROP_PCT_5D}%"
        )
    if metrics["valuation_history_score"] is None or metrics["valuation_history_score"] < DIP_MIN_VALUATION_HISTORY:
        failed.append(
            f"valuation history pillar {metrics['valuation_history_score']} < {DIP_MIN_VALUATION_HISTORY}"
        )
    if metrics["peer_valuation_score"] is None or metrics["peer_valuation_score"] < DIP_MIN_PEER_VALUATION:
        failed.append(
            f"peer valuation pillar {metrics['peer_valuation_score']} < {DIP_MIN_PEER_VALUATION}"
        )
    if metrics["rsi14"] is None or metrics["rsi14"] < DIP_MIN_RSI:
        failed.append(f"RSI14 {metrics['rsi14']} < {DIP_MIN_RSI}: still a falling knife, not an exhausted dip")
    if metrics["rvol_20d"] is None or metrics["rvol_20d"] < DIP_MIN_RVOL:
        failed.append(
            f"RVOL {metrics['rvol_20d']} < {DIP_MIN_RVOL}: no volume confirmation the catalyst was absorbed"
        )
    if metrics["composite"] is None or metrics["composite"] < DIP_MIN_COMPOSITE:
        failed.append(f"composite {metrics['composite']} < catalyst floor {DIP_MIN_COMPOSITE}")
    if (metrics["change_5d_pct"] or 0.0) < -DIP_MAX_DROP_PCT_5D:
        failed.append(
            f"5-day move {metrics['change_5d_pct']}% is a collapse worse than "
            f"-{DIP_MAX_DROP_PCT_5D}%: treat as a broken thesis, not a discount"
        )
    return failed


def _extension_reasons(metrics: dict) -> list[str]:
    failed: list[str] = []
    if (metrics["change_5d_pct"] or 0.0) < EXTENSION_MIN_RUNUP_PCT_5D:
        failed.append(
            f"5-day move {metrics['change_5d_pct']}% is not a run-up of at least "
            f"+{EXTENSION_MIN_RUNUP_PCT_5D}%"
        )
    if metrics["rsi14"] is None or metrics["rsi14"] < EXTENSION_MIN_RSI:
        failed.append(f"RSI14 {metrics['rsi14']} < {EXTENSION_MIN_RSI}: not overextended")
    if metrics["peer_valuation_score"] is not None and metrics["peer_valuation_score"] > EXTENSION_MAX_PEER_VALUATION:
        failed.append(
            f"peer valuation pillar {metrics['peer_valuation_score']} is not rich enough to book "
            f"profit (> {EXTENSION_MAX_PEER_VALUATION})"
        )
    if (
        metrics["valuation_history_score"] is not None
        and metrics["valuation_history_score"] > EXTENSION_MAX_VALUATION_HISTORY
    ):
        failed.append(
            f"valuation history pillar {metrics['valuation_history_score']} is not rich enough to book "
            f"profit (> {EXTENSION_MAX_VALUATION_HISTORY})"
        )
    if metrics["range_proximity"] is None or metrics["range_proximity"] < EXTENSION_MIN_RANGE_PROXIMITY:
        failed.append(
            f"price sits at {metrics['range_proximity']} of the 20-day range, below the "
            f"{EXTENSION_MIN_RANGE_PROXIMITY} required for an extended high"
        )
    return failed


def evaluate_catalyst_setup(
    technical_data: dict,
    deterministic_scores: dict,
    news_items: Iterable[dict],
    needles: Sequence[str],
    today: datetime.date | None = None,
) -> dict:
    """Classify a symbol as a certified catalyst setup, or explain the refusal."""
    technical = technical_data if isinstance(technical_data, dict) else {}
    scores = deterministic_scores if isinstance(deterministic_scores, dict) else {}

    price = _to_float(technical.get("current_price"))
    ema50 = _to_float(technical.get("ema50"))
    metrics = {
        "change_5d_pct": _to_float(technical.get("change_5d_pct")),
        "rsi14": _to_float(technical.get("rsi14")),
        "rvol_20d": _to_float(technical.get("rvol_20d")),
        "composite": _to_float(scores.get("composite_quantitative_score")),
        "valuation_history_score": _to_float(scores.get("valuation_history_score")),
        "peer_valuation_score": _to_float(scores.get("peer_valuation_score")),
        "range_proximity": _range_proximity(technical),
        "above_ema50": bool(price is not None and ema50 is not None and price > ema50),
    }

    fresh = find_fresh_catalysts(news_items, needles, today=today)
    reasons: list[str] = []
    result: dict = {
        "direction": NO_CATALYST,
        "qualified": False,
        "reasons": reasons,
        "metrics": metrics,
        "catalysts": [
            {"title": str(item.get("title") or ""), "age_days": item["age_days"]}
            for item in fresh[:3]
        ],
        "catalyst_count": len(fresh),
    }

    if not fresh:
        result["reasons"] = [
            f"no symbol-relevant catalyst published within {CATALYST_MAX_AGE_DAYS} day(s)"
        ]
        return result

    dip_failures = _dip_reasons(metrics)
    if not dip_failures:
        result.update(direction=DIP_BUY, qualified=True)
        result["reasons"] = [
            f"adverse catalyst {fresh[0]['age_days']}d old with a {metrics['change_5d_pct']}% "
            f"discount, valuation pillars intact "
            f"(history {metrics['valuation_history_score']}, peer {metrics['peer_valuation_score']})"
        ]
        return result

    extension_failures = _extension_reasons(metrics)
    if not extension_failures:
        result.update(direction=EXTENSION_SELL, qualified=True)
        result["reasons"] = [
            f"favourable catalyst {fresh[0]['age_days']}d old with a {metrics['change_5d_pct']}% "
            f"run-up into an extended, richly valued high"
        ]
        return result

    result["reasons"] = [f"dip: {reason}" for reason in dip_failures[:3]] + [
        f"extension: {reason}" for reason in extension_failures[:2]
    ]
    return result
