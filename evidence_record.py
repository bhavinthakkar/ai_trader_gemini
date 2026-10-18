"""Auditable Evidence Record for RAG and LLM Analysis (Step 4.1).

Defines the explicit 10-field auditable evidence schema:
1. document_id: Unique deterministic content identifier.
2. ticker: Target instrument symbol.
3. source_type: Primary vs secondary source classification (SEC_FILING, COMPANY_IR, etc.).
4. source_url: Direct SEC EDGAR link, press release URI, or source feed accession.
5. publication_timestamp: ISO 8601 publication datetime.
6. available_timestamp: ISO 8601 market dissemination / availability datetime.
7. retrieval_timestamp: ISO 8601 RAG query retrieval datetime.
8. passage_or_excerpt: Exact extracted text chunk.
9. event_type: Standardized event classification (EARNINGS_RELEASE, GUIDANCE_UPDATE, etc.).
10. event_horizon: Standardized temporal horizon (IMMEDIATE_24H, SHORT_TERM_7D, etc.).

Supports strict point-in-time as-of querying for zero-lookahead historical testing.
"""

from __future__ import annotations

import datetime
import hashlib
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class SourceType(str, Enum):
    """Standardized source types with primary source prioritization."""
    SEC_FILING = "SEC_FILING"
    COMPANY_IR = "COMPANY_IR"
    EARNINGS_CALL_TRANSCRIPT = "EARNINGS_CALL_TRANSCRIPT"
    MACRO_RELEASE = "MACRO_RELEASE"
    ANALYST_RESEARCH = "ANALYST_RESEARCH"
    REPUTABLE_NEWS = "REPUTABLE_NEWS"
    GENERAL_NEWS = "GENERAL_NEWS"


class EventType(str, Enum):
    """Standardized corporate and macroeconomic event categories."""
    EARNINGS_RELEASE = "EARNINGS_RELEASE"
    GUIDANCE_UPDATE = "GUIDANCE_UPDATE"
    MERGER_ACQUISITION = "MERGER_ACQUISITION"
    REGULATORY_APPROVAL = "REGULATORY_APPROVAL"
    PRODUCT_LAUNCH = "PRODUCT_LAUNCH"
    MACRO_RATE_DECISION = "MACRO_RATE_DECISION"
    LITIGATION_LEGAL = "LITIGATION_LEGAL"
    CAPITAL_ALLOCATION = "CAPITAL_ALLOCATION"
    INSIDER_TRANSACTION = "INSIDER_TRANSACTION"
    GENERAL_BUSINESS = "GENERAL_BUSINESS"


class EventHorizon(str, Enum):
    """Standardized temporal event horizons."""
    IMMEDIATE_24H = "IMMEDIATE_24H"
    SHORT_TERM_7D = "SHORT_TERM_7D"
    MEDIUM_TERM_20D = "MEDIUM_TERM_20D"
    MULTI_YEAR_FUNDAMENTAL = "MULTI_YEAR_FUNDAMENTAL"


# Source reliability weights reflecting primary vs secondary source hierarchy
SOURCE_RELIABILITY_WEIGHTS: Dict[str, float] = {
    SourceType.SEC_FILING.value: 1.00,
    SourceType.MACRO_RELEASE.value: 1.00,
    SourceType.COMPANY_IR.value: 0.95,
    SourceType.EARNINGS_CALL_TRANSCRIPT.value: 0.90,
    SourceType.ANALYST_RESEARCH.value: 0.75,
    SourceType.REPUTABLE_NEWS.value: 0.70,
    SourceType.GENERAL_NEWS.value: 0.60,
}


def infer_source_type(source_name: str = "", document_type: str = "") -> SourceType:
    """Infer standardized SourceType from source name or document type string."""
    s = f"{source_name} {document_type}".lower()
    if any(k in s for k in ("sec", "edgar", "10-k", "10-q", "8-k", "form 4", "form-4", "13f")):
        return SourceType.SEC_FILING
    if any(k in s for k in ("fed", "fomc", "fred", "macro", "treasury", "cftc", "cot", "bls", "bea")):
        return SourceType.MACRO_RELEASE
    if any(k in s for k in ("investor relations", "ir", "press release", "pr", "company profile")):
        return SourceType.COMPANY_IR
    if any(k in s for k in ("transcript", "earnings call", "call takeaway")):
        return SourceType.EARNINGS_CALL_TRANSCRIPT
    if any(k in s for k in ("analyst", "target price", "bank action", "rating")):
        return SourceType.ANALYST_RESEARCH
    if any(k in s for k in ("reuters", "bloomberg", "wsj", "wall street journal", "ft", "financial times", "cnbc", "barron")):
        return SourceType.REPUTABLE_NEWS
    return SourceType.GENERAL_NEWS


def infer_event_type(text: str = "", document_type: str = "") -> EventType:
    """Infer standardized EventType from text passage and document type."""
    t = f"{text} {document_type}".lower()
    if any(k in t for k in ("earnings", "eps", "revenue growth", "quarterly result", "10-q", "10-k")):
        return EventType.EARNINGS_RELEASE
    if any(k in t for k in ("guidance", "outlook", "forecast", "target price", "analyst")):
        return EventType.GUIDANCE_UPDATE
    if any(k in t for k in ("rate decision", "fed funds", "fomc", "yield curve", "treasury", "inflation")):
        return EventType.MACRO_RATE_DECISION
    if any(k in t for k in ("insider", "form 4", "form-4", "share purchase", "share sale")):
        return EventType.INSIDER_TRANSACTION
    if any(k in t for k in ("merger", "acquisition", "buyout", "takeover", "acquire")):
        return EventType.MERGER_ACQUISITION
    if any(k in t for k in ("fda", "approval", "regulatory", "sec investigation", "doj", "patent")):
        return EventType.REGULATORY_APPROVAL
    if any(k in t for k in ("product launch", "chip", "platform", "release", "architecture")):
        return EventType.PRODUCT_LAUNCH
    if any(k in t for k in ("dividend", "buyback", "share repurchase", "capital allocation", "debt")):
        return EventType.CAPITAL_ALLOCATION
    if any(k in t for k in ("lawsuit", "litigation", "court", "settlement", "legal")):
        return EventType.LITIGATION_LEGAL
    return EventType.GENERAL_BUSINESS


def infer_event_horizon(published_at_str: str = "", time_horizon: str = "") -> EventHorizon:
    """Infer standardized EventHorizon from publication date and temporal horizon."""
    th = str(time_horizon).upper()
    if th == "HISTORICAL":
        return EventHorizon.MULTI_YEAR_FUNDAMENTAL
    if not published_at_str:
        return EventHorizon.SHORT_TERM_7D
    try:
        pub_dt = parse_iso_timestamp(published_at_str)
        now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        days = abs((now - pub_dt).days)
        if days <= 1:
            return EventHorizon.IMMEDIATE_24H
        elif days <= 7:
            return EventHorizon.SHORT_TERM_7D
        elif days <= 30:
            return EventHorizon.MEDIUM_TERM_20D
        else:
            return EventHorizon.MULTI_YEAR_FUNDAMENTAL
    except Exception:
        return EventHorizon.SHORT_TERM_7D


def parse_iso_timestamp(ts: Union[str, datetime.datetime, datetime.date, None]) -> datetime.datetime:
    """Parse string or date into UTC-normalized naive datetime for robust comparison."""
    if ts is None:
        return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    if isinstance(ts, datetime.datetime):
        if ts.tzinfo is not None:
            return ts.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        return ts
    if isinstance(ts, datetime.date):
        return datetime.datetime(ts.year, ts.month, ts.day)

    s = str(ts).strip()
    if not s or s.upper() == "N/A":
        return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)

    # Clean ISO string
    s = s.replace("Z", "+00:00")
    try:
        dt = datetime.datetime.fromisoformat(s)
        if dt.tzinfo is not None:
            return dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        return dt
    except ValueError:
        # Fallback to date prefix YYYY-MM-DD
        if len(s) >= 10 and s[4] == "-" and s[7] == "-":
            try:
                return datetime.datetime.strptime(s[:10], "%Y-%m-%d")
            except ValueError:
                pass
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


def format_iso_timestamp(dt: Union[datetime.datetime, datetime.date, str, None]) -> str:
    """Format any date-like input into standardized ISO 8601 string."""
    parsed = parse_iso_timestamp(dt)
    return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")


def generate_document_id(
    ticker: str,
    source_type: str,
    publication_timestamp: str,
    passage: str,
) -> str:
    """Generate deterministic, collision-resistant unique document identifier."""
    raw = f"{str(ticker).upper()}:{str(source_type).upper()}:{str(publication_timestamp)[:19]}:{str(passage).strip()[:150]}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"doc_{str(ticker).lower()}_{digest}"


def is_point_in_time(
    publication_timestamp: Union[str, datetime.datetime],
    available_timestamp: Union[str, datetime.datetime],
    as_of_timestamp: Union[str, datetime.datetime],
) -> bool:
    """Check whether a source document was publicly available at or before as_of_timestamp.

    Enforces the zero-lookahead rule:
        publication_timestamp <= available_timestamp <= as_of_timestamp
    """
    pub_dt = parse_iso_timestamp(publication_timestamp)
    avail_dt = parse_iso_timestamp(available_timestamp)
    as_of_dt = parse_iso_timestamp(as_of_timestamp)

    return pub_dt <= as_of_dt and avail_dt <= as_of_dt


@dataclass
class EvidenceRecord:
    """Auditable Evidence Record for RAG Source Passages."""
    document_id: str
    ticker: str
    source_type: str
    source_url: str
    publication_timestamp: str
    available_timestamp: str
    retrieval_timestamp: str
    passage_or_excerpt: str
    event_type: str
    event_horizon: str

    # Optional quality and attribution metrics
    reliability: float = 0.75
    importance: float = 0.75
    hybrid_score: float = 0.0
    sub_query: str = ""
    as_of_timestamp: Optional[str] = None

    def __post_init__(self):
        self.ticker = str(self.ticker).strip().upper()
        self.publication_timestamp = format_iso_timestamp(self.publication_timestamp)
        self.available_timestamp = format_iso_timestamp(self.available_timestamp)
        self.retrieval_timestamp = format_iso_timestamp(self.retrieval_timestamp)
        if self.as_of_timestamp:
            self.as_of_timestamp = format_iso_timestamp(self.as_of_timestamp)

        if not self.document_id:
            self.document_id = generate_document_id(
                self.ticker,
                self.source_type,
                self.publication_timestamp,
                self.passage_or_excerpt,
            )

        # Set default reliability if not explicitly provided
        if self.reliability == 0.75 and self.source_type in SOURCE_RELIABILITY_WEIGHTS:
            self.reliability = SOURCE_RELIABILITY_WEIGHTS[self.source_type]

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvidenceRecord:
        """Construct record from dictionary."""
        return cls(
            document_id=data.get("document_id", ""),
            ticker=data.get("ticker", ""),
            source_type=data.get("source_type", SourceType.GENERAL_NEWS.value),
            source_url=data.get("source_url", ""),
            publication_timestamp=data.get("publication_timestamp", ""),
            available_timestamp=data.get("available_timestamp", ""),
            retrieval_timestamp=data.get("retrieval_timestamp", ""),
            passage_or_excerpt=data.get("passage_or_excerpt", ""),
            event_type=data.get("event_type", EventType.GENERAL_BUSINESS.value),
            event_horizon=data.get("event_horizon", EventHorizon.SHORT_TERM_7D.value),
            reliability=float(data.get("reliability", 0.75)),
            importance=float(data.get("importance", 0.75)),
            hybrid_score=float(data.get("hybrid_score", 0.0)),
            sub_query=str(data.get("sub_query", "")),
            as_of_timestamp=data.get("as_of_timestamp"),
        )
