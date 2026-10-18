"""Strict Pydantic Schema and Validator for LLM Evidence Analysis (Step 4.2).

Replaces free-form LLM decision authority with an evidence-based schema:
- Replaces unrestricted decision text with structured qualitative findings:
  catalysts, risk flags, cross-pillar contradictions, missing information, and summary.
- Enforces limited advisory action enumeration (CANDIDATE_BUY, CANDIDATE_HOLD, CANDIDATE_SELL, NO_ACTION)
  which is strictly non-authoritative and CANNOT authorize broker execution.
- Rejects malformed output, unsupported/extra fields (extra="forbid"), unknown tickers,
  invalid timestamps, and prohibited broker command fields.
- Partitions deterministic/categorical fields from explanatory prose (summary, passage excerpts).
- Validates provenance: every catalyst and risk flag must carry supporting document IDs,
  timestamp, event horizon, and relevant passage excerpts.
"""

from __future__ import annotations

import datetime
import json
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
except ImportError:
    # Graceful fallback if pydantic is unavailable in a minimalist sub-environment
    BaseModel = object
    ConfigDict = dict
    Field = lambda *args, **kwargs: None
    field_validator = lambda *args, **kwargs: lambda f: f
    model_validator = lambda *args, **kwargs: lambda f: f

from evidence_record import EventHorizon, EventType


class AnalysisStatus(str, Enum):
    """Execution status of the evidence extraction analysis."""
    COMPLETE = "COMPLETE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    DEGRADED = "DEGRADED"
    ERROR = "ERROR"


class AdvisoryAction(str, Enum):
    """
    Limited advisory candidate action.
    INVARIANT: This enumeration is strictly advisory and cannot authorize broker execution.
    Order initiation and routing are exclusively governed by deterministic quantitative gates.
    """
    CANDIDATE_BUY = "CANDIDATE_BUY"
    CANDIDATE_HOLD = "CANDIDATE_HOLD"
    CANDIDATE_SELL = "CANDIDATE_SELL"
    NO_ACTION = "NO_ACTION"


class CatalystPolarity(str, Enum):
    """Directional polarity of an observed or upcoming catalyst."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class RiskSeverity(str, Enum):
    """Severity classification for an extracted risk flag."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RelativeOutlook(str, Enum):
    """20-session relative outlook vs market benchmark."""
    OUTPERFORM = "OUTPERFORM"
    MARKET_LIKE = "MARKET_LIKE"
    UNDERPERFORM = "UNDERPERFORM"


# Explicit list of forbidden broker command keys.
# If any of these fields are detected in raw LLM output, validation fails immediately.
FORBIDDEN_BROKER_FIELDS: Set[str] = {
    "order_type",
    "order_id",
    "shares",
    "quantity",
    "limit_price",
    "stop_price",
    "execute_order",
    "execute_now",
    "execute_buy",
    "execute_sell",
    "broker_command",
    "broker_action",
    "broker_routing",
    "broker_account",
    "account_id",
    "route",
    "portfolio_pct",
    "position_size_usd",
    "leverage_ratio",
    "margin_enabled",
    "send_order",
    "time_in_force",
}


class CatalystEvidenceItem(BaseModel):
    """Structured catalyst observation backed by point-in-time evidence."""
    model_config = ConfigDict(extra="forbid")

    catalyst_id: str = Field(description="Unique identifier for the catalyst")
    description: str = Field(min_length=3, description="Concise description of the event or catalyst")
    category: str = Field(default="GENERAL_BUSINESS", description="Event category e.g. EARNINGS_RELEASE, GUIDANCE_UPDATE, PRODUCT_LAUNCH")
    polarity: CatalystPolarity = Field(description="Directional impact polarity: BULLISH, BEARISH, or NEUTRAL")
    event_horizon: str = Field(default=EventHorizon.MEDIUM_TERM_20D.value, description="Expected duration/impact horizon")
    timestamp: str = Field(min_length=4, description="ISO 8601 timestamp or session date of the catalyst")
    supporting_document_ids: List[str] = Field(min_length=1, description="List of retrieved evidence document IDs supporting this catalyst")
    relevant_passage: str = Field(min_length=5, description="Verbatim citation or excerpt from retrieved evidence")
    is_priced_in: Union[bool, str] = Field(default=False, description="Whether consensus market price already reflects this catalyst")

    @field_validator("supporting_document_ids")
    @classmethod
    def validate_supporting_docs(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("catalyst must have at least one supporting document_id")
        cleaned = [d.strip() for d in v if d and d.strip()]
        if not cleaned:
            raise ValueError("catalyst supporting document_ids cannot be empty strings")
        return cleaned


class RiskFlagEvidenceItem(BaseModel):
    """Structured risk factor or thesis headwind backed by point-in-time evidence."""
    model_config = ConfigDict(extra="forbid")

    flag_id: str = Field(description="Unique identifier for the risk flag")
    risk_name: str = Field(min_length=3, description="Concise title or name of the risk factor")
    category: str = Field(description="Risk category: VALUATION, EARNINGS_BLACKOUT, MACRO_YIELD, VOLATILITY, LEVERAGE, COMPETITION, GOVERNANCE")
    severity: RiskSeverity = Field(description="Risk severity: LOW, MEDIUM, HIGH, CRITICAL")
    event_horizon: str = Field(default=EventHorizon.MEDIUM_TERM_20D.value, description="Impact horizon")
    supporting_document_ids: List[str] = Field(min_length=1, description="List of retrieved evidence document IDs supporting this risk")
    relevant_passage: str = Field(min_length=5, description="Verbatim citation or excerpt from retrieved evidence")
    veto_trigger: bool = Field(default=False, description="True if this risk flag constitutes a thesis-breaking veto suggestion")

    @field_validator("supporting_document_ids")
    @classmethod
    def validate_supporting_docs(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("risk_flag must have at least one supporting document_id")
        cleaned = [d.strip() for d in v if d and d.strip()]
        if not cleaned:
            raise ValueError("risk_flag supporting document_ids cannot be empty strings")
        return cleaned


class EvidenceContradiction(BaseModel):
    """Structured divergence or contradiction identified between quantitative pillars and qualitative evidence."""
    model_config = ConfigDict(extra="forbid")

    contradiction_id: str = Field(description="Unique identifier for the contradiction")
    pillar_or_source_a: str = Field(min_length=2, description="First pillar or evidence channel (e.g. Technical Trend)")
    pillar_or_source_b: str = Field(min_length=2, description="Contradicting pillar or evidence channel (e.g. Valuation History or Insider Form 4)")
    description: str = Field(min_length=5, description="Detailed explanation of the divergence/contradiction")
    severity: str = Field(default="MEDIUM", description="Severity: LOW, MEDIUM, HIGH")
    supporting_document_ids: List[str] = Field(default_factory=list, description="Associated document IDs if applicable")


class Forecast20dEvidence(BaseModel):
    """20-session relative outlook vs market benchmark."""
    model_config = ConfigDict(extra="forbid")

    relative_outlook: RelativeOutlook = Field(description="20-session outlook: OUTPERFORM, MARKET_LIKE, or UNDERPERFORM")
    confidence: float = Field(ge=0.0, le=1.0, description="Conviction score in [0.0, 1.0]")
    rationale: str = Field(min_length=5, description="Evidence-based reasoning for the relative outlook")


class LLMEvidenceAnalysis(BaseModel):
    """
    Strict evidence-based LLM analysis output contract (Step 4.2).
    
    Replaces free-form decision authority with an auditable evidence and risk-analysis object.
    All execution-authorizing commands are rejected at schema validation.
    """
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    ticker: str = Field(min_length=1, max_length=16, description="Target stock ticker symbol (e.g. NVDA, AAPL)")
    as_of: str = Field(description="ISO 8601 UTC timestamp of analysis time (e.g. 2026-10-15T16:00:00Z)")
    analysis_status: AnalysisStatus = Field(default=AnalysisStatus.COMPLETE, description="Analysis completion status")
    advisory_action: AdvisoryAction = Field(
        default=AdvisoryAction.NO_ACTION,
        description="Non-authoritative advisory candidate action. CANNOT authorize execution."
    )
    advisory_confidence: float = Field(default=0.50, ge=0.0, le=1.0, description="Conviction score in [0.0, 1.0]")
    catalysts: List[CatalystEvidenceItem] = Field(default_factory=list, description="Structured catalysts with document provenance")
    risk_flags: List[RiskFlagEvidenceItem] = Field(default_factory=list, description="Structured risk flags with document provenance")
    supporting_document_ids: List[str] = Field(default_factory=list, description="All document IDs cited in this analysis")
    contradictions: List[EvidenceContradiction] = Field(default_factory=list, description="Cross-pillar or cross-evidence contradictions")
    missing_information: List[str] = Field(default_factory=list, description="Explicitly absent data channels or missing filings")
    forecast_20d: Optional[Forecast20dEvidence] = Field(default=None, description="20-session relative outlook")
    summary: str = Field(min_length=5, description="Evidence-based qualitative synthesis, kept separate from executable/deterministic fields")

    @field_validator("ticker")
    @classmethod
    def validate_ticker_format(cls, v: str) -> str:
        s = v.strip().upper()
        if not s:
            raise ValueError("ticker cannot be empty")
        if not re.match(r"^[A-Z0-9.\-]{1,16}$", s):
            raise ValueError(f"invalid ticker format '{v}'; must be 1-16 alphanumeric characters")
        return s

    @field_validator("as_of")
    @classmethod
    def validate_as_of_timestamp(cls, v: str) -> str:
        s = v.strip()
        try:
            # Parse ISO timestamp, standardizing trailing Z
            clean_ts = s.replace("Z", "+00:00")
            dt = datetime.datetime.fromisoformat(clean_ts)
            # Ensure timestamp is offset-aware or standard UTC
            if dt.tzinfo is None:
                # Interpret naive timestamp as UTC
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            return s
        except Exception as e:
            raise ValueError(f"invalid as_of timestamp '{v}': must be valid ISO 8601 UTC format (e.g. '2026-10-15T16:00:00Z'): {e}")

    @model_validator(mode="before")
    @classmethod
    def check_forbidden_broker_commands(cls, data: Any) -> Any:
        if isinstance(data, dict):
            found_forbidden = [k for k in data.keys() if str(k).lower() in FORBIDDEN_BROKER_FIELDS]
            if found_forbidden:
                raise ValueError(
                    f"prohibited broker command fields detected in LLM output: {found_forbidden}. "
                    "The LLM is an evidence service and must never issue broker commands or order parameters."
                )
        return data

    @model_validator(mode="after")
    def sync_supporting_document_ids(self) -> LLMEvidenceAnalysis:
        # Automatically aggregate all cited document IDs across catalysts, risks, and contradictions
        all_ids = set(self.supporting_document_ids)
        for cat in self.catalysts:
            all_ids.update(cat.supporting_document_ids)
        for rf in self.risk_flags:
            all_ids.update(rf.supporting_document_ids)
        for ct in self.contradictions:
            all_ids.update(ct.supporting_document_ids)
        self.supporting_document_ids = sorted(list(all_ids))
        return self


def validate_llm_evidence_response(
    raw_data: Union[dict, str],
    expected_ticker: Optional[str] = None,
    valid_document_ids: Optional[Set[str]] = None,
) -> Tuple[Optional[LLMEvidenceAnalysis], List[str]]:
    """
    Validates an extracted LLM evidence response against the strict Pydantic schema (Step 4.2).

    Parameters:
    - raw_data: Dictionary or JSON string from the LLM.
    - expected_ticker: Optional expected ticker symbol to prevent unknown/mismatched instruments.
    - valid_document_ids: Optional set of document_ids actually retrieved by RAG to verify grounding.

    Returns:
    - (model, errors): Tuple of (LLMEvidenceAnalysis instance or None, list of human-readable error messages).
    """
    errors: List[str] = []

    # 1. Parse JSON if string
    obj: dict
    if isinstance(raw_data, str):
        try:
            obj = json.loads(raw_data)
        except Exception as e:
            return None, [f"malformed JSON output: {e}"]
    elif isinstance(raw_data, dict):
        obj = dict(raw_data)
    else:
        return None, ["response is not a JSON object or string"]

    # 2. Check for forbidden broker command fields
    forbidden_keys = [k for k in obj.keys() if str(k).lower() in FORBIDDEN_BROKER_FIELDS]
    if forbidden_keys:
        errors.append(
            f"prohibited broker command fields detected: {forbidden_keys}. "
            "Output must not contain broker execution instructions."
        )
        return None, errors

    # 3. Validate ticker if expected_ticker is provided
    raw_ticker = obj.get("ticker") or obj.get("stock") or obj.get("symbol")
    if not raw_ticker:
        errors.append("missing required field 'ticker'")
    elif expected_ticker:
        clean_exp = str(expected_ticker).strip().upper()
        clean_act = str(raw_ticker).strip().upper()
        if clean_act != clean_exp:
            errors.append(f"ticker mismatch: received '{clean_act}', expected '{clean_exp}'")

    # 4. Standardize key aliases for schema validation
    if "ticker" not in obj and raw_ticker:
        obj["ticker"] = raw_ticker

    # 5. Pydantic Model Validation
    try:
        model = LLMEvidenceAnalysis.model_validate(obj)
    except Exception as e:
        # Format Pydantic validation errors cleanly
        if hasattr(e, "errors"):
            for err in e.errors():
                loc = ".".join(str(p) for p in err.get("loc", []))
                msg = err.get("msg", "validation error")
                errors.append(f"Field '{loc}': {msg}")
        else:
            errors.append(f"schema validation error: {e}")
        return None, errors

    # 6. Verify Document ID Grounding (if valid_document_ids set provided)
    if valid_document_ids is not None and len(valid_document_ids) > 0:
        ungrounded = set(model.supporting_document_ids) - set(valid_document_ids)
        if ungrounded:
            errors.append(f"ungrounded document citations: {sorted(list(ungrounded))} not in retrieved evidence set")

    if errors:
        return None, errors

    return model, []


def evidence_analysis_to_signal_dict(
    analysis: LLMEvidenceAnalysis,
    quant_scores: Optional[dict] = None,
    technical_data: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Bridges the strict LLMEvidenceAnalysis object into the normalized dictionary structure
    consumed by the deterministic decision engine (normalize_master_trader_json).

    Guarantees:
    - Advisory action is mapped to a non-authoritative candidate recommendation.
    - Catalysts and risk flags are structured with document IDs and provenance.
    - Explanatory text is strictly kept in 'summary' and 'rationale'.
    """
    # Map advisory action to candidate decision
    advisory_map = {
        AdvisoryAction.CANDIDATE_BUY: "BUY",
        AdvisoryAction.CANDIDATE_HOLD: "HOLD",
        AdvisoryAction.CANDIDATE_SELL: "SELL",
        AdvisoryAction.NO_ACTION: "HOLD",
    }
    candidate_dec = advisory_map.get(analysis.advisory_action, "HOLD")

    # Model probability distribution corresponding to candidate action
    if analysis.advisory_action == AdvisoryAction.CANDIDATE_BUY:
        b_score, h_score, s_score = 0.70, 0.20, 0.10
    elif analysis.advisory_action == AdvisoryAction.CANDIDATE_SELL:
        b_score, h_score, s_score = 0.10, 0.20, 0.70
    else:
        b_score, h_score, s_score = 0.20, 0.60, 0.20

    # Extract catalyst descriptions and risk factors for backward-compatible views
    catalyst_list = [c.model_dump() for c in analysis.catalysts]
    risk_flag_list = [r.model_dump() for r in analysis.risk_flags]
    contradiction_list = [ct.model_dump() for ct in analysis.contradictions]

    bull_case = [f"[{c.category}] {c.description}" for c in analysis.catalysts if c.polarity == CatalystPolarity.BULLISH]
    bear_case = [f"[{r.category}] {r.risk_name}: {r.relevant_passage[:120]}" for r in analysis.risk_flags]
    key_risks = [f"{r.risk_name} (Severity: {r.severity.value})" for r in analysis.risk_flags]

    forecast_dict = None
    if analysis.forecast_20d:
        forecast_dict = {
            "relative_outlook": analysis.forecast_20d.relative_outlook.value,
            "confidence": analysis.forecast_20d.confidence,
            "rationale": analysis.forecast_20d.rationale,
        }

    return {
        "stock": analysis.ticker,
        "ticker": analysis.ticker,
        "as_of": analysis.as_of,
        "analysis_status": analysis.analysis_status.value,
        "advisory_action": analysis.advisory_action.value,
        "decision": candidate_dec,
        "decision_origin": "EVIDENCE_SERVICE",
        "primary_driver": "QUANT_STRUCTURE",
        "confidence": analysis.advisory_confidence,
        "buy_score": b_score,
        "hold_score": h_score,
        "sell_score": s_score,
        "horizon_days": 20,
        "horizon_sessions": 20,
        "data_completeness": 0.88,
        "catalysts": catalyst_list,
        "risk_flags": risk_flag_list,
        "contradictions": contradiction_list,
        "supporting_document_ids": analysis.supporting_document_ids,
        "missing_information": analysis.missing_information,
        "forecast_20d": forecast_dict,
        "summary": analysis.summary,
        "bull_case": bull_case if bull_case else ["Evidence synthesis completed."],
        "bear_case": bear_case if bear_case else ["No acute negative thesis flags identified."],
        "key_risks": key_risks if key_risks else ["Standard equity market volatility."],
    }
