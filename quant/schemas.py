"""Data schemas and types for the quantitative scoring engine."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class PillarStatus(str, Enum):
    MEASURED = "measured"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


@dataclass
class InputTracking:
    name: str
    value: Any
    available: bool
    freshness: str = "fresh"  # fresh, stale, unavailable
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value if self.available else None,
            "available": self.available,
            "freshness": self.freshness,
            "timestamp": self.timestamp,
        }


@dataclass
class PillarResult:
    name: str
    score: Optional[float]
    status: PillarStatus
    inputs: Dict[str, InputTracking]
    weight: float
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "score": round(self.score, 2) if self.score is not None else None,
            "status": self.status.value,
            "inputs": {k: v.to_dict() for k, v in self.inputs.items()},
            "weight": self.weight,
            "description": self.description,
        }


@dataclass
class FinancialQualityResult:
    score: Optional[float]
    status: PillarStatus
    metrics: Dict[str, Any]
    flags: List[str]
    is_value_trap: bool
    value_trap_reasons: List[str]
    is_buy_eligible: bool
    compounder_bonus: float = 0.0
    inputs: Dict[str, InputTracking] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 2) if self.score is not None else None,
            "status": self.status.value,
            "metrics": self.metrics,
            "flags": self.flags,
            "is_value_trap": self.is_value_trap,
            "value_trap_reasons": self.value_trap_reasons,
            "is_buy_eligible": self.is_buy_eligible,
            "compounder_bonus": self.compounder_bonus,
            "inputs": {k: v.to_dict() for k, v in self.inputs.items()},
        }


@dataclass
class QuantCompositeResult:
    composite_quantitative_score: Optional[float]
    raw_composite: Optional[float]
    vol_factor: float
    quant_signal: str
    is_tradeable: bool
    no_trade_reason: Optional[str]
    available_pillars_count: int
    total_pillars_count: int
    pillars: Dict[str, PillarResult]
    financial_quality: FinancialQualityResult
    active_weights: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "composite_quantitative_score": self.composite_quantitative_score,
            "raw_composite": self.raw_composite,
            "vol_factor": self.vol_factor,
            "quant_signal": self.quant_signal,
            "is_tradeable": self.is_tradeable,
            "no_trade_reason": self.no_trade_reason,
            "available_pillars_count": self.available_pillars_count,
            "total_pillars_count": self.total_pillars_count,
            "trend_score": self.pillars["trend"].score if "trend" in self.pillars else None,
            "sector_relative_score": self.pillars["sector"].score if "sector" in self.pillars else None,
            "market_alpha_score": self.pillars["alpha"].score if "alpha" in self.pillars else None,
            "valuation_history_score": self.pillars["valuation_history"].score if "valuation_history" in self.pillars else None,
            "peer_valuation_score": self.pillars["peer_valuation"].score if "peer_valuation" in self.pillars else None,
            "financial_quality_score": self.financial_quality.score,
            "is_value_trap": self.financial_quality.is_value_trap,
            "value_trap_reasons": self.financial_quality.value_trap_reasons,
            "pillar_status": {k: v.status.value for k, v in self.pillars.items()},
            "pillar_input_tracking": {k: {ik: iv.to_dict() for ik, iv in v.inputs.items()} for k, v in self.pillars.items()},
            "active_weights": self.active_weights,
            "financial_quality": self.financial_quality.to_dict(),
        }
