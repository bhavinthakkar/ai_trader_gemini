"""
Formalized Signal Strength and Confidence Engine (Step 5.5).

Formalizes the mathematical calculation of Signal Strength combining:
1. Score Decisiveness (D): D = |S - 50| / 50 in [0.0, 1.0]
2. Pillar Agreement (A): Consensus dispersion across normalized pillar scores in [0.0, 1.0]
3. Data Completeness (C): Percentage of required inputs that are valid and fresh in [0.0, 1.0]

IMPORTANT EPISTEMIC DISTINCTION:
Until the calculation is empirically calibrated against realized forward return distributions,
the resulting index is explicitly designated 'Signal Strength' (an index of signal decisiveness,
cross-pillar consensus, and data integrity), NOT the empirical probability that a trade will be profitable.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


class DispersionMetric(str, Enum):
    """Dispersion metric across pillar scores."""
    MAD = "mad"                     # Mean Absolute Deviation from pillar mean
    STANDARD_DEVIATION = "std"      # Sample / population standard deviation
    RANGE = "range"                 # Max score minus Min score


class CombinationMethod(str, Enum):
    """Method for combining decisiveness, agreement, and completeness."""
    HIERARCHICAL_GATED = "hierarchical_gated"  # Production default (Step 5.3 continuity)
    LINEAR_WEIGHTED = "linear_weighted"        # Direct weighted sum: w_D*D + w_A*A + w_C*C
    GEOMETRIC = "geometric"                    # Multiplicative: D^w_D * A^w_A * C^w_C


@dataclass
class SignalStrengthWeights:
    """Configurable weights for signal strength combination."""
    weight_decisiveness: float = 0.70
    weight_agreement: float = 0.30
    completeness_blend_weight: float = 0.50
    linear_weight_decisiveness: float = 0.50
    linear_weight_agreement: float = 0.25
    linear_weight_completeness: float = 0.25
    hold_ceiling: float = 0.60
    min_threshold: float = 0.65
    catalyst_confidence_boost: float = 0.15
    catalyst_confidence_ceiling: float = 0.85


@dataclass
class RequiredInputRecord:
    """Individual required input tracking for data completeness calculation."""
    name: str
    available: bool
    valid: bool
    fresh: bool
    weight: float = 1.0
    value: Any = None
    reason: Optional[str] = None

    @property
    def is_valid_and_fresh(self) -> bool:
        return bool(self.available and self.valid and self.fresh)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "valid": self.valid,
            "fresh": self.fresh,
            "is_valid_and_fresh": self.is_valid_and_fresh,
            "weight": self.weight,
            "value": self.value if self.available else None,
            "reason": self.reason,
        }


@dataclass
class SignalStrengthResult:
    """
    Formal audit result of the Signal Strength and Confidence calculation.
    """
    signal_strength: float
    raw_signal_strength: float
    score_decisiveness: float
    pillar_agreement: float
    pillar_dispersion: float
    data_completeness: float
    valid_pillars_count: int
    total_pillars_count: int
    combination_method: CombinationMethod
    decision: str
    is_calibrated_probability: bool = False
    label: str = "SIGNAL_STRENGTH"
    calibration_status: str = "UNCALIBRATED_HEURISTIC_INDEX"
    metrics: Dict[str, Any] = field(default_factory=dict)
    input_records: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["combination_method"] = self.combination_method.value
        return res


# =============================================================================
# 1. SCORE DECISIVENESS (D)
# =============================================================================
def compute_score_decisiveness(
    composite_score: Optional[float],
    neutral_anchor: float = 50.0,
    max_scale: float = 50.0,
) -> float:
    """
    Calculates score decisiveness D:
        D = |S - 50.0| / 50.0
    where S is clamped to [0.0, 100.0].

    Properties:
    - S = 50.0  -> D = 0.00 (completely neutral, zero directional conviction)
    - S = 60.0  -> D = 0.20
    - S = 70.0  -> D = 0.40
    - S = 85.0  -> D = 0.70
    - S = 100.0 -> D = 1.00 (maximum conviction)
    - S = 0.0   -> D = 1.00 (maximum bearish conviction)

    Returns:
        float in [0.0, 1.0]
    """
    if composite_score is None:
        return 0.0
    if isinstance(composite_score, bool):
        return 0.0
    try:
        score_val = float(composite_score)
        if math.isnan(score_val) or math.isinf(score_val):
            return 0.0
    except (TypeError, ValueError):
        return 0.0

    clamped_score = max(0.0, min(100.0, score_val))
    distance = abs(clamped_score - neutral_anchor)
    decisiveness = distance / max_scale if max_scale > 0 else 0.0
    return max(0.0, min(1.0, decisiveness))


# =============================================================================
# 2. PILLAR DISPERSION & AGREEMENT (A)
# =============================================================================
def compute_pillar_agreement(
    pillar_scores: Optional[Dict[str, Optional[float]]],
    dispersion_metric: Union[DispersionMetric, str] = DispersionMetric.MAD,
    max_dispersion_scale: float = 50.0,
    pillar_keys: Optional[Sequence[str]] = None,
) -> Tuple[float, float, int, float]:
    """
    Calculates pillar dispersion and agreement A from normalized pillar scores:
        A = 1.0 - clamp(Dispersion / max_dispersion_scale, 0.0, 1.0)

    For pillar scores in [0.0, 100.0], the theoretical maximum Mean Absolute Deviation (MAD)
    occurs when half the scores are 0 and half are 100, which equals 50.0.
    Thus, max_dispersion_scale = 50.0 normalizes dispersion perfectly to [0.0, 1.0].

    Returns:
        (agreement, normalized_dispersion, valid_count, mean_score)
    """
    if isinstance(dispersion_metric, str):
        dispersion_metric = DispersionMetric(dispersion_metric.lower())

    keys = pillar_keys or ["trend", "sector", "alpha", "valuation_history", "peer_valuation"]
    valid_values: List[float] = []

    if pillar_scores:
        for k in keys:
            v = pillar_scores.get(k)
            if v is not None:
                try:
                    vf = float(v)
                    valid_values.append(max(0.0, min(100.0, vf)))
                except (TypeError, ValueError):
                    pass

    valid_count = len(valid_values)
    if valid_count == 0:
        return 0.50, 0.50, 0, 50.0

    mean_score = sum(valid_values) / valid_count

    if valid_count == 1:
        # A single pillar cannot establish cross-sectional consensus
        return 0.50, 0.50, 1, mean_score

    # Compute raw dispersion
    if dispersion_metric == DispersionMetric.MAD:
        raw_dispersion = sum(abs(v - mean_score) for v in valid_values) / valid_count
    elif dispersion_metric == DispersionMetric.STANDARD_DEVIATION:
        variance = sum((v - mean_score) ** 2 for v in valid_values) / valid_count
        raw_dispersion = math.sqrt(variance)
    elif dispersion_metric == DispersionMetric.RANGE:
        raw_dispersion = (max(valid_values) - min(valid_values)) / 2.0
    else:
        raw_dispersion = sum(abs(v - mean_score) for v in valid_values) / valid_count

    normalized_dispersion = max(0.0, min(1.0, raw_dispersion / max_dispersion_scale))
    agreement = max(0.0, min(1.0, 1.0 - normalized_dispersion))

    return agreement, normalized_dispersion, valid_count, mean_score


# =============================================================================
# 3. DATA COMPLETENESS (C)
# =============================================================================
def compute_data_completeness(
    input_records: Optional[Sequence[RequiredInputRecord]] = None,
    direct_completeness: Optional[float] = None,
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Calculates data completeness C as the percentage of required inputs that are valid and fresh:
        C = sum(w_j * I(input_j is valid and fresh)) / sum(w_j)

    Returns:
        (completeness, audit_records_list)
    """
    if input_records is not None and len(input_records) > 0:
        total_weight = sum(rec.weight for rec in input_records)
        if total_weight <= 0.0:
            return 0.0, [r.to_dict() for r in input_records]

        satisfied_weight = sum(
            rec.weight for rec in input_records if rec.is_valid_and_fresh
        )
        completeness = max(0.0, min(1.0, satisfied_weight / total_weight))
        return completeness, [r.to_dict() for r in input_records]

    if direct_completeness is not None:
        try:
            c_val = float(direct_completeness)
            return max(0.0, min(1.0, c_val)), []
        except (TypeError, ValueError):
            return 0.0, []

    return 0.0, []


# =============================================================================
# 4. SIGNAL STRENGTH CALCULATION & COMBINATION ENGINE
# =============================================================================
class SignalStrengthEngine:
    """
    Orchestrates the calculation of Signal Strength from Decisiveness,
    Pillar Agreement, and Data Completeness.
    """

    def __init__(self, weights: Optional[SignalStrengthWeights] = None):
        self.weights = weights or SignalStrengthWeights()

    def calculate(
        self,
        composite_score: Optional[float],
        pillar_scores: Optional[Dict[str, Optional[float]]],
        data_completeness: Optional[Union[float, Sequence[RequiredInputRecord]]] = None,
        decision: str = "BUY",
        method: CombinationMethod = CombinationMethod.HIERARCHICAL_GATED,
        dispersion_metric: DispersionMetric = DispersionMetric.MAD,
        is_certified_dip_buy: bool = False,
    ) -> SignalStrengthResult:
        """
        Combines Decisiveness (D), Pillar Agreement (A), and Data Completeness (C).
        """
        # 1. Decisiveness D
        decisiveness = compute_score_decisiveness(composite_score)

        # 2. Pillar Agreement A
        agreement, dispersion, valid_pillars_count, mean_pillar = compute_pillar_agreement(
            pillar_scores=pillar_scores,
            dispersion_metric=dispersion_metric,
        )

        # 3. Data Completeness C
        if isinstance(data_completeness, (int, float)):
            comp_val, input_audit = compute_data_completeness(direct_completeness=float(data_completeness))
        elif isinstance(data_completeness, Sequence):
            comp_val, input_audit = compute_data_completeness(input_records=data_completeness)
        else:
            comp_val, input_audit = compute_data_completeness(direct_completeness=0.87)

        # 4. Combine based on method
        edge_strength = 0.0
        base_signal = 0.0

        if method == CombinationMethod.HIERARCHICAL_GATED:
            # Step 5.3 Production Model:
            # Edge strength: w_D * D + w_A * (A - 0.5)
            edge_strength = (
                self.weights.weight_decisiveness * decisiveness
                + self.weights.weight_agreement * (agreement - 0.50)
            )
            base_signal = 0.50 + 0.50 * edge_strength
            raw_signal = base_signal * (
                (1.0 - self.weights.completeness_blend_weight)
                + self.weights.completeness_blend_weight * comp_val
            )

        elif method == CombinationMethod.LINEAR_WEIGHTED:
            # Direct Linear Sum: w_D*D + w_A*A + w_C*C
            raw_signal = (
                self.weights.linear_weight_decisiveness * decisiveness
                + self.weights.linear_weight_agreement * agreement
                + self.weights.linear_weight_completeness * comp_val
            )
            edge_strength = decisiveness
            base_signal = raw_signal

        elif method == CombinationMethod.GEOMETRIC:
            # Multiplicative Geometric Mean
            # Avoid math domain error with zero
            d_safe = max(0.01, decisiveness)
            a_safe = max(0.01, agreement)
            c_safe = max(0.01, comp_val)
            raw_signal = (
                (d_safe ** self.weights.linear_weight_decisiveness)
                * (a_safe ** self.weights.linear_weight_agreement)
                * (c_safe ** self.weights.linear_weight_completeness)
            )
            edge_strength = decisiveness
            base_signal = raw_signal

        else:
            raw_signal = 0.50

        # Boost for certified dip-buy dislocation
        if is_certified_dip_buy:
            raw_signal = min(
                raw_signal + self.weights.catalyst_confidence_boost,
                self.weights.catalyst_confidence_ceiling,
            )

        # Cap neutral / non-directional HOLD signals
        final_signal = raw_signal
        if isinstance(decision, str) and decision.upper() == "HOLD":
            final_signal = min(final_signal, self.weights.hold_ceiling)

        rounded_signal = round(max(0.0, min(1.0, final_signal)), 2)

        metrics = {
            "decisiveness_D": round(decisiveness, 4),
            "agreement_A": round(agreement, 4),
            "dispersion": round(dispersion, 4),
            "mean_pillar_score": round(mean_pillar, 2),
            "completeness_C": round(comp_val, 4),
            "edge_strength": round(edge_strength, 4),
            "base_signal": round(base_signal, 4),
            "raw_signal": round(raw_signal, 4),
            "final_signal": rounded_signal,
            "is_certified_dip_buy": is_certified_dip_buy,
            "catalyst_confidence_boost": self.weights.catalyst_confidence_boost if is_certified_dip_buy else 0.0,
            "weights": asdict(self.weights),
            "combination_method": method.value,
            "dispersion_metric": dispersion_metric.value,
        }

        return SignalStrengthResult(
            signal_strength=rounded_signal,
            raw_signal_strength=round(raw_signal, 4),
            score_decisiveness=round(decisiveness, 4),
            pillar_agreement=round(agreement, 4),
            pillar_dispersion=round(dispersion, 4),
            data_completeness=round(comp_val, 4),
            valid_pillars_count=valid_pillars_count,
            total_pillars_count=5,
            combination_method=method,
            decision=str(decision).upper() if decision is not None else "HOLD",
            is_calibrated_probability=False,
            label="SIGNAL_STRENGTH",
            calibration_status="UNCALIBRATED_HEURISTIC_INDEX",
            metrics=metrics,
            input_records=input_audit,
        )


def calculate_signal_strength(
    composite_score: Optional[float],
    pillar_scores: Optional[Dict[str, Optional[float]]],
    data_completeness: Optional[Union[float, Sequence[RequiredInputRecord]]] = None,
    decision: str = "BUY",
    weights: Optional[SignalStrengthWeights] = None,
    method: CombinationMethod = CombinationMethod.HIERARCHICAL_GATED,
    dispersion_metric: DispersionMetric = DispersionMetric.MAD,
    is_certified_dip_buy: bool = False,
) -> SignalStrengthResult:
    """Convenience wrapper for SignalStrengthEngine calculation."""
    engine = SignalStrengthEngine(weights)
    return engine.calculate(
        composite_score=composite_score,
        pillar_scores=pillar_scores,
        data_completeness=data_completeness,
        decision=decision,
        method=method,
        dispersion_metric=dispersion_metric,
        is_certified_dip_buy=is_certified_dip_buy,
    )
