"""Top-level re-export of quant.signal_strength for pipeline convenience (Step 5.5)."""

from quant.signal_strength import (
    DispersionMetric,
    CombinationMethod,
    SignalStrengthWeights,
    RequiredInputRecord,
    SignalStrengthResult,
    compute_score_decisiveness,
    compute_pillar_agreement,
    compute_data_completeness,
    SignalStrengthEngine,
    calculate_signal_strength,
)

__all__ = [
    "DispersionMetric",
    "CombinationMethod",
    "SignalStrengthWeights",
    "RequiredInputRecord",
    "SignalStrengthResult",
    "compute_score_decisiveness",
    "compute_pillar_agreement",
    "compute_data_completeness",
    "SignalStrengthEngine",
    "calculate_signal_strength",
]
