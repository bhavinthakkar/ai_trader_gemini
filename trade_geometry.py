"""Top-level re-export of quant.trade_geometry for pipeline convenience (Step 5.2)."""

from quant.trade_geometry import (
    TradeDirection,
    PriceOrderingStatus,
    ExecutionCostModel,
    RewardRiskResult,
    validate_price_ordering,
    compute_breakeven_win_rate,
    estimate_empirical_win_probability,
    compute_expected_value_r,
    compute_validated_reward_risk,
    derive_deterministic_channel_geometry,
)

__all__ = [
    "TradeDirection",
    "PriceOrderingStatus",
    "ExecutionCostModel",
    "RewardRiskResult",
    "validate_price_ordering",
    "compute_breakeven_win_rate",
    "estimate_empirical_win_probability",
    "compute_expected_value_r",
    "compute_validated_reward_risk",
    "derive_deterministic_channel_geometry",
]
