"""Quant Package: Deterministic Quantitative Scoring Engine."""

from quant.schemas import (
    PillarStatus,
    InputTracking,
    PillarResult,
    FinancialQualityResult,
    QuantCompositeResult,
)
from quant.factors import (
    compute_ema_alignment,
    compute_rsi_velocity,
    compute_sector_excess,
    compute_market_alpha,
    compute_earnings_yield,
    compute_ebitda_yield,
    compute_fcf_yield,
)
from quant.normalization import (
    clamp,
    logistic_scale,
    normal_cdf_scale,
    cross_sectional_robust_zscore,
)
from quant.pillar_scores import (
    calculate_trend_pillar,
    calculate_sector_pillar,
    calculate_alpha_pillar,
    calculate_valuation_history_pillar,
    calculate_peer_valuation_pillar,
)
from quant.quality_adjustments import (
    evaluate_financial_quality,
    evaluate_value_trap,
    apply_quality_adjustment,
)
from quant.volatility import (
    compute_vol_factor,
    apply_volatility_dampener,
)
from quant.composite_score import (
    compute_deterministic_composite,
)
from quant.build_training_table import (
    FROZEN_16_FACTORS,
    FEATURE_ALIASES,
    FACTOR_SPECIFICATIONS,
    build_ml_training_table,
    generate_and_save_training_tables,
)

try:
    from quant.lambdamart_ranker import (
        LambdaMARTRanker,
        DEFAULT_LAMBDAMART_PARAMS,
    )
    from quant.walk_forward_evaluation import (
        run_purged_walk_forward_evaluation,
        train_production_ranker,
    )
except ImportError:
    LambdaMARTRanker = None
    DEFAULT_LAMBDAMART_PARAMS = None
    run_purged_walk_forward_evaluation = None
    train_production_ranker = None

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
from quant.compliance_gates import (
    GateStatus,
    GateResult,
    ComplianceReport,
    BuyFloorGate,
    RewardToRiskGate,
    VolatilityEventLiquidityGate,
    SellExitSymmetryGate,
    GatedConfidenceGate,
    CompliancePolicyEngine,
)
from quant.decision_precedence import (
    InternalActionType,
    DecisionStage,
    PositionState,
    PrecedenceEvaluationResult,
    DecisionPrecedenceEngine,
    evaluate_decision_precedence,
)
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
    "PillarStatus",
    "InputTracking",
    "PillarResult",
    "FinancialQualityResult",
    "QuantCompositeResult",
    "compute_ema_alignment",
    "compute_rsi_velocity",
    "compute_sector_excess",
    "compute_market_alpha",
    "compute_earnings_yield",
    "compute_ebitda_yield",
    "compute_fcf_yield",
    "clamp",
    "logistic_scale",
    "normal_cdf_scale",
    "cross_sectional_robust_zscore",
    "calculate_trend_pillar",
    "calculate_sector_pillar",
    "calculate_alpha_pillar",
    "calculate_valuation_history_pillar",
    "calculate_peer_valuation_pillar",
    "evaluate_financial_quality",
    "evaluate_value_trap",
    "apply_quality_adjustment",
    "compute_vol_factor",
    "apply_volatility_dampener",
    "compute_deterministic_composite",
    "FROZEN_16_FACTORS",
    "FEATURE_ALIASES",
    "FACTOR_SPECIFICATIONS",
    "build_ml_training_table",
    "generate_and_save_training_tables",
    "LambdaMARTRanker",
    "DEFAULT_LAMBDAMART_PARAMS",
    "run_purged_walk_forward_evaluation",
    "train_production_ranker",
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
    "GateStatus",
    "GateResult",
    "ComplianceReport",
    "BuyFloorGate",
    "RewardToRiskGate",
    "VolatilityEventLiquidityGate",
    "SellExitSymmetryGate",
    "GatedConfidenceGate",
    "CompliancePolicyEngine",
    "InternalActionType",
    "DecisionStage",
    "PositionState",
    "PrecedenceEvaluationResult",
    "DecisionPrecedenceEngine",
    "evaluate_decision_precedence",
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


