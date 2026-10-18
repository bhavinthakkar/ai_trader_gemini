"""Top-level re-export for Shadow Trading Engine, Metrics Analyzer, and Promotion Evaluator (Step 8)."""

from quant.shadow_trading import (
    DriftStatus,
    PromotionStatus,
    ShadowDecisionRecord,
    ShadowMetricsReport,
    PromotionDecision,
    compute_spearman_rank_ic,
    compute_population_stability_index,
    ShadowTradingEngine,
    ShadowMetricsAnalyzer,
    ShadowPromotionEvaluator,
)

__all__ = [
    "DriftStatus",
    "PromotionStatus",
    "ShadowDecisionRecord",
    "ShadowMetricsReport",
    "PromotionDecision",
    "compute_spearman_rank_ic",
    "compute_population_stability_index",
    "ShadowTradingEngine",
    "ShadowMetricsAnalyzer",
    "ShadowPromotionEvaluator",
]
