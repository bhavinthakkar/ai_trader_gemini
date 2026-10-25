"""Comprehensive Test Suite for Historical Evaluation, Shadow Mode, and Dual Promotion (Step 8)."""

from datetime import datetime, timezone
import json
import os
import tempfile
import pytest

from quant.policy_config import TradingPolicy, get_default_policy
from quant.historical_evaluation import (
    HistoricalEvaluationConfig,
    HistoricalEvaluationReport,
    HistoricalSetupMetrics,
    HistoricalTrade,
    compute_setup_metrics,
    run_historical_setups_evaluation,
)
from quant.shadow_trading import (
    DualPromotionDecision,
    DualPromotionEvaluator,
    DualPromotionStatus,
    PromotionDecision,
    PromotionStatus,
    ShadowDecisionRecord,
    ShadowMetricsAnalyzer,
    ShadowMetricsReport,
    ShadowPromotionEvaluator,
    run_dual_promotion_evaluation,
    run_shadow_trading_evaluation,
)


@pytest.fixture
def policy() -> TradingPolicy:
    return get_default_policy()


@pytest.fixture
def sample_trades() -> list[HistoricalTrade]:
    """Builds a realistic sample of 10 historical trades with trims, stops, and rebounds."""
    t1 = HistoricalTrade(
        trade_id="TRD-001",
        ticker="AAPL",
        setup_type="PULLBACK_BUY",
        entry_date="2023-03-01",
        entry_price=150.0,
        entry_commission=1.0,
        shares=50,
        orig_shares=50,
        stop_loss=140.0,
        target_price=165.0,
        highest_price=170.0,
        lowest_price=148.0,
        days_held=12,
        exit_date="2023-03-15",
        exit_price=168.0,
        exit_reason="TIME_HORIZON",
        was_stopped_out=False,
        was_trimmed=True,
        realized_trim_pnl=374.0,  # 25 shares x ($165 - $150) - comm
        gross_pnl=824.0,
        net_pnl=822.0,
        net_return_pct=10.96,
        profit_retained_pct=82.2,
        was_missed_rebound=False,
    )
    t2 = HistoricalTrade(
        trade_id="TRD-002",
        ticker="NVDA",
        setup_type="PULLBACK_BUY",
        entry_date="2023-04-01",
        entry_price=200.0,
        entry_commission=1.0,
        shares=30,
        orig_shares=30,
        stop_loss=185.0,
        target_price=220.0,
        highest_price=205.0,
        lowest_price=180.0,
        days_held=4,
        exit_date="2023-04-05",
        exit_price=184.0,
        exit_reason="STOP_LOSS",
        was_stopped_out=True,
        was_trimmed=False,
        realized_trim_pnl=0.0,
        gross_pnl=-480.0,
        net_pnl=-482.0,
        net_return_pct=-8.03,
        profit_retained_pct=0.0,
        was_missed_rebound=True,  # NVDA subsequently rebounded >= 5%
    )
    t3 = HistoricalTrade(
        trade_id="TRD-003",
        ticker="MSFT",
        setup_type="CATALYST_DIP_BUY",
        entry_date="2024-02-01",
        entry_price=300.0,
        entry_commission=1.0,
        shares=20,
        orig_shares=20,
        stop_loss=280.0,
        target_price=330.0,
        highest_price=335.0,
        lowest_price=295.0,
        days_held=10,
        exit_date="2024-02-15",
        exit_price=332.0,
        exit_reason="TIME_HORIZON",
        was_stopped_out=False,
        was_trimmed=True,
        realized_trim_pnl=299.0,
        gross_pnl=619.0,
        net_pnl=617.0,
        net_return_pct=10.28,
        profit_retained_pct=88.1,
        was_missed_rebound=False,
    )
    return [t1, t2, t3]


# =============================================================================
# 1. HISTORICAL SETUP METRICS & ATTRIBUTION TESTS
# =============================================================================

def test_compute_setup_metrics_calculations(sample_trades):
    """Verifies that win rate, stop-out rate, trim retention, and shakeouts are correctly computed."""
    import pandas as pd

    equity_curve = [
        (pd.Timestamp("2023-03-01"), 100_000.0),
        (pd.Timestamp("2023-03-15"), 100_822.0),
        (pd.Timestamp("2023-04-05"), 100_340.0),
        (pd.Timestamp("2024-02-15"), 100_957.0),
    ]

    metrics = compute_setup_metrics(
        trades=sample_trades,
        daily_equity=equity_curve,
        total_candidates=15,
        setup_type="PULLBACK_BUY",
        initial_equity=100_000.0,
    )

    assert metrics.total_trades_executed == 3
    assert metrics.winning_trades == 2
    assert metrics.losing_trades == 1
    assert pytest.approx(metrics.win_rate_pct, abs=0.1) == 66.67
    assert metrics.net_pnl == pytest.approx(822.0 - 482.0 + 617.0, abs=0.1)

    # Stop-out tracking: 1 stopped out out of 3
    assert metrics.stop_out_count == 1
    assert pytest.approx(metrics.stop_out_rate_pct, abs=0.1) == 33.33

    # Missed rebound tracking: 1 shakeout out of 1 stop-out
    assert metrics.missed_rebounds_count == 1
    assert pytest.approx(metrics.missed_rebounds_rate_pct, abs=0.1) == 100.0

    # Profit retained after trims
    assert metrics.profit_retained_after_trims_pct > 80.0
    assert metrics.annualized_turnover_pct > 0.0


def test_separate_pullback_and_catalyst_dip_reporting():
    """Verifies that historical evaluation runs and returns separate metrics for pullbacks and catalyst dips."""
    rep = run_historical_setups_evaluation(export_json=False)

    assert isinstance(rep, HistoricalEvaluationReport)
    assert rep.pullbacks.setup_type == "PULLBACK_BUY"
    assert rep.catalyst_dips.setup_type == "CATALYST_DIP_BUY"
    assert rep.combined.setup_type == "COMBINED"

    # Separate trade counts
    assert rep.pullbacks.total_trades_executed > 100
    assert rep.catalyst_dips.total_trades_executed > 50

    # Both must have separate period breakdowns
    for p_name in ["2022_bear_shock", "2023_recovery", "2024_bull_expansion", "2025_2026_holdout"]:
        assert p_name in rep.pullbacks.period_breakdown
        assert p_name in rep.catalyst_dips.period_breakdown
        assert p_name in rep.combined.period_breakdown

    # Drawdown must be bounded
    assert rep.pullbacks.max_drawdown_pct <= 20.0
    assert rep.catalyst_dips.max_drawdown_pct <= 20.0

    # Profit retention after trims must be >= 50%
    assert rep.pullbacks.profit_retained_after_trims_pct >= 50.0
    assert rep.catalyst_dips.profit_retained_after_trims_pct >= 50.0

    # The corrected combined book reconciles, but the strategy fails the
    # drawdown hurdle and this technical proxy cannot certify production rules.
    assert rep.combined.pnl_reconciliation_error <= 0.05
    assert rep.oos_gate_checks["max_historical_drawdown_within_bounds"]["passed"] is False
    assert rep.oos_gate_checks["production_rule_fidelity"]["passed"] is False
    assert rep.passed_oos_gate is False


# =============================================================================
# 2. DUAL PROMOTION EVALUATOR TESTS (INVARIANT VERIFICATION)
# =============================================================================

def test_dual_evaluator_promotes_when_both_pillars_pass(policy):
    """Verifies promotion to production occurs ONLY when both historical OOS and shadow trading pass."""
    dual_eval = DualPromotionEvaluator(policy=policy)

    mock_hist = {
        "passed_oos_gate": True,
        "production_rule_fidelity": True,
        "oos_gate_checks": {
            "min_trades": {"passed": True, "required": 50, "actual": 250},
            "pullback_expectancy": {"passed": True, "required": "> 0.0%", "actual": "+4.5%"},
            "max_drawdown": {"passed": True, "required": "<= 20.0%", "actual": "14.9%"},
        },
    }

    mock_shadow = ShadowMetricsReport(
        session_id="SESS_DUAL_PASS",
        total_observations=25,
        reproducibility_rate=1.0,
        freshness_pass_rate=1.0,
        spearman_ic=0.15,
        directional_accuracy=0.65,
        net_avg_return_pct=1.8,
        overall_psi=0.03,
        temporal_leakage_rate=0.0,
        reconciliation_error_rate=0.0,
    )

    decision = dual_eval.evaluate(mock_hist, mock_shadow)
    assert decision.is_promoted is True
    assert decision.status == DualPromotionStatus.PROMOTED_TO_PRODUCTION.value
    assert decision.historical_passed is True
    assert decision.shadow_passed is True
    assert len(decision.failed_criteria) == 0


def test_historical_replay_shadow_metrics_cannot_promote(policy):
    report = ShadowMetricsReport(
        session_id="REPLAY_ONLY",
        total_observations=100,
        source_type="HISTORICAL_REPLAY",
        reproducibility_rate=1.0,
        freshness_pass_rate=1.0,
        spearman_ic=0.5,
        directional_accuracy=0.9,
        net_avg_return_pct=2.0,
        overall_psi=0.01,
        temporal_leakage_rate=0.0,
        reconciliation_error_rate=0.0,
    )
    decision = ShadowPromotionEvaluator(policy=policy).evaluate(report)
    assert decision.is_promoted is False
    assert decision.status == PromotionStatus.REJECTED_FOR_PROMOTION.value
    assert "HISTORICAL_REPLAY" in decision.failed_criteria[0]


def test_dual_evaluator_rejects_when_backtest_passes_but_shadow_fails(policy):
    """Verifies that good backtest results ALONE cannot promote when shadow trading fails."""
    dual_eval = DualPromotionEvaluator(policy=policy)

    mock_hist = {
        "passed_oos_gate": True,
        "production_rule_fidelity": True,
        "oos_gate_checks": {"drawdown": {"passed": True, "actual": "12.0%", "required": "<= 20%"}},
    }

    # Shadow fails on directional accuracy and net expectancy
    mock_shadow = ShadowMetricsReport(
        session_id="SESS_SHADOW_FAIL",
        total_observations=25,
        reproducibility_rate=1.0,
        freshness_pass_rate=1.0,
        spearman_ic=-0.10,
        directional_accuracy=0.40,  # Below 50%
        net_avg_return_pct=-1.2,    # Negative net expectancy
        overall_psi=0.04,
        temporal_leakage_rate=0.0,
        reconciliation_error_rate=0.0,
    )

    decision = dual_eval.evaluate(mock_hist, mock_shadow)
    assert decision.is_promoted is False
    assert decision.status == DualPromotionStatus.REJECTED_SHADOW_PERIOD_FAILED.value
    assert decision.historical_passed is True
    assert decision.shadow_passed is False
    assert len(decision.failed_criteria) > 0


def test_dual_evaluator_rejects_when_shadow_passes_but_backtest_fails(policy):
    """Verifies that good shadow results cannot promote if out-of-sample backtest failed."""
    dual_eval = DualPromotionEvaluator(policy=policy)

    mock_hist = {
        "passed_oos_gate": False,  # Historical OOS failed
        "oos_gate_checks": {
            "max_historical_drawdown_within_bounds": {
                "passed": False,
                "required": "<= 20.0%",
                "actual": "32.5%",
            }
        },
    }

    mock_shadow = ShadowMetricsReport(
        session_id="SESS_SHADOW_PASS",
        total_observations=25,
        reproducibility_rate=1.0,
        freshness_pass_rate=1.0,
        spearman_ic=0.10,
        directional_accuracy=0.60,
        net_avg_return_pct=2.0,
        overall_psi=0.02,
        temporal_leakage_rate=0.0,
        reconciliation_error_rate=0.0,
    )

    decision = dual_eval.evaluate(mock_hist, mock_shadow)
    assert decision.is_promoted is False
    assert decision.status == DualPromotionStatus.REJECTED_HISTORICAL_OOS_FAILED.value
    assert decision.historical_passed is False
    assert decision.shadow_passed is True
    assert "HISTORICAL_OOS_CALIBRATION_REMEDIATION" in decision.remediation_step


def test_dual_evaluator_rejects_when_both_fail(policy):
    """Verifies that dual failure routes to DUAL_FAILURE_COMPREHENSIVE_REMEDIATION."""
    dual_eval = DualPromotionEvaluator(policy=policy)

    mock_hist = {"passed_oos_gate": False}
    mock_shadow = ShadowMetricsReport(
        session_id="SESS_DUAL_FAIL",
        total_observations=25,
        directional_accuracy=0.35,
        net_avg_return_pct=-2.0,
    )

    decision = dual_eval.evaluate(mock_hist, mock_shadow)
    assert decision.is_promoted is False
    assert decision.status == DualPromotionStatus.REJECTED_DUAL_FAILURE.value
    assert decision.remediation_step == "DUAL_FAILURE_COMPREHENSIVE_REMEDIATION"


def test_dual_evaluator_insufficient_shadow_observations(policy):
    """Verifies that evaluation rejects with INSUFFICIENT_SHADOW_OBSERVATIONS when n < min_obs."""
    dual_eval = DualPromotionEvaluator(policy=policy)

    mock_hist = {"passed_oos_gate": True}
    mock_shadow = ShadowMetricsReport(
        session_id="SESS_LOW_N",
        total_observations=8,  # 8 < 20
    )

    decision = dual_eval.evaluate(mock_hist, mock_shadow)
    assert decision.is_promoted is False
    assert decision.status == DualPromotionStatus.INSUFFICIENT_SHADOW_OBSERVATIONS.value
    assert decision.remediation_step == "EXTEND_SHADOW_TRADING_OBSERVATION_WINDOW"


# =============================================================================
# 3. END-TO-END ORCHESTRATOR INTEGRATION TEST
# =============================================================================

def test_run_dual_promotion_evaluation_integration(policy):
    """Verifies full execution of historical evaluation, shadow trading, and dual promotion gating."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_out = os.path.join(tmp_dir, "test_dual_promotion.json")
        hist_rep, shadow_rep, dual_dec = run_dual_promotion_evaluation(
            policy=policy,
            persist_db=False,
            export_json=True,
            json_path=json_out,
        )

        assert hist_rep.passed_oos_gate is False
        assert shadow_rep.source_type == "HISTORICAL_REPLAY"
        assert shadow_rep.total_observations >= 20
        assert shadow_rep.reproducibility_rate >= 0.99
        assert shadow_rep.freshness_pass_rate >= 0.99
        assert shadow_rep.directional_accuracy >= 0.50
        assert shadow_rep.net_avg_return_pct >= 0.0
        assert shadow_rep.reconciliation_error_rate == 0.0

        assert dual_dec.is_promoted is False
        assert dual_dec.status in (
            DualPromotionStatus.REJECTED_HISTORICAL_OOS_FAILED.value,
            DualPromotionStatus.REJECTED_DUAL_FAILURE.value,
        )
        assert os.path.isfile(json_out)

        with open(json_out, "r", encoding="utf-8") as f:
            saved_data = json.load(f)
            assert saved_data["dual_promotion_decision"]["is_promoted"] is False
