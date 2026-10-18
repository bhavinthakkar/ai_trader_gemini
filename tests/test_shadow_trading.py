"""Comprehensive Test Suite for Shadow Trading Engine, Measurements, and Promotion Gate (Step 8)."""

from datetime import datetime, timedelta, timezone
import math
import os
import tempfile
import pytest

from quant.policy_config import TradingPolicy, get_default_policy
from quant.portfolio_controls import PortfolioSnapshot, Position
from quant.order_execution import (
    ExecutionLedger,
    MarketQuote,
    OperationalKillSwitch,
    OrderStatus,
    PositionReconciler,
    SimulatedBroker,
)
from quant.shadow_trading import (
    DriftStatus,
    PromotionDecision,
    PromotionStatus,
    ShadowDecisionRecord,
    ShadowMetricsAnalyzer,
    ShadowMetricsReport,
    ShadowPromotionEvaluator,
    ShadowTradingEngine,
    compute_population_stability_index,
    compute_spearman_rank_ic,
)
import db


@pytest.fixture
def policy() -> TradingPolicy:
    return get_default_policy()


@pytest.fixture
def fresh_quote() -> MarketQuote:
    return MarketQuote(
        ticker="NVDA",
        last_price=100.0,
        bid_price=99.98,
        ask_price=100.02,
        timestamp=datetime.now(timezone.utc),
        is_halted=False,
        adv_shares_20d=5_000_000.0,
        atr_14d=2.50,
    )


@pytest.fixture
def sample_portfolio() -> PortfolioSnapshot:
    return PortfolioSnapshot(
        portfolio_equity=100_000.0,
        cash=100_000.0,
        peak_equity=100_000.0,
        positions={},
    )


# =============================================================================
# 1. STATISTICAL & MATHEMATICAL HELPER TESTS
# =============================================================================

def test_spearman_rank_ic_perfect_positive_correlation():
    """Verifies Spearman IC is +1.0 for perfectly monotonic increasing data."""
    x = [10.0, 20.0, 30.0, 40.0, 50.0]
    y = [1.0, 2.0, 3.0, 4.0, 5.0]
    ic = compute_spearman_rank_ic(x, y)
    assert pytest.approx(ic, abs=1e-4) == 1.0


def test_spearman_rank_ic_perfect_negative_correlation():
    """Verifies Spearman IC is -1.0 for perfectly monotonic decreasing data."""
    x = [10.0, 20.0, 30.0, 40.0, 50.0]
    y = [5.0, 4.0, 3.0, 2.0, 1.0]
    ic = compute_spearman_rank_ic(x, y)
    assert pytest.approx(ic, abs=1e-4) == -1.0


def test_spearman_rank_ic_small_sample():
    """Verifies Spearman IC returns 0.0 for samples smaller than 3."""
    assert compute_spearman_rank_ic([1.0, 2.0], [2.0, 1.0]) == 0.0
    assert compute_spearman_rank_ic([], []) == 0.0


def test_population_stability_index_no_drift():
    """Verifies PSI is near 0.0 when baseline and current distributions are identical."""
    base = [float(i) for i in range(100)]
    curr = [float(i) for i in range(100)]
    psi = compute_population_stability_index(base, curr)
    assert psi < 0.05


def test_population_stability_index_significant_drift():
    """Verifies PSI detects significant feature distribution shift."""
    base = [float(i) for i in range(100)]  # Uniform 0 to 99
    curr = [float(i + 150) for i in range(100)]  # Shifted 150 to 249
    psi = compute_population_stability_index(base, curr)
    assert psi >= 0.25


# =============================================================================
# 2. END-TO-END SHADOW DECISION EXECUTION & DB PERSISTENCE
# =============================================================================

def test_shadow_decision_execution_and_persistence(policy, fresh_quote, sample_portfolio):
    """Verifies end-to-end shadow decision generation, execution simulation, and SQLite persistence."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        db.init_db(db_path=db_path)
        engine = ShadowTradingEngine(policy=policy)

        features = {"rsi14": 45.0, "ema_alignment": 0.5, "atr_pct": 0.025}
        record = engine.execute_shadow_decision(
            ticker="NVDA",
            quote=fresh_quote,
            features=features,
            portfolio=sample_portfolio,
            candidate_decision="BUY",
            composite_score=75.0,
            signal_strength=0.88,
            ml_score=76.0,
            ml_predicted_return=0.03,
            pillar_scores={"trend": 80.0, "momentum": 70.0},
            session_id="TEST_SESSION_01",
            persist_db=True,
            db_path=db_path,
        )

        assert record.decision == "BUY"
        assert record.position_action == "BUY"
        assert record.is_order_proposed is True
        assert record.proposed_shares > 0
        assert record.pre_flight_valid is True
        assert record.simulated_order_id is not None
        assert record.simulated_fill_price is not None
        assert record.simulated_status == "FILLED"
        assert record.decision_id is not None

        # Verify record in database
        saved_records = db.get_shadow_decisions(session_id="TEST_SESSION_01", db_path=db_path)
        assert len(saved_records) == 1
        row = saved_records[0]
        assert row["ticker"] == "NVDA"
        assert row["decision"] == "BUY"
        assert row["proposed_shares"] == record.proposed_shares
        assert row["simulated_status"] == "FILLED"

        # Update realized outcome
        db.update_shadow_decision_outcome(
            decision_id=record.decision_id,
            realized_forward_return_pct=2.5,
            realized_net_return_pct=2.45,
            hit_target=True,
            hit_stop=False,
            is_profitable=True,
            db_path=db_path,
        )

        updated_records = db.get_shadow_decisions(session_id="TEST_SESSION_01", db_path=db_path)
        assert updated_records[0]["realized_forward_return_pct"] == 2.5
        assert updated_records[0]["is_profitable"] == 1
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


# =============================================================================
# 3. MEASUREMENT 1: SIGNAL REPRODUCIBILITY
# =============================================================================

def test_signal_reproducibility_deterministic_consistency(policy, fresh_quote):
    """Verifies that re-evaluating identical snapshots produces 100% deterministic decisions."""
    engine = ShadowTradingEngine(policy=policy)
    features = {"rsi14": 52.0, "ema_alignment": 0.6}
    port1 = PortfolioSnapshot(portfolio_equity=100_000.0, cash=100_000.0, peak_equity=100_000.0, positions={})
    port2 = PortfolioSnapshot(portfolio_equity=100_000.0, cash=100_000.0, peak_equity=100_000.0, positions={})

    rec1 = engine.execute_shadow_decision(
        ticker="NVDA",
        quote=fresh_quote,
        features=features,
        portfolio=port1,
        candidate_decision="BUY",
        composite_score=70.0,
        run_id="run-001",
    )

    rec2 = engine.execute_shadow_decision(
        ticker="NVDA",
        quote=fresh_quote,
        features=features,
        portfolio=port2,
        candidate_decision="BUY",
        composite_score=70.0,
        run_id="run-002",
    )

    assert rec1.decision == rec2.decision
    assert rec1.position_action == rec2.position_action
    assert rec1.composite_score == rec2.composite_score
    assert rec1.proposed_shares == rec2.proposed_shares
    assert rec1.passed_gates == rec2.passed_gates
    assert rec1.failed_gates == rec2.failed_gates
    assert rec1.reason_codes == rec2.reason_codes


# =============================================================================
# 4. MEASUREMENT 2: DECISION LATENCY & DATA FRESHNESS
# =============================================================================

def test_stale_data_rejection_and_freshness_tracking(policy, sample_portfolio):
    """Verifies that stale market quotes trip the freshness gate and prevent order execution."""
    engine = ShadowTradingEngine(policy=policy)
    stale_quote = MarketQuote(
        ticker="NVDA",
        last_price=100.0,
        bid_price=99.98,
        ask_price=100.02,
        timestamp=datetime.now(timezone.utc) - timedelta(seconds=600),  # 600s old > 300s limit
        adv_shares_20d=5_000_000.0,
        atr_14d=2.50,
    )

    rec = engine.execute_shadow_decision(
        ticker="NVDA",
        quote=stale_quote,
        features={},
        portfolio=sample_portfolio,
        candidate_decision="BUY",
        composite_score=75.0,
    )

    assert rec.is_fresh is False
    assert "DATA_FRESHNESS" in rec.failed_gates
    assert "MARKET_DATA_STALE" in rec.reason_codes
    assert rec.is_order_proposed is False
    assert rec.proposed_shares == 0



# =============================================================================
# 5. MEASUREMENT 3: PREDICTED VS. REALIZED RETURNS
# =============================================================================

def test_predicted_vs_realized_returns_metrics(policy):
    """Verifies Spearman IC, directional accuracy, and quintile spread calculation."""
    analyzer = ShadowMetricsAnalyzer(policy=policy)
    
    # 5 test records with scores 50, 60, 70, 80, 90 and corresponding positive returns
    records = []
    for i, (score, ret, action) in enumerate([
        (50.0, -0.02, "HOLD"),
        (60.0, -0.01, "HOLD"),
        (70.0, 0.01, "BUY"),
        (80.0, 0.03, "BUY"),
        (90.0, 0.05, "BUY"),
    ]):
        rec = ShadowDecisionRecord(
            session_id="SESS_IC",
            run_id=f"run-{i}",
            ticker="NVDA",
            decision="BUY" if action == "BUY" else "HOLD",
            position_action=action,
            composite_score=score,
            ml_predicted_return=ret,
            realized_forward_return_pct=ret * 100.0,
            realized_net_return_pct=(ret - 0.001) * 100.0,
            is_order_proposed=(action == "BUY"),
            is_fresh=True,
            decision_latency_seconds=0.005,
        )
        records.append(rec)

    report = analyzer.analyze_session(records=records)
    assert report.spearman_ic > 0.90
    assert report.directional_accuracy == 1.0  # All 3 BUY trades had positive returns
    assert report.top_bottom_spread_pct > 0.0
    assert report.predictive_status == "PASS"


# =============================================================================
# 6. MEASUREMENT 4: COST-ADJUSTED PERFORMANCE
# =============================================================================

def test_cost_adjusted_performance_and_slippage_drag(policy):
    """Verifies calculation of slippage drag in bps, net Sharpe, and profit factor."""
    analyzer = ShadowMetricsAnalyzer(policy=policy)
    records = [
        ShadowDecisionRecord(
            session_id="SESS_COST",
            run_id="run-1",
            ticker="NVDA",
            is_order_proposed=True,
            realized_forward_return_pct=2.0,   # +2.0% gross
            realized_net_return_pct=1.85,      # +1.85% net (15 bps friction)
            is_fresh=True,
        ),
        ShadowDecisionRecord(
            session_id="SESS_COST",
            run_id="run-2",
            ticker="MSFT",
            is_order_proposed=True,
            realized_forward_return_pct=3.0,   # +3.0% gross
            realized_net_return_pct=2.80,      # +2.80% net (20 bps friction)
            is_fresh=True,
        ),
    ]

    report = analyzer.analyze_session(records=records)
    assert report.gross_avg_return_pct == pytest.approx(2.5, abs=0.01)
    assert report.net_avg_return_pct == pytest.approx(2.325, abs=0.01)
    assert report.slippage_drag_bps == pytest.approx(17.5, abs=0.1)
    assert report.profit_factor >= 1.0
    assert report.performance_status == "PASS"


# =============================================================================
# 7. MEASUREMENT 5: GATE REJECTION RATE & ATTRIBUTION
# =============================================================================

def test_gate_rejection_tracking_and_filter_efficiency(policy):
    """Verifies that gate rejections track properly and quantify counterfactual filter efficiency."""
    analyzer = ShadowMetricsAnalyzer(policy=policy)
    records = [
        ShadowDecisionRecord(
            session_id="SESS_GATES",
            run_id="run-1",
            ticker="NVDA",
            is_order_proposed=True,
            realized_forward_return_pct=4.0,  # Certified trade gained 4%
            is_fresh=True,
        ),
        ShadowDecisionRecord(
            session_id="SESS_GATES",
            run_id="run-2",
            ticker="INTC",
            is_order_proposed=False,
            failed_gates=["BUY_FLOOR"],
            reason_codes=["COMPOSITE_BELOW_FLOOR"],
            realized_forward_return_pct=-3.0, # Rejected trade lost 3%
            is_fresh=True,
        ),
    ]

    report = analyzer.analyze_session(records=records)
    assert report.certified_trade_count == 1
    assert report.rejected_trade_count == 1
    assert report.rejection_rate_pct == 50.0
    assert report.gate_breakdown.get("BUY_FLOOR") == 1
    assert report.gate_filter_efficiency == pytest.approx(7.0, abs=0.01)  # 4.0 - (-3.0) = +7.0%
    assert report.gate_status == "PASS"


# =============================================================================
# 8. MEASUREMENT 6: MODEL DRIFT ANALYSIS (PSI)
# =============================================================================

def test_feature_drift_detection_via_psi(policy):
    """Verifies PSI drift tracking against baseline feature statistics."""
    analyzer = ShadowMetricsAnalyzer(policy=policy)
    
    # Baseline distributions
    baseline_stats = {
        "rsi14": [float(x) for x in range(20, 80)],
        "atr_pct": [0.01 + 0.001 * x for x in range(60)],
    }

    # Live records with shifted rsi (severe drift)
    records = []
    for i in range(20):
        records.append(ShadowDecisionRecord(
            session_id="SESS_DRIFT",
            run_id=f"run-{i}",
            ticker="NVDA",
            input_snapshot={"features": {"rsi14": 95.0, "atr_pct": 0.02}},
            is_fresh=True,
        ))

    report = analyzer.analyze_session(records=records, baseline_feature_stats=baseline_stats)
    assert report.overall_psi >= 0.25
    assert report.drift_status == DriftStatus.SIGNIFICANT_DRIFT.value


# =============================================================================
# 9. MEASUREMENT 7: LLM QUALITY & TEMPORAL INTEGRITY
# =============================================================================

def test_llm_quality_and_temporal_lookahead_detection(policy):
    """Verifies detection of Point-in-Time future data leakage in evidence passages."""
    analyzer = ShadowMetricsAnalyzer(policy=policy)
    
    # Record with future publication timestamp relative to quote
    leaked_record = ShadowDecisionRecord(
        session_id="SESS_LEAK",
        run_id="run-1",
        ticker="NVDA",
        quote_timestamp="2026-10-18 10:00:00",
        input_snapshot={
            "passages": [
                {"publication_timestamp": "2026-10-18 12:00:00", "reliability": 0.90}  # Future passage!
            ]
        },
        is_fresh=True,
    )

    report = analyzer.analyze_session(records=[leaked_record])
    assert report.temporal_leakage_rate == 1.0
    assert report.llm_quality_status == "FAIL"


# =============================================================================
# 10. MEASUREMENT 8: RISK & LEDGER RECONCILIATION
# =============================================================================

def test_risk_and_order_state_reconciliation_compliance(policy, fresh_quote, sample_portfolio):
    """Verifies that executed shadow trades reconcile perfectly with execution ledger."""
    engine = ShadowTradingEngine(policy=policy)
    engine.execute_shadow_decision(
        ticker="NVDA",
        quote=fresh_quote,
        features={},
        portfolio=sample_portfolio,
        candidate_decision="BUY",
        composite_score=75.0,
    )

    analyzer = ShadowMetricsAnalyzer(policy=policy)
    report = analyzer.analyze_session(
        records=[],
        session_id="SESS_RECON",
        ledger=engine.ledger,
        portfolio=sample_portfolio,
    )

    assert report.reconciliation_error_rate == 0.0
    assert report.risk_compliance_status == "PASS"


# =============================================================================
# 11. INSTITUTIONAL PROMOTION CRITERIA EVALUATION & REMEDIATION ROUTING
# =============================================================================

def test_promotion_evaluation_insufficient_observations(policy):
    """Verifies that evaluation rejects with INSUFFICIENT_OBSERVATIONS when n < min_observations."""
    evaluator = ShadowPromotionEvaluator(policy=policy)
    report = ShadowMetricsReport(
        session_id="SESS_LOW_N",
        total_observations=5,  # 5 < 20
    )

    decision = evaluator.evaluate(report)
    assert decision.is_promoted is False
    assert decision.status == PromotionStatus.INSUFFICIENT_OBSERVATIONS.value
    assert decision.remediation_step == "EXTEND_SHADOW_TRADING_OBSERVATION_WINDOW"


def test_promotion_evaluation_remediation_on_predictive_failure(policy):
    """Verifies that sub-zero IC routes to STEP_2_3_FACTOR_REMEDIATION."""
    evaluator = ShadowPromotionEvaluator(policy=policy)
    report = ShadowMetricsReport(
        session_id="SESS_BAD_IC",
        total_observations=25,
        reproducibility_rate=1.0,
        freshness_pass_rate=1.0,
        spearman_ic=-0.15,  # Negative IC
        directional_accuracy=0.40,
        net_avg_return_pct=-0.50,
        overall_psi=0.02,
        temporal_leakage_rate=0.0,
        reconciliation_error_rate=0.0,
    )

    decision = evaluator.evaluate(report)
    assert decision.is_promoted is False
    assert decision.status == PromotionStatus.REJECTED_FOR_PROMOTION.value
    assert "STEP_2_3_FACTOR_REMEDIATION" in decision.remediation_step


def test_promotion_evaluation_remediation_on_temporal_leakage(policy):
    """Verifies that temporal lookahead leakage routes to STEP_4_RAG_POINT_IN_TIME_REMEDIATION."""
    evaluator = ShadowPromotionEvaluator(policy=policy)
    report = ShadowMetricsReport(
        session_id="SESS_LEAK_FAIL",
        total_observations=25,
        reproducibility_rate=1.0,
        freshness_pass_rate=1.0,
        spearman_ic=0.10,
        directional_accuracy=0.60,
        net_avg_return_pct=1.5,
        overall_psi=0.02,
        temporal_leakage_rate=0.05,  # 5% look-ahead leakage
        reconciliation_error_rate=0.0,
    )

    decision = evaluator.evaluate(report)
    assert decision.is_promoted is False
    assert decision.remediation_step == "STEP_4_RAG_POINT_IN_TIME_REMEDIATION"


def test_promotion_evaluation_full_pass(policy):
    """Verifies that meeting all criteria results in PROMOTED_TO_PRODUCTION."""
    evaluator = ShadowPromotionEvaluator(policy=policy)
    report = ShadowMetricsReport(
        session_id="SESS_PROMOTION_PASS",
        total_observations=25,
        reproducibility_rate=1.0,
        freshness_pass_rate=1.0,
        spearman_ic=0.12,
        directional_accuracy=0.65,
        net_avg_return_pct=2.10,
        overall_psi=0.04,
        temporal_leakage_rate=0.0,
        reconciliation_error_rate=0.0,
    )

    decision = evaluator.evaluate(report)
    assert decision.is_promoted is True
    assert decision.status == PromotionStatus.PROMOTED_TO_PRODUCTION.value
    assert len(decision.failed_criteria) == 0
    assert len(decision.passed_criteria) >= 7
