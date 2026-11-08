"""Unit Tests for Monte Carlo Risk Simulation Engine (Step 8.2)."""

import sys
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np

from quant.monte_carlo import (
    MonteCarloConfig,
    MonteCarloResults,
    MonteCarloSimulator,
)


@pytest.fixture
def sample_trades():
    """Sample trade records with mixed positive and negative returns."""
    return [
        {"trade_id": "T1", "portfolio_return_pct": 1.5, "net_pnl": 1500.0},
        {"trade_id": "T2", "portfolio_return_pct": -0.8, "net_pnl": -800.0},
        {"trade_id": "T3", "portfolio_return_pct": 2.2, "net_pnl": 2200.0},
        {"trade_id": "T4", "portfolio_return_pct": -1.0, "net_pnl": -1000.0},
        {"trade_id": "T5", "portfolio_return_pct": 0.5, "net_pnl": 500.0},
        {"trade_id": "T6", "portfolio_return_pct": -0.7, "net_pnl": -700.0},
        {"trade_id": "T7", "portfolio_return_pct": -1.2, "net_pnl": -1200.0},
        {"trade_id": "T8", "portfolio_return_pct": 3.0, "net_pnl": 3000.0},
    ]


def test_default_simulation_initialization():
    """Verify default simulator loads cached trades or fallback correctly."""
    sim = MonteCarloSimulator()
    assert len(sim.trades) > 0
    assert len(sim._pnl_fractions) > 0

    cfg = MonteCarloConfig(n_simulations=500, horizon_trades=30, random_seed=42)
    res = sim.run_simulation(config=cfg)

    assert isinstance(res, MonteCarloResults)
    assert res.n_simulations == 500
    assert res.horizon_trades == 30
    assert res.initial_equity == 100_000.0
    assert res.median_final_equity > 0.0
    assert 0.0 <= res.prob_circuit_breaker_pct <= 100.0
    assert 0.0 <= res.prob_de_risking_pct <= 100.0
    assert 0.0 <= res.prob_ruin_pct <= 100.0
    assert len(res.risk_sensitivity_sweep) == 4
    assert "verdict" in res.safety_verdict.lower() or "safe" in res.safety_verdict.lower() or "risk" in res.safety_verdict.lower()


def test_simulation_determinism_with_seed(sample_trades):
    """Verify identical random seeds produce exact deterministic results."""
    sim = MonteCarloSimulator(trades=sample_trades)
    cfg1 = MonteCarloConfig(n_simulations=1000, horizon_trades=50, random_seed=123)
    cfg2 = MonteCarloConfig(n_simulations=1000, horizon_trades=50, random_seed=123)

    res1 = sim.run_simulation(config=cfg1)
    res2 = sim.run_simulation(config=cfg2)

    assert res1.median_final_equity == res2.median_final_equity
    assert res1.max_drawdown_percentiles == res2.max_drawdown_percentiles
    assert res1.prob_circuit_breaker_pct == res2.prob_circuit_breaker_pct
    assert res1.consecutive_loss_streaks == res2.consecutive_loss_streaks


def test_regime_dampening_effect(sample_trades):
    """Verify regime dampening (de-risking at 5%, halt at 10%) reduces extreme drawdown."""
    sim = MonteCarloSimulator(trades=sample_trades)

    # 1. Without dampening
    cfg_raw = MonteCarloConfig(n_simulations=2000, horizon_trades=60, apply_regime_dampening=False, random_seed=99)
    res_raw = sim.run_simulation(config=cfg_raw)

    # 2. With dampening
    cfg_dampened = MonteCarloConfig(n_simulations=2000, horizon_trades=60, apply_regime_dampening=True, random_seed=99)
    res_dampened = sim.run_simulation(config=cfg_dampened)

    # 99th percentile drawdown must be equal or lower with dampening enabled
    assert res_dampened.max_drawdown_percentiles["p99"] <= res_raw.max_drawdown_percentiles["p99"] + 0.1


def test_risk_scaling_monotonicity(sample_trades):
    """Verify increasing risk fraction r increases max drawdown percentiles."""
    sim = MonteCarloSimulator(trades=sample_trades)

    cfg_low = MonteCarloConfig(n_simulations=1000, horizon_trades=50, risk_per_trade_pct=0.5, random_seed=77)
    cfg_high = MonteCarloConfig(n_simulations=1000, horizon_trades=50, risk_per_trade_pct=2.0, random_seed=77)

    res_low = sim.run_simulation(config=cfg_low)
    res_high = sim.run_simulation(config=cfg_high)

    assert res_high.max_drawdown_percentiles["p95"] > res_low.max_drawdown_percentiles["p95"]
    assert res_high.prob_de_risking_pct >= res_low.prob_de_risking_pct


def test_overnight_gap_stress_injection(sample_trades):
    """Verify overnight gap stress injection increases drawdown risk."""
    sim = MonteCarloSimulator(trades=sample_trades)

    cfg_normal = MonteCarloConfig(n_simulations=1500, horizon_trades=50, overnight_gap_prob=0.0, random_seed=42)
    cfg_gap = MonteCarloConfig(n_simulations=1500, horizon_trades=50, overnight_gap_prob=0.08, overnight_gap_loss_pct=5.0, random_seed=42)

    res_normal = sim.run_simulation(config=cfg_normal)
    res_gap = sim.run_simulation(config=cfg_gap)

    assert res_gap.max_drawdown_percentiles["worst"] >= res_normal.max_drawdown_percentiles["worst"]


def test_consecutive_losses_calculation(sample_trades):
    """Verify consecutive losing streaks are within valid bounds."""
    sim = MonteCarloSimulator(trades=sample_trades)
    cfg = MonteCarloConfig(n_simulations=500, horizon_trades=40, random_seed=42)
    res = sim.run_simulation(config=cfg)

    streaks = res.consecutive_loss_streaks
    assert 1.0 <= streaks["median"] <= 40.0
    assert streaks["median"] <= streaks["p95"] <= streaks["worst"] <= 40.0


def test_fan_chart_percentiles_order(sample_trades):
    """Verify p5 <= p25 <= p50 <= p75 <= p95 at every step in fan chart."""
    sim = MonteCarloSimulator(trades=sample_trades)
    cfg = MonteCarloConfig(n_simulations=1000, horizon_trades=40, random_seed=42)
    res = sim.run_simulation(config=cfg)

    fc = res.equity_fan_chart
    steps_count = len(fc["steps"])
    assert steps_count > 1

    for i in range(steps_count):
        p5 = fc["p5"][i]
        p25 = fc["p25"][i]
        p50 = fc["p50"][i]
        p75 = fc["p75"][i]
        p95 = fc["p95"][i]
        assert p5 <= p25 + 0.01
        assert p25 <= p50 + 0.01
        assert p50 <= p75 + 0.01
        assert p75 <= p95 + 0.01
