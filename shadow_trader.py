#!/usr/bin/env python3
"""Shadow Trader CLI & Orchestrator (Step 8).

Runs the complete end-to-end analytical and execution workflow in shadow mode:
- Feeds market quotes, factor predictions, structured decisions, and portfolio sizing.
- Validates pre-flight execution constraints and simulates realistic broker fills with market impact slippage.
- Records all actions in the immutable ExecutionLedger and reconciles positions.
- Computes the 8 formal shadow trading measurement suites:
    1. Signal Reproducibility (deterministic consistency across identical snapshots)
    2. Decision Latency & Data Freshness (P95 latency, quote age distribution, staleness violation rate)
    3. Predicted vs. Realized Returns (Spearman rank IC, directional accuracy, top/bottom quintile spread)
    4. Cost-Adjusted Performance (gross vs. net return, slippage drag bps, net Sharpe, profit factor)
    5. Gate Rejection Rate & Attribution (filter efficiency, certified vs. counterfactual returns)
    6. Model Feature Drift (Population Stability Index / PSI across core feature distributions)
    7. LLM Quality & Temporal Integrity (Point-in-Time availability, zero look-ahead bias)
    8. Risk & Order State Compliance (risk budget adherence, concentration limits, zero reconciliation errors)
- Predefines institutional promotion criteria and evaluates production readiness with remediation routing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from quant.policy_config import TradingPolicy, get_default_policy
from quant.portfolio_controls import (
    DEFAULT_UNIVERSE_TAXONOMY,
    PortfolioSnapshot,
    Position,
)
from quant.order_execution import (
    ExecutionLedger,
    MarketQuote,
    OperationalKillSwitch,
    PositionReconciler,
    SimulatedBroker,
)
from quant.shadow_trading import (
    PromotionDecision,
    PromotionStatus,
    ShadowDecisionRecord,
    ShadowMetricsAnalyzer,
    ShadowMetricsReport,
    ShadowPromotionEvaluator,
    ShadowTradingEngine,
)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def generate_synthetic_quote(
    ticker: str,
    base_price: float = 150.0,
    spread_bps: float = 4.0,
    adv: float = 5_000_000.0,
    atr_pct: float = 0.02,
) -> MarketQuote:
    """Generates a realistic test quote for shadow trading orchestration."""
    half_spread = (base_price * (spread_bps / 10000.0)) / 2.0
    return MarketQuote(
        ticker=ticker,
        last_price=base_price,
        bid_price=round(base_price - half_spread, 2),
        ask_price=round(base_price + half_spread, 2),
        timestamp=_now_utc(),
        is_halted=False,
        adv_shares_20d=adv,
        atr_14d=round(base_price * atr_pct, 2),
    )


def run_shadow_universe_session(
    tickers: List[str],
    policy: Optional[TradingPolicy] = None,
    portfolio: Optional[PortfolioSnapshot] = None,
    session_id: Optional[str] = None,
    persist_db: bool = False,
    db_path: Optional[str] = None,
    simulate_forward_outcomes: bool = False,
    baseline_features: Optional[Dict[str, List[float]]] = None,
) -> Tuple[List[ShadowDecisionRecord], ShadowMetricsReport, PromotionDecision]:
    """
    Orchestrates a complete shadow trading cycle across a target universe of tickers.
    """
    pol = policy or get_default_policy()
    sid = session_id or f"SHADOW_SESSION_{int(time.time())}"
    
    # Initialize Portfolio if not supplied
    port = portfolio or PortfolioSnapshot(
        portfolio_equity=100_000.0,
        cash=100_000.0,
        peak_equity=100_000.0,
        positions={},
    )

    engine = ShadowTradingEngine(policy=pol)
    records: List[ShadowDecisionRecord] = []

    # Ticker Price Mapping (sample baseline prices for the 25-stock universe)
    sample_prices = {
        "NVDA": 120.0, "AMD": 150.0, "AVGO": 170.0, "QCOM": 160.0, "INTC": 25.0,
        "TSM": 170.0, "ASML": 800.0, "MSFT": 420.0, "AAPL": 225.0, "GOOGL": 170.0,
        "META": 580.0, "AMZN": 185.0, "TSLA": 220.0, "JPM": 220.0, "BAC": 40.0,
        "GS": 500.0, "V": 280.0, "MA": 480.0, "XOM": 115.0, "CVX": 150.0,
        "JNJ": 160.0, "LLY": 900.0, "UNH": 580.0, "PG": 170.0, "COST": 900.0,
    }

    # Execute decisions for each ticker
    for sym in tickers:
        base_p = sample_prices.get(sym, 100.0)
        quote = generate_synthetic_quote(ticker=sym, base_price=base_p)
        
        # Synthetic features and score generation
        rsi = 55.0 + (hash(sym) % 25) - 12
        features = {
            "rsi14": rsi,
            "ema_alignment": 0.8,
            "atr_pct": 0.02,
            "market_alpha": 0.015,
        }

        # Simulated scores: higher scores for top half
        is_buy_candidate = (hash(sym) % 2 == 0)
        score = 72.0 if is_buy_candidate else 42.0
        candidate_dec = "BUY" if is_buy_candidate else "HOLD"
        predicted_ret = 0.035 if is_buy_candidate else -0.01

        rec = engine.execute_shadow_decision(
            ticker=sym,
            quote=quote,
            features=features,
            portfolio=port,
            candidate_decision=candidate_dec,
            composite_score=score,
            signal_strength=0.85,
            ml_score=score,
            ml_predicted_return=predicted_ret,
            pillar_scores={"trend": 75.0, "momentum": 70.0, "quality": 80.0},
            session_id=sid,
            persist_db=persist_db,
            db_path=db_path,
        )

        if simulate_forward_outcomes:
            # Simulate forward realized return consistent with model forecast (positive IC)
            if rec.composite_score and rec.composite_score >= 60.0:
                realized = 0.032  # 3.2% gain
            else:
                realized = -0.015  # -1.5% loss
            
            # Net return accounts for commissions and slippage
            net_ret = realized - ((rec.simulated_slippage_bps or 4.0) / 10000.0) - 0.0005
            rec.realized_forward_return_pct = realized * 100.0
            rec.realized_net_return_pct = net_ret * 100.0
            rec.is_profitable = (net_ret > 0)
            rec.hit_target = (realized >= 0.03)
            rec.hit_stop = False
            rec.outcome_evaluated_at = _now_utc().strftime("%Y-%m-%d %H:%M:%S")

        records.append(rec)

    # 2. Analyze the 8 Measurement Areas
    analyzer = ShadowMetricsAnalyzer(policy=pol)
    metrics_report = analyzer.analyze_session(
        records=records,
        baseline_feature_stats=baseline_features,
        session_id=sid,
        ledger=engine.ledger,
        portfolio=port,
        source_type="SYNTHETIC_SIMULATION",
    )

    # 3. Predefined Institutional Promotion Gate Evaluation
    evaluator = ShadowPromotionEvaluator(policy=pol)
    promotion_dec = evaluator.evaluate(metrics_report)

    return records, metrics_report, promotion_dec


def format_cli_report(
    records: List[ShadowDecisionRecord],
    report: ShadowMetricsReport,
    promo: PromotionDecision,
) -> str:
    """Formats a structured Markdown / terminal report of shadow trading results."""
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append(f"  SHADOW TRADING AUDIT & PROMOTION REPORT — SESSION: {report.session_id}")
    lines.append("=" * 80)
    lines.append(f"Evaluated At:          {report.evaluated_at} UTC")
    lines.append(f"Total Observations:    {report.total_observations}")
    lines.append(f"Orders Proposed:       {report.certified_trade_count} (Rejection Rate: {report.rejection_rate_pct:.1f}%)")
    lines.append(f"Promotion Verdict:     {promo.status}")
    lines.append("-" * 80)
    lines.append("  THE EIGHT FORMAL SHADOW TRADING MEASUREMENTS (STEP 8)")
    lines.append("-" * 80)

    # 1. Reproducibility
    lines.append(f"  [1] Signal Reproducibility:          {report.reproducibility_rate * 100:.1f}% ({report.reproducibility_status})")
    lines.append(f"      - Exact Composite Score Match:   {report.composite_exact_match_rate * 100:.1f}%")
    lines.append(f"      - Gate & Reason Code Match:      {report.gate_match_rate * 100:.1f}%")

    # 2. Freshness & Latency
    lines.append(f"  [2] Latency & Data Freshness:        {report.freshness_pass_rate * 100:.1f}% Pass ({report.freshness_status})")
    lines.append(f"      - Mean Latency:                  {report.mean_latency_seconds * 1000:.1f} ms (P95: {report.p95_latency_seconds * 1000:.1f} ms)")
    lines.append(f"      - Mean Quote Age:                {report.mean_quote_age_seconds:.2f} s (P95: {report.p95_quote_age_seconds:.2f} s)")

    # 3. Predicted vs Realized Returns
    lines.append(f"  [3] Predicted vs. Realized Returns:  IC {report.spearman_ic:+.4f} | Dir Acc: {report.directional_accuracy * 100:.1f}% ({report.predictive_status})")
    lines.append(f"      - Top-Bottom Quintile Spread:    {report.top_bottom_spread_pct:+.2f}%")
    lines.append(f"      - Forecast RMSE:                 {report.forecast_rmse:.4f}")

    # 4. Cost-Adjusted Performance
    lines.append(f"  [4] Cost-Adjusted Performance:       Net Return: {report.net_avg_return_pct:+.2f}% ({report.performance_status})")
    lines.append(f"      - Slippage Friction Drag:        {report.slippage_drag_bps:.1f} bps")
    lines.append(f"      - Net Sharpe Ratio:              {report.net_sharpe_ratio:.2f}")
    lines.append(f"      - Profit Factor:                 {report.profit_factor:.2f}")

    # 5. Gate Rejection Rate & Attribution
    eff_str = f"{report.gate_filter_efficiency:+.2f}%" if report.gate_filter_efficiency is not None else "N/A"
    lines.append(f"  [5] Gate Filtering Efficiency:       Delta: {eff_str} ({report.gate_status})")
    lines.append(f"      - Certified Avg Return:          {report.certified_avg_return_pct if report.certified_avg_return_pct is not None else 0.0:+.2f}%")
    lines.append(f"      - Rejected Avg Return:           {report.rejected_avg_return_pct if report.rejected_avg_return_pct is not None else 0.0:+.2f}%")

    # 6. Feature Drift (PSI)
    lines.append(f"  [6] Model Feature Drift (PSI):       Max PSI {report.overall_psi:.4f} ({report.drift_status})")

    # 7. LLM Quality & Temporal Integrity
    lines.append(f"  [7] Temporal Integrity (No-Leak):    Leakage Rate: {report.temporal_leakage_rate * 100:.1f}% ({report.llm_quality_status})")

    # 8. Risk & Order State Compliance
    lines.append(f"  [8] Risk & Ledger Reconciliation:    Recon Errors: {report.reconciliation_error_rate * 100:.1f}% ({report.risk_compliance_status})")
    lines.append(f"      - Risk Budget Adherence:         {report.risk_budget_compliance_rate * 100:.1f}%")
    lines.append(f"      - Concentration Compliance:      {report.concentration_compliance_rate * 100:.1f}%")

    lines.append("-" * 80)
    lines.append("  INSTITUTIONAL PROMOTION GATE EVALUATION")
    lines.append("-" * 80)
    lines.append(f"Promotion Status:      {promo.status}")
    lines.append(f"Promoted to Prod:      {'YES' if promo.is_promoted else 'NO'}")
    
    if promo.passed_criteria:
        lines.append("\nPassed Criteria:")
        for c in promo.passed_criteria:
            lines.append(f"  ✓ {c}")

    if promo.failed_criteria:
        lines.append("\nFailed Criteria:")
        for c in promo.failed_criteria:
            lines.append(f"  ✗ {c}")

    if promo.remediation_step:
        lines.append(f"\nRemediation Step:      {promo.remediation_step}")
        lines.append(f"Remediation Guidance:  {promo.remediation_guidance}")

    lines.append("=" * 80)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Shadow Trader CLI Orchestrator (Step 8)")
    parser.add_argument("--tickers", type=str, default="", help="Comma-separated tickers (e.g. NVDA,AMD,MSFT)")
    parser.add_argument("--all-universe", action="store_true", help="Run full 25-stock ETF leader universe")
    parser.add_argument("--session-id", type=str, default="", help="Custom session ID")
    parser.add_argument("--equity", type=float, default=100_000.0, help="Initial portfolio equity")
    parser.add_argument("--cash", type=float, default=100_000.0, help="Initial cash balance")
    parser.add_argument("--persist-db", action="store_true", help="Persist shadow decisions to SQLite database")
    parser.add_argument("--db-path", type=str, default="", help="SQLite database path")
    parser.add_argument("--format", type=str, choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("--output-file", type=str, default="", help="File to write report to")
    args = parser.parse_args()

    # Determine universe
    if args.all_universe:
        ticker_list = list(DEFAULT_UNIVERSE_TAXONOMY.keys())
    elif args.tickers:
        ticker_list = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        # Default to 25-stock universe
        ticker_list = list(DEFAULT_UNIVERSE_TAXONOMY.keys())

    policy = get_default_policy()
    portfolio = PortfolioSnapshot(
        portfolio_equity=args.equity,
        cash=args.cash,
        peak_equity=args.equity,
        positions={},
    )

    db_path = args.db_path or None
    records, report, promo = run_shadow_universe_session(
        tickers=ticker_list,
        policy=policy,
        portfolio=portfolio,
        session_id=args.session_id or None,
        persist_db=args.persist_db,
        db_path=db_path,
        simulate_forward_outcomes=False,
    )

    if args.format == "json":
        output = json.dumps({
            "metrics_report": report.to_dict(),
            "promotion_decision": promo.to_dict(),
            "decision_records_count": len(records),
        }, indent=2)
    else:
        output = format_cli_report(records, report, promo)

    print(output)

    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(output)
        print(f"Report written to: {args.output_file}")

    return 0 if promo.is_promoted else 1


if __name__ == "__main__":
    sys.exit(main())
