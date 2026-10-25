"""
Unit tests for Step 5: Sizing Staged Entries Against One Risk Budget.

Verifies:
1. Sizing uses existing position-sizing and portfolio-control code for the total position.
2. The maximum total position risk is decided in advance.
3. Tranches divide the single risk budget and are not treated as independent risk allowances.
4. Explicit evidence permits each additional tranche (profit >= 0.5 ATR, composite score intact).
5. Refusal conditions (no further additions allowed):
   - Averaging down a losing position is strictly prohibited (position in loss).
   - Insufficient profit cushion (< 0.50 ATR).
   - Max tranches reached.
   - Risk budget exhausted.
   - Event risk blackout or extreme volatility.
   - Portfolio circuit breaker halt.
6. Open orders and existing exposure explicitly reduce the remaining available risk budget.
7. Summed risk of all tranches strictly stays within configured account and portfolio limits.
"""

import pytest

from quant.policy_config import TradingPolicy, get_default_policy, load_policy_config
from quant.portfolio_controls import (
    PortfolioRiskEngine,
    PortfolioSnapshot,
    Position,
)
from quant.position_sizing import (
    InvestmentMandate,
    calculate_risk_based_position_size,
)
from quant.staged_entries import (
    StagedEntryPlan,
    StagedTrancheEvaluation,
    TranchePlan,
    TrancheRejectionReason,
    TrancheStatus,
    build_staged_entry_plan,
    evaluate_staged_tranche_addition,
)


class TestStagedEntriesGovernance:
    """Tests the staged entries governance engine and single risk budget enforcement."""

    def test_decide_max_total_position_risk_in_advance(self):
        """Sizes total position using existing code and sets total risk budget upfront."""
        policy = load_policy_config()
        # Equity = $100,000, max_risk = 1.0% ($1,000)
        # Entry = 100.0, Stop = 96.0, ATR = 2.0
        plan = build_staged_entry_plan(
            entry_price=100.0,
            stop_loss=96.0,
            ticker="AAPL",
            direction="LONG",
            portfolio_equity=100_000.0,
            risk_fraction=0.010,
            atr=2.0,
            policy=policy,
        )

        assert plan.ticker == "AAPL"
        assert plan.direction == "LONG"
        assert plan.total_risk_budget_dollars <= 1000.0
        assert plan.total_allowed_shares > 0
        assert plan.max_tranches == 3
        assert len(plan.tranches) == 3

        # Weights sum to 1.0 (50%, 30%, 20%)
        weights_sum = sum(t.target_weight for t in plan.tranches)
        assert pytest.approx(weights_sum, 0.01) == 1.0

        # Planned tranche shares sum up to total allowed shares
        total_planned_shares = sum(t.planned_shares for t in plan.tranches)
        assert total_planned_shares <= plan.total_allowed_shares

        # Planned tranche risk sums up to total risk budget (not independent budgets!)
        total_planned_risk = sum(t.planned_risk_dollars for t in plan.tranches)
        assert total_planned_risk <= plan.total_risk_budget_dollars + 1.0

    def test_evidence_permits_additional_tranche_when_in_profit(self):
        """Additional tranche is permitted when existing position is in profit >= 0.50 ATR and score intact."""
        policy = load_policy_config()
        plan = build_staged_entry_plan(
            entry_price=100.0,
            stop_loss=96.0,
            ticker="AAPL",
            direction="LONG",
            portfolio_equity=100_000.0,
            risk_fraction=0.010,
            atr=2.0,
            existing_shares=50,
            policy=policy,
        )

        # Price advances from 100.0 to 102.50 ($2.50 profit = 1.25 ATR > 0.50 ATR)
        eval_res = evaluate_staged_tranche_addition(
            plan=plan,
            current_price=102.50,
            tranche_index=2,
            proposed_entry_price=102.50,
            proposed_stop_loss=99.50,
            atr=2.0,
            composite_score=65.0,
            days_to_earnings=30,
            policy=policy,
        )

        assert eval_res.allowed is True
        assert eval_res.shares > 0
        assert eval_res.risk_dollars > 0
        assert eval_res.rejection_reason is None
        # Summed risk after addition stays within total budget
        assert eval_res.total_position_risk_after <= plan.total_risk_budget_dollars + 1.0

    def test_averaging_down_loser_strictly_prohibited(self):
        """Strict Invariant: Adding to a losing position (averaging down) is categorically prohibited."""
        policy = load_policy_config()
        plan = build_staged_entry_plan(
            entry_price=100.0,
            stop_loss=96.0,
            ticker="AAPL",
            direction="LONG",
            portfolio_equity=100_000.0,
            risk_fraction=0.010,
            atr=2.0,
            existing_shares=50,
            policy=policy,
        )

        # Price drops from 100.0 to 98.0 (position is in a loss)
        eval_res = evaluate_staged_tranche_addition(
            plan=plan,
            current_price=98.0,
            tranche_index=2,
            proposed_entry_price=98.0,
            proposed_stop_loss=95.0,
            atr=2.0,
            composite_score=65.0,
            policy=policy,
        )

        assert eval_res.allowed is False
        assert eval_res.shares == 0
        assert eval_res.rejection_reason == TrancheRejectionReason.POSITION_IN_LOSS.value
        assert eval_res.binding_constraint == "AVERAGING_DOWN_PROHIBITED"

    def test_insufficient_profit_buffer_blocks_addition(self):
        """Tranche addition is blocked if unrealized profit is below the required 0.50 ATR cushion."""
        policy = load_policy_config()
        plan = build_staged_entry_plan(
            entry_price=100.0,
            stop_loss=96.0,
            ticker="AAPL",
            direction="LONG",
            portfolio_equity=100_000.0,
            risk_fraction=0.010,
            atr=2.0,
            existing_shares=50,
            policy=policy,
        )

        # Price is at 100.40 (profit = $0.40 = 0.20 ATR < 0.50 ATR minimum)
        eval_res = evaluate_staged_tranche_addition(
            plan=plan,
            current_price=100.40,
            tranche_index=2,
            proposed_entry_price=100.40,
            proposed_stop_loss=97.50,
            atr=2.0,
            composite_score=65.0,
            policy=policy,
        )

        assert eval_res.allowed is False
        assert eval_res.shares == 0
        assert eval_res.rejection_reason == TrancheRejectionReason.INSUFFICIENT_PROFIT_BUFFER.value

    def test_max_tranches_blocks_further_additions(self):
        """Attempting to add beyond configured max_tranches (3) is rejected."""
        policy = load_policy_config()
        plan = build_staged_entry_plan(
            entry_price=100.0,
            stop_loss=96.0,
            ticker="AAPL",
            direction="LONG",
            portfolio_equity=100_000.0,
            risk_fraction=0.010,
            atr=2.0,
            existing_shares=80,
            policy=policy,
        )

        # Requesting tranche 4 when max_tranches is 3
        eval_res = evaluate_staged_tranche_addition(
            plan=plan,
            current_price=108.0,
            tranche_index=4,
            proposed_entry_price=108.0,
            proposed_stop_loss=104.0,
            atr=2.0,
            composite_score=65.0,
            policy=policy,
        )

        assert eval_res.allowed is False
        assert eval_res.shares == 0
        assert eval_res.rejection_reason == TrancheRejectionReason.MAX_TRANCHES_REACHED.value

    def test_existing_exposure_reduces_available_budget(self):
        """Existing exposure explicitly consumes risk budget; tranches are not independent allowances."""
        policy = load_policy_config()
        # Sizing total position with existing exposure
        plan = build_staged_entry_plan(
            entry_price=100.0,
            stop_loss=96.0,
            ticker="AAPL",
            direction="LONG",
            portfolio_equity=100_000.0,
            risk_fraction=0.010,  # $1,000 total risk
            atr=2.0,
            existing_shares=100,  # 100 shares * ~$4.50 risk = ~$450
            policy=policy,
        )

        assert plan.existing_shares == 100
        assert plan.existing_risk_dollars > 400.0
        # Available risk budget is strictly less than the total budget
        assert plan.remaining_risk_budget_dollars <= plan.total_risk_budget_dollars - plan.existing_risk_dollars

    def test_open_orders_consume_available_budget(self):
        """Open orders commit both risk and share capacity from the single position budget."""
        policy = load_policy_config()
        # Open order for 60 shares
        open_orders = [
            {
                "ticker": "AAPL",
                "shares": 60,
                "limit_price": 101.0,
                "stop_loss": 97.0,
                "side": "BUY",
                "is_open": True,
            }
        ]

        plan = build_staged_entry_plan(
            entry_price=100.0,
            stop_loss=96.0,
            ticker="AAPL",
            direction="LONG",
            portfolio_equity=100_000.0,
            risk_fraction=0.010,
            atr=2.0,
            existing_shares=50,
            open_orders=open_orders,
            policy=policy,
        )

        assert plan.open_order_shares == 60
        assert plan.open_order_risk_dollars > 200.0
        # Total committed = existing (50) + open order (60) = 110 shares
        assert plan.total_committed_shares == 110
        assert plan.remaining_shares == max(0, plan.total_allowed_shares - 110)

    def test_open_orders_exhausting_budget_blocks_additional_tranche(self):
        """If open orders and existing exposure exhaust the total budget, new addition is blocked."""
        policy = load_policy_config()
        # Total allowed is around ~200 shares. Commit 120 existing + 100 open = 220 shares (exhausted)
        open_orders = [
            {
                "ticker": "AAPL",
                "shares": 150,
                "limit_price": 101.0,
                "stop_loss": 97.0,
                "side": "BUY",
                "is_open": True,
            }
        ]

        plan = build_staged_entry_plan(
            entry_price=100.0,
            stop_loss=96.0,
            ticker="AAPL",
            direction="LONG",
            portfolio_equity=100_000.0,
            risk_fraction=0.010,
            atr=2.0,
            existing_shares=100,
            open_orders=open_orders,
            policy=policy,
        )

        eval_res = evaluate_staged_tranche_addition(
            plan=plan,
            current_price=103.0,
            tranche_index=2,
            proposed_entry_price=103.0,
            proposed_stop_loss=100.0,
            atr=2.0,
            composite_score=65.0,
            policy=policy,
        )

        assert eval_res.allowed is False
        assert eval_res.shares == 0
        assert eval_res.rejection_reason in (
            TrancheRejectionReason.RISK_BUDGET_EXHAUSTED.value,
            TrancheRejectionReason.PORTFOLIO_LIMIT_REACHED.value,
        )

    def test_trailing_stop_tightening_frees_budget_safely(self):
        """Moving stop to breakeven eliminates original principal risk, safely freeing budget for scale-in."""
        policy = load_policy_config()
        plan = build_staged_entry_plan(
            entry_price=100.0,
            stop_loss=96.0,
            ticker="AAPL",
            direction="LONG",
            portfolio_equity=100_000.0,
            risk_fraction=0.010,
            atr=2.0,
            existing_shares=50,
            policy=policy,
        )

        # Stop tightened to entry 100.0 (breakeven)
        eval_res = evaluate_staged_tranche_addition(
            plan=plan,
            current_price=106.0,
            tranche_index=2,
            proposed_entry_price=106.0,
            proposed_stop_loss=102.0,
            atr=2.0,
            composite_score=65.0,
            trailing_stop_tightened_to=100.0,  # Stop moved to breakeven
            policy=policy,
        )

        assert eval_res.allowed is True
        assert eval_res.shares > 0
        # Total position risk after addition remains strictly bounded by original max budget
        assert eval_res.total_position_risk_after <= plan.total_risk_budget_dollars + 1.0

    def test_event_risk_blackout_blocks_addition(self):
        """Corporate binary earnings blackout blocks subsequent staged additions."""
        policy = load_policy_config()
        plan = build_staged_entry_plan(
            entry_price=100.0,
            stop_loss=96.0,
            ticker="AAPL",
            direction="LONG",
            portfolio_equity=100_000.0,
            risk_fraction=0.010,
            atr=2.0,
            existing_shares=50,
            policy=policy,
        )

        eval_res = evaluate_staged_tranche_addition(
            plan=plan,
            current_price=103.0,
            tranche_index=2,
            proposed_entry_price=103.0,
            proposed_stop_loss=100.0,
            atr=2.0,
            composite_score=65.0,
            days_to_earnings=2,  # Earnings in 2 days (blackout)
            policy=policy,
        )

        assert eval_res.allowed is False
        assert eval_res.shares == 0
        assert eval_res.rejection_reason == TrancheRejectionReason.EVENT_RISK_BLACKOUT.value

    def test_extreme_volatility_blocks_addition(self):
        """Extreme volatility regime (IV > 100%) blocks subsequent staged additions."""
        policy = load_policy_config()
        plan = build_staged_entry_plan(
            entry_price=100.0,
            stop_loss=96.0,
            ticker="AAPL",
            direction="LONG",
            portfolio_equity=100_000.0,
            risk_fraction=0.010,
            atr=2.0,
            existing_shares=50,
            policy=policy,
        )

        eval_res = evaluate_staged_tranche_addition(
            plan=plan,
            current_price=103.0,
            tranche_index=2,
            proposed_entry_price=103.0,
            proposed_stop_loss=100.0,
            atr=2.0,
            composite_score=65.0,
            iv_pct=115.0,  # Extreme IV
            policy=policy,
        )

        assert eval_res.allowed is False
        assert eval_res.shares == 0
        assert eval_res.rejection_reason == TrancheRejectionReason.VOLATILITY_REGIME_EXTREME.value

    def test_drawdown_circuit_breaker_blocks_addition(self):
        """Portfolio drawdown halt (>= 10%) freezes all new risk entries, blocking additions."""
        policy = load_policy_config()
        snapshot = PortfolioSnapshot(
            portfolio_equity=88_000.0,
            cash=30_000.0,
            peak_equity=100_000.0,  # 12% drawdown >= 10% halt threshold
            positions={"AAPL": Position("AAPL", "LONG", 50, 100.0, 103.0)},
        )

        plan = build_staged_entry_plan(
            entry_price=100.0,
            stop_loss=96.0,
            ticker="AAPL",
            direction="LONG",
            portfolio_equity=88_000.0,
            risk_fraction=0.010,
            atr=2.0,
            existing_shares=50,
            portfolio_snapshot=snapshot,
            policy=policy,
        )

        eval_res = evaluate_staged_tranche_addition(
            plan=plan,
            current_price=103.0,
            tranche_index=2,
            proposed_entry_price=103.0,
            proposed_stop_loss=100.0,
            atr=2.0,
            composite_score=65.0,
            portfolio_snapshot=snapshot,
            policy=policy,
        )

        assert eval_res.allowed is False
        assert eval_res.shares == 0
        assert eval_res.rejection_reason == TrancheRejectionReason.DRAWDOWN_CIRCUIT_BREAKER.value

    def test_summed_risk_stays_within_account_and_portfolio_limits(self):
        """End-to-end multi-tranche lifecycle: summed risk of all tranches stays within limits."""
        policy = load_policy_config()
        equity = 100_000.0
        max_risk_frac = 0.010
        total_risk_cap = equity * max_risk_frac  # $1,000

        # 1. Plan upfront
        plan = build_staged_entry_plan(
            entry_price=100.0,
            stop_loss=96.0,
            ticker="AAPL",
            direction="LONG",
            portfolio_equity=equity,
            risk_fraction=max_risk_frac,
            atr=2.0,
            policy=policy,
        )

        # 2. Execute Tranche 1 (Pilot)
        t1 = plan.tranches[0]
        shares_t1 = t1.planned_shares
        risk_t1 = t1.planned_risk_dollars

        # Update position state with Tranche 1
        plan_after_t1 = build_staged_entry_plan(
            entry_price=100.0,
            stop_loss=96.0,
            ticker="AAPL",
            direction="LONG",
            portfolio_equity=equity,
            risk_fraction=max_risk_frac,
            atr=2.0,
            existing_shares=shares_t1,
            existing_entry_price=100.0,
            existing_stop_loss=96.0,
            policy=policy,
        )

        # 3. Evaluate Tranche 2 (Confirmation) at price 103.0
        t2_eval = evaluate_staged_tranche_addition(
            plan=plan_after_t1,
            current_price=103.0,
            tranche_index=2,
            proposed_entry_price=103.0,
            proposed_stop_loss=100.0,
            atr=2.0,
            composite_score=65.0,
            policy=policy,
        )

        assert t2_eval.allowed is True
        shares_t2 = t2_eval.shares
        risk_t2 = t2_eval.risk_dollars

        # Verify combined risk of T1 + T2
        combined_risk_1_2 = risk_t1 + risk_t2
        assert combined_risk_1_2 <= total_risk_cap + 1.0

        # Update position state with Tranche 1 + Tranche 2 (100 shares at avg 101.50)
        total_shares_1_2 = shares_t1 + shares_t2
        avg_entry = ((shares_t1 * 100.0) + (shares_t2 * 103.0)) / total_shares_1_2

        # 4. Evaluate Tranche 3 (Continuation) with stop raised to 101.0
        plan_after_t2 = build_staged_entry_plan(
            entry_price=106.0,
            stop_loss=103.0,
            ticker="AAPL",
            direction="LONG",
            portfolio_equity=equity,
            risk_fraction=max_risk_frac,
            atr=2.0,
            existing_shares=total_shares_1_2,
            existing_entry_price=avg_entry,
            existing_stop_loss=101.0,  # Stop moved up
            policy=policy,
        )

        t3_eval = evaluate_staged_tranche_addition(
            plan=plan_after_t2,
            current_price=106.0,
            tranche_index=3,
            proposed_entry_price=106.0,
            proposed_stop_loss=103.0,
            atr=2.0,
            composite_score=65.0,
            trailing_stop_tightened_to=101.0,
            policy=policy,
        )

        assert t3_eval.allowed is True
        assert t3_eval.total_position_risk_after <= total_risk_cap + 1.0
        assert t3_eval.total_position_shares_after <= plan.total_allowed_shares
