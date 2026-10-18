"""Comprehensive Test Suite for Order Execution, Pre-Flight Validation, Kill Switch, and Reconciliation (Step 7.3)."""

from datetime import datetime, timedelta, timezone
import pytest
import tempfile
import os

from quant.policy_config import TradingPolicy, get_default_policy
from quant.portfolio_controls import (
    PortfolioSnapshot,
    Position,
    DrawdownRegime,
    PortfolioBindingConstraint,
)
from quant.position_sizing import calculate_risk_based_position_size
from quant.order_execution import (
    OrderStatus,
    OrderSide,
    OrderType,
    KillSwitchReason,
    PreFlightRejectionReason,
    MarketQuote,
    Order,
    OrderFill,
    ExecutionEvent,
    PreFlightValidationResult,
    ReconciliationDiscrepancy,
    ReconciliationResult,
    generate_idempotency_key,
    OperationalKillSwitch,
    PreFlightOrderValidator,
    ExecutionLedger,
    PositionReconciler,
    SimulatedBroker,
)
import db


@pytest.fixture
def policy() -> TradingPolicy:
    """Returns default institutional trading compliance policy."""
    return get_default_policy()


@pytest.fixture
def fresh_quote() -> MarketQuote:
    """Generates a pristine live market quote."""
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
    """Generates a standard institutional portfolio snapshot ($100k equity, $50k cash)."""
    return PortfolioSnapshot(
        portfolio_equity=100_000.0,
        cash=50_000.0,
        peak_equity=100_000.0,
        positions={
            "NVDA": Position(
                ticker="NVDA",
                direction="LONG",
                shares=50,
                entry_price=95.0,
                current_price=100.0,
                sector="Technology",
                factor_tags=["AI_SEMICONDUCTORS"],
            )
        },
    )


# =============================================================================
# 1. STATE MACHINE SEPARATION
# =============================================================================

def test_state_machine_distinct_lifecycle_progression(policy):
    """Verifies that signals, orders, and fills maintain strictly distinct states."""
    ledger = ExecutionLedger()

    # 1. Upstream recommendation is merely a signal (SIGNAL_GENERATED)
    signal_status = OrderStatus.SIGNAL_GENERATED
    assert signal_status.value == "SIGNAL_GENERATED"
    # An un-submitted signal is NOT in the execution ledger
    assert len(ledger.orders) == 0

    # 2. Creating an order transitions to ORDER_SUBMITTED
    order = ledger.create_order(
        ticker="NVDA",
        side=OrderSide.BUY,
        requested_shares=100,
        signal_price=100.0,
        limit_price=100.0,
        stop_loss_price=95.0,
        target_price=110.0,
    )
    assert order.status == OrderStatus.ORDER_SUBMITTED
    assert order.requested_shares == 100
    assert order.filled_shares == 0
    assert order.remaining_shares == 100
    assert order.is_open is True
    assert order.is_terminal is False

    # 3. Broker acknowledgement transitions to BROKER_ACKNOWLEDGED
    order = ledger.acknowledge_order(order.order_id, broker_order_id="BRK-12345")
    assert order.status == OrderStatus.BROKER_ACKNOWLEDGED
    assert order.broker_order_id == "BRK-12345"

    # 4. Partial fill transitions to PARTIALLY_FILLED
    fill_1 = OrderFill(
        fill_id="FIL-1",
        order_id=order.order_id,
        ticker="NVDA",
        side=OrderSide.BUY,
        shares=40,
        price=100.10,
        commission=1.0,
    )
    order = ledger.record_fill(fill_1)
    assert order.status == OrderStatus.PARTIALLY_FILLED
    assert order.filled_shares == 40
    assert order.remaining_shares == 60
    assert order.avg_fill_price == pytest.approx(100.10)
    assert order.is_open is True

    # 5. Final fill completes order to FILLED
    fill_2 = OrderFill(
        fill_id="FIL-2",
        order_id=order.order_id,
        ticker="NVDA",
        side=OrderSide.BUY,
        shares=60,
        price=100.20,
        commission=1.0,
    )
    order = ledger.record_fill(fill_2)
    assert order.status == OrderStatus.FILLED
    assert order.filled_shares == 100
    assert order.remaining_shares == 0
    # Weighted avg price: (40 * 100.10 + 60 * 100.20) / 100 = 100.16
    assert order.avg_fill_price == pytest.approx(100.16)
    assert order.total_commission == pytest.approx(2.0)
    assert order.is_terminal is True
    assert order.is_open is False


def test_order_cancellation_and_rejection(policy):
    """Verifies cancellation and pre-execution rejection state transitions."""
    ledger = ExecutionLedger()

    # Cancellation test
    order_a = ledger.create_order("AMD", OrderSide.BUY, 50, 150.0)
    ledger.cancel_order(order_a.order_id, reason="Operator cancelled order")
    assert order_a.status == OrderStatus.CANCELLED
    assert order_a.cancellation_reason == "Operator cancelled order"
    assert order_a.is_terminal is True

    # Rejection test
    order_b = ledger.create_order("MSFT", OrderSide.BUY, 20, 400.0)
    ledger.reject_order(order_b.order_id, reason="Broker margin check rejected")
    assert order_b.status == OrderStatus.REJECTED
    assert order_b.rejection_reason == "Broker margin check rejected"
    assert order_b.is_terminal is True


# =============================================================================
# 2. PRE-FLIGHT VALIDATION LAYER
# =============================================================================

def test_pre_flight_validation_success(policy, fresh_quote, sample_portfolio):
    """Verifies that an order passing all constraints is certified valid."""
    validator = PreFlightOrderValidator(policy=policy)
    res = validator.validate(
        ticker="NVDA",
        side=OrderSide.BUY,
        requested_shares=50,
        quote=fresh_quote,
        portfolio=sample_portfolio,
        signal_price=100.0,
        limit_price=100.0,
        stop_loss_price=95.0,
        target_price=110.0,
    )
    assert res.is_valid is True
    assert len(res.rejection_reasons) == 0
    assert res.permitted_shares == 50
    assert res.live_reward_risk_ratio == pytest.approx(2.0)  # (110 - 100) / (100 - 95) = 2.0


def test_pre_flight_stale_market_data(policy, sample_portfolio):
    """Verifies that stale quotes (> 300 seconds) are rejected."""
    kill_switch = OperationalKillSwitch()
    validator = PreFlightOrderValidator(policy=policy, kill_switch=kill_switch)

    stale_time = datetime.now(timezone.utc) - timedelta(seconds=350)
    stale_quote = MarketQuote(
        ticker="NVDA",
        last_price=100.0,
        bid_price=99.98,
        ask_price=100.02,
        timestamp=stale_time,
        is_halted=False,
    )

    res = validator.validate(
        ticker="NVDA",
        side=OrderSide.BUY,
        requested_shares=50,
        quote=stale_quote,
        portfolio=sample_portfolio,
    )
    assert res.is_valid is False
    assert PreFlightRejectionReason.MARKET_DATA_STALE.value in res.rejection_reasons
    assert res.permitted_shares == 0
    # Auto-tripped kill switch on data failure
    assert kill_switch.is_tripped is True
    assert kill_switch.trip_reason == KillSwitchReason.DATA_QUALITY_FAILURE


def test_pre_flight_market_halt_rejection(policy, fresh_quote, sample_portfolio):
    """Verifies that halted securities are rejected."""
    fresh_quote.is_halted = True
    validator = PreFlightOrderValidator(policy=policy)
    res = validator.validate(
        ticker="NVDA",
        side=OrderSide.BUY,
        requested_shares=20,
        quote=fresh_quote,
        portfolio=sample_portfolio,
    )
    assert res.is_valid is False
    assert PreFlightRejectionReason.MARKET_HALTED.value in res.rejection_reasons


def test_pre_flight_cash_sufficiency_and_reserve(policy, fresh_quote):
    """Verifies that orders which would breach the 10% cash reserve are rejected."""
    # Portfolio with only $6,000 cash on $50,000 equity (reserve requirement is $5,000)
    tight_portfolio = PortfolioSnapshot(
        portfolio_equity=50_000.0,
        cash=6_000.0,
        peak_equity=50_000.0,
    )
    validator = PreFlightOrderValidator(policy=policy)

    # 50 shares of $100 stock = ~$5,000. Remaining cash would be ~$1,000 < $5,000 min reserve
    res = validator.validate(
        ticker="NVDA",
        side=OrderSide.BUY,
        requested_shares=50,
        quote=fresh_quote,
        portfolio=tight_portfolio,
    )
    assert res.is_valid is False
    assert PreFlightRejectionReason.INSUFFICIENT_CASH.value in res.rejection_reasons


def test_pre_flight_liquidity_cap_and_market_impact(policy, fresh_quote, sample_portfolio):
    """Verifies that orders exceeding 1.0% ADV or 15 bps square-root impact are rejected."""
    # Set ADV to 10,000 shares (1.0% cap = 100 shares)
    fresh_quote.adv_shares_20d = 10_000.0
    validator = PreFlightOrderValidator(policy=policy)

    # Requesting 150 shares exceeds 100 shares limit
    res = validator.validate(
        ticker="NVDA",
        side=OrderSide.BUY,
        requested_shares=150,
        quote=fresh_quote,
        portfolio=sample_portfolio,
    )
    assert res.is_valid is False
    assert PreFlightRejectionReason.LIQUIDITY_LIMIT_EXCEEDED.value in res.rejection_reasons


def test_pre_flight_dead_on_arrival_stop_loss(policy, fresh_quote, sample_portfolio):
    """Verifies that Long orders where current price <= stop loss are rejected as dead on arrival."""
    validator = PreFlightOrderValidator(policy=policy)

    # Live price is $100, but stop loss is $102 (live price has already fallen below stop!)
    res = validator.validate(
        ticker="NVDA",
        side=OrderSide.BUY,
        requested_shares=50,
        quote=fresh_quote,
        portfolio=sample_portfolio,
        stop_loss_price=102.0,
        target_price=120.0,
    )
    assert res.is_valid is False
    assert PreFlightRejectionReason.INVALID_STOP_PRICE.value in res.rejection_reasons


def test_pre_flight_target_already_achieved(policy, fresh_quote, sample_portfolio):
    """Verifies that Long orders where current price >= target are rejected."""
    validator = PreFlightOrderValidator(policy=policy)

    # Live price is $100, but target price is $98 (target already hit or inverted)
    res = validator.validate(
        ticker="NVDA",
        side=OrderSide.BUY,
        requested_shares=50,
        quote=fresh_quote,
        portfolio=sample_portfolio,
        stop_loss_price=90.0,
        target_price=98.0,
    )
    assert res.is_valid is False
    assert PreFlightRejectionReason.INVALID_TARGET_PRICE.value in res.rejection_reasons


def test_pre_flight_excessive_price_drift(policy, fresh_quote, sample_portfolio):
    """Verifies that price drift exceeding 2.0% between signal and live price is rejected."""
    validator = PreFlightOrderValidator(policy=policy)

    # Signal was generated at $95.0, but live price is $100.0 (drift = 5.26% > 2.0%)
    res = validator.validate(
        ticker="NVDA",
        side=OrderSide.BUY,
        requested_shares=50,
        quote=fresh_quote,
        portfolio=sample_portfolio,
        signal_price=95.0,
        stop_loss_price=90.0,
        target_price=120.0,
    )
    assert res.is_valid is False
    assert PreFlightRejectionReason.EXCESSIVE_PRICE_DRIFT.value in res.rejection_reasons


def test_pre_flight_position_direction_conflict(policy, fresh_quote, sample_portfolio):
    """Verifies that opening a SHORT when already LONG (or BUY when SHORT) is blocked."""
    validator = PreFlightOrderValidator(policy=policy)

    # sample_portfolio currently holds 50 shares LONG of NVDA
    res = validator.validate(
        ticker="NVDA",
        side=OrderSide.SELL_SHORT,
        requested_shares=20,
        quote=fresh_quote,
        portfolio=sample_portfolio,
    )
    assert res.is_valid is False
    assert PreFlightRejectionReason.POSITION_DIRECTION_CONFLICT.value in res.rejection_reasons


# =============================================================================
# 3. IDEMPOTENCY & DUPLICATE ORDER PREVENTION
# =============================================================================

def test_idempotency_key_deterministic_and_duplicate_rejection(policy, fresh_quote, sample_portfolio):
    """Verifies deterministic SHA-256 key generation and duplicate rejection."""
    key1 = generate_idempotency_key("NVDA", OrderSide.BUY, 50, 100.0, "2026-10-18", "RUN-001")
    key2 = generate_idempotency_key("NVDA", OrderSide.BUY, 50, 100.0, "2026-10-18", "RUN-001")
    assert key1 == key2

    # Different share count produces different key
    key3 = generate_idempotency_key("NVDA", OrderSide.BUY, 60, 100.0, "2026-10-18", "RUN-001")
    assert key1 != key3

    # Validator rejects duplicate
    validator = PreFlightOrderValidator(policy=policy)
    res = validator.validate(
        ticker="NVDA",
        side=OrderSide.BUY,
        requested_shares=50,
        quote=fresh_quote,
        portfolio=sample_portfolio,
        limit_price=100.0,
        session_date="2026-10-18",
        run_id="RUN-001",
        existing_idempotency_keys={key1},
    )
    assert res.is_valid is False
    assert PreFlightRejectionReason.DUPLICATE_ORDER_IDEMPOTENCY.value in res.rejection_reasons


# =============================================================================
# 4. OPERATIONAL KILL SWITCH
# =============================================================================

def test_operational_kill_switch_blocks_entries_permits_exits():
    """Verifies fail-closed entry blocking while defensive exits remain authorized."""
    ks = OperationalKillSwitch()
    assert ks.is_tripped is False

    # All permitted when not tripped
    can_buy, _ = ks.can_submit_order(OrderSide.BUY)
    can_sell, _ = ks.can_submit_order(OrderSide.SELL)
    assert can_buy is True
    assert can_sell is True

    # Trip kill switch on manual operator emergency
    ks.trip(
        reason=KillSwitchReason.MANUAL_OPERATOR_TRIP,
        details="Operator triggered emergency desk stop",
        actor="CHIEF_RISK_OFFICER",
    )
    assert ks.is_tripped is True
    assert ks.trip_reason == KillSwitchReason.MANUAL_OPERATOR_TRIP

    # Entries are BLOCKED
    can_buy_tripped, buy_reason = ks.can_submit_order(OrderSide.BUY)
    can_short_tripped, short_reason = ks.can_submit_order(OrderSide.SELL_SHORT)
    assert can_buy_tripped is False
    assert "kill switch active" in buy_reason.lower()
    assert can_short_tripped is False

    # Defensive exits ARE PERMITTED
    can_sell_tripped, sell_reason = ks.can_submit_order(OrderSide.SELL)
    can_cover_tripped, cover_reason = ks.can_submit_order(OrderSide.BUY_TO_COVER)
    assert can_sell_tripped is True
    assert sell_reason is None
    assert can_cover_tripped is True
    assert cover_reason is None

    # Resetting restores entry permission
    ks.reset(actor="OPERATOR", reason="Circuit breaker review cleared")
    assert ks.is_tripped is False
    can_buy_after, _ = ks.can_submit_order(OrderSide.BUY)
    assert can_buy_after is True


# =============================================================================
# 5. MULTI-FILL & BROKER SIMULATION
# =============================================================================

def test_simulated_broker_multi_fill_and_slippage(policy, fresh_quote):
    """Verifies simulated broker execution with partial fills and square-root impact slippage."""
    ledger = ExecutionLedger()
    broker = SimulatedBroker(policy=policy)

    order = ledger.create_order(
        ticker="NVDA",
        side=OrderSide.BUY,
        requested_shares=100,
        signal_price=100.0,
        limit_price=100.0,
    )

    # Execute with two 50% partial fills
    updated_order, fills = broker.execute_order(
        order=order,
        quote=fresh_quote,
        ledger=ledger,
        partial_fill_ratios=[0.5, 0.5],
    )

    assert updated_order.status == OrderStatus.FILLED
    assert updated_order.filled_shares == 100
    assert len(fills) == 2
    assert fills[0].shares == 50
    assert fills[1].shares == 50
    # Slippage increased fill price slightly above quoted last_price of 100.0
    assert updated_order.avg_fill_price > 100.0
    assert updated_order.total_commission > 0.0


def test_simulated_broker_rejection(policy, fresh_quote):
    """Verifies simulated broker order rejection handling."""
    ledger = ExecutionLedger()
    broker = SimulatedBroker(policy=policy)

    order = ledger.create_order(
        ticker="NVDA",
        side=OrderSide.BUY,
        requested_shares=50,
        signal_price=100.0,
    )

    updated_order, fills = broker.execute_order(
        order=order,
        quote=fresh_quote,
        ledger=ledger,
        simulate_rejection=True,
        rejection_reason="BORROW_LOCATE_FAILED",
    )

    assert updated_order.status == OrderStatus.REJECTED
    assert updated_order.rejection_reason == "BORROW_LOCATE_FAILED"
    assert len(fills) == 0


# =============================================================================
# 6. DATABASE INTEGRATION & AUDIT TRAIL
# =============================================================================

def test_database_persistence_orders_and_fills(policy, fresh_quote):
    """Verifies SQLite database persistence of execution orders, fills, and events."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        temp_db_path = tf.name

    try:
        db.init_db(temp_db_path)
        ledger = ExecutionLedger()
        broker = SimulatedBroker(policy=policy)

        order = ledger.create_order(
            ticker="NVDA",
            side=OrderSide.BUY,
            requested_shares=50,
            signal_price=100.0,
        )

        broker.execute_order(
            order=order,
            quote=fresh_quote,
            ledger=ledger,
            partial_fill_ratios=[1.0],
            persist_db=True,
            db_path=temp_db_path,
        )

        # Query database tables directly
        db_orders = db.get_execution_orders(ticker="NVDA", db_path=temp_db_path)
        assert len(db_orders) == 1
        assert db_orders[0]["order_id"] == order.order_id
        assert db_orders[0]["status"] == "FILLED"
        assert db_orders[0]["filled_shares"] == 50

        db_fills = db.get_execution_fills(order_id=order.order_id, db_path=temp_db_path)
        assert len(db_fills) == 1
        assert db_fills[0]["shares"] == 50

        db_events = db.get_execution_events(order_id=order.order_id, db_path=temp_db_path)
        assert len(db_events) >= 2  # Acknowledged and Fill events
    finally:
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)


# =============================================================================
# 7. ACCEPTANCE GATE: RISK SIZING, CONCENTRATION & RECONCILIATION
# =============================================================================

def test_acceptance_gate_full_simulated_portfolio_lifecycle():
    """
    Step 7.3 Acceptance Gate:
    1. Position sizing respects the risk budget and all concentration limits in simulated portfolios.
    2. The system can reconcile its internal position state with the execution ledger.
    3. Any discrepancy automatically trips the operational kill switch.
    """
    policy = get_default_policy()
    kill_switch = OperationalKillSwitch()
    reconciler = PositionReconciler(policy=policy, kill_switch=kill_switch)
    ledger = ExecutionLedger()
    broker = SimulatedBroker(policy=policy)

    initial_equity = 100_000.0
    initial_cash = 100_000.0

    # Start with empty internal portfolio
    internal_portfolio = PortfolioSnapshot(
        portfolio_equity=initial_equity,
        cash=initial_cash,
        peak_equity=initial_equity,
        positions={},
    )

    # 1. Size Position Respecting Risk Budget & Concentration Caps
    sizing = calculate_risk_based_position_size(
        entry_price=100.0,
        stop_loss=95.0,
        ticker="NVDA",
        direction="LONG",
        portfolio_equity=internal_portfolio.portfolio_equity,
        portfolio_snapshot=internal_portfolio,
        adv_shares_20d=2_000_000.0,
        atr=2.50,
        composite_score=75.0,
    )
    assert sizing.shares_allowed > 0
    # Single stock cap (10% of $100k = $10,000 / $100 = 100 shares max)
    assert sizing.shares_allowed <= 100
    assert sizing.position_equity_pct <= 0.10

    # 2. Pre-Flight Validation
    quote_nvda = MarketQuote(
        ticker="NVDA",
        last_price=100.0,
        bid_price=99.98,
        ask_price=100.02,
        timestamp=datetime.now(timezone.utc),
        adv_shares_20d=2_000_000.0,
        atr_14d=2.50,
    )
    validator = PreFlightOrderValidator(policy=policy, kill_switch=kill_switch)
    pre_flight = validator.validate(
        ticker="NVDA",
        side=OrderSide.BUY,
        requested_shares=sizing.shares_allowed,
        quote=quote_nvda,
        portfolio=internal_portfolio,
        signal_price=100.0,
        limit_price=100.0,
        stop_loss_price=95.0,
        target_price=110.0,
    )
    assert pre_flight.is_valid is True

    # 3. Create, Submit, and Execute Order through Broker into Execution Ledger
    order = ledger.create_order(
        ticker="NVDA",
        side=OrderSide.BUY,
        requested_shares=sizing.shares_allowed,
        signal_price=100.0,
        limit_price=100.0,
        stop_loss_price=95.0,
        target_price=110.0,
    )
    executed_order, fills = broker.execute_order(
        order=order,
        quote=quote_nvda,
        ledger=ledger,
        partial_fill_ratios=[1.0],
    )
    assert executed_order.status == OrderStatus.FILLED

    # 4. Update Internal Portfolio State based on Execution Fills
    trade_fill = fills[0]
    internal_portfolio.positions["NVDA"] = Position(
        ticker="NVDA",
        direction="LONG",
        shares=trade_fill.shares,
        entry_price=trade_fill.price,
        current_price=quote_nvda.last_price,
        sector="Technology",
        factor_tags=["AI_SEMICONDUCTORS"],
    )
    internal_portfolio.cash -= (trade_fill.shares * trade_fill.price + trade_fill.commission)

    # 5. Reconcile Internal Portfolio against Execution Ledger -> Zero Discrepancy
    recon = reconciler.reconcile(
        portfolio=internal_portfolio,
        ledger=ledger,
        initial_cash=initial_cash,
    )
    assert recon.is_reconciled is True
    assert len(recon.discrepancies) == 0
    assert recon.internal_positions["NVDA"] == sizing.shares_allowed
    assert recon.ledger_positions["NVDA"] == sizing.shares_allowed
    assert recon.internal_cash == pytest.approx(recon.ledger_cash)
    assert kill_switch.is_tripped is False

    # 6. Introduce An Intentional Discrepancy (e.g. Desynchronized internal position)
    internal_portfolio.positions["NVDA"].shares += 10  # Phantom 10 shares added
    recon_bad = reconciler.reconcile(
        portfolio=internal_portfolio,
        ledger=ledger,
        initial_cash=initial_cash,
    )
    assert recon_bad.is_reconciled is False
    assert len(recon_bad.discrepancies) > 0
    assert recon_bad.discrepancies[0].discrepancy_type == "POSITION_SHARES_MISMATCH"

    # 7. Verify Operational Kill Switch Automatically Tripped and Froze New Entries
    assert kill_switch.is_tripped is True
    assert kill_switch.trip_reason == KillSwitchReason.ORDER_RECONCILIATION_FAILURE

    # Next candidate trade is immediately blocked by kill switch
    can_enter, reason = kill_switch.can_submit_order(OrderSide.BUY)
    assert can_enter is False
    assert "ORDER_RECONCILIATION_FAILURE" in reason
