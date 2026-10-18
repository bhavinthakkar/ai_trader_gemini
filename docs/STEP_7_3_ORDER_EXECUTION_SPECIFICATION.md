# Specification: Order Execution Assumptions, Pre-Flight Validation, Kill Switch, and Ledger Reconciliation (Step 7.3)

## 1. Overview & Institutional Mandate

In institutional quantitative trading, generating an alpha signal or a trade recommendation is only the first step. Translating that recommendation into an actual market position introduces real-world execution friction, slippage, market impact, exchange latency, data freshness risks, and potential operational errors.

A fundamental rule of institutional systems is that **a trade signal is not an executed order**:
> **Invariant**: An analytical `BUY` or `SELL_SHORT` recommendation (`SIGNAL_GENERATED`) must **never** be recorded as an executed trade. The execution layer must maintain strict state machine separation across the entire lifecycle:
> `SIGNAL_GENERATED` $\to$ `ORDER_SUBMITTED` $\to$ `BROKER_ACKNOWLEDGED` $\to$ `PARTIALLY_FILLED` $\to$ `FILLED` (or `CANCELLED` / `REJECTED`).

Furthermore, before an order can be transmitted to an execution gateway, it must pass a deterministic **Pre-Flight Order Validation Layer**, and the overall platform must be guarded by an **Operational Kill Switch** that halts all new risk entries when market data staleness, position reconciliation discrepancies, or risk breaches occur. Finally, internal portfolio states must be continually reconciled against the immutable **Execution Ledger** to guarantee zero-discrepancy integrity.

```mermaid
flowchart TD
    subgraph SignalLayer ["1. Signal Generation Layer"]
        Sig["Upstream Recommendation<br/>(Structured Decision / Compliance Gates)"]
        SigState["Status: SIGNAL_GENERATED<br/>(NOT an executed trade)"]
        Sig --> SigState
    end

    subgraph PreFlightLayer ["2. Pre-Flight Order Validation Layer"]
        KS_Check{"Operational Kill Switch Active?"}
        Data_Check{"Market Data Fresh & Not Halted?<br/>(Age <= 300s, Last > 0)"}
        Pos_Check{"Position Alignment OK?<br/>(No Long/Short Direction Conflict)"}
        Cash_Check{"Free Cash >= 10% Reserve?<br/>(Trade Cost + Margin Covered)"}
        Liq_Check{"Liquidity & Market Impact OK?<br/>(<= 1% ADV, Impact <= 15 bps)"}
        Geometry_Check{"Stop & Target Price Valid?<br/>(0 < Stop < P_live < Target, Drift <= 2%)"}
        Idemp_Check{"Duplicate Order Check?<br/>(Deterministic SHA-256 Idempotency Key)"}
    end

    subgraph KillSwitch ["Operational Kill Switch (Fail-Closed)"]
        Tripped["Kill Switch Tripped!<br/>- DATA_QUALITY_FAILURE<br/>- ORDER_RECONCILIATION_FAILURE<br/>- RISK_MONITORING_FAILURE<br/>- MANUAL_OPERATOR_TRIP"]
        Tripped -->|Block| BlockNew["Block New Entries (BUY, SELL_SHORT)"]
        Tripped -->|Allow| AllowExit["Permit Defensive Exits (SELL, BUY_TO_COVER)"]
    end

    subgraph ExecutionLedger ["3. Execution Ledger & Broker Interaction"]
        OrderSub["Status: ORDER_SUBMITTED"]
        BrokerAck["Status: BROKER_ACKNOWLEDGED"]
        PartFill["Status: PARTIALLY_FILLED"]
        FullFill["Status: FILLED"]
        OrderSub --> BrokerAck --> PartFill --> FullFill
    end

    subgraph Reconciliation ["4. Position & Cash Reconciliation"]
        LedgerDerived["Derived Ledger Net Shares:<br/>sum(+BUY -SELL -SHORT +COVER)"]
        InternalSnapshot["Internal Portfolio Snapshot:<br/>Active Positions & Available Cash"]
        Reconciler{"Zero Discrepancy Reconciliation"}
        LedgerDerived & InternalSnapshot --> Reconciler
        Reconciler -->|Discrepancy Detected| Tripped
        Reconciler -->|Zero Discrepancy| Reconciled["Portfolio & Ledger Synchronized"]
    end

    SigState --> KS_Check
    KS_Check -->|Active (Entry Order)| RejectKS["REJECTED: KILL_SWITCH_ACTIVE"]
    KS_Check -->|Pass| Data_Check
    Data_Check -->|Stale / Halted| RejectData["REJECTED: MARKET_DATA_STALE / HALTED"]
    Data_Check -->|Pass| Pos_Check
    Pos_Check -->|Conflict| RejectPos["REJECTED: POSITION_DIRECTION_CONFLICT"]
    Pos_Check -->|Pass| Cash_Check
    Cash_Check -->|Cash Reserve Breached| RejectCash["REJECTED: INSUFFICIENT_CASH / MARGIN"]
    Cash_Check -->|Pass| Liq_Check
    Liq_Check -->|Participation / Impact Breached| RejectLiq["REJECTED: LIQUIDITY_LIMIT_EXCEEDED"]
    Liq_Check -->|Pass| Geometry_Check
    Geometry_Check -->|Inverted Stop / Target Hit / Drift > 2%| RejectGeom["REJECTED: INVALID_STOP / TARGET / DRIFT"]
    Geometry_Check -->|Pass| Idemp_Check
    Idemp_Check -->|Duplicate Key| RejectIdemp["REJECTED: DUPLICATE_ORDER_IDEMPOTENCY"]
    Idemp_Check -->|Pass All Pre-Flight| OrderSub
    FullFill --> LedgerDerived
```

---

## 2. Order Lifecycle State Machine

The order state machine decouples alpha decisions from execution results:

| State | Definition | Is Terminal | Permitted Transitions |
| :--- | :--- | :---: | :--- |
| `SIGNAL_GENERATED` | An analytical buy/sell signal has been produced by the quantitative engine. No capital is at risk. | No | $\to$ `ORDER_SUBMITTED`, `REJECTED` |
| `ORDER_SUBMITTED` | An order has been constructed with a deterministic client order ID and idempotency key, validated, and sent to the execution router. | No | $\to$ `BROKER_ACKNOWLEDGED`, `REJECTED`, `CANCELLED` |
| `BROKER_ACKNOWLEDGED` | The execution venue or broker has received the order and assigned an exchange order ID. | No | $\to$ `PARTIALLY_FILLED`, `FILLED`, `CANCELLED`, `REJECTED` |
| `PARTIALLY_FILLED` | At least one trade fill has been matched ($0 < \text{filled\_shares} < \text{requested\_shares}$). | No | $\to$ `PARTIALLY_FILLED`, `FILLED`, `CANCELLED` |
| `FILLED` | All requested shares have been completely executed ($\text{filled\_shares} == \text{requested\_shares}$). | **Yes** | None (terminal) |
| `CANCELLED` | The order was cancelled by the user, time-in-force expiry, or the kill switch before full completion. Partial fills remain in the ledger. | **Yes** | None (terminal) |
| `REJECTED` | The order was rejected by pre-flight validation or by the broker/exchange gateway. | **Yes** | None (terminal) |

---

## 3. Pre-Flight Order Validation Requirements

Before any order can be dispatched, `PreFlightOrderValidator.validate()` executes the following mandatory pre-flight checks:

### 3.1 Market-Data Freshness & Tradability
- **Non-Positive Price Guard**: Rejects quotes where $P_{\text{last}} \le 0$ or quote data is missing (`NON_POSITIVE_PRICE`).
- **Trading Halt Guard**: Rejects instruments with active trading halts (`MARKET_HALTED`).
- **Staleness Threshold**: Calculates quote age in seconds:
  $$\Delta t = t_{\text{ref}} - t_{\text{quote}}$$
  If $\Delta t > \text{max\_quote\_staleness\_seconds}$ (default: 300.0s), the order is rejected with `MARKET_DATA_STALE`. If configured, this immediately trips the operational kill switch (`DATA_QUALITY_FAILURE`).

### 3.2 Position Alignment & Directional Conflicts
- **No Long/Short Overlap**:
  - If the portfolio holds a `SHORT` position in the ticker, opening a `BUY` order without an explicit `BUY_TO_COVER` is blocked (`POSITION_DIRECTION_CONFLICT`).
  - If the portfolio holds a `LONG` position, opening a `SELL_SHORT` order without an explicit `SELL` is blocked (`POSITION_DIRECTION_CONFLICT`).
- **Holding Quantity Enforcement**:
  - A `SELL` (exit) order cannot request more shares than currently held long (`EXCEEDS_HOLDING_QUANTITY`).
  - A `BUY_TO_COVER` order cannot request more shares than currently held short (`EXCEEDS_HOLDING_QUANTITY`).

### 3.3 Cash Balance & Reserve Buffer
- **Long Entry (`BUY`)**:
  $$\text{Required Cash} = \text{shares} \times P_{\text{ask}} \times (1 + \text{slippage}) + \text{commission}$$
  $$\text{Post-Trade Cash} = \text{Cash} - \text{Required Cash}$$
  $$\text{Condition}: \text{Post-Trade Cash} \ge E \times \text{min\_cash\_reserve\_pct} \quad (10\% \text{ of Equity})$$
  Failing this check rejects with `INSUFFICIENT_CASH`.
- **Short Entry (`SELL_SHORT`)**:
  $$\text{Required Margin} = \text{shares} \times P_{\text{bid}} \times \text{short\_margin\_requirement\_pct} + \text{commission}$$
  $$\text{Post-Trade Cash} = \text{Cash} - \text{Required Margin} \ge E \times \text{min\_cash\_reserve\_pct}$$
  Failing this check rejects with `INSUFFICIENT_MARGIN`.

### 3.4 Liquidity & Market Impact Constraints
- **Participation Limit**: Order shares must not exceed $1.0\%$ of 20-day Average Daily Volume ($0.01 \times \text{ADV}_{20d}$). Exceeding this cap rejects with `LIQUIDITY_LIMIT_EXCEEDED`.
- **Square-Root Market Impact**:
  $$\text{Impact}_{\text{bps}} = 10000 \times \left(\frac{\text{ATR}_{14d}}{P_{\text{last}}}\right) \times \sqrt{\frac{\text{shares}}{\text{ADV}_{20d}}}$$
  If $\text{Impact}_{\text{bps}} > 15.0 \text{ bps}$, the order is rejected with `MARKET_IMPACT_EXCEEDED`.

### 3.5 Stop and Target Validity Against Live Price
Guards against slippage drift, inverted setup geometry, and dead-on-arrival trades:
- **Price Drift Guard**:
  $$\text{Drift} = \frac{|P_{\text{live}} - P_{\text{signal}}|}{P_{\text{signal}}}$$
  If $\text{Drift} > 2.0\%$, the order is rejected with `EXCESSIVE_PRICE_DRIFT`.
- **Long Geometry**:
  - Invariant: $0 < \text{Stop} < P_{\text{live}} < \text{Target}$.
  - If $P_{\text{live}} \le \text{Stop}$: Stock already breached stop loss! Rejected with `INVALID_STOP_PRICE`.
  - If $P_{\text{live}} \ge \text{Target}$: Profit target already reached! Rejected with `INVALID_TARGET_PRICE`.
  - Live Reward-to-Risk:
    $$\text{RR}_{\text{live}} = \frac{\text{Target} - P_{\text{live}}}{P_{\text{live}} - \text{Stop}} \ge 1.20$$
    If below hurdle, rejected with `SUB_MINIMAL_REWARD_RISK`.
- **Short Geometry**:
  - Invariant: $0 < \text{Target} < P_{\text{live}} < \text{Stop}$.
  - If $P_{\text{live}} \ge \text{Stop}$: Stock already breached short stop! Rejected with `INVALID_STOP_PRICE`.
  - If $P_{\text{live}} \le \text{Target}$: Short target already hit! Rejected with `INVALID_TARGET_PRICE`.
  - Live Short Reward-to-Risk:
    $$\text{RR}_{\text{live}} = \frac{P_{\text{live}} - \text{Target}}{\text{Stop} - P_{\text{live}}} \ge 1.50$$
    If below hurdle, rejected with `SUB_MINIMAL_REWARD_RISK`.

### 3.6 Deterministic Idempotency Key & Duplicate Prevention
To prevent duplicate execution from retries or race conditions, each order generates a deterministic SHA-256 hash:
$$\text{Key} = \text{SHA256}\left(\text{Ticker} \parallel \text{Side} \parallel \text{Shares} \parallel P_{\text{bucket}} \parallel \text{SessionDate} \parallel \text{RunId}\right)$$
If an order with this key is already active or filled, subsequent submissions are blocked with `DUPLICATE_ORDER_IDEMPOTENCY`.

---

## 4. Operational Kill Switch Mechanism

The `OperationalKillSwitch` provides fail-closed emergency circuit breaker protection:

### 4.1 Trip Triggers
1. `DATA_QUALITY_FAILURE`: Market quote feed outage, price staleness $> 300$s, or corrupted non-positive prices.
2. `ORDER_RECONCILIATION_FAILURE`: Position state desynchronization between internal portfolio holdings and the execution ledger.
3. `RISK_MONITORING_FAILURE`: Drawdown circuit breaker triggered (drawdown $\ge 10\%$), severe cash deficit, or margin violation.
4. `MANUAL_OPERATOR_TRIP`: Operator emergency desk stop.

### 4.2 Asymmetric Entry vs. Exit Protection
When the kill switch is tripped:
- **New risk entries are blocked**: Candidate `BUY` and `SELL_SHORT` orders are rejected immediately with `KILL_SWITCH_ACTIVE`.
- **Defensive risk-reducing exits are permitted**: `SELL` (closing long) and `BUY_TO_COVER` (closing short) orders remain fully executable so the trading desk can derisk and mitigate losses.

---

## 5. Execution Ledger & Position Reconciliation

The `ExecutionLedger` is an append-only, audit-logged store of all orders, fills, and transition events.

### 5.1 Mathematical Position Derivation
The ledger is the single source of truth. Net shares for each ticker are derived directly from matched fills:
$$\text{Shares}_{\text{ticker}} = \sum_{\text{fills}} \left(\text{shares} \times \text{sign}(\text{side})\right)$$
where $\text{sign}(\text{BUY}) = +1$, $\text{sign}(\text{SELL}) = -1$, $\text{sign}(\text{SELL\_SHORT}) = -1$, and $\text{sign}(\text{BUY\_TO_COVER}) = +1$.

### 5.2 Mathematical Cash Derivation
$$\text{Cash}_{\text{ledger}} = \text{Cash}_{\text{initial}} - \sum_{\text{BUY}} (P \cdot S + C) + \sum_{\text{SELL}} (P \cdot S - C) + \sum_{\text{SHORT}} (P \cdot S - C) - \sum_{\text{COVER}} (P \cdot S + C)$$

### 5.3 Automated Position Reconciliation
`PositionReconciler.reconcile()` compares internal portfolio state against derived ledger positions:
- **Share Discrepancies**: Checks $\text{Shares}_{\text{internal}} == \text{Shares}_{\text{ledger}}$. Flags `POSITION_SHARES_MISMATCH`.
- **Direction Inversions**: Flags `DIRECTION_MISMATCH`.
- **Phantom Positions**: Flags `PHANTOM_INTERNAL_POSITION` or `PHANTOM_LEDGER_POSITION`.
- **Cash Reconciliation**: Checks $|\text{Cash}_{\text{internal}} - \text{Cash}_{\text{ledger}}| \le \$0.01$. Flags `CASH_MISMATCH`.
- **Kill Switch Tripping**: Any detected discrepancy immediately engages the operational kill switch (`ORDER_RECONCILIATION_FAILURE`), freezing further risk.

---

## 6. Database Schema & Persistence

Step 7.3 adds four persistent tables to `db.py`:
1. `execution_orders`: Full order lifecycle, prices, status, shares, commission, idempotency key.
2. `execution_fills`: Individual fill prices, quantities, commission, exchange execution ID, slippage bps.
3. `execution_events`: Comprehensive audit log tracking all state transitions from `SIGNAL_GENERATED` to `FILLED`.
4. `operational_kill_switch_state`: Audit log of kill switch trips, resets, actors, and reasons.

---

## 7. Verification & Acceptance Gate Health

Automated test suite [`tests/test_order_execution.py`](file:///home/bhavin/Desktop/git/ai_trader_gemini/tests/test_order_execution.py) verifies:
- **17/17 tests passing**:
  - `test_state_machine_distinct_lifecycle_progression`: State machine separation and multi-leg fill tracking.
  - `test_order_cancellation_and_rejection`: Terminal state handling.
  - `test_pre_flight_validation_success`: Full validation certification.
  - `test_pre_flight_stale_market_data`: 300s quote staleness check and automatic kill switch engagement.
  - `test_pre_flight_market_halt_rejection`: Trading halt check.
  - `test_pre_flight_cash_sufficiency_and_reserve`: 10% cash reserve enforcement.
  - `test_pre_flight_liquidity_cap_and_market_impact`: 1% ADV and 15 bps square-root impact limits.
  - `test_pre_flight_dead_on_arrival_stop_loss`: Inverted stop loss validation.
  - `test_pre_flight_target_already_achieved`: Inverted target price validation.
  - `test_pre_flight_excessive_price_drift`: 2.0% price drift guard.
  - `test_pre_flight_position_direction_conflict`: Direction conflict prevention.
  - `test_idempotency_key_deterministic_and_duplicate_rejection`: SHA-256 idempotency.
  - `test_operational_kill_switch_blocks_entries_permits_exits`: Entry-blocking vs. exit-permitting behavior.
  - `test_simulated_broker_multi_fill_and_slippage`: Multi-leg execution and square-root slippage modeling.
  - `test_simulated_broker_rejection`: Broker rejection handling.
  - `test_database_persistence_orders_and_fills`: SQLite ledger persistence.
  - `test_acceptance_gate_full_simulated_portfolio_lifecycle`: Sizing under risk budget and concentration limits + zero-discrepancy ledger reconciliation + automatic kill switch engagement on discrepancy.
- **Repository Regression Suite**: **559 passed, 1 skipped** across all test files.
- **Unit Test Discovery**: **255 passed** across all unit tests.
