# Institutional Quantitative Scoring Engine Specification
## Step 2.1: Factor Calculation & Mathematical Transformation Standards

**Document Version**: 2.1.0  
**Status**: APPROVED SPECIFICATION  
**Scope**: Deterministic 5-Pillar Engine, Financial Quality Guard, Volatility Factor, and Setup Geometry  
**Repository**: `ai_trader_gemini`  

---

## 1. Executive Summary & Design Principles

The deterministic quantitative engine evaluates candidate equities across five independent fundamental and technical dimensions (the **5 Pillars**), incorporates a **Financial Quality & Value-Trap Guard**, and scales through an **ATR% Volatility Factor Dampener**.

### Core Mathematical Constraints
1. **Directional Consistency**: For every factor, metric, and pillar score $S \in [0.0, 100.0]$, **higher values MUST strictly indicate a more favorable setup for a long position** (e.g., stronger trend, higher market/sector alpha, more attractive valuation discount, higher financial solvency).
2. **Yield Space Multiples**: Raw valuation multiples ($P/E$, $EV/EBITDA$, $P/FCF$) exhibit fatal mathematical singularities (division by zero, asymptotic sign-flipping for negative earnings). All valuation metrics MUST be computed in **Yield Space** ($E/P$, $EBITDA/EV$, $FCF/EV$), which remain continuous, monotonic, and bounded across $[-\infty, +\infty]$.
3. **Strict Point-in-Time Availability**: No calculation may ingest data whose publication or availability timestamp succeeds the decision timestamp $t$ (e.g., Form 10-Q/10-K filing availability dates, 45-day lag for 13F holdings).
4. **Transparent Missingness & No Artificial Neutrality**: Missing inputs must never be masked with an arbitrary $50.0$ default. Missing inputs produce an explicit `None` pillar score, trigger status tracking (`"measured"`, `"partial"`, `"unavailable"`), and invoke dynamic weight re-normalization across verified pillars.

---

## 2. Factor Specification by Pillar

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           5-PILLAR QUANTITATIVE ENGINE                           │
├──────────────────┬──────────────────┬──────────────────┬─────────────────────────┤
│ Pillar           │ Weight ($w_i$)   │ Normalization    │ Primary Horizon         │
├──────────────────┼──────────────────┼──────────────────┼─────────────────────────┤
│ Historical Trend │ 25%              │ Time-Series      │ Daily / 20 / 50 / 200d  │
│ Sector Relative  │ 20%              │ Cross-Sectional  │ 5d (20%), 20d (50%), 60d│
│ Market Alpha     │ 20%              │ Cross-Sectional  │ 20d Jensen's α + Excess │
│ Valuation History│ 15%              │ Time-Series      │ 3-Year Rolling (756d)   │
│ Peer Valuation   │ 20%              │ Cross-Sectional  │ Point-in-Time Peer Group│
└──────────────────┴──────────────────┴──────────────────┴─────────────────────────┘
```

---

### Pillar 1: Historical Trend ($S_{\text{trend}}$) — 25% Weight

#### 1.1 Mathematical Formulation
The Historical Trend pillar combines hierarchical Exponential Moving Average (EMA) structural alignment with 14-day Relative Strength Index (RSI) momentum and velocity.

$$S_{\text{trend}}(t) = 0.60 \times S_{\text{EMA}}(t) + 0.40 \times S_{\text{RSI}}(t)$$

#### 1.2 EMA Structural Alignment & Slope ($S_{\text{EMA}}$)
- **Lookback Windows**: $\text{EMA}_{20}$, $\text{EMA}_{50}$, and $\text{EMA}_{200}$ computed on adjusted closing prices.
- **Hierarchical Alignment State**:
  - Perfect Bullish Stack: $P_t > \text{EMA}_{20}(t) > \text{EMA}_{50}(t) > \text{EMA}_{200}(t)$ (Base = 85 pts)
  - Golden Cross State: $P_t > \text{EMA}_{20}(t) > \text{EMA}_{50}(t)$ and $\text{EMA}_{50}(t) \le \text{EMA}_{200}(t)$ (Base = 65 pts)
  - Mixed / Consolidation: $P_t$ oscillating between $\text{EMA}_{20}$ and $\text{EMA}_{50}$ (Base = 50 pts)
  - Breakdown State: $P_t < \text{EMA}_{20}(t)$ or $\text{EMA}_{20}(t) < \text{EMA}_{50}(t)$ (Base = 35 pts)
  - Perfect Bearish Stack: $P_t < \text{EMA}_{20}(t) < \text{EMA}_{50}(t) < \text{EMA}_{200}(t)$ (Base = 15 pts)
- **Slope Velocity Adjustment**:
  $$\text{Slope}_{20}(t) = \frac{\text{EMA}_{20}(t) - \text{EMA}_{20}(t-5)}{5 \times \text{ATR}_{14}(t)}$$
  $$\Delta S_{\text{slope}} = \text{clamp}\left(\text{Slope}_{20}(t) \times 15.0, -15.0, +15.0\right)$$
  $$S_{\text{EMA}}(t) = \text{clamp}\left(\text{Base}_{\text{align}} + \Delta S_{\text{slope}}, 0.0, 100.0\right)$$

#### 1.3 RSI Level & RSI Velocity ($S_{\text{RSI}}$)
- **RSI Calculation**: Standard Wilder 14-period RSI on closing prices.
- **RSI Velocity ($\Delta \text{RSI}$)**: 5-session rate of change:
  $$\text{RSI\_Velocity}_5(t) = \frac{\text{RSI}_{14}(t) - \text{RSI}_{14}(t-5)}{5}$$
- **Momentum Regime Scoring**:
  - **Bullish Momentum Acceleration**: $50.0 \le \text{RSI}_{14}(t) \le 68.0$ and $\text{RSI\_Velocity}_5(t) > 0$:
    $$S_{\text{RSI}} = 65.0 + \left(\frac{\text{RSI}_{14}(t) - 50.0}{18.0}\right) \times 20.0 + \text{clamp}(\text{RSI\_Velocity}_5 \times 3.0, 0, 15)$$
  - **Mean-Reversion Oversold Bounce**: $\text{RSI}_{14}(t) < 32.0$ with positive inflection ($\text{RSI\_Velocity}_5 > +1.5$ pts/day):
    $$S_{\text{RSI}} = 45.0 + (\text{RSI\_Velocity}_5 \times 5.0)$$
  - **Severe Downtrend / Unreclaimed Washout**: $\text{RSI}_{14}(t) < 32.0$ with flat/negative velocity:
    $$S_{\text{RSI}} = \max(5.0, \text{RSI}_{14}(t) \times 0.8)$$
  - **Overbought Blow-off / Exhaustion**: $\text{RSI}_{14}(t) > 75.0$:
    $$S_{\text{RSI}} = \max(20.0, 85.0 - (\text{RSI}_{14}(t) - 75.0) \times 4.0)$$

---

### Pillar 2: Sector Relative Performance ($S_{\text{sector}}$) — 20% Weight

#### 2.1 Multi-Horizon Sector Excess Return
Evaluates the equity's total return relative to its primary SPDR Sector ETF across three distinct session lookbacks:
- Short-term tactical: $\tau = 5$ sessions (weight = 20%)
- Core swing objective: $\tau = 20$ sessions (weight = 50%)
- Intermediate trend context: $\tau = 60$ sessions (weight = 30%)

For each horizon $\tau \in \{5, 20, 60\}$:
$$R_{i, \tau}(t) = \frac{P_{i, t}}{P_{i, t-\tau}} - 1, \quad R_{S, \tau}(t) = \frac{P_{S, t}}{P_{S, t-\tau}} - 1$$
$$\Delta R_{i/S, \tau}(t) = R_{i, \tau}(t) - R_{S, \tau}(t)$$

#### 2.2 Composite Sector Excess Metric
$$\text{Sector\_Excess}(t) = 0.20 \times \Delta R_{i/S, 5}(t) + 0.50 \times \Delta R_{i/S, 20}(t) + 0.30 \times \Delta R_{i/S, 60}(t)$$

#### 2.3 Comprehensive GICS-11 Sector ETF Mapping Table

| GICS Sector Name | Sector Code | Primary Sector Benchmark ETF | Secondary / Non-US Fallback |
|---|---|---|---|
| Information Technology | 45 | **XLK** (Technology Select Sector SPDR) | IXN / EXV6 |
| Communication Services | 50 | **XLC** (Communication Services Select SPDR) | IXP |
| Consumer Discretionary | 25 | **XLY** (Consumer Discretionary SPDR) | RXI |
| Consumer Staples | 30 | **XLP** (Consumer Staples Select SPDR) | KXI |
| Financials | 40 | **XLF** (Financial Select Sector SPDR) | IXG / EXH1 |
| Health Care | 35 | **XLV** (Health Care Select Sector SPDR) | IXJ |
| Energy | 10 | **XLE** (Energy Select Sector SPDR) | IXC |
| Industrials | 20 | **XLI** (Industrial Select Sector SPDR) | EXI |
| Materials | 15 | **XLB** (Materials Select Sector SPDR) | MXI |
| Real Estate | 60 | **XLRE** (Real Estate Select Sector SPDR) | REET |
| Utilities | 55 | **XLU** (Utilities Select Sector SPDR) | JXI |

**Fallback Hierarchy**:
1. If the sector is unmapped or unknown, map to broad market benchmark **SPY** and mark `sector_fallback: "BROAD_MARKET"`.
2. For cross-listed international equities without an active US sector ETF quote, map to the regional benchmark (e.g. `VGK` for Europe, `EWY` for South Korea).

#### 2.4 Score Normalization
The composite excess return is mapped to $[0, 100]$ using a symmetric logistic transformation calibrated to sector return dispersion ($\sigma_{\text{sector}} \approx 4.0\%$ over 20 sessions):
$$S_{\text{sector}}(t) = 100.0 \times \frac{1}{1 + e^{-k \cdot \text{Sector\_Excess}(t)}}, \quad \text{where } k = 25.0$$
- $\text{Sector\_Excess} = 0.0\% \implies S_{\text{sector}} = 50.0$
- $\text{Sector\_Excess} = +4.0\% \implies S_{\text{sector}} = 73.1$
- $\text{Sector\_Excess} = -4.0\% \implies S_{\text{sector}} = 26.9$

---

### Pillar 3: Market Alpha vs. SPY ($S_{\text{alpha}}$) — 20% Weight

#### 3.1 Resolving Single-Horizon vs. Multi-Horizon Excess Return
> [!IMPORTANT]
> A single 5-session return window is excessively noisy and conflates market beta with genuine idiosyncratic alpha. The Market Alpha pillar MUST integrate **rolling 60-day beta-adjusted Jensen's Alpha** ($\alpha_{20}$) and **20-session excess return**, with 5-session momentum serving strictly as a minor tactical confirmation.

#### 3.2 Formulation
1. **Rolling 60-Day Market Beta**:
   $$\beta_{i, 60}(t) = \frac{\text{Cov}_{60}\left(r_{i}, r_{\text{SPY}}\right)}{\text{Var}_{60}\left(r_{\text{SPY}}\right)}$$
2. **20-Session Jensen's Alpha**:
   $$\alpha_{i, 20}(t) = R_{i, 20}(t) - \left[ r_{f, 20} + \beta_{i, 60}(t) \cdot \left(R_{\text{SPY}, 20}(t) - r_{f, 20}\right) \right]$$
   *(where $r_{f, 20} = \text{US 3M T-bill rate} \times \frac{20}{252}$)*
3. **Total Excess Return**:
   $$\Delta R_{i/\text{SPY}, 20}(t) = R_{i, 20}(t) - R_{\text{SPY}, 20}(t)$$
   $$\Delta R_{i/\text{SPY}, 5}(t) = R_{i, 5}(t) - R_{\text{SPY}, 5}(t)$$
4. **Composite Alpha Metric**:
   $$\text{Alpha\_Metric}(t) = 0.50 \cdot \alpha_{i, 20}(t) + 0.35 \cdot \Delta R_{i/\text{SPY}, 20}(t) + 0.15 \cdot \Delta R_{i/\text{SPY}, 5}(t)$$

#### 3.3 Score Transformation
Standardized by 60-session residual idiosyncratic volatility $\sigma_{\epsilon, i}$:
$$\text{Appraisal\_Ratio}(t) = \frac{\text{Alpha\_Metric}(t)}{\sigma_{\epsilon, i} \cdot \sqrt{20 / 252}}$$
$$S_{\text{alpha}}(t) = 100.0 \times \Phi\left(\frac{\text{Appraisal\_Ratio}(t)}{0.75}\right)$$
*(where $\Phi$ is the cumulative standard normal distribution function).*

---

### Pillar 4: Valuation History ($S_{\text{val\_hist}}$) — 15% Weight

#### 4.1 Data Sources & Enterprise Value Definitions
- **Earnings Yield ($EY$)**: $\text{Diluted EPS}_{\text{TTM}} / P_t$
- **Enterprise Value EBITDA Yield ($EBITDA\_EV$)**: $\text{EBITDA}_{\text{TTM}} / \text{Enterprise Value}_t$
  $$\text{Enterprise Value}_t = \text{MarketCap}_t + \text{TotalDebt}_t - \text{Cash\&Equiv}_t + \text{MinorityInterest}_t + \text{PreferredEquity}_t$$
- **Free Cash Flow Yield ($FCFY$)**: $\text{FCF}_{\text{TTM}} / \text{Enterprise Value}_t$

#### 4.2 Lookback Horizon & As-Of Point-in-Time Distribution
- **Lookback Period**: Rolling 756 trading sessions (3 calendar years) of historical yields.
- **Sampling Frequency**: Weekly (every Friday session close) across the trailing 756 sessions ($N \approx 156$ observations).
- **Point-in-Time Constraint**: For any historical date $\tau \le t$, fundamental data ($EPS, EBITDA, Debt$) must reflect only filings whose public availability timestamp preceded $\tau$.

#### 4.3 Time-Series Percentile Transformation
For each yield metric $X \in \{EY, EBITDA\_EV, FCFY\}$:
$$PR_X(t) = \frac{1}{N} \sum_{\tau=1}^N \mathbb{I}\left(X_\tau \le X_t\right) \times 100.0$$
Since higher yield = cheaper valuation = more attractive entry:
$$S_{\text{val\_hist}}(t) = 0.40 \cdot PR_{EY}(t) + 0.40 \cdot PR_{EBITDA\_EV}(t) + 0.20 \cdot PR_{FCFY}(t)$$

#### 4.4 Explicit Handling of Loss-Making Companies ($EPS \le 0$ or $EBITDA \le 0$)
1. **Mathematical Continuity**: Yields naturally accept negative numbers without asymptotes (e.g. $EY = -0.06$). A company whose yield moves from $-12\%$ to $-4\%$ is improving and its percentile rank will appropriately rise.
2. **Chronic Loss-Maker Penalty**:
   - If TTM EBITDA is negative and the historical 3-year median yield is positive, the current observation falls in the lowest historical decile ($PR < 10$).
   - If TTM EBITDA has been negative for $> 6$ consecutive quarters (structural non-viability), $S_{\text{val\_hist}}$ is **strictly capped at a ceiling of 20.0**, regardless of revenue growth.

---

### Pillar 5: Peer Valuation Multiples ($S_{\text{peer\_val}}$) — 20% Weight

#### 5.1 Dynamic Peer Selection Algorithm
Rather than relying on static hardcoded tables, peers are selected dynamically on decision date $t$:
1. **Universe Filter**: Same GICS Sub-Industry (8-digit).
2. **Market Capitalization Screen**: Peer market cap $M_{\text{peer}} \in [0.33 \times M_i, 3.0 \times M_i]$.
3. **Liquidity Hurdle**: 60-day Average Daily Traded Volume ($\text{ADTV}_{60}$) $> \$5.0\text{M}$.
4. **Peer Count Thresholds**:
   - Ideal peer group: $5 \le K \le 12$.
   - If $K < 5$: Broaden to GICS Industry (6-digit).
   - If still $K < 5$: Ingest top 10 liquid non-target constituents of the corresponding SPDR Sector ETF.

#### 5.2 Handling Non-Comparable Businesses & Outliers
- **Yield Space Calculation**: Multiples are compared strictly in yield space ($EY_{\text{peer}}, EBITDA\_EV_{\text{peer}}$).
- **Robust Central Tendency**: To protect against distorted single-peer outliers or conglomerates, the peer benchmark is computed using the **Median** and **Median Absolute Deviation (MAD)**:
  $$\text{Median\_Peer\_EY}(t) = \text{Median}\left(\{EY_j(t)\}_{j=1}^K\right)$$
  $$\text{MAD}_{EY}(t) = \text{Median}\left(\left| EY_j(t) - \text{Median\_Peer\_EY}(t) \right|\right)$$
  $$\sigma_{\text{robust}} = 1.4826 \times \text{MAD}_{EY}(t)$$

#### 5.3 Cross-Sectional Score Transformation
$$\Delta EY_{\text{peer}}(t) = EY_i(t) - \text{Median\_Peer\_EY}(t)$$
$$z_{EY}(t) = \text{clamp}\left(\frac{\Delta EY_{\text{peer}}(t)}{\max(\sigma_{\text{robust}}, 0.01)}, -3.0, +3.0\right)$$
$$S_{\text{peer\_val}}(t) = 100.0 \times \Phi\left(0.60 \cdot z_{EY}(t) + 0.40 \cdot z_{EBITDA\_EV}(t)\right)$$
- If the stock trades at an earnings yield higher than its peer median ($z > 0$), it is cheaper $\implies S_{\text{peer\_val}} > 50.0$.
- If both stock and peer median are loss-making ($EY < 0$), the stock with smaller burn (higher yield) scores higher.

---

## 3. Financial Quality & Value-Trap Guard

The base composite is adjusted for balance-sheet solvency and earnings sustainability before volatility scaling.

### 3.1 Financial Quality Score ($S_{\text{FQ}} \in [0, 100]$)
Evaluates 6 fundamental health metrics:
1. **Free Cash Flow Margin** ($\text{FCF} / \text{Revenue}$): $> 20\% \to +15\text{ pts}; < 0\% \to -15\text{ pts}$.
2. **Operating Margin**: $> 25\% \to +10\text{ pts}; < 0\% \to -10\text{ pts}$.
3. **Net Debt / EBITDA**: $\le 0.0$ (Net Cash) $\to +15\text{ pts}$; $> 3.5\text{x} \to -15\text{ pts}; > 4.5\text{x} \to -20\text{ pts}$.
4. **Interest Coverage** ($\text{EBIT} / \text{Interest}$): Net Cash or $> 8.0\text{x} \to +10\text{ pts}; < 1.5\text{x} \to -15\text{ pts}$.
5. **Earnings Quality** ($\text{OCF} / \text{Net Income}$ accrual test): $\ge 1.1\text{x} \to +10\text{ pts}; < 0.5\text{x} \to -15\text{ pts}$.
6. **Share Dilution Rate**: Net buybacks $\to +5\text{ pts}$; annual share count growth $> 5\% \to -8\text{ pts}$.

### 3.2 Deterministic Value-Trap Penalty (-15 pts) & Algorithmic Veto
A security is certified as a **Value Trap** if it appears quantitatively "cheap" on multiples but exhibits balance-sheet distress:
$$\text{Is\_Cheap} = \left(EY_i \ge \text{Median\_Peer\_EY} + 1.5\%\right) \lor \left(PR_{EY} \ge 80.0\right)$$
$$\text{Is\_Distressed} = \left(\frac{\text{Net Debt}}{\text{EBITDA}} > 3.5\text{x}\right) \lor \left(\text{FCF Margin} < 0.0\right) \lor \left(\text{Interest Coverage} < 2.0\text{x}\right) \lor \left(S_{\text{FQ}} < 40.0\right)$$

$$\text{Value\_Trap} = \text{Is\_Cheap} \land \text{Is\_Distressed}$$

**Consequences**:
1. Raw Composite penalized by **$-15.0$ points**.
2. Directional BUY decisions are **mechanically vetoed** and downgraded to `HOLD` (`no_trade_reason = "VALUE_TRAP"`).

---

## 4. ATR% Volatility Factor Dampener

To prevent high-beta or hyper-volatile equities from triggering buy signals during volatility explosions, the composite score is scaled by a deterministic dampener:

$$\text{ATR\%} = \frac{\text{ATR}_{14}(t)}{P_t} \times 100.0$$

$$\text{Vol\_Factor} = \begin{cases} 
1.00 & \text{if } \text{ATR\%} \le 1.5\% \quad (\text{Calm}) \\
0.95 & \text{if } 1.5\% < \text{ATR\%} \le 2.5\% \quad (\text{Normal}) \\
0.90 & \text{if } 2.5\% < \text{ATR\%} \le 4.0\% \quad (\text{Elevated}) \\
0.85 & \text{if } \text{ATR\%} > 4.0\% \quad (\text{Extreme Volatility})
\end{cases}$$

### Centred Volatility Transformation Formula
To avoid pushing neutral ($50.0$) scores into artificial sell territory ($50.0 \times 0.85 = 42.5 \implies$ SELL) and distorting directional meaning, the engine applies a **centred transformation anchored at the neutral midpoint (50.0)**:

$$Q_{\text{final}} = \text{clamp}\left(50.0 + \left(Q_{\text{adjusted}} - 50.0\right) \times f_{\text{vol}}, 0.0, 100.0\right)$$

* **Neutral Invariance**: $Q_{\text{adj}} = 50.0 \implies Q_{\text{final}} = 50.0$ for all $f_{\text{vol}} \in [0.0, 1.0]$.
* **Symmetric Conviction Contraction**:
  * Bullish setup ($Q_{\text{adj}} = 70.0$, $f_{\text{vol}} = 0.85$): $50.0 + (20.0) \times 0.85 = 67.0$.
  * Bearish setup ($Q_{\text{adj}} = 30.0$, $f_{\text{vol}} = 0.85$): $50.0 + (-20.0) \times 0.85 = 33.0$ (conviction reduced toward neutral, rather than artificially amplified toward extreme oversold).

### Score Dampening vs. Volatility-Based Position Sizing
* **Empirical Comparison**:
  * Applying volatility dampening directly to the directional score ($Q_{\text{final}}$) reduces predictive Information Coefficient (Rank IC drops from **+0.0560** to **+0.0496**) because cross-sectional ranking is distorted by ATR differences.
  * Preserving pure directional scores ($Q_{\text{adj}}$) for signal ranking and handling volatility **solely in portfolio position sizing** ($w_i \propto \frac{1}{\text{ATR\%}_i}$) preserves maximum signal alpha ($\text{Rank IC} = +0.0560$) while cutting realized portfolio volatility from **30.47% to 27.87%**.
  * The centred score dampener is retained specifically for downstream deterministic threshold gates (e.g. Gate 1 BUY floor $\ge 60.0$, Gate 4 SELL floor $\le 45.0$).

---

## 5. Normalization Architecture: Cross-Sectional vs. Time-Series

| Factor / Pillar | Transformation Type | Distribution Window | Lookahead Prevention Rule |
|---|---|---|---|
| **Pillar 1: Trend** | Time-Series | Rolling 200 sessions | Rolling past $t$ only; no future session prices |
| **Pillar 2: Sector** | Cross-Sectional & Time-Series | Trailing 252-day sector spread | Point-in-time sector ETF returns as of session $t$ close |
| **Pillar 3: Market Alpha** | Cross-Sectional Appraisal | Rolling 60-day idiosyncratic variance | SPY and stock returns strictly preceding decision date $t$ |
| **Pillar 4: Valuation History** | Time-Series | Rolling 756 sessions (3 years) | Yield history formed strictly from point-in-time filing dates |
| **Pillar 5: Peer Valuation** | Cross-Sectional | As-of decision date $t$ peer group | Only contemporaneous peer filings available by date $t$ |

---

## 6. Mathematical Handling of Edge Cases

```
                                    VALUATION INPUT
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    │                                             │
             EPS / EBITDA > 0                              EPS / EBITDA ≤ 0
                    │                                             │
            Standard Yields                               Yield Space Formula
         EY = EPS / P > 0                             EY = EPS / P ≤ 0 (Continuous)
      EBITDA_EV = EBITDA / EV                      EBITDA_EV = EBITDA / EV ≤ 0
                    │                                             │
                    └──────────────────────┬──────────────────────┘
                                           │
                            Is Historical Lookback Missing?
                                           │
                             ┌─────────────┴─────────────┐
                            YES                          NO
                             │                           │
                    Pillar UNAVAILABLE            3-Year Rolling PR
                    Dynamic Weight Re-norm        Loss History Evaluated
```

1. **Negative Earnings ($EPS \le 0$)**: Evaluated as negative earnings yield ($EY \le 0$). Prevents sign inversion and ranks deeper losses below shallower losses.
2. **Negative Enterprise Value ($EV \le 0$)**: Occurs when cash exceeds market capitalization plus debt. In this scenario, $EBITDA / EV$ is mathematically degenerate. The engine automatically switches valuation weighting to Market Cap yields: $EY$ (60%) and $FCF / \text{MarketCap}$ (40%).
3. **Missing Fundamentals**: If financial statements are missing or filing date $> 180$ days stale, Pillar 4 and Pillar 5 are flagged as `unavailable` (`score = None`). The composite dynamically re-normalizes base weights across remaining verified pillars:
   $$W_A = \sum_{i \in \text{Available}} w_i, \quad \text{Raw Composite} = \frac{\sum_{i \in \text{Available}} w_i \cdot S_i}{W_A}$$
   If Available Pillars count $< 3$, the engine aborts evaluation and returns `INSUFFICIENT_EVIDENCE`.
