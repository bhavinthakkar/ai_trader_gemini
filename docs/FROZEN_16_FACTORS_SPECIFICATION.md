# Institutional 16-Factor Frozen Specification & Reproducible Training Table

**Document ID**: `SPEC-ML-16F-V1`  
**Status**: APPROVED & FROZEN  
**Effective Date**: October 2026  
**Implementation**: [`quant/build_training_table.py`](file:///home/bhavin-thakkar/gitClient/ai_trader_gemini/quant/build_training_table.py)  
**Artifacts Generated**:  
- [`data/ml_training_table.parquet`](file:///home/bhavin-thakkar/gitClient/ai_trader_gemini/data/ml_training_table.parquet)  
- [`data/ml_training_table.csv`](file:///home/bhavin-thakkar/gitClient/ai_trader_gemini/data/ml_training_table.csv)  
- Sibling clone: `/home/bhavin-thakkar/gitClient/stock-ml/data/ml_training_table.parquet`

---

## 1. Executive Summary & Purpose

Step 3 of the quantitative engine hardening mandates an independent validation of the Machine Learning model. Before training, evaluating, or blending ML predictions with deterministic quant scores, the quantitative engine must establish **one canonical, reproducible point-in-time training table** where:
1. Every observation represents exactly one stock and decision date.
2. The feature suite is permanently frozen across 16 economically grounded factors spanning 5 alpha pillars (Momentum, Value, Quality & Profitability, Growth & Revisions, and Low Volatility / Risk).
3. Every feature has an exact mathematical formula, vendor/source definition, measurement unit, lookback window, zero-lookahead timestamp rule, and missing-value imputation policy.
4. Valuation features handle negative earnings and non-positive EBITDA continuously without numerical singularities.
5. Labels strictly adhere to the NYSE trading calendar (20 trading sessions ahead).

---

## 2. Frozen 16-Factor Specification Matrix

| Col ID | Canonical Factor Name | Pillar | Alpha Direction | Measurement Units | Lookback Window | Source Data |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `feature_1` | `ret_252d_ex_21d` | Momentum | Higher is better | Decimal return | 252 sessions (skipping recent 21) | Daily split/div-adjusted close |
| `feature_2` | `risk_adj_mom_6m` | Momentum | Higher is better | Sharpe-like ratio | 126 sessions scaled by 60d vol | Daily adjusted close |
| `feature_3` | `rel_sector_6m` | Momentum | Higher is better | Decimal excess return | 126 sessions | Daily adjusted close + GICS mapping |
| `feature_4` | `fcf_yield_ttm` | Value | Higher is better | Decimal cash yield | TTM (trailing 4 quarters) | SEC EDGAR 10-Q/10-K (CFO - CapEx) / MCap |
| `feature_5` | `earnings_yield_ttm` | Value | Higher is better | Decimal yield (E/P) | TTM (trailing 4 quarters) | SEC EDGAR 10-Q/10-K (Net Income) / MCap |
| `feature_6` | `book_to_market` | Value | Higher is better | Decimal ratio (B/M) | Latest quarterly balance sheet | SEC EDGAR 10-Q/10-K (Equity) / MCap |
| `feature_7` | `gross_profit_to_assets` | Quality | Higher is better | Decimal ratio | TTM GP / latest Total Assets | SEC EDGAR 10-Q/10-K (Rev - COGS) / Assets |
| `feature_8` | `fcf_margin_ttm` | Quality | Higher is better | Decimal ratio | TTM FCF / TTM Revenue | SEC EDGAR 10-Q/10-K |
| `feature_9` | `roe_ttm` | Quality | Higher is better | Decimal return on equity | TTM Net Income / Equity | SEC EDGAR 10-Q/10-K |
| `feature_10` | `accruals_ratio` | Quality | Lower is better | Decimal ratio | TTM (NI - CFO) / Assets | SEC EDGAR 10-Q/10-K (Sloan Accruals) |
| `feature_11` | `revenue_growth_yoy` | Growth | Higher is better | Decimal YoY % growth | 1-year trailing (4-quarter shift) | SEC EDGAR 10-Q/10-K TTM Revenue |
| `feature_12` | `gross_margin_expansion`| Growth | Higher is better | Decimal margin change | 1-year trailing change | SEC EDGAR 10-Q/10-K TTM Gross Margin |
| `feature_13` | `eps_surprise_pct` | Revisions | Higher is better | Decimal percentage | Most recent quarterly report | Consensus EPS announcement releases |
| `feature_14` | `eps_surprise_direction`| Revisions | Higher is better | Discrete `{-1, 0, +1}` | Most recent quarterly report | Consensus EPS announcement releases |
| `feature_15` | `volatility_60_ann` | Risk | Lower is better | Annualized decimal vol | 60 trading sessions | Daily log returns |
| `feature_16` | `downside_vol_60_ann` | Risk | Lower is better | Annualized semi-dev | 60 trading sessions | Daily negative return deviations |

---

## 3. Factor-by-Factor Mathematical Formulas & Rules

### Pillar 1: Intermediate & Risk-Adjusted Momentum
1. **`ret_252d_ex_21d` (Feature 1)**:
   $$\text{Score}(t) = \frac{1 + R_{252}(t)}{1 + R_{21}(t)} - 1.0 = \frac{P_{t-21}}{P_{t-252}} - 1.0$$
   *Rationale*: Captures intermediate momentum (Jegadeesh-Titman 1993, Fama-French 2012) while skipping the most recent 21 trading sessions to eradicate microstructure bid-ask bounce and 1-month short-term reversal noise.
   *Timestamp*: As of close on decision date $t$.

2. **`risk_adj_mom_6m` (Feature 2)**:
   $$\text{Score}(t) = \frac{P_t / P_{t-126} - 1.0}{\sigma_{60}(t) \cdot \sqrt{252}}$$
   *Rationale*: Normalizes cumulative 6-month price trend by recent 60-day realized volatility, penalizing high-volatility speculative rallies.
   *Timestamp*: As of close on decision date $t$.

3. **`rel_sector_6m` (Feature 3)**:
   $$\text{Score}(t) = R_{i, 126\text{d}}(t) - \frac{1}{|S_k|}\sum_{j \in S_k} R_{j, 126\text{d}}(t)$$
   *Rationale*: Measures idiosyncratic outperformance relative to equal-weighted sector peers ($S_k$), immunizing the signal against broad sector rotations.
   *Timestamp*: As of close on decision date $t$.

---

### Pillar 2: Valuation History & Yields (Handling Negative Earnings)
Traditional valuation multiples (such as $P/E$ or $P/\text{FCF}$) become mathematically invalid and directionally reversed when earnings or cash flows are negative:
- A company with Net Income of $-\$0.01\text{M}$ and Market Cap of $\$100\text{M}$ produces $P/E = -10,000$.
- A company with Net Income of $-\$10\text{M}$ produces $P/E = -10$.
- Under naive ranking, $-10$ ranks "higher" than $-10,000$, perversely declaring the catastrophic loss-maker more attractive than the near-breakeven firm!

To solve this fundamentally and permanently, all valuation metrics are strictly formulated as **Yields**:

4. **`fcf_yield_ttm` (Feature 4)**:
   $$\text{FCF Yield} = \frac{\text{TTM Operating Cash Flow} - \text{TTM Capital Expenditures}}{\text{Close}_t \times \text{Shares Outstanding}_t}$$
   *Continuous Ranking*: When $\text{FCF} < 0$, the yield is negative (e.g. $-8\%$). Mild cash-burners rank higher than severe cash-burners, which in turn rank strictly below cash-generative firms ($+5\%$).

5. **`earnings_yield_ttm` (Feature 5)**:
   $$\text{Earnings Yield} = \frac{\text{TTM Net Income}}{\text{Close}_t \times \text{Shares Outstanding}_t} = \frac{1}{P/E}$$
   *Continuous Ranking*: Continuous across $(-\infty, +\infty)$. Unprofitable firms smoothly receive negative scores directly proportional to the magnitude of their losses.

6. **`book_to_market` (Feature 6)**:
   $$\text{Book-to-Market} = \frac{\text{Stockholders' Equity}_t}{\text{Market Capitalization}_t}$$
   *Treatment*: Negative equity (frequently caused by extensive debt-funded share repurchases or accumulated deficits) produces negative B/M, appropriately penalizing solvency risk.

---

### Pillar 3: Quality & Profitability
7. **`gross_profit_to_assets` (Feature 7)**:
   $$\text{GPA} = \frac{\text{TTM Revenue} - \text{TTM Cost of Goods Sold}}{\text{Total Assets}_t}$$
   *Rationale*: Novy-Marx (2013) demonstrated that gross profitability is the cleanest predictor of long-term excess returns because it is untainted by capital structure, tax timing, or discretionary overhead.
   *Financials Treatment*: For commercial banks and financial institutions that do not report Gross Profit, GPA is cross-sectionally imputed using the contemporaneous financial peer median.

8. **`fcf_margin_ttm` (Feature 8)**:
   $$\text{FCF Margin} = \frac{\text{TTM Free Cash Flow}}{\text{TTM Revenue}}$$
   *Rationale*: Measures cash conversion efficiency per dollar of topline sales.

9. **`roe_ttm` (Feature 9)**:
   $$\text{ROE} = \frac{\text{TTM Net Income}}{\text{Stockholders' Equity}_t}$$
   *Distress Handling*: Robustly winsorized at 1st and 99th percentiles to eliminate extreme ratios caused by near-zero equity denominators.

10. **`accruals_ratio` (Feature 10)**:
    $$\text{Accruals} = \frac{\text{TTM Net Income} - \text{TTM Operating Cash Flow}}{\text{Total Assets}_t}$$
    *Direction*: **Lower is better**. Sloan (1996) proved that firms with high positive accruals (accounting earnings far exceeding cash receipts) suffer subsequent mean-reversion and earnings collapse. Negative accruals indicate conservative accounting with high cash backing.

---

### Pillar 4: Growth & Revisions
11. **`revenue_growth_yoy` (Feature 11)**:
    $$\text{YoY Rev Growth} = \frac{\text{Revenue (TTM)}_t - \text{Revenue (TTM)}_{t-1\text{yr}}}{|\text{Revenue (TTM)}_{t-1\text{yr}}|}$$
    *Rationale*: Trailing 12-month topline expansion across 4 quarters.

12. **`gross_margin_expansion` (Feature 12)**:
    $$\Delta \text{GM} = \left(\frac{\text{Gross Profit}}{\text{Revenue}}\right)_t - \left(\frac{\text{Gross Profit}}{\text{Revenue}}\right)_{t-1\text{yr}}$$
    *Rationale*: Captures operational leverage and pricing power.

13. **`eps_surprise_pct` (Feature 13)**:
    $$\text{Surprise \%} = \frac{\text{Reported EPS} - \text{Consensus Estimate EPS}}{|\text{Consensus Estimate EPS}|}$$
    *Missingness*: If no earnings announcement occurred in the lookback window or consensus estimates are unobserved, defaults to `0.0` (neutral).

14. **`eps_surprise_direction` (Feature 14)**:
    $$\text{Direction} = \begin{cases} +1.0 & \text{if Reported} > \text{Consensus} \\ 0.0 & \text{if Meet or Unobserved} \\ -1.0 & \text{if Reported} < \text{Consensus} \end{cases}$$

---

### Pillar 5: Low Volatility & Risk
15. **`volatility_60_ann` (Feature 15)**:
    $$\sigma_{60} = \sqrt{252} \cdot \sqrt{\frac{1}{59}\sum_{s=0}^{59}(r_{t-s} - \bar{r})^2}$$
    *Direction*: **Lower is better** (Low-Volatility Anomaly: Baker, Bradley, & Wurgler 2011).

16. **`downside_vol_60_ann` (Feature 16)**:
    $$\sigma_{60,\text{down}} = \sqrt{252} \cdot \sqrt{\frac{1}{60}\sum_{s=0}^{59}\min(0, r_{t-s})^2}$$
    *Direction*: **Lower is better** (penalizes asymmetric tail downside risk).

---

## 4. Missing-Value Treatment & Zero-Lookahead Audit

### Zero-Lookahead Audit
- **Price and Volume inputs**: Strictly use closing prices on or before decision date $t$.
- **SEC EDGAR Fundamentals**: Aligned strictly using `filing_date < decision_date`. When a filing is stamped on decision date $t$, it is treated as available on session $t+1$ (execution date) to guarantee zero lookahead from after-market filings.
- **Future Information Barrier**: No future data from $s > t$ is accessible during feature construction.

### Imputation Policy
1. **Contemporaneous Cross-Sectional Median Imputation**:
   For any fundamental feature missing on decision date $t$, the missing value is replaced with:
   $$\hat{x}_{i, t} = \text{median}(\{x_{j, t} \mid j \in U_t, x_{j, t} \neq \text{NaN}\})$$
   where $U_t$ is the active universe on decision date $t$. This ensures the imputation uses **only contemporaneous market information**.
2. **Revisions Imputation**:
   Unobserved earnings surprises default to `0.0` (meet expectations).
3. **Winsorization**:
   All 16 features are robustly clipped to their empirical $[1^{\text{st}}, 99^{\text{th}}]$ percentiles:
   $$x_{i, t}^{\text{clean}} = \text{clip}(x_{i, t}, Q_{0.01}, Q_{0.99})$$
   preventing outlier distortions in gradient boosting tree splits or linear factor regressions.

---

## 5. Training Dataset Structure & Validation

### Table Schema
The dataset [`data/ml_training_table.parquet`](file:///home/bhavin-thakkar/gitClient/ai_trader_gemini/data/ml_training_table.parquet) contains:
- `decision_date`: string (`YYYY-MM-DD`), date signal is formed.
- `ticker`: string (e.g. `NVDA`, `AAPL`).
- `feature_1` ... `feature_16`: numeric floating-point values for each frozen factor.
- `ret_252d_ex_21d` ... `downside_vol_60_ann`: named factor columns identical to features.
- `label_excess_20d`: float, 20-session market excess return vs SPY ($R_{\text{stock}, 20\text{d}} - R_{\text{SPY}, 20\text{d}}$).
- `label_end_date`: string (`YYYY-MM-DD`), exact trading session 20 sessions ahead following NYSE calendar.
- `universe_version`: string (`universe_v3`).
- `has_label`: bool (`True` for mature labeled sessions, `False` for recent live scoring sessions).

### Dataset Summary Statistics
- **Total Rows**: 1,209 observations
- **Date Range**: `2016-01-29` to `2026-10-06` (130 monthly decision dates)
- **Mature Labeled Rows**: 1,189 observations
- **Live Scoring Rows**: 20 observations
- **Unhandled NaNs**: **0.00%** across all 16 features.
- **Unit Test Suite**: 8 tests in [`tests/test_training_table.py`](file:///home/bhavin-thakkar/gitClient/ai_trader_gemini/tests/test_training_table.py) passing cleanly (total 30 unit tests passing in repository).
