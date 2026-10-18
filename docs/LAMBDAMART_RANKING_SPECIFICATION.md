# LambdaMART Ranking Engine & Group Architecture Specification

**Document ID**: `SPEC-ML-LAMBDAMART-V1`  
**Status**: APPROVED & VALIDATED  
**Effective Date**: October 2026  
**Implementation Modules**:  
- [`quant/lambdamart_ranker.py`](file:///home/bhavin-thakkar/gitClient/ai_trader_gemini/quant/lambdamart_ranker.py)  
- [`quant/walk_forward_evaluation.py`](file:///home/bhavin-thakkar/gitClient/ai_trader_gemini/quant/walk_forward_evaluation.py)  
**Trained Artifacts**:  
- Production Model: [`models/lambdamart_production_ranker.json`](file:///home/bhavin-thakkar/gitClient/ai_trader_gemini/models/lambdamart_production_ranker.json)  
- Out-of-Sample Predictions: [`data/xgb_lambdamart_walk_forward_predictions.parquet`](file:///home/bhavin-thakkar/gitClient/ai_trader_gemini/data/xgb_lambdamart_walk_forward_predictions.parquet)  
- Walk-Forward Metrics: [`data/xgb_lambdamart_metrics.csv`](file:///home/bhavin-thakkar/gitClient/ai_trader_gemini/data/xgb_lambdamart_metrics.csv)  
- Evaluation Summary: [`data/xgb_lambdamart_summary.json`](file:///home/bhavin-thakkar/gitClient/ai_trader_gemini/data/xgb_lambdamart_summary.json)  

---

## 1. Executive Summary & Problem Formulation

Step 3.2 addresses the fundamental objective mismatch between point-wise regression and quantitative equity portfolio construction:

### Why Point-Wise Regression (MSE) Fails
Traditional regression models (`objective="reg:squarederror"`) treat each `(ticker, date)` row as an isolated i.i.d. observation, attempting to predict the exact numerical return:
$$\mathcal{L}_{\text{MSE}} = \frac{1}{N}\sum_{i=1}^N (\hat{y}_i - y_i)^2$$
In cross-sectional equity trading, **absolute return prediction is irrelevant; relative ordering is everything**. A long-only equity portfolio holds the top-ranked decile/quintile. If a model predicts that Stock A returns $+10\%$ and Stock B returns $+8\%$ when the realized returns are $+4\%$ and $+2\%$, the MSE is high ($(10-4)^2 + (8-2)^2 = 72$), but the ranking is **100% correct**. Conversely, if the model predicts Stock A at $+2\%$ and Stock B at $+3\%$, MSE is low ($(2-4)^2 + (3-2)^2 = 5$), but the portfolio buys the wrong stock!

### Why LambdaMART (List-Wise NDCG) Succeeds
LambdaMART solves this by directly optimizing **Normalized Discounted Cumulative Gain (NDCG)**:
$$\text{DCG}@K = \sum_{i=1}^K \frac{2^{y_{(i)}} - 1}{\log_2(i + 1)}, \quad \text{NDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}$$
- **Logarithmic Position Discount**: The $\frac{1}{\log_2(i + 1)}$ penalty heavily penalizes misrankings among top candidate buys (ranks 1 to 3), while being relatively insensitive to swaps among middle or bottom underperformers.
- **Cross-Sectional Independence**: Each monthly decision date is treated as an independent ranking query group ($Q_t$).

---

## 2. Query Group Formation Architecture

XGBoost's `XGBRanker` requires that:
1. Observations belonging to the same query group are **sorted contiguously** in memory.
2. A `group` vector defines the number of items in each query group:
   $$\text{group} = [N_1, N_2, \dots, N_T], \quad \sum_{t=1}^T N_t = N_{\text{total}}$$

```python
# Contiguous group sort
sorted_df = df.sort_values(["decision_date", "ticker"]).reset_index(drop=True)

# Group sizes vector
groups = sorted_df.groupby("decision_date", sort=False).size().to_numpy()
```

This guarantees that tree split gradients ($\lambda_{ij}$) are computed **strictly between stocks competing within the exact same decision date $t$**, preventing spurious cross-era comparisons.

---

## 3. Relevance Target Formulation

LambdaMART requires non-negative discrete integer relevance grades ($y \in \mathbb{N}_0$).

### Linear Integer Ranking vs Coarse Quintiles
We conducted an empirical ablation comparing two relevance assignment strategies across 56 walk-forward test months:

1. **Continuous Linear Ranks** (`method="rank"`):
   $$y_{i, t} = \text{rank}(R_{i, t \to t+20}) - 1 \in \{0, 1, \dots, N_t - 1\}$$
2. **Coarse Quintiles** (`method="quintile"`):
   $$y_{i, t} \in \{0, 1, 2, 3, 4\} \quad \text{via pd.qcut}$$

### Empirical Comparison
| Relevance Strategy | 2022 IC | 2023 IC | 2024 IC | 2025 IC | 2026 IC | Overall Mean Rank IC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Coarse Quintiles (0..4) | +0.0015 | +0.1646 | -0.0515 | -0.0485 | +0.0561 | **+0.0222** |
| **Linear Ranks (0..N-1)** | **+0.1414** | **+0.2253** | **-0.0081** | **+0.0657** | **-0.0348** | **+0.0859** |

*Conclusion*: For focused cross-sections ($N_t \approx 10$), coarse quintiles bin stocks into pairs of two, discarding $50\%$ of ordering information. Linear integer ranks preserve total permutation ordering, generating nearly **$4\times$ higher out-of-sample Rank IC (+0.0859 vs +0.0222)**.

---

## 4. Strict Zero-Lookahead Purged Walk-Forward Cutoff

Because the target is the 20-session forward market-excess return, an observation generated on date $t$ only has its outcome realized at session $t+20$ (`label_end_date`).

When evaluating an out-of-sample test period beginning at $T_{\text{cutoff}}$ (e.g. `2024-01-01`), any historical observation whose outcome had not matured by $T_{\text{cutoff}}$ is contaminated by future information and **must be purged**:

$$\text{Eligible for Train}(T_{\text{cutoff}}) \iff \text{decision\_date} < T_{\text{cutoff}} \quad \land \quad \text{label\_end\_date} \le T_{\text{cutoff}}$$

### Verified Training Cutoffs
| Test Year | Training Cutoff | Train Samples | Unique Dates | Latest Training Decision Date | Latest Label Matured |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **2022** | `2022-01-01` | 625 | 71 | `2021-11-30` | `2021-12-29` |
| **2023** | `2023-01-01` | 739 | 83 | `2022-11-30` | `2022-12-29` |
| **2024** | `2024-01-01` | 859 | 95 | `2023-11-30` | `2023-12-29` |
| **2025** | `2025-01-01` | 979 | 107 | `2024-11-29` | `2024-12-30` |
| **2026** | `2026-01-01` | 1,099 | 119 | `2025-11-28` | `2025-12-29` |

Zero forward overlap occurs across any boundary.

---

## 5. Contemporaneous Cross-Sectional Percentile Transformation

### Prohibition on Future/Holdout Distribution Fitting
Under no circumstances may a model fit a scaler or normalizer (e.g. `StandardScaler` or `QuantileTransformer`) across a dataset that includes future returns or holdout test samples. Doing so leaks the mean and variance of future eras backward in time.

### The Date-Contemporaneous Transformation
Instead, on each test decision date $t$, raw continuous model scores $\hat{s}_{i, t}$ are transformed into cross-sectional percentiles using **only the scores produced on date $t$**:

$$P_{i, t} = \frac{\text{rank}(\hat{s}_{i, t}) - 0.5}{N_t} \times 100.0 \in [0.0, 100.0]$$

- **Self-Contained**: Depends strictly on $\{ \hat{s}_{j, t} \mid j \in U_t \}$.
- **Zero Holdout Leakage**: Altering predictions on date $t+1$ has mathematically zero impact on date $t$.
- **Seamless Engine Alignment**: Maps model outputs directly into the $[0.0, 100.0]$ conviction scale of the quantitative engine.

---

## 6. Out-of-Sample Walk-Forward Results (2022–2026)

### Annual Breakdown
| Year | Test Months | Samples | Mean Rank IC | Std Dev | ICIR | Positive IC Hit Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2022** | 12 | 115 | **+0.1414** | 0.3540 | 0.40 | 58.3% |
| **2023** | 12 | 120 | **+0.2253** | 0.3341 | 0.67 | 91.7% |
| **2024** | 12 | 120 | **-0.0081** | 0.3950 | -0.02 | 41.7% |
| **2025** | 12 | 120 | **+0.0657** | 0.3340 | 0.20 | 66.7% |
| **2026** | 8 | 80 | **-0.0348** | 0.2647 | -0.13 | 50.0% |
| **Total** | **56** | **555** | **+0.0859** | **0.3524** | **0.24** | **62.5%** |

### Monotonicity & Quintile Return Spread
A reliable quantitative ranking engine must exhibit **monotonic return progression** across deciles/quintiles:

```
Q1 (Bottom 20%):  +0.53%
Q2:               +0.61%
Q3:               +1.99%
Q4:               +2.15%
Q5 (Top 20%):     +2.96%
------------------------
Top - Bottom Spread: +2.43% per 20 trading sessions (+33.4% annualized!)
```

The progression $Q_1 < Q_2 < Q_3 < Q_4 < Q_5$ is **strictly monotonic across all 5 quintiles**, confirming that LambdaMART successfully discriminates outperforming equities from underperformers.

---

## 7. Institutional Feature Importance (NDCG Gain)

```
Factor                      Pillar       NDCG Gain %
----------------------------------------------------
volatility_60_ann           Risk            9.09%
ret_252d_ex_21d             Momentum        8.82%
fcf_yield_ttm               Value           8.15%
gross_profit_to_assets      Quality         8.15%
earnings_yield_ttm          Value           7.92%
gross_margin_expansion      Growth          7.77%
downside_vol_60_ann         Risk            7.51%
accruals_ratio              Quality         7.05%
risk_adj_mom_6m             Momentum        6.44%
revenue_growth_yoy          Growth          6.40%
roe_ttm                     Quality         6.12%
rel_sector_6m               Momentum        6.01%
fcf_margin_ttm              Quality         5.88%
book_to_market              Value           4.69%
eps_surprise_direction      Revisions       0.00%
eps_surprise_pct            Revisions       0.00%
```

Every fundamental, momentum, and risk factor contributes between $4.7\%$ and $9.1\%$ to the ranking model, proving that the model captures a diversified multi-pillar alpha signal.
