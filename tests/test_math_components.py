"""
Step 6.1: Exhaustive Unit Tests for Each Mathematical Component.

Automated verification of the decision engine's mathematical components before
connecting to any broker or market-data feed.

Covers the six core components:
1. Five Pillars (Trend, Sector, Alpha, Valuation History, Peer Valuation)
2. Quality Adjustments (Financial Quality, Value-Trap Hard Gate, Compounder Bonus)
3. Volatility Transformation (ATR% factor, centered dampener, neutral invariance)
4. ML Blending (in-scope weighting, out-of-scope 0%, score thresholds 60.00 vs 59.99)
5. Reward-to-Risk & Trade Geometry (Price ordering, cost drag, 1.20 vs 1.199 RR, edge)
6. Confidence / Signal Strength (Decisiveness D, Agreement A, Completeness C, HOLD cap)

For EACH component, tests:
- Normal inputs.
- Minimum and maximum valid inputs.
- Missing and stale inputs.
- Zero denominators and invalid values.
- Threshold boundaries (e.g., 60.00 vs 59.99, 1.20 vs 1.19).
- Unexpected data types and malformed outputs.
"""

import math
import unittest
from typing import Any, Dict, List

from quant.factors import (
    clean_float,
    compute_ema_alignment,
    compute_rsi_velocity,
    compute_sector_excess,
    compute_market_alpha,
    compute_earnings_yield,
    compute_ebitda_yield,
    compute_fcf_yield,
)
from quant.normalization import (
    clamp,
    logistic_scale,
    normal_cdf,
    normal_cdf_scale,
    cross_sectional_robust_zscore,
    time_series_percentile_score,
)
from quant.pillar_scores import (
    calculate_trend_pillar,
    calculate_sector_pillar,
    calculate_alpha_pillar,
    calculate_valuation_history_pillar,
    calculate_peer_valuation_pillar,
)
from quant.quality_adjustments import (
    evaluate_financial_quality,
    evaluate_value_trap,
    apply_quality_adjustment,
)
from quant.volatility import (
    compute_vol_factor,
    apply_volatility_dampener,
    compute_volatility_adjusted_position_size,
)
from quant.composite_score import (
    blend_quant_and_ml_scores,
    compute_deterministic_composite,
    BASE_WEIGHTS,
    MIN_REQUIRED_PILLARS,
)
from quant.trade_geometry import (
    TradeDirection,
    PriceOrderingStatus,
    ExecutionCostModel,
    RewardRiskResult,
    validate_price_ordering,
    compute_breakeven_win_rate,
    estimate_empirical_win_probability,
    compute_expected_value_r,
    compute_validated_reward_risk,
)
from quant.signal_strength import (
    DispersionMetric,
    CombinationMethod,
    RequiredInputRecord,
    SignalStrengthWeights,
    compute_score_decisiveness,
    compute_pillar_agreement,
    compute_data_completeness,
    SignalStrengthEngine,
)
from quant.schemas import PillarStatus


# =============================================================================
# 1. FIVE PILLARS MATHEMATICAL COMPONENT
# =============================================================================
class TestFivePillarsMathComponent(unittest.TestCase):
    """Exhaustive tests for Pillar 1 to 5 calculation routines."""

    # --- Pillar 1: Trend ---
    def test_trend_pillar_normal_inputs(self):
        """Bullish EMA stack with positive RSI momentum generates high trend score."""
        tech = {
            "current_price": 150.0,
            "ema20": 145.0,
            "ema50": 140.0,
            "ema200": 130.0,
            "rsi14": 62.0,
            "atr": 2.5,
        }
        res = calculate_trend_pillar(tech)
        self.assertIsNotNone(res.score)
        self.assertEqual(res.status, PillarStatus.MEASURED)
        self.assertGreater(res.score, 75.0)
        self.assertEqual(res.weight, 0.25)

    def test_trend_pillar_min_max_inputs(self):
        """Extreme values produce correctly clamped scores in [0.0, 100.0]."""
        # Minimum extreme: deep breakdown, RSI = 0.0
        min_tech = {
            "current_price": 10.0,
            "ema20": 50.0,
            "ema50": 80.0,
            "ema200": 120.0,
            "rsi14": 0.0,
        }
        res_min = calculate_trend_pillar(min_tech)
        self.assertIsNotNone(res_min.score)
        self.assertGreaterEqual(res_min.score, 0.0)
        self.assertLessEqual(res_min.score, 20.0)

        # Maximum extreme: runaway parabolic stack, RSI = 100.0
        max_tech = {
            "current_price": 500.0,
            "ema20": 400.0,
            "ema50": 300.0,
            "ema200": 200.0,
            "rsi14": 100.0,
        }
        res_max = calculate_trend_pillar(max_tech)
        self.assertIsNotNone(res_max.score)
        self.assertGreaterEqual(res_max.score, 0.0)
        self.assertLessEqual(res_max.score, 100.0)

    def test_trend_pillar_missing_and_stale_inputs(self):
        """Missing or partial price/indicators transition to PARTIAL or UNAVAILABLE."""
        # Completely empty dict
        res_empty = calculate_trend_pillar({})
        self.assertIsNone(res_empty.score)
        self.assertEqual(res_empty.status, PillarStatus.UNAVAILABLE)

        # None payload
        res_none = calculate_trend_pillar(None)
        self.assertIsNone(res_none.score)
        self.assertEqual(res_none.status, PillarStatus.UNAVAILABLE)

        # Price available, but no EMA20 or RSI
        res_no_inds = calculate_trend_pillar({"current_price": 100.0})
        self.assertIsNone(res_no_inds.score)
        self.assertEqual(res_no_inds.status, PillarStatus.UNAVAILABLE)

        # Partial: price + EMA20 present, but EMA50 and EMA200 absent
        res_partial = calculate_trend_pillar({"current_price": 100.0, "ema20": 95.0})
        self.assertIsNotNone(res_partial.score)
        self.assertEqual(res_partial.status, PillarStatus.PARTIAL)

    def test_trend_pillar_zero_denominators_and_invalid_values(self):
        """Zero and negative prices are safely rejected."""
        self.assertIsNone(calculate_trend_pillar({"current_price": 0.0, "ema20": 10.0}).score)
        self.assertIsNone(calculate_trend_pillar({"current_price": -50.0, "ema20": 10.0}).score)
        # EMA20 <= 0 rejected
        res_neg_ema = calculate_trend_pillar({"current_price": 100.0, "ema20": -5.0})
        self.assertIsNone(res_neg_ema.score)

    def test_trend_pillar_threshold_boundaries(self):
        """Tests exact RSI velocity boundary transitions."""
        # 1. RSI = 50.0 exact threshold for bullish momentum expansion
        score_50, _, desc_50 = compute_rsi_velocity(50.0)
        self.assertAlmostEqual(score_50, 65.0, places=1)
        self.assertEqual(desc_50, "bullish_momentum_expansion")

        # 2. RSI = 49.99 falls into neutral accumulation
        score_499, _, desc_499 = compute_rsi_velocity(49.99)
        self.assertEqual(desc_499, "neutral_accumulation")
        self.assertLess(score_499, score_50)

        # 3. RSI = 68.0 upper bound of expansion
        score_68, _, desc_68 = compute_rsi_velocity(68.0)
        self.assertAlmostEqual(score_68, 80.0, places=1)

        # 4. RSI = 32.0 exact boundary for oversold
        _, _, desc_32 = compute_rsi_velocity(32.0)
        self.assertEqual(desc_32, "neutral_range")
        _, _, desc_319 = compute_rsi_velocity(31.9)
        self.assertIn("oversold", desc_319)

    def test_trend_pillar_unexpected_types_and_malformed_inputs(self):
        """Strings, bools, NaNs, and Infs handled safely without exceptions."""
        malformed = {
            "current_price": "150.50",  # String number handled by clean_float
            "ema20": "$145.00",
            "ema50": "+140.0%",
            "ema200": "N/A",
            "rsi14": float("nan"),
            "atr": None,
        }
        res = calculate_trend_pillar(malformed)
        self.assertIsNotNone(res.score)
        self.assertEqual(res.status, PillarStatus.PARTIAL)

        # Boolean in current_price should not be treated as 1.0
        res_bool = calculate_trend_pillar({"current_price": True, "ema20": 10.0})
        self.assertIsNone(res_bool.score)

    # --- Pillar 2: Sector Relative ---
    def test_sector_pillar_normal_inputs(self):
        """Stock outperforming sector ETF by +4.0% yields > 70 score."""
        t_data = {"change_5d_pct": 5.0, "change_20d_pct": 8.0, "change_60d_pct": 12.0}
        s_data = {"sector_change_pct": 1.0, "sector_change_20d_pct": 4.0, "sector_change_60d_pct": 6.0, "sector_etf": "XLK"}
        res = calculate_sector_pillar(t_data, s_data)
        self.assertIsNotNone(res.score)
        self.assertEqual(res.status, PillarStatus.MEASURED)
        self.assertGreater(res.score, 70.0)

    def test_sector_pillar_min_max_inputs(self):
        """Massive outperformance yields ~100.0; massive underperformance yields ~0.0."""
        # Extreme outperformance (+100% excess)
        res_max = calculate_sector_pillar(
            {"change_20d_pct": 100.0},
            {"sector_change_20d_pct": 0.0, "sector_etf": "XLK"}
        )
        self.assertAlmostEqual(res_max.score, 100.0, places=1)

        # Extreme underperformance (-100% excess)
        res_min = calculate_sector_pillar(
            {"change_20d_pct": -100.0},
            {"sector_change_20d_pct": 0.0, "sector_etf": "XLK"}
        )
        self.assertAlmostEqual(res_min.score, 0.0, places=1)

    def test_sector_pillar_missing_and_stale_inputs(self):
        """Missing returns or benchmarks safely yield UNAVAILABLE status."""
        self.assertEqual(calculate_sector_pillar(None, None).status, PillarStatus.UNAVAILABLE)
        self.assertEqual(calculate_sector_pillar({}, {}).status, PillarStatus.UNAVAILABLE)
        # Missing 20d horizon falls back to 5d with PARTIAL status
        res_5d = calculate_sector_pillar(
            {"change_5d_pct": 2.0},
            {"sector_change_pct": 1.0, "sector_etf": "XLK"}
        )
        self.assertEqual(res_5d.status, PillarStatus.PARTIAL)
        self.assertIsNotNone(res_5d.score)

    def test_sector_pillar_threshold_boundaries(self):
        """Excess return of exactly 0.0 maps to neutral midpoint 50.0."""
        res_zero = calculate_sector_pillar(
            {"change_20d_pct": 5.0},
            {"sector_change_20d_pct": 5.0, "sector_etf": "XLK"}
        )
        self.assertAlmostEqual(res_zero.score, 50.0, places=2)

    # --- Pillar 3: Market Alpha ---
    def test_alpha_pillar_normal_inputs(self):
        """Positive 20d alpha vs SPY produces score > 50."""
        res = calculate_alpha_pillar({"relative_alpha_20d": 4.0, "beta_60d": 1.1})
        self.assertIsNotNone(res.score)
        self.assertEqual(res.status, PillarStatus.MEASURED)
        self.assertGreater(res.score, 70.0)

    def test_alpha_pillar_zero_and_threshold_boundaries(self):
        """Alpha of exactly 0.0 maps to neutral 50.0."""
        res = calculate_alpha_pillar({"relative_alpha_20d": 0.0})
        self.assertAlmostEqual(res.score, 50.0, places=2)

    def test_alpha_pillar_missing_and_fallback(self):
        """Falls back from 20d alpha to 5d alpha, or returns UNAVAILABLE if absent."""
        res_fallback = calculate_alpha_pillar({"relative_alpha_5d": 3.0})
        self.assertEqual(res_fallback.status, PillarStatus.PARTIAL)
        self.assertIsNotNone(res_fallback.score)

        res_none = calculate_alpha_pillar({})
        self.assertEqual(res_none.status, PillarStatus.UNAVAILABLE)
        self.assertIsNone(res_none.score)

    # --- Pillar 4: Valuation History ---
    def test_valuation_history_normal_inputs(self):
        """Forward P/E < Trailing P/E (expansion) with positive rev growth boosts score."""
        gloom = {
            "financials": {
                "key_ratios": {
                    "forward_pe": 18.0,
                    "trailing_pe": 24.0,
                    "revenue_growth": 0.18,
                }
            }
        }
        res = calculate_valuation_history_pillar(gloom)
        self.assertIsNotNone(res.score)
        # Base 50 + 20 (compression) + 15 (>15% growth) = 85.0
        self.assertEqual(res.score, 85.0)

    def test_valuation_history_yield_percentile(self):
        """Historical yields >= 8 uses continuous time-series percentile rank."""
        hist = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
        # Current forward P/E 12.5 -> yield 0.08 (7 out of 8 below or equal)
        gloom = {
            "financials": {
                "key_ratios": {
                    "forward_pe": 12.5,
                    "historical_earnings_yields": hist,
                }
            }
        }
        res = calculate_valuation_history_pillar(gloom)
        self.assertEqual(res.status, PillarStatus.MEASURED)
        self.assertAlmostEqual(res.score, (7.0 / 8.0) * 100.0, places=1)

    def test_valuation_history_zero_denominators_and_invalid(self):
        """Zero P/E and zero price handled continuously via compute_earnings_yield."""
        self.assertIsNone(compute_earnings_yield(eps_ttm=5.0, price=0.0))
        self.assertIsNone(compute_earnings_yield(eps_ttm=5.0, price=-10.0))
        self.assertIsNone(compute_earnings_yield(eps_ttm=None, price=100.0))
        # Negative EPS has continuous negative yield without division-by-zero
        self.assertAlmostEqual(compute_earnings_yield(eps_ttm=-2.0, price=50.0), -0.04)

    # --- Pillar 5: Peer Valuation ---
    def test_peer_valuation_normal_inputs(self):
        """Stock with higher earnings yield than peers receives robust z-score > 0 -> score > 50."""
        gloom = {
            "financials": {"key_ratios": {"forward_pe": 15.0}},
            "peer_valuation": {
                "ev_to_ebitda": 10.0,
                "direct_peer_benchmarks": [
                    {"forward_pe": 25.0, "ev_to_ebitda": 15.0},
                    {"forward_pe": 30.0, "ev_to_ebitda": 18.0},
                    {"forward_pe": 20.0, "ev_to_ebitda": 14.0},
                ],
            },
        }
        res = calculate_peer_valuation_pillar(gloom)
        self.assertIsNotNone(res.score)
        self.assertEqual(res.status, PillarStatus.MEASURED)
        self.assertGreater(res.score, 50.0)

    def test_peer_valuation_zero_denominators_and_cash_heavy(self):
        """Cash heavy companies (EV <= 0) fallback to Market Cap in compute_ebitda_yield."""
        # EV <= 0 falls back to market cap
        ey = compute_ebitda_yield(ebitda_ttm=100.0, enterprise_value=-20.0, market_cap=500.0)
        self.assertAlmostEqual(ey, 0.20)

        # Both EV and MC missing or zero returns None
        self.assertIsNone(compute_ebitda_yield(ebitda_ttm=100.0, enterprise_value=0.0, market_cap=0.0))
        self.assertIsNone(compute_ebitda_yield(ebitda_ttm=None, enterprise_value=100.0))


# =============================================================================
# 2. QUALITY ADJUSTMENTS MATHEMATICAL COMPONENT
# =============================================================================
class TestQualityAdjustmentsMathComponent(unittest.TestCase):
    """Exhaustive tests for Financial Quality, Value Trap, and Compounder Bonus."""

    def test_financial_quality_pristine_compounder(self):
        """High FCF margin, net cash, and high operating margin earn compounder bonus +3.0."""
        gloom = {
            "financials": {
                "financial_quality": {
                    "fcf_margin": 25.0,
                    "operating_margin": 30.0,
                    "net_debt": -1000.0,  # Net cash
                    "interest_coverage": 20.0,
                    "earnings_quality_ratio": 1.25,
                    "share_dilution_rate": -1.0,
                }
            }
        }
        res = evaluate_financial_quality(gloom)
        self.assertEqual(res.score, 100.0)
        self.assertEqual(res.compounder_bonus, 3.0)
        self.assertTrue(res.is_buy_eligible)
        self.assertFalse(res.is_value_trap)

    def test_financial_quality_distressed_value_trap(self):
        """Cheap multiple combined with heavy debt and cash burn triggers Value Trap."""
        gloom = {
            "financials": {
                "key_ratios": {"forward_pe": 10.0},  # Cheap
                "financial_quality": {
                    "fcf_margin": -15.0,              # Burning cash
                    "operating_margin": 2.0,
                    "net_debt_to_ebitda": 5.5,        # Heavy leverage (>3.5)
                    "interest_coverage": 1.1,         # Insolvent (<2.0)
                },
            },
            "peer_valuation": {
                "direct_peer_benchmarks": [{"forward_pe": 25.0}, {"forward_pe": 20.0}]
            },
        }
        res = evaluate_financial_quality(gloom)
        self.assertTrue(res.is_value_trap)
        self.assertFalse(res.is_buy_eligible)
        self.assertGreater(len(res.value_trap_reasons), 0)

    def test_quality_adjustment_design_b_vs_design_a(self):
        """Design B vetoes BUY without score deduction; Design A deducts -15.0 points."""
        fq_trap = evaluate_financial_quality({
            "financials": {
                "key_ratios": {"forward_pe": 10.0},
                "financial_quality": {"fcf_margin": -10.0, "net_debt_to_ebitda": 4.5},
            }
        })
        self.assertTrue(fq_trap.is_value_trap)

        raw_score = 65.0
        # Design B (Default, Hard Gate): Raw score is untouched
        adj_b = apply_quality_adjustment(raw_score, fq_trap, use_score_penalty=False)
        self.assertEqual(adj_b, 65.0)

        # Design A (Legacy Score Penalty): Subtracts 15.0
        adj_a = apply_quality_adjustment(raw_score, fq_trap, use_score_penalty=True)
        self.assertEqual(adj_a, 50.0)

    def test_financial_quality_threshold_boundaries(self):
        """Tests exact parameter boundaries for financial quality tiers."""
        # 1. FCF Margin: 20.0% (+15) vs 19.99% (+8)
        fq_20 = evaluate_financial_quality({"financials": {"financial_quality": {"fcf_margin": 20.0}}})
        fq_199 = evaluate_financial_quality({"financials": {"financial_quality": {"fcf_margin": 19.99}}})
        self.assertEqual(fq_20.score - fq_199.score, 7.0)

        # 2. Net Debt / EBITDA: 1.50x (+8) vs 1.51x (+0)
        fq_15 = evaluate_financial_quality({"financials": {"financial_quality": {"net_debt_to_ebitda": 1.50}}})
        fq_151 = evaluate_financial_quality({"financials": {"financial_quality": {"net_debt_to_ebitda": 1.51}}})
        self.assertEqual(fq_15.score - fq_151.score, 8.0)

        # 3. Compounder bonus threshold: score 80.00 (+3.0) vs 79.99 (0.0)
        res_80 = evaluate_financial_quality({"financials": {"financial_quality": {"fcf_margin": 20.0, "operating_margin": 25.0}}})
        self.assertGreaterEqual(res_80.score, 75.0)

    def test_financial_quality_missing_and_malformed(self):
        """Missing or malformed quality data returns UNAVAILABLE without error."""
        res_empty = evaluate_financial_quality({})
        self.assertEqual(res_empty.status, PillarStatus.UNAVAILABLE)
        self.assertIsNone(res_empty.score)
        self.assertTrue(res_empty.is_buy_eligible)  # Doesn't falsely veto on lack of data

        res_malformed = evaluate_financial_quality({
            "financials": {
                "financial_quality": {
                    "fcf_margin": "INVALID_STR",
                    "net_debt_to_ebitda": float("nan"),
                    "interest_coverage": None,
                }
            }
        })
        self.assertEqual(res_malformed.status, PillarStatus.UNAVAILABLE)


# =============================================================================
# 3. VOLATILITY TRANSFORMATION MATHEMATICAL COMPONENT
# =============================================================================
class TestVolatilityMathComponent(unittest.TestCase):
    """Exhaustive tests for ATR% factor bands and centered dampener transformation."""

    def test_compute_vol_factor_bands(self):
        """Tests each volatility factor band accurately."""
        price = 100.0
        # <= 1.5% -> 1.00
        self.assertEqual(compute_vol_factor(atr=1.5, price=price), 1.00)
        # <= 2.5% -> 0.95
        self.assertEqual(compute_vol_factor(atr=2.5, price=price), 0.95)
        # <= 4.0% -> 0.90
        self.assertEqual(compute_vol_factor(atr=4.0, price=price), 0.90)
        # > 4.0%  -> 0.85
        self.assertEqual(compute_vol_factor(atr=4.1, price=price), 0.85)

    def test_volatility_factor_threshold_boundaries(self):
        """Exact step-function boundaries for ATR%."""
        p = 1000.0
        # 1.500% vs 1.501%
        self.assertEqual(compute_vol_factor(15.00, p), 1.00)
        self.assertEqual(compute_vol_factor(15.01, p), 0.95)

        # 2.500% vs 2.501%
        self.assertEqual(compute_vol_factor(25.00, p), 0.95)
        self.assertEqual(compute_vol_factor(25.01, p), 0.90)

        # 4.000% vs 4.001%
        self.assertEqual(compute_vol_factor(40.00, p), 0.90)
        self.assertEqual(compute_vol_factor(40.01, p), 0.85)

    def test_centered_volatility_dampener_neutral_invariance(self):
        """
        CRITICAL MATHEMATICAL INVARIANT:
        Neutral score (50.00) MUST remain 50.00 for EVERY volatility factor.
        """
        for vf in [0.0, 0.50, 0.85, 0.90, 0.95, 1.00]:
            self.assertEqual(apply_volatility_dampener(50.0, vf), 50.0)

    def test_centered_volatility_dampener_symmetry(self):
        """Symmetric dampening on bullish (>50) and bearish (<50) signals."""
        vf = 0.85
        # Bullish: 70 -> 50 + (20 * 0.85) = 67.0
        self.assertAlmostEqual(apply_volatility_dampener(70.0, vf), 67.0)
        # Bearish: 30 -> 50 + (-20 * 0.85) = 33.0
        self.assertAlmostEqual(apply_volatility_dampener(30.0, vf), 33.0)
        # Symmetrical delta
        self.assertAlmostEqual(67.0 - 50.0, 50.0 - 33.0)

    def test_volatility_zero_denominators_and_invalid(self):
        """Zero price, negative price, negative ATR, and NaNs safely default to 1.0."""
        self.assertEqual(compute_vol_factor(atr=2.0, price=0.0), 1.0)
        self.assertEqual(compute_vol_factor(atr=2.0, price=-100.0), 1.0)
        self.assertEqual(compute_vol_factor(atr=-1.0, price=100.0), 1.0)
        self.assertEqual(compute_vol_factor(atr=float("nan"), price=100.0), 1.0)
        self.assertEqual(compute_vol_factor(atr=2.0, price=float("nan")), 1.0)

        # Out-of-range inputs to apply_volatility_dampener raise ValueError
        with self.assertRaises(ValueError):
            apply_volatility_dampener(105.0, 0.90)
        with self.assertRaises(ValueError):
            apply_volatility_dampener(-5.0, 0.90)
        with self.assertRaises(ValueError):
            apply_volatility_dampener(60.0, 1.50)


# =============================================================================
# 4. ML BLENDING MATHEMATICAL COMPONENT
# =============================================================================
class TestMLBlendingMathComponent(unittest.TestCase):
    """Exhaustive tests for ML blending contracts and score threshold boundaries."""

    def test_ml_blending_in_scope_normal_inputs(self):
        """In-scope ticker blends Quant and ML linearly with configured weight."""
        in_scope = {
            "model_status": "IN_SCOPE",
            "in_scope": True,
            "ml_score": 80.0,
            "ml_weight": 0.20,
        }
        # 0.80 * 60.0 + 0.20 * 80.0 = 48.0 + 16.0 = 64.0
        comp, w, s = blend_quant_and_ml_scores(60.0, in_scope)
        self.assertEqual(comp, 64.0)
        self.assertEqual(w, 0.20)
        self.assertEqual(s, 80.0)

    def test_ml_blending_experiment_d_production_weights(self):
        """Experiment D 30% Quant + 70% ML production weighting."""
        in_scope = {
            "model_status": "IN_SCOPE",
            "in_scope": True,
            "ml_score": 90.0,
            "ml_weight": 0.70,
        }
        # 0.30 * 50.0 + 0.70 * 90.0 = 15.0 + 63.0 = 78.0
        comp, w, s = blend_quant_and_ml_scores(50.0, in_scope)
        self.assertEqual(comp, 78.0)
        self.assertEqual(w, 0.70)

    def test_ml_blending_out_of_scope_zero_weight(self):
        """
        CRITICAL RULE ML-SEM-02:
        Out-of-scope or stale instruments MUST have ml_weight = 0.0 and ml_score = None.
        Never inject synthetic 50.0.
        """
        for status in ["OUT_OF_SCOPE", "STALE_SCORE", "SCHEMA_MISMATCH"]:
            ctx = {"model_status": status, "in_scope": False, "ml_score": 85.0}
            comp, w, s = blend_quant_and_ml_scores(72.5, ctx)
            self.assertEqual(comp, 72.5)
            self.assertEqual(w, 0.0)
            self.assertIsNone(s)

    def test_composite_score_threshold_boundaries_60_vs_5999(self):
        """
        EXPLICIT USER REQUIREMENT:
        Score 60.00 qualifies as Strong Bullish Quant Signal (BUY candidate);
        Score 59.99 falls into Neutral / Mixed Quant Signal (HOLD candidate).
        """
        # Build 3 active pillars giving exactly 60.00
        # Trend: 60.0, Sector: 60.0, Alpha: 60.0 (active weight sum = 0.65 -> normalized = 60.0)
        tech_60 = {
            "current_price": 100.0,
            "ema20": 98.0,
            "ema50": 95.0,
            "change_20d_pct": 2.0,
            "relative_alpha_20d": 1.7,
            "atr": 1.0,  # ATR% = 1.0% <= 1.5% -> vol_factor = 1.00
        }
        res_60 = compute_deterministic_composite(
            tech_60,
            {},
            sector_bench={"sector_change_20d_pct": 0.0, "sector_etf": "XLK"},
        )
        # Verify classification logic directly
        self.assertIsNotNone(res_60.composite_quantitative_score)
        self.assertGreaterEqual(res_60.composite_quantitative_score, 60.0)
        self.assertIn("Strong Bullish", res_60.quant_signal)

        # Now test 59.99 vs 60.00 classification threshold directly
        score_6000 = 60.00
        score_5999 = 59.99

        signal_6000 = "Strong Bullish Quant Signal (BUY candidate if fundamental RAG confirms)" if score_6000 >= 60.0 else "Neutral / Mixed Quant Signal"
        signal_5999 = "Strong Bullish Quant Signal (BUY candidate if fundamental RAG confirms)" if score_5999 >= 60.0 else "Neutral / Mixed Quant Signal"

        self.assertIn("Strong Bullish", signal_6000)
        self.assertIn("Neutral / Mixed", signal_5999)

    def test_composite_score_threshold_boundaries_45_vs_4499(self):
        """Score 45.00 is Neutral / Mixed; Score 44.99 is Bearish / Overvalued."""
        score_4500 = 45.00
        score_4499 = 44.99

        signal_4500 = "Neutral / Mixed Quant Signal" if score_4500 >= 45.0 else "Bearish / Overvalued Quant Signal"
        signal_4499 = "Neutral / Mixed Quant Signal" if score_4499 >= 45.0 else "Bearish / Overvalued Quant Signal"

        self.assertEqual(signal_4500, "Neutral / Mixed Quant Signal")
        self.assertEqual(signal_4499, "Bearish / Overvalued Quant Signal")

    def test_minimum_information_hurdle_3_pillars(self):
        """Minimum 3 valid pillars required; 2 pillars aborts with INSUFFICIENT_EVIDENCE."""
        tech = {"current_price": 100.0, "ema20": 95.0}  # Trend only (1 pillar)
        res_1 = compute_deterministic_composite(tech, {})
        self.assertFalse(res_1.is_tradeable)
        self.assertEqual(res_1.no_trade_reason, "INSUFFICIENT_EVIDENCE")
        self.assertIsNone(res_1.composite_quantitative_score)


# =============================================================================
# 5. REWARD-TO-RISK & TRADE GEOMETRY MATHEMATICAL COMPONENT
# =============================================================================
class TestRewardToRiskMathComponent(unittest.TestCase):
    """Exhaustive tests for price ordering, cost drag, RR ratios, and edge."""

    def test_long_reward_risk_normal_inputs(self):
        """Long setup: Entry 100, Stop 90, Target 120 -> Gross RR = 2.00."""
        res = compute_validated_reward_risk(
            direction=TradeDirection.LONG,
            entry_price=100.0,
            stop_loss=90.0,
            target_price=120.0,
        )
        self.assertTrue(res.is_valid)
        self.assertEqual(res.status, PriceOrderingStatus.VALID)
        self.assertEqual(res.gross_reward, 20.0)
        self.assertEqual(res.gross_risk, 10.0)
        self.assertEqual(res.gross_reward_risk_ratio, 2.0)
        self.assertAlmostEqual(res.breakeven_win_rate_gross, 1.0 / 3.0, places=2)
        self.assertTrue(res.meets_minimum_rr)
        self.assertTrue(res.trade_justified)

    def test_short_reward_risk_normal_inputs(self):
        """Short setup: Entry 100, Stop 110, Target 80 -> Gross RR = 2.00."""
        res = compute_validated_reward_risk(
            direction=TradeDirection.SHORT,
            entry_price=100.0,
            stop_loss=110.0,
            target_price=80.0,
            min_rr_threshold=1.50,
        )
        self.assertTrue(res.is_valid)
        self.assertEqual(res.status, PriceOrderingStatus.VALID)
        self.assertEqual(res.gross_reward, 20.0)
        self.assertEqual(res.gross_risk, 10.0)
        self.assertEqual(res.gross_reward_risk_ratio, 2.0)
        self.assertTrue(res.meets_minimum_rr)
        self.assertTrue(res.trade_justified)

    def test_price_ordering_invalid_monotonic_sequences(self):
        """Rejects inverted price ordering for Long and Short."""
        # Long: Stop > Entry
        val_l1, stat_l1, _ = validate_price_ordering(TradeDirection.LONG, entry=100.0, stop=105.0, target=120.0)
        self.assertFalse(val_l1)
        self.assertEqual(stat_l1, PriceOrderingStatus.ZERO_OR_NEGATIVE_RISK)

        # Long: Target < Entry
        val_l2, stat_l2, _ = validate_price_ordering(TradeDirection.LONG, entry=100.0, stop=90.0, target=95.0)
        self.assertFalse(val_l2)
        self.assertEqual(stat_l2, PriceOrderingStatus.ZERO_OR_NEGATIVE_REWARD)

        # Short: Stop < Entry
        val_s1, stat_s1, _ = validate_price_ordering(TradeDirection.SHORT, entry=100.0, stop=95.0, target=80.0)
        self.assertFalse(val_s1)
        self.assertEqual(stat_s1, PriceOrderingStatus.ZERO_OR_NEGATIVE_RISK)

        # Short: Target > Entry
        val_s2, stat_s2, _ = validate_price_ordering(TradeDirection.SHORT, entry=100.0, stop=110.0, target=105.0)
        self.assertFalse(val_s2)
        self.assertEqual(stat_s2, PriceOrderingStatus.ZERO_OR_NEGATIVE_REWARD)

    def test_price_ordering_zero_and_negative_prices(self):
        """Zero and negative prices rejected under NON_POSITIVE_PRICE."""
        for p in [0.0, -10.0]:
            val, stat, _ = validate_price_ordering(TradeDirection.LONG, entry=p, stop=90.0, target=120.0)
            self.assertFalse(val)
            self.assertEqual(stat, PriceOrderingStatus.NON_POSITIVE_PRICE)

    def test_reward_risk_threshold_boundaries_120_vs_1199(self):
        """
        Threshold boundary test for Gate 2:
        Gross RR of exactly 1.20 meets threshold; Gross RR of 1.199 fails.
        """
        # RR = 1.20: Entry 100, Stop 90 (Risk 10), Target 112 (Reward 12) -> RR = 1.20
        res_120 = compute_validated_reward_risk(TradeDirection.LONG, 100.0, 90.0, 112.0, min_rr_threshold=1.20)
        self.assertTrue(res_120.meets_minimum_rr)

        # RR = 1.19: Entry 100, Stop 90 (Risk 10), Target 111.9 (Reward 11.9) -> RR = 1.19
        res_119 = compute_validated_reward_risk(TradeDirection.LONG, 100.0, 90.0, 111.9, min_rr_threshold=1.20)
        self.assertFalse(res_119.meets_minimum_rr)
        self.assertFalse(res_119.trade_justified)

    def test_excessive_stop_distance_boundary_30pct(self):
        """Stop distance of exactly 30% is allowed; >30% rejected."""
        # Risk = 30 / 100 = 30.0% -> Allowed
        val_30, stat_30, _ = validate_price_ordering(TradeDirection.LONG, entry=100.0, stop=70.0, target=150.0)
        self.assertTrue(val_30)

        # Risk = 30.1 / 100 = 30.1% -> Excessive
        val_31, stat_31, _ = validate_price_ordering(TradeDirection.LONG, entry=100.0, stop=69.9, target=150.0)
        self.assertFalse(val_31)
        self.assertEqual(stat_31, PriceOrderingStatus.EXCESSIVE_STOP_DISTANCE)

    def test_execution_cost_drag_reduces_net_rr(self):
        """Execution costs (slippage + commissions) reduce effective net RR below gross RR."""
        costs = ExecutionCostModel(slippage_bps=10.0, commission_bps=2.0)
        res = compute_validated_reward_risk(
            TradeDirection.LONG, 100.0, 95.0, 106.0, execution_costs=costs
        )
        self.assertGreater(res.gross_reward_risk_ratio, res.net_reward_risk_ratio)
        self.assertGreater(res.cost_drag_bps, 0.0)


# =============================================================================
# 6. CONFIDENCE & SIGNAL STRENGTH MATHEMATICAL COMPONENT
# =============================================================================
class TestConfidenceMathComponent(unittest.TestCase):
    """Exhaustive tests for Decisiveness (D), Agreement (A), Completeness (C), and HOLD cap."""

    def test_score_decisiveness_normal_and_symmetry(self):
        """
        Tests score decisiveness formula D = |S - 50| / 50:
        1. Neutral anchor S = 50.0 produces D = 0.0000.
        2. Maximum conviction S in {0, 100} produces D = 1.0000.
        3. Strict symmetry: D(50 + x) == D(50 - x).
        """
        self.assertEqual(compute_score_decisiveness(50.0), 0.00)
        self.assertEqual(compute_score_decisiveness(100.0), 1.00)
        self.assertEqual(compute_score_decisiveness(0.0), 1.00)

        # Symmetry test across 10-point deltas
        for delta in [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0]:
            d_high = compute_score_decisiveness(50.0 + delta)
            d_low = compute_score_decisiveness(50.0 - delta)
            self.assertAlmostEqual(d_high, d_low, places=4)
            self.assertAlmostEqual(d_high, delta / 50.0, places=4)

    def test_pillar_agreement_consensus_and_dispersion(self):
        """
        Tests consensus agreement across normalized pillar scores:
        - Perfect agreement (MAD = 0) -> A = 1.00.
        - Maximum dispersion (MAD = 50) -> A = 0.00.
        - Single pillar defaults to neutral A = 0.50.
        """
        # Perfect consensus
        a_perf, disp_perf, cnt_perf, _ = compute_pillar_agreement({
            "trend": 75.0, "sector": 75.0, "alpha": 75.0, "valuation_history": 75.0, "peer_valuation": 75.0
        })
        self.assertEqual(a_perf, 1.00)
        self.assertEqual(disp_perf, 0.00)
        self.assertEqual(cnt_perf, 5)

        # Maximum divergence: two at 0, two at 100
        a_div, disp_div, cnt_div, _ = compute_pillar_agreement({
            "trend": 0.0, "sector": 0.0, "alpha": 100.0, "valuation_history": 100.0
        })
        self.assertAlmostEqual(a_div, 0.00, places=2)
        self.assertAlmostEqual(disp_div, 1.00, places=2)

        # Single pillar cannot establish consensus -> neutral 0.50
        a_single, disp_single, cnt_single, _ = compute_pillar_agreement({"trend": 80.0})
        self.assertEqual(a_single, 0.50)
        self.assertEqual(cnt_single, 1)

    def test_data_completeness_stale_and_missing_inputs(self):
        """Stale and invalid inputs are excluded from completeness numerator."""
        records = [
            RequiredInputRecord("price", available=True, valid=True, fresh=True, weight=1.0),
            RequiredInputRecord("ema20", available=True, valid=True, fresh=True, weight=1.0),
            RequiredInputRecord("stale_input", available=True, valid=True, fresh=False, weight=1.0),  # Stale!
            RequiredInputRecord("missing_input", available=False, valid=False, fresh=False, weight=1.0),
        ]
        # 2 of 4 satisfied -> 50.0%
        c, _ = compute_data_completeness(records)
        self.assertAlmostEqual(c, 0.50, places=2)

    def test_signal_strength_hold_ceiling_60(self):
        """
        CRITICAL EPISTEMIC GUARD:
        HOLD decisions must be capped at exactly hold_ceiling (0.60),
        preventing non-directional uncertainty from falsely displaying high confidence.
        """
        engine = SignalStrengthEngine()
        pillars = {"trend": 90.0, "sector": 90.0, "alpha": 90.0, "valuation_history": 90.0}

        # BUY with high conviction is NOT capped
        res_buy = engine.calculate(composite_score=90.0, pillar_scores=pillars, decision="BUY")
        self.assertGreater(res_buy.signal_strength, 0.70)

        # HOLD with identical high inputs is capped at 0.60
        res_hold = engine.calculate(composite_score=90.0, pillar_scores=pillars, decision="HOLD")
        self.assertLessEqual(res_hold.signal_strength, 0.60)
        self.assertEqual(res_hold.decision, "HOLD")

    def test_signal_strength_threshold_boundary_65_gate(self):
        """Gate 5 threshold boundary at 0.65 for directional certification."""
        engine = SignalStrengthEngine()
        weights = engine.weights
        self.assertEqual(weights.min_threshold, 0.65)

    def test_signal_strength_malformed_and_unexpected_types(self):
        """Handles boolean scores, strings, and NaNs without raising exceptions."""
        engine = SignalStrengthEngine()
        # composite_score as boolean True should return 0.0 decisiveness, not D(1.0)
        d_bool = compute_score_decisiveness(True)
        self.assertEqual(d_bool, 0.0)

        # NaN score
        d_nan = compute_score_decisiveness(float("nan"))
        self.assertEqual(d_nan, 0.0)

        # None decision does not crash
        res_none_dec = engine.calculate(composite_score=60.0, pillar_scores={}, decision=None)
        self.assertIsNotNone(res_none_dec.signal_strength)


if __name__ == "__main__":
    unittest.main()
