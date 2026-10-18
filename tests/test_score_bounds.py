"""Unit tests for quant score bounds [0.0, 100.0]."""

import unittest
from quant.normalization import clamp, logistic_scale, normal_cdf_scale
from quant.pillar_scores import (
    calculate_trend_pillar,
    calculate_sector_pillar,
    calculate_alpha_pillar,
    calculate_valuation_history_pillar,
    calculate_peer_valuation_pillar,
)
from quant.composite_score import compute_deterministic_composite


class TestScoreBounds(unittest.TestCase):
    def test_clamp_bounds(self):
        self.assertEqual(clamp(150.0), 100.0)
        self.assertEqual(clamp(-50.0), 0.0)
        self.assertEqual(clamp(50.0), 50.0)
        self.assertEqual(clamp(float("nan")), 50.0)

    def test_logistic_scale_bounds(self):
        # Extreme positive
        self.assertLessEqual(logistic_scale(100.0), 100.0)
        self.assertGreaterEqual(logistic_scale(100.0), 99.9)
        # Extreme negative
        self.assertGreaterEqual(logistic_scale(-100.0), 0.0)
        self.assertLessEqual(logistic_scale(-100.0), 0.1)
        # Exact midpoint
        self.assertEqual(logistic_scale(0.0), 50.0)

    def test_normal_cdf_bounds(self):
        self.assertLessEqual(normal_cdf_scale(10.0), 100.0)
        self.assertGreaterEqual(normal_cdf_scale(-10.0), 0.0)
        self.assertAlmostEqual(normal_cdf_scale(0.0), 50.0, places=4)

    def test_pillar_scores_under_extremes(self):
        # 1. Trend under extreme bullish and bearish
        tech_extreme_bull = {
            "current_price": 500.0, "ema20": 400.0, "ema50": 300.0, "ema200": 200.0,
            "rsi14": 99.0, "atr": 5.0
        }
        res_bull = calculate_trend_pillar(tech_extreme_bull)
        self.assertGreaterEqual(res_bull.score, 0.0)
        self.assertLessEqual(res_bull.score, 100.0)

        tech_extreme_bear = {
            "current_price": 50.0, "ema20": 100.0, "ema50": 150.0, "ema200": 200.0,
            "rsi14": 5.0, "atr": 10.0
        }
        res_bear = calculate_trend_pillar(tech_extreme_bear)
        self.assertGreaterEqual(res_bear.score, 0.0)
        self.assertLessEqual(res_bear.score, 100.0)

        # 2. Sector under extreme +200% and -200%
        res_sec_high = calculate_sector_pillar(
            {"change_5d_pct": 200.0, "change_20d_pct": 300.0},
            {"sector_change_pct": -50.0, "sector_change_20d_pct": -80.0}
        )
        self.assertGreaterEqual(res_sec_high.score, 0.0)
        self.assertLessEqual(res_sec_high.score, 100.0)

        # 3. Market Alpha under extreme
        res_alpha = calculate_alpha_pillar({"relative_alpha_20d": -150.0})
        self.assertGreaterEqual(res_alpha.score, 0.0)
        self.assertLessEqual(res_alpha.score, 100.0)

    def test_composite_bounds_under_all_scenarios(self):
        # Pristine compounder + super trend
        tech = {"current_price": 100.0, "ema20": 95.0, "ema50": 90.0, "ema200": 80.0, "rsi14": 65.0, "atr": 1.0}
        sec = {"sector_change_pct": 0.0}
        gloom = {
            "financials": {
                "financial_quality": {"fcf_margin": 45.0, "operating_margin": 40.0, "net_debt": -100.0, "interest_coverage": 50.0, "earnings_quality_ratio": 1.5},
                "key_ratios": {"forward_pe": 18.0, "trailing_pe": 25.0, "revenue_growth": 0.25}
            },
            "peer_valuation": {"direct_peer_benchmarks": [{"forward_pe": 22.0}]}
        }
        res = compute_deterministic_composite(tech, gloom, sec)
        self.assertIsNotNone(res.composite_quantitative_score)
        self.assertGreaterEqual(res.composite_quantitative_score, 0.0)
        self.assertLessEqual(res.composite_quantitative_score, 100.0)
        self.assertGreaterEqual(res.raw_composite, 0.0)
        self.assertLessEqual(res.raw_composite, 100.0)


if __name__ == "__main__":
    unittest.main()
