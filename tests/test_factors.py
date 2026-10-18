"""Unit tests for quant/factors.py."""

import unittest
from quant.factors import (
    clean_float,
    compute_ema_alignment,
    compute_rsi_velocity,
    compute_sector_excess,
    compute_market_alpha,
    compute_earnings_yield,
    compute_ebitda_yield,
    compute_fcf_yield,
    GICS_SECTOR_ETF_MAP,
)


class TestFactors(unittest.TestCase):
    def test_clean_float(self):
        self.assertEqual(clean_float("12.5%"), 12.5)
        self.assertEqual(clean_float("+4.2"), 4.2)
        self.assertEqual(clean_float("$150.25"), 150.25)
        self.assertEqual(clean_float("1,250.50"), 1250.50)
        self.assertIsNone(clean_float("N/A"))
        self.assertIsNone(clean_float("None"))
        self.assertIsNone(clean_float(None))
        self.assertEqual(clean_float(None, default=50.0), 50.0)

    def test_ema_alignment(self):
        # 1. Bullish stack above EMA200: Price > EMA20 > EMA50 > EMA200
        score_bull, desc_bull = compute_ema_alignment(price=110, ema20=105, ema50=100, ema200=90)
        self.assertGreaterEqual(score_bull, 85.0)
        self.assertIn("bullish", desc_bull)

        # 2. Bearish stack: Price < EMA20 < EMA50 < EMA200
        score_bear, desc_bear = compute_ema_alignment(price=80, ema20=85, ema50=90, ema200=100)
        self.assertLessEqual(score_bear, 25.0)
        self.assertIn("bearish", desc_bear)

        # 3. Slope adjustment with rising EMA20
        score_rising, _ = compute_ema_alignment(price=110, ema20=105, ema50=100, ema200=90, atr14=2.0, ema20_prev5=100)
        self.assertGreater(score_rising, score_bull)

    def test_rsi_velocity(self):
        # 1. Bullish momentum expansion (50-68) with positive velocity
        score_mom, vel, desc = compute_rsi_velocity(rsi14=60.0, rsi14_prev5=55.0)
        self.assertEqual(vel, 1.0)
        self.assertGreaterEqual(score_mom, 70.0)

        # 2. Oversold inflection bounce: RSI < 32 with strong positive velocity
        score_bounce, vel_b, desc_b = compute_rsi_velocity(rsi14=28.0, rsi14_prev5=18.0)
        self.assertEqual(vel_b, 2.0)
        self.assertEqual(score_bounce, 55.0)  # 45.0 + 2.0*5.0
        self.assertEqual(desc_b, "oversold_positive_inflection")

        # 3. Severe oversold washout without inflection
        score_wash, vel_w, desc_w = compute_rsi_velocity(rsi14=20.0, rsi14_prev5=22.0)
        self.assertEqual(score_wash, 16.0)
        self.assertEqual(desc_w, "severe_oversold_washout")

        # 4. Overbought exhaustion
        score_ob, _, desc_ob = compute_rsi_velocity(rsi14=80.0)
        self.assertEqual(score_ob, 65.0)  # 85 - (80-75)*4 = 65
        self.assertEqual(desc_ob, "overbought_exhaustion")

    def test_sector_excess_multi_horizon(self):
        # Stock beats sector across 5d (+2%), 20d (+5%), 60d (+8%)
        excess = compute_sector_excess(
            stock_ret_5d=0.03, sector_ret_5d=0.01,   # +2% (wt 0.20) -> +0.004
            stock_ret_20d=0.08, sector_ret_20d=0.03, # +5% (wt 0.50) -> +0.025
            stock_ret_60d=0.18, sector_ret_60d=0.10, # +8% (wt 0.30) -> +0.024
        )
        self.assertAlmostEqual(excess, 0.053, places=4)

        # Fallback when only 5-day is present
        excess_5d_only = compute_sector_excess(stock_ret_5d=0.02, sector_ret_5d=0.01)
        self.assertAlmostEqual(excess_5d_only, 0.01, places=4)

    def test_gics_sector_mapping(self):
        self.assertEqual(GICS_SECTOR_ETF_MAP["Technology"], "XLK")
        self.assertEqual(GICS_SECTOR_ETF_MAP["Financials"], "XLF")
        self.assertEqual(GICS_SECTOR_ETF_MAP["Energy"], "XLE")
        self.assertEqual(GICS_SECTOR_ETF_MAP["Health Care"], "XLV")

    def test_market_alpha_jensen(self):
        # Stock: +10% 20d, SPY: +4% 20d, Beta: 1.0, rf: 0.16%
        # Jensen alpha = 10% - [0.16% + 1.0*(4% - 0.16%)] = 6.0%
        # Excess 20d = 6.0%
        # Excess 5d = 1.0%
        # Composite = 0.50*6.0% + 0.35*6.0% + 0.15*1.0% = 5.25%
        alpha = compute_market_alpha(
            stock_ret_20d=0.10, spy_ret_20d=0.04,
            beta_60d=1.0, rf_20d=0.0016,
            stock_ret_5d=0.02, spy_ret_5d=0.01
        )
        self.assertAlmostEqual(alpha, 0.0525, places=4)

    def test_yield_space_multiples(self):
        # 1. Earnings Yield
        self.assertAlmostEqual(compute_earnings_yield(eps_ttm=5.0, price=100.0), 0.05)
        # Negative EPS: no error, preserves negative yield
        self.assertAlmostEqual(compute_earnings_yield(eps_ttm=-2.5, price=50.0), -0.05)
        self.assertIsNone(compute_earnings_yield(eps_ttm=None, price=100.0))

        # 2. EBITDA Yield
        self.assertAlmostEqual(compute_ebitda_yield(ebitda_ttm=10.0, enterprise_value=100.0), 0.10)
        # Non-positive EBITDA
        self.assertAlmostEqual(compute_ebitda_yield(ebitda_ttm=-5.0, enterprise_value=100.0), -0.05)
        # Negative EV (cash > debt + mcap) falls back to market cap
        self.assertAlmostEqual(compute_ebitda_yield(ebitda_ttm=10.0, enterprise_value=-20.0, market_cap=50.0), 0.20)

        # 3. FCF Yield
        self.assertAlmostEqual(compute_fcf_yield(fcf_ttm=8.0, enterprise_value=80.0), 0.10)
        self.assertAlmostEqual(compute_fcf_yield(fcf_ttm=-4.0, enterprise_value=80.0), -0.05)


if __name__ == "__main__":
    unittest.main()
