"""Unit tests for volatility bands, centred dampening, and boundary conditions."""

import unittest
from quant.volatility import (
    compute_vol_factor,
    apply_volatility_dampener,
    compute_volatility_adjusted_position_size,
)


class TestVolatilityBoundaries(unittest.TestCase):
    def test_compute_vol_factor_bands(self):
        # Calm: <= 1.5%
        self.assertEqual(compute_vol_factor(atr=1.4, price=100.0), 1.00)
        self.assertEqual(compute_vol_factor(atr=1.5, price=100.0), 1.00)

        # Normal: <= 2.5%
        self.assertEqual(compute_vol_factor(atr=1.51, price=100.0), 0.95)
        self.assertEqual(compute_vol_factor(atr=2.5, price=100.0), 0.95)

        # Elevated: <= 4.0%
        self.assertEqual(compute_vol_factor(atr=2.51, price=100.0), 0.90)
        self.assertEqual(compute_vol_factor(atr=4.0, price=100.0), 0.90)

        # Extreme: > 4.0%
        self.assertEqual(compute_vol_factor(atr=4.01, price=100.0), 0.85)
        self.assertEqual(compute_vol_factor(atr=8.0, price=100.0), 0.85)

        # Edge cases: missing, zero price, negative atr
        self.assertEqual(compute_vol_factor(atr=None, price=100.0), 1.0)
        self.assertEqual(compute_vol_factor(atr=2.0, price=0.0), 1.0)
        self.assertEqual(compute_vol_factor(atr=-1.0, price=100.0), 1.0)

    def test_neutral_invariance_across_all_bands(self):
        # Neutral signal (50.0) MUST NOT become bearish under any volatility factor
        bands = [1.00, 0.95, 0.90, 0.85, 0.70, 0.50, 0.0]
        for f in bands:
            score = apply_volatility_dampener(50.0, f)
            self.assertEqual(score, 50.0, f"Neutral 50.0 became {score} under f={f}")

    def test_symmetric_conviction_dampening(self):
        # Bullish signal (70.0) with f=0.85 -> 67.0
        # Formula: 50.0 + (70 - 50) * 0.85 = 50 + 17 = 67.0
        bull_dampened = apply_volatility_dampener(70.0, 0.85)
        self.assertEqual(bull_dampened, 67.0)

        # Bearish signal (30.0) with f=0.85 -> 33.0
        # Formula: 50.0 + (30 - 50) * 0.85 = 50 - 17 = 33.0
        # Crucial: conviction contracts toward neutral 50, does NOT push down to 25.5!
        bear_dampened = apply_volatility_dampener(30.0, 0.85)
        self.assertEqual(bear_dampened, 33.0)

        # Distances to neutral are perfectly symmetric: |67 - 50| == |33 - 50| == 17
        self.assertEqual(abs(bull_dampened - 50.0), abs(bear_dampened - 50.0))

    def test_input_validation_errors(self):
        # Score out of bounds [0, 100]
        with self.assertRaises(ValueError):
            apply_volatility_dampener(105.0, 0.90)
        with self.assertRaises(ValueError):
            apply_volatility_dampener(-5.0, 0.90)

        # Vol factor out of bounds [0, 1]
        with self.assertRaises(ValueError):
            apply_volatility_dampener(70.0, 1.20)
        with self.assertRaises(ValueError):
            apply_volatility_dampener(70.0, -0.10)

    def test_position_sizing_volatility_adjustment(self):
        # Target ATR is 2.5%
        # Calm stock (ATR = 1.5%): keeps full base weight 10%
        w_calm = compute_volatility_adjusted_position_size(base_weight=0.10, atr_pct=1.5, target_atr_pct=2.5)
        self.assertEqual(w_calm, 0.10)

        # High-vol stock (ATR = 5.0%): weight halved to 5%
        w_vol = compute_volatility_adjusted_position_size(base_weight=0.10, atr_pct=5.0, target_atr_pct=2.5)
        self.assertEqual(w_vol, 0.05)


if __name__ == "__main__":
    unittest.main()
