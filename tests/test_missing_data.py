"""Unit tests for missing data tracking, weight re-normalization, and insufficient evidence gates."""

import unittest
from quant.composite_score import compute_deterministic_composite, BASE_WEIGHTS
from quant.schemas import PillarStatus


class TestMissingData(unittest.TestCase):
    def test_completely_empty_payload(self):
        # Empty inputs should cleanly return insufficient evidence without exceptions
        res = compute_deterministic_composite({}, {}, {})
        self.assertIsNone(res.composite_quantitative_score)
        self.assertIsNone(res.raw_composite)
        self.assertFalse(res.is_tradeable)
        self.assertEqual(res.no_trade_reason, "INSUFFICIENT_EVIDENCE")
        self.assertEqual(res.available_pillars_count, 0)

    def test_missing_price_aborts_evaluation(self):
        # Price is mandatory for trend and trading
        tech = {"change_5d_pct": 5.0}  # No price
        res = compute_deterministic_composite(tech, {}, {})
        self.assertIsNone(res.composite_quantitative_score)
        self.assertFalse(res.is_tradeable)
        self.assertEqual(res.no_trade_reason, "INSUFFICIENT_EVIDENCE")

    def test_insufficient_pillars_gate_below_three(self):
        # Only 2 pillars available (Trend and Alpha)
        tech = {
            "current_price": 100.0,
            "ema20": 95.0,
            "rsi14": 55.0,
            "relative_alpha_5d": 3.0,
        }
        # No sector benchmark, no valuation history, no peer valuation
        res = compute_deterministic_composite(tech, {}, {})
        self.assertEqual(res.available_pillars_count, 2)
        self.assertIsNone(res.composite_quantitative_score)
        self.assertFalse(res.is_tradeable)
        self.assertEqual(res.no_trade_reason, "INSUFFICIENT_EVIDENCE")

    def test_dynamic_weight_renormalization_three_pillars(self):
        # 3 pillars available: Trend (0.25), Sector (0.20), Alpha (0.20)
        # Base weight sum = 0.65
        # Expected re-normalized weights:
        # Trend: 0.25 / 0.65 = 0.3846
        # Sector: 0.20 / 0.65 = 0.3077
        # Alpha: 0.20 / 0.65 = 0.3077
        tech = {
            "current_price": 100.0,
            "ema20": 95.0,
            "rsi14": 55.0,
            "change_5d_pct": 2.0,
            "relative_alpha_5d": 1.0,
        }
        sec = {"sector_change_pct": 0.5, "sector_etf": "XLK"}
        res = compute_deterministic_composite(tech, {}, sec)

        self.assertEqual(res.available_pillars_count, 3)
        self.assertTrue(res.is_tradeable)
        self.assertIsNotNone(res.composite_quantitative_score)

        weights = res.active_weights
        self.assertIn("trend", weights)
        self.assertIn("sector", weights)
        self.assertIn("alpha", weights)
        self.assertNotIn("valuation_history", weights)
        self.assertNotIn("peer_valuation", weights)

        self.assertAlmostEqual(sum(weights.values()), 1.0, places=2)
        self.assertAlmostEqual(weights["trend"], 0.25 / 0.65, places=3)
        self.assertAlmostEqual(weights["sector"], 0.20 / 0.65, places=3)
        self.assertAlmostEqual(weights["alpha"], 0.20 / 0.65, places=3)

    def test_dynamic_weight_renormalization_four_pillars(self):
        # 4 pillars available: Trend (0.25), Sector (0.20), Alpha (0.20), Peer Valuation (0.20)
        # Base weight sum = 0.85
        tech = {
            "current_price": 100.0,
            "ema20": 95.0,
            "rsi14": 55.0,
            "change_5d_pct": 2.0,
            "relative_alpha_5d": 1.0,
        }
        sec = {"sector_change_pct": 0.5, "sector_etf": "XLK"}
        gloom = {
            "financials": {"key_ratios": {"forward_pe": 20.0}},
            "peer_valuation": {"direct_peer_benchmarks": [{"forward_pe": 22.0}]}
        }
        res = compute_deterministic_composite(tech, gloom, sec)

        self.assertEqual(res.available_pillars_count, 4)
        self.assertTrue(res.is_tradeable)
        weights = res.active_weights
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=2)
        self.assertAlmostEqual(weights["trend"], 0.25 / 0.85, places=3)


if __name__ == "__main__":
    unittest.main()
