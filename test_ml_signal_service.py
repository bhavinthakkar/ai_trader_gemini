import unittest
from ml_signal_service import (
    classify_ml_signal,
    get_ml_context,
    get_all_ml_scores,
)
from db import _enrich_signal_dict


class TestMLSignalService(unittest.TestCase):
    def test_classify_ml_signal_tiers(self):
        # Top quintile (>= 80)
        c85 = classify_ml_signal(85.0)
        self.assertEqual(c85["signal"], "STRONG OUTPERFORM")
        self.assertEqual(c85["stance"], "BULLISH")
        self.assertIn("Top Quintile", c85["tier"])

        # Quintile 2 (60-80)
        c70 = classify_ml_signal(70.0)
        self.assertEqual(c70["signal"], "OUTPERFORM")
        self.assertEqual(c70["stance"], "BULLISH")

        # Quintile 3 (40-60)
        c50 = classify_ml_signal(50.0)
        self.assertEqual(c50["signal"], "MARKET PERFORM")
        self.assertEqual(c50["stance"], "NEUTRAL")

        # Quintile 4 (20-40)
        c30 = classify_ml_signal(30.0)
        self.assertEqual(c30["signal"], "UNDERPERFORM")
        self.assertEqual(c30["stance"], "BEARISH")

        # Bottom quintile (< 20)
        c10 = classify_ml_signal(10.0)
        self.assertEqual(c10["signal"], "STRONG UNDERPERFORM")
        self.assertEqual(c10["stance"], "BEARISH")

        # None / unavailable
        c_none = classify_ml_signal(None)
        self.assertEqual(c_none["signal"], "UNAVAILABLE")
        self.assertEqual(c_none["stance"], "UNAVAILABLE")

    def test_get_ml_context_valid_ticker(self):
        ctx = get_ml_context("EXPE")
        self.assertIsNotNone(ctx)
        self.assertEqual(ctx["ticker"], "EXPE")
        self.assertEqual(ctx["target_horizon_sessions"], 20)
        self.assertGreaterEqual(ctx["score_0_100"], 0.0)
        self.assertLessEqual(ctx["score_0_100"], 100.0)
        self.assertIn("signal", ctx)
        self.assertIn("badge", ctx)
        self.assertIn("predicted_20d_excess_return", ctx)
        self.assertIn("in_top_10_portfolio", ctx)

    def test_get_ml_context_invalid_or_european(self):
        self.assertIsNone(get_ml_context("NONEXISTENT_TICKER_12345"))
        self.assertIsNone(get_ml_context("SAP.DE"))
        self.assertIsNone(get_ml_context(""))
        self.assertIsNone(get_ml_context(None))

    def test_get_all_ml_scores(self):
        scores = get_all_ml_scores()
        self.assertIsInstance(scores, list)
        self.assertGreater(len(scores), 100)
        # Should be ordered by rank ascending
        self.assertEqual(scores[0]["rank"], 1)
        self.assertLessEqual(scores[0]["rank"], scores[1]["rank"])
        self.assertIn("signal", scores[0])
        self.assertIn("predicted_20d_excess_return", scores[0])

    def test_enrich_signal_dict(self):
        sample = {
            "symbol": "EXPE",
            "decision": "HOLD",
            "quant_score": 74.34,
            "raw_json": None,
        }
        enriched = _enrich_signal_dict(sample)
        self.assertIsNotNone(enriched.get("ml_score"))
        self.assertEqual(enriched.get("ml_signal_label"), "STRONG OUTPERFORM")
        self.assertIsNotNone(enriched.get("ml_predicted_return"))
        self.assertIsNotNone(enriched.get("ml_rank"))


if __name__ == "__main__":
    unittest.main()
