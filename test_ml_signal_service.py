import unittest
from ml_signal_service import (
    classify_ml_signal,
    get_ml_context,
    get_all_ml_scores,
    get_out_of_scope_context,
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

    def test_get_ml_context_focused_ticker(self):
        ctx = get_ml_context("MRVL")
        self.assertIsNotNone(ctx)
        self.assertEqual(ctx["ticker"], "MRVL")
        self.assertTrue(ctx["in_scope"])
        self.assertEqual(ctx["rank"], 1)
        self.assertEqual(ctx["universe_size"], 25)
        self.assertEqual(ctx["target_horizon_sessions"], 20)
        self.assertGreaterEqual(ctx["score_0_100"], 0.0)
        self.assertLessEqual(ctx["score_0_100"], 100.0)
        self.assertIn("signal", ctx)
        self.assertIn("badge", ctx)
        self.assertIn("predicted_20d_excess_return", ctx)
        self.assertTrue(ctx["in_top_5_portfolio"])
        self.assertTrue(ctx["in_top_10_portfolio"])

    def test_get_ml_context_out_of_scope_returns_sensible_value(self):
        # A ticker outside the scope of both the focused model and broad model
        ctx = get_ml_context("NONEXISTENT_TICKER_12345")
        self.assertIsNotNone(ctx)
        self.assertEqual(ctx["ticker"], "NONEXISTENT_TICKER_12345")
        self.assertFalse(ctx["in_scope"])
        self.assertEqual(ctx["status"], "OUT_OF_SCOPE")
        self.assertEqual(ctx["score_0_100"], 50.0)
        self.assertIsNone(ctx["raw_score_0_100"])
        self.assertEqual(ctx["predicted_20d_excess_return"], 0.0)
        self.assertEqual(ctx["signal"], "OUT_OF_SCOPE")
        self.assertEqual(ctx["stance"], "NEUTRAL")
        self.assertIn("OUT OF SCOPE", ctx["badge"])
        self.assertFalse(ctx["in_top_10_portfolio"])
        self.assertIn("outside", ctx["message"].lower())

        # Empty / None inputs still return None
        self.assertIsNone(get_ml_context(""))
        self.assertIsNone(get_ml_context(None))

    def test_get_all_ml_scores_returns_focused_universe(self):
        scores = get_all_ml_scores()
        self.assertIsInstance(scores, list)
        self.assertGreaterEqual(len(scores), 25)
        # Should be ordered by rank ascending
        self.assertEqual(scores[0]["rank"], 1)
        self.assertEqual(scores[0]["ticker"], "MRVL")
        self.assertLessEqual(scores[0]["rank"], scores[1]["rank"])
        self.assertIn("signal", scores[0])
        self.assertIn("predicted_20d_excess_return", scores[0])

    def test_enrich_signal_dict(self):
        sample = {
            "symbol": "MRVL",
            "decision": "BUY",
            "quant_score": 85.0,
            "raw_json": None,
        }
        enriched = _enrich_signal_dict(sample)
        self.assertIsNotNone(enriched.get("ml_score"))
        self.assertEqual(enriched.get("ml_signal_label"), "STRONG OUTPERFORM")
        self.assertIsNotNone(enriched.get("ml_predicted_return"))
        self.assertEqual(enriched.get("ml_rank"), 1)


if __name__ == "__main__":
    unittest.main()
