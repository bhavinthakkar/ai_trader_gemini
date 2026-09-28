import datetime
import unittest

from catalyst_service import (
    CATALYST_CONFIDENCE_CEILING,
    DIP_BUY,
    EXTENSION_SELL,
    NO_CATALYST,
    build_relevance_needles,
    evaluate_catalyst_setup,
    find_fresh_catalysts,
)

TODAY = datetime.date(2026, 9, 28)


def news(title, days_ago=0, summary=""):
    published = TODAY - datetime.timedelta(days=days_ago)
    return {"title": title, "summary": summary, "published_at": published.isoformat(), "category": "News"}


def strong_pillars():
    return {
        "composite_quantitative_score": 52.0,
        "valuation_history_score": 71.0,
        "peer_valuation_score": 66.0,
    }


def rich_pillars():
    return {
        "composite_quantitative_score": 61.0,
        "valuation_history_score": 34.0,
        "peer_valuation_score": 29.0,
    }


def dip_technical(**overrides):
    values = {
        "current_price": 118.0,
        "ema50": 122.0,
        "high_20d": 132.0,
        "low_20d": 115.0,
        "change_5d_pct": -7.4,
        "rsi14": 38.0,
        "rvol_20d": 1.9,
    }
    values.update(overrides)
    return values


def extension_technical(**overrides):
    values = {
        "current_price": 171.0,
        "ema50": 150.0,
        "high_20d": 172.0,
        "low_20d": 140.0,
        "change_5d_pct": 11.2,
        "rsi14": 78.0,
        "rvol_20d": 1.6,
    }
    values.update(overrides)
    return values


class RelevanceNeedlesTests(unittest.TestCase):
    def test_strips_exchange_suffix(self):
        self.assertIn("000660", build_relevance_needles("000660.KS"))

    def test_strips_class_share_suffix(self):
        self.assertIn("brk", build_relevance_needles("BRK.B"))

    def test_adds_company_name_and_head_token(self):
        needles = build_relevance_needles("NVDA", "NVIDIA Corporation")
        self.assertIn("nvidia corporation", needles)
        self.assertIn("nvidia", needles)

    def test_company_name_sanitised(self):
        needles = build_relevance_needles("NVDA", "NVIDIA Corp. (St)")
        self.assertTrue(any("nvidia" in n for n in needles))

    def test_deduplicates(self):
        needles = build_relevance_needles("NVDA", "NVDA")
        self.assertEqual(len(needles), len(set(needles)))

    def test_empty_symbol_yields_no_needles(self):
        self.assertEqual(build_relevance_needles(""), [])


class FreshCatalystFilterTests(unittest.TestCase):
    def setUp(self):
        self.needles = build_relevance_needles("NVDA", "NVIDIA Corporation")

    def test_keeps_only_ticker_relevant_items(self):
        items = [
            news("Nvidia slips as China weighs export limits", summary="NVDA"),
            news("I Keep Adding to This Pipeline Stock"),
            news("A Financial Stock Can Be a Great Business"),
        ]
        fresh = find_fresh_catalysts(items, self.needles, today=TODAY)
        self.assertEqual(len(fresh), 1)
        self.assertIn("Nvidia", fresh[0]["title"])

    def test_rejects_substring_false_positives(self):
        fresh = find_fresh_catalysts([news("GENERAL Motors rallies")], ["ge"], today=TODAY)
        self.assertEqual(fresh, [])

    def test_rejects_stale(self):
        self.assertEqual(find_fresh_catalysts([news("NVDA news", days_ago=18)], self.needles, today=TODAY), [])

    def test_accepts_boundary_age(self):
        fresh = find_fresh_catalysts([news("NVDA news", days_ago=3)], self.needles, today=TODAY)
        self.assertEqual(len(fresh), 1)

    def test_rejects_undated(self):
        self.assertEqual(find_fresh_catalysts([{"title": "NVDA", "summary": ""}], self.needles, today=TODAY), [])

    def test_rejects_future_dated(self):
        self.assertEqual(find_fresh_catalysts([news("NVDA news", days_ago=-1)], self.needles, today=TODAY), [])

    def test_no_needles_never_matches(self):
        self.assertEqual(find_fresh_catalysts([news("NVDA news")], [], today=TODAY), [])

    def test_sorted_newest_first(self):
        items = [news("NVDA old", days_ago=2), news("NVDA new", days_ago=0)]
        self.assertEqual([i["age_days"] for i in find_fresh_catalysts(items, self.needles, today=TODAY)], [0, 2])

    def test_skips_non_dict_entries(self):
        self.assertEqual(find_fresh_catalysts(["garbage", None], self.needles, today=TODAY), [])


class DipBuyCertificateTests(unittest.TestCase):
    def setUp(self):
        self.needles = build_relevance_needles("NVDA", "NVIDIA Corporation")
        self.news = [news("Nvidia slips as China weighs export limits", summary="NVDA")]

    def qualify(self, technical, scores=None):
        return evaluate_catalyst_setup(technical, scores or strong_pillars(), self.news, self.needles, today=TODAY)

    def test_certifies_bad_news_with_strong_fundamentals(self):
        result = self.qualify(dip_technical())
        self.assertTrue(result["qualified"])
        self.assertEqual(result["direction"], DIP_BUY)

    def test_rejects_falling_knife(self):
        result = self.qualify(dip_technical(rsi14=24.0))
        self.assertFalse(result["qualified"])
        self.assertEqual(result["direction"], NO_CATALYST)
        self.assertTrue(any("falling knife" in r for r in result["reasons"]))

    def test_rejects_collapse(self):
        result = self.qualify(dip_technical(change_5d_pct=-22.0))
        self.assertFalse(result["qualified"])
        self.assertTrue(any("collapse" in r for r in result["reasons"]))

    def test_rejects_shallow_dip(self):
        result = self.qualify(dip_technical(change_5d_pct=-1.2))
        self.assertFalse(result["qualified"])

    def test_rejects_value_trap(self):
        weak = {
            "composite_quantitative_score": 38.0,
            "valuation_history_score": 41.0,
            "peer_valuation_score": 35.0,
        }
        result = self.qualify(dip_technical(), weak)
        self.assertFalse(result["qualified"])
        self.assertTrue(any("valuation history pillar" in r for r in result["reasons"]))

    def test_rejects_missing_volume(self):
        result = self.qualify(dip_technical(rvol_20d=0.8))
        self.assertFalse(result["qualified"])
        self.assertTrue(any("RVOL" in r for r in result["reasons"]))

    def test_rejects_low_composite(self):
        scores = dict(strong_pillars(), composite_quantitative_score=30.0)
        result = self.qualify(dip_technical(), scores)
        self.assertFalse(result["qualified"])

    def test_requires_fresh_news(self):
        result = evaluate_catalyst_setup(
            dip_technical(), strong_pillars(), [news("NVDA", days_ago=30)], self.needles, today=TODAY
        )
        self.assertFalse(result["qualified"])
        self.assertTrue(any("no symbol-relevant catalyst" in r for r in result["reasons"]))

    def test_works_without_company_name(self):
        result = evaluate_catalyst_setup(
            dip_technical(), strong_pillars(), self.news, build_relevance_needles("NVDA"), today=TODAY
        )
        self.assertTrue(result["qualified"])


class ExtensionSellCertificateTests(unittest.TestCase):
    def setUp(self):
        self.needles = build_relevance_needles("NVDA", "NVIDIA Corporation")
        self.news = [news("Nvidia raises datacenter guidance", summary="NVDA")]

    def qualify(self, technical, scores=None):
        return evaluate_catalyst_setup(technical, scores or rich_pillars(), self.news, self.needles, today=TODAY)

    def test_certifies_runup_into_rich_valuation(self):
        result = self.qualify(extension_technical())
        self.assertTrue(result["qualified"])
        self.assertEqual(result["direction"], EXTENSION_SELL)

    def test_rejects_runup_into_cheap_valuation(self):
        cheap = {
            "composite_quantitative_score": 61.0,
            "valuation_history_score": 68.0,
            "peer_valuation_score": 72.0,
        }
        result = self.qualify(extension_technical(), cheap)
        self.assertFalse(result["qualified"])
        self.assertTrue(any("not rich enough" in r for r in result["reasons"]))

    def test_rejects_not_overextended(self):
        result = self.qualify(extension_technical(rsi14=58.0))
        self.assertFalse(result["qualified"])

    def test_rejects_price_not_near_high(self):
        result = self.qualify(extension_technical(current_price=150.0))
        self.assertFalse(result["qualified"])
        self.assertTrue(any("20-day range" in r for r in result["reasons"]))

    def test_rejects_missing_range_data(self):
        technical = extension_technical()
        technical.pop("high_20d")
        self.assertFalse(self.qualify(technical)["qualified"])


class DirectionExclusivityTests(unittest.TestCase):
    def test_dip_setup_is_not_extension(self):
        needles = build_relevance_needles("NVDA")
        result = evaluate_catalyst_setup(
            dip_technical(), strong_pillars(), [news("NVDA", summary="")], needles, today=TODAY
        )
        self.assertEqual(result["direction"], DIP_BUY)
        self.assertNotEqual(result["direction"], EXTENSION_SELL)

    def test_extension_setup_is_not_dip(self):
        needles = build_relevance_needles("NVDA")
        result = evaluate_catalyst_setup(
            extension_technical(), rich_pillars(), [news("NVDA", summary="")], needles, today=TODAY
        )
        self.assertEqual(result["direction"], EXTENSION_SELL)

    def test_ambiguous_market_qualifies_nothing(self):
        needles = build_relevance_needles("NVDA")
        flat = {
            "current_price": 100.0,
            "ema50": 100.0,
            "high_20d": 110.0,
            "low_20d": 95.0,
            "change_5d_pct": 0.5,
            "rsi14": 52.0,
            "rvol_20d": 1.0,
        }
        result = evaluate_catalyst_setup(
            flat, strong_pillars(), [news("NVDA", summary="")], needles, today=TODAY
        )
        self.assertFalse(result["qualified"])
        self.assertEqual(result["direction"], NO_CATALYST)


class MalformedInputTests(unittest.TestCase):
    def test_non_dict_inputs_do_not_raise(self):
        result = evaluate_catalyst_setup(None, None, None, [], today=TODAY)
        self.assertFalse(result["qualified"])
        self.assertEqual(result["direction"], NO_CATALYST)

    def test_missing_metrics_reported_as_none(self):
        result = evaluate_catalyst_setup({}, {}, [], ["nvda"], today=TODAY)
        self.assertIsNone(result["metrics"]["rsi14"])
        self.assertIsNone(result["metrics"]["range_proximity"])
        self.assertFalse(result["metrics"]["above_ema50"])

    def test_flat_range_gives_no_proximity(self):
        flat = {"current_price": 100.0, "high_20d": 100.0, "low_20d": 100.0}
        self.assertIsNone(evaluate_catalyst_setup(flat, {}, [], ["nvda"], today=TODAY)["metrics"]["range_proximity"])


class ConfidencePolicyTests(unittest.TestCase):
    def test_ceiling_leaves_headroom(self):
        self.assertGreater(CATALYST_CONFIDENCE_CEILING, 0.5)
        self.assertLessEqual(CATALYST_CONFIDENCE_CEILING, 1.0)


if __name__ == "__main__":
    unittest.main()
