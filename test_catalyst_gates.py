import datetime
import unittest

from main import normalize_master_trader_json

TODAY = datetime.date.today()


def news(title, days_ago=0, summary=""):
    published = TODAY - datetime.timedelta(days=days_ago)
    return {
        "title": title,
        "summary": summary,
        "published_at": published.isoformat(),
        "category": "News",
    }


def make_m_data(**overrides):
    """A dip-shaped m_data that clears every non-catalyst BUY gate.

    Price sits just above 20d support so the channel-anchored reward:risk clears 1.5:
    stop = min(99, 99 - 0.5*atr) and target = min(price + 2.5*atr, high + 0.5*atr).
    """
    values = {
        "current_price": 100.0,
        "atr": 3.0,
        "suggested_stop_loss": 98.0,
        "suggested_target_price": 108.0,
        "rsi14": 38.0,
        "rvol_20d": 1.9,
        "ema20": 102.0,
        "ema50": 101.0,
        "change_5d_pct": -7.4,
        "high_20d": 110.0,
        "low_20d": 99.0,
        "avg_dollar_vol_20d": 50_000_000.0,
        "deterministic_5pillar_scores": {
            "composite_quantitative_score": 52.0,
            "valuation_history_score": 71.0,
            "peer_valuation_score": 66.0,
            "trend": 30.0,
            "sector": 60.0,
            "alpha": 45.0,
        },
    }
    values.update(overrides)
    return values


def make_data(**overrides):
    values = {
        "stock": "NVDA",
        "decision": "BUY",
        "primary_driver": "NEWS_CATALYST",
        "confidence": 0.7,
        "buy_score": 0.6,
        "hold_score": 0.25,
        "sell_score": 0.15,
        "data_completeness": 0.95,
        "horizon_days": 10,
        "quant_score": 52,
        "pillar_scores": {
            "trend": 30.0,
            "sector": 60.0,
            "alpha": 45.0,
            "valuation_history": 71.0,
            "peer_valuation": 68.0,
        },
        "bull_case": ["valuation intact"],
        "bear_case": ["weak trend"],
        "key_risks": [],
        "missing_information": [],
        "falsification_bull": ["n/a"],
        "falsification_bear": ["n/a"],
    }
    values.update(overrides)
    return values


GOOD_CATALYST = [news("Nvidia slips as China weighs export limits", summary="NVDA guidance")]


def run(data=None, m_data=None, news_items=None, symbol="NVDA"):
    return normalize_master_trader_json(
        data if data is not None else make_data(),
        symbol,
        m_data if m_data is not None else make_m_data(),
        {},
        {"news": GOOD_CATALYST if news_items is None else news_items},
    )


class VetoStillFiresTests(unittest.TestCase):
    """The overlay must not have disabled the existing news veto."""

    def test_no_news_holds(self):
        out = run(news_items=[])
        self.assertEqual(out["decision"], "HOLD")
        self.assertEqual(out["no_trade_reason"], "INSUFFICIENT_EVIDENCE")

    def test_irrelevant_news_holds(self):
        out = run(news_items=[news("I Keep Adding to This Pipeline Stock")])
        self.assertEqual(out["decision"], "HOLD")

    def test_stale_news_holds(self):
        out = run(news_items=[news("NVDA export limits", days_ago=20, summary="NVDA")])
        self.assertEqual(out["decision"], "HOLD")

    def test_undated_news_holds(self):
        out = run(news_items=[{"title": "NVDA export limits", "summary": "NVDA"}])
        self.assertEqual(out["decision"], "HOLD")

    def test_macro_event_never_unlocks(self):
        out = run(make_data(primary_driver="MACRO_EVENT"))
        self.assertEqual(out["decision"], "HOLD")
        self.assertEqual(out["no_trade_reason"], "INSUFFICIENT_EVIDENCE")

    def test_falling_knife_holds(self):
        out = run(m_data=make_m_data(rsi14=24.0))
        self.assertEqual(out["decision"], "HOLD")

    def test_value_trap_holds(self):
        scores = make_m_data()["deterministic_5pillar_scores"]
        scores.update({"valuation_history_score": 40.0, "peer_valuation_score": 35.0})
        out = run(m_data=make_m_data(deterministic_5pillar_scores=scores))
        self.assertEqual(out["decision"], "HOLD")

    def test_collapse_holds(self):
        out = run(m_data=make_m_data(change_5d_pct=-24.0))
        self.assertEqual(out["decision"], "HOLD")


class DirectionMatchingTests(unittest.TestCase):
    def test_dip_certificate_does_not_unlock_a_sell(self):
        out = run(make_data(decision="SELL", buy_score=0.2, hold_score=0.2, sell_score=0.6))
        self.assertEqual(out["decision"], "HOLD")

    def test_extension_certificate_unlocks_a_sell(self):
        scores = {
            "composite_quantitative_score": 61.0,
            "valuation_history_score": 34.0,
            "peer_valuation_score": 29.0,
            "trend": 80.0,
            "sector": 60.0,
            "alpha": 70.0,
        }
        m_data = make_m_data(
            current_price=110.0,
            ema20=100.0,
            ema50=95.0,
            change_5d_pct=11.2,
            rsi14=78.0,
            rvol_20d=1.6,
            high_20d=111.0,
            low_20d=80.0,
            atr=3.0,
            suggested_stop_loss=95.0,
            suggested_target_price=130.0,
            deterministic_5pillar_scores=scores,
        )
        out = run(make_data(decision="SELL", buy_score=0.2, hold_score=0.2, sell_score=0.6), m_data)
        self.assertEqual(out["decision"], "SELL")


class CatalystUnlocksTests(unittest.TestCase):
    def test_dip_buy_survives_the_veto(self):
        out = run()
        self.assertEqual(out["decision"], "BUY")
        self.assertTrue(
            any("Catalyst certificate 'DIP_BUY' unlocked" in r for r in out["key_risks"]),
            "BUY must be attributable to the certificate unlocking the veto",
        )

    def test_unlocked_buy_still_reports_gate_context(self):
        out = run()
        self.assertFalse(
            any("capped to HOLD" in r for r in out["key_risks"]),
            "the news veto must not also have fired on an unlocked path",
        )

    def test_catalyst_floor_is_still_enforced(self):
        scores = make_m_data()["deterministic_5pillar_scores"]
        scores["composite_quantitative_score"] = 30.0
        out = run(m_data=make_m_data(deterministic_5pillar_scores=scores))
        self.assertEqual(out["decision"], "HOLD")

    def test_earnings_blackout_still_blocks(self):
        m_data = make_m_data()
        m_data["days_to_earnings"] = 1
        out = run(m_data=m_data)
        self.assertEqual(out["decision"], "HOLD")

    def test_liquidity_still_blocks(self):
        out = run(m_data=make_m_data(avg_dollar_vol_20d=1000.0))
        self.assertEqual(out["decision"], "HOLD")

    def test_poor_reward_risk_still_blocks(self):
        m_data = make_m_data()
        m_data["suggested_stop_loss"] = 80.0
        m_data["high_20d"] = 101.0
        out = run(m_data=m_data)
        self.assertEqual(out["decision"], "HOLD")

    def test_confidence_is_capped(self):
        out = run()
        self.assertLessEqual(out["confidence"], 0.85)


class AdditiveOnlyTests(unittest.TestCase):
    """Proof that non-catalyst paths are untouched by the overlay."""

    def test_quant_structure_buy_unaffected_by_news_presence(self):
        data = make_data(primary_driver="QUANT_STRUCTURE")
        strong = make_m_data(
            change_5d_pct=1.0,
            rsi14=55.0,
            low_20d=99.0,
            deterministic_5pillar_scores={
                "composite_quantitative_score": 78.0,
                "valuation_history_score": 70.0,
                "peer_valuation_score": 68.0,
                "trend": 75.0,
                "sector": 72.0,
                "alpha": 70.0,
            },
        )
        with_news = run(data, strong, GOOD_CATALYST)
        without_news = run(data, strong, [])
        self.assertEqual(with_news["decision"], "BUY")
        self.assertEqual(with_news["decision"], without_news["decision"])
        self.assertEqual(with_news["confidence"], without_news["confidence"])
        self.assertEqual(with_news["no_trade_reason"], without_news["no_trade_reason"])

    def test_normal_buy_composite_floor_still_70(self):
        data = make_data(primary_driver="QUANT_STRUCTURE")
        medium = make_m_data(
            change_5d_pct=1.0,
            rsi14=55.0,
            low_20d=99.0,
            deterministic_5pillar_scores={
                "composite_quantitative_score": 55.0,
                "valuation_history_score": 70.0,
                "peer_valuation_score": 68.0,
                "trend": 55.0,
                "sector": 60.0,
                "alpha": 55.0,
            },
        )
        out = run(data, medium, [])
        self.assertEqual(out["decision"], "HOLD")
        self.assertEqual(out["no_trade_reason"], "INSUFFICIENT_EVIDENCE")

    def test_hold_is_untouched(self):
        out = run(
            make_data(
                decision="HOLD",
                primary_driver="QUANT_STRUCTURE",
                buy_score=0.2,
                hold_score=0.6,
                sell_score=0.2,
            )
        )
        self.assertEqual(out["decision"], "HOLD")

    def test_missing_inputs_do_not_raise(self):
        out = normalize_master_trader_json({}, "NVDA", {}, {}, {})
        self.assertEqual(out["decision"], "HOLD")

    def test_no_news_key_present(self):
        out = normalize_master_trader_json(make_data(), "NVDA", make_m_data(), {}, {})
        self.assertEqual(out["decision"], "HOLD")


if __name__ == "__main__":
    unittest.main()
