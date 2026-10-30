#!/usr/bin/env python3
"""
Unit test suite for Step 3: Distinct Pullback Candidate Setup Rule.

Verifies:
1. A normal trend pullback qualifies without requiring any news catalyst.
2. The setup evidence records exactly which of the 5 condition areas passed/failed:
   - trend_intact
   - pullback_area_tested
   - stabilization_confirmed
   - volume_and_relative_strength
   - exclusions_passed
3. Threshold violations in each condition area are caught and documented with exact reasons.
4. Clean economic separation from the catalyst dip path (no copied thresholds).
5. Full pipeline integration via `normalize_master_trader_json`.
"""

import unittest
from datetime import datetime, timezone
from main import normalize_master_trader_json
from quant.entry_setups import evaluate_pullback_setup, PullbackSetupResult
from quant.policy_config import get_default_policy, TradingPolicy, PullbackConfig
from catalyst_service import evaluate_catalyst_setup, build_relevance_needles


class TestEntrySetups(unittest.TestCase):
    def setUp(self):
        self.policy = get_default_policy()
        self.symbol = "AAPL"

        # Baseline textbook trend pullback setup
        self.base_pullback_tech = {
            "current_price": 150.0,
            "ema20": 149.0,
            "ema50": 144.0,
            "ema200": 135.0,
            "atr": 3.0,
            "change_5d_pct": -3.2,
            "rsi14": 47.0,
            "rvol_20d": 1.05,
            "high_20d": 156.0,
            "low_20d": 146.0,
            "suggested_stop_loss": 144.0,
            "suggested_target_price": 162.0,
            "reward_risk_ratio": 2.0,
            "avg_dollar_vol_20d": 80_000_000.0,
            "days_to_earnings": 45,
            "hv_20d": 1.2,
        }

        self.base_pullback_scores = {
            "composite_quantitative_score": 70.0,
            "trend_score": 75.0,
            "sector_relative_score": 68.0,
            "market_alpha_score": 72.0,
            "valuation_history_score": 65.0,
            "peer_valuation_score": 62.0,
        }

    def test_normal_trend_pullback_qualifies_without_news(self):
        """
        A normal trend pullback qualifies purely on quantitative and technical merits,
        without requiring any news catalyst.
        """
        res = evaluate_pullback_setup(
            technical_data=self.base_pullback_tech,
            deterministic_scores=self.base_pullback_scores,
            policy=self.policy,
        )

        self.assertTrue(res.qualified)
        self.assertEqual(res.candidate_action, "BUY")
        self.assertEqual(res.candidate_type, "PULLBACK_BUY")

        # Must record exactly which conditions it met
        cond = res.conditions_met
        self.assertTrue(cond["trend_intact"])
        self.assertTrue(cond["pullback_area_tested"])
        self.assertTrue(cond["stabilization_confirmed"])
        self.assertTrue(cond["volume_and_relative_strength"])
        self.assertTrue(cond["exclusions_passed"])

        # Evidence must contain complete audit details
        ev = res.evidence
        self.assertEqual(ev["trigger"], "pullback_in_uptrend")
        self.assertEqual(ev["setup_type"], "PULLBACK_BUY")
        self.assertEqual(ev["composite_score"], 70.0)
        self.assertEqual(ev["rsi14"], 47.0)
        self.assertEqual(ev["conditions_met"], cond)
        self.assertIn("metrics", ev)
        self.assertIn("reasons", ev)

    def test_trend_intact_condition_failures(self):
        """
        Verifies rejection when longer-term trend criteria fail:
        - composite score < 60.0
        - trend pillar < 60.0
        - price < ema50
        - ema50 < ema200
        """
        # 1. Composite below minimum 60.0
        scores_low_comp = dict(self.base_pullback_scores, composite_quantitative_score=57.0)
        res = evaluate_pullback_setup(self.base_pullback_tech, scores_low_comp, policy=self.policy)
        self.assertFalse(res.qualified)
        self.assertFalse(res.conditions_met["trend_intact"])
        self.assertTrue(any("Composite score" in r for r in res.reasons))

        # 2. Trend pillar score below 60.0
        scores_low_trend = dict(self.base_pullback_scores, trend_score=52.0)
        res = evaluate_pullback_setup(self.base_pullback_tech, scores_low_trend, policy=self.policy)
        self.assertFalse(res.qualified)
        self.assertFalse(res.conditions_met["trend_intact"])
        self.assertTrue(any("Trend pillar score" in r for r in res.reasons))

        # 3. Price below EMA50 (intermediate trend broken)
        tech_below_ema50 = dict(self.base_pullback_tech, current_price=142.0, ema50=144.0)
        res = evaluate_pullback_setup(tech_below_ema50, self.base_pullback_scores, policy=self.policy)
        self.assertFalse(res.qualified)
        self.assertFalse(res.conditions_met["trend_intact"])
        self.assertTrue(any("EMA50" in r for r in res.reasons))

        # 4. EMA50 below EMA200 (death cross / structural downtrend)
        tech_death_cross = dict(self.base_pullback_tech, ema50=130.0, ema200=138.0)
        res = evaluate_pullback_setup(tech_death_cross, self.base_pullback_scores, policy=self.policy)
        self.assertFalse(res.qualified)
        self.assertFalse(res.conditions_met["trend_intact"])
        self.assertTrue(any("EMA200" in r for r in res.reasons))

    def test_pullback_area_definition_failures(self):
        """
        Verifies rejection when price level / pullback area geometry fails:
        - drop < 1.5% (too shallow, hasn't pulled back)
        - drop > 8.0% (too deep, excessive breakdown)
        - distance to anchor EMA > 1.5 ATR
        - range proximity < 0.20 or > 0.75
        """
        # 1. Drop too shallow (-0.8% < 1.5%)
        tech_shallow = dict(self.base_pullback_tech, change_5d_pct=-0.8)
        res = evaluate_pullback_setup(tech_shallow, self.base_pullback_scores, policy=self.policy)
        self.assertFalse(res.qualified)
        self.assertFalse(res.conditions_met["pullback_area_tested"])
        self.assertTrue(any("below minimum" in r for r in res.reasons))

        # 2. Drop too deep (-10.5% > 8.0%)
        tech_deep = dict(self.base_pullback_tech, change_5d_pct=-10.5)
        res = evaluate_pullback_setup(tech_deep, self.base_pullback_scores, policy=self.policy)
        self.assertFalse(res.qualified)
        self.assertFalse(res.conditions_met["pullback_area_tested"])
        self.assertTrue(any("exceeds maximum" in r for r in res.reasons))

        # 3. Distance to anchor EMA > 1.5 ATR (|150 - 142| / 3 = 2.67 ATR)
        tech_far_ema = dict(self.base_pullback_tech, ema20=142.0, atr=3.0)
        res = evaluate_pullback_setup(tech_far_ema, self.base_pullback_scores, policy=self.policy)
        self.assertFalse(res.qualified)
        self.assertFalse(res.conditions_met["pullback_area_tested"])
        self.assertTrue(any("Distance to ema20" in r for r in res.reasons))

        # 4. Range proximity < 0.20 (trading near 20d lows)
        tech_low_prox = dict(self.base_pullback_tech, low_20d=149.0, high_20d=160.0, current_price=150.0) # prox = 0.09
        res = evaluate_pullback_setup(tech_low_prox, self.base_pullback_scores, policy=self.policy)
        self.assertFalse(res.qualified)
        self.assertFalse(res.conditions_met["pullback_area_tested"])
        self.assertTrue(any("Range proximity" in r for r in res.reasons))

        # 5. Range proximity > 0.75 (trading near 20d highs, not pulling back)
        tech_high_prox = dict(self.base_pullback_tech, low_20d=140.0, high_20d=152.0, current_price=151.0) # prox = 0.916
        res = evaluate_pullback_setup(tech_high_prox, self.base_pullback_scores, policy=self.policy)
        self.assertFalse(res.qualified)
        self.assertFalse(res.conditions_met["pullback_area_tested"])
        self.assertTrue(any("Range proximity" in r for r in res.reasons))

    def test_stabilization_reclaim_confirmation_failures(self):
        """
        Verifies rejection when stabilization / reclaim criteria fail:
        - RSI < 40.0 (momentum breakdown)
        - RSI > 55.0 (overextended / not in cooling band)
        - Price fails stabilization below anchor floor (anchor - 0.5 ATR)
        - Reward/Risk < 1.20
        """
        # 1. RSI below 40.0 (e.g. 36.0)
        tech_low_rsi = dict(self.base_pullback_tech, rsi14=36.0)
        res = evaluate_pullback_setup(tech_low_rsi, self.base_pullback_scores, policy=self.policy)
        self.assertFalse(res.qualified)
        self.assertFalse(res.conditions_met["stabilization_confirmed"])
        self.assertTrue(any("RSI 36.0 below minimum" in r for r in res.reasons))

        # 2. RSI above 55.0 (e.g. 64.0)
        tech_high_rsi = dict(self.base_pullback_tech, rsi14=64.0)
        res = evaluate_pullback_setup(tech_high_rsi, self.base_pullback_scores, policy=self.policy)
        self.assertFalse(res.qualified)
        self.assertFalse(res.conditions_met["stabilization_confirmed"])
        self.assertTrue(any("RSI 64.0 exceeds maximum" in r for r in res.reasons))

        # 3. Price crashing through anchor floor (anchor 149.0, atr 3.0 -> floor 147.5, price 146.0)
        tech_failed_floor = dict(self.base_pullback_tech, current_price=146.0, ema20=149.0, atr=3.0)
        res = evaluate_pullback_setup(tech_failed_floor, self.base_pullback_scores, policy=self.policy)
        self.assertFalse(res.qualified)
        self.assertFalse(res.conditions_met["stabilization_confirmed"])
        self.assertTrue(any("failed stabilization" in r for r in res.reasons))

        # 4. Reward/risk ratio below 1.20 (e.g. 0.90)
        tech_low_rr = dict(self.base_pullback_tech, reward_risk_ratio=0.90)
        res = evaluate_pullback_setup(tech_low_rr, self.base_pullback_scores, policy=self.policy)
        self.assertFalse(res.qualified)
        self.assertFalse(res.conditions_met["stabilization_confirmed"])
        self.assertTrue(any("Reward/risk ratio" in r for r in res.reasons))

    def test_volume_and_relative_strength_failures(self):
        """
        Verifies rejection when volume or relative strength conditions fail:
        - RVOL > 1.50 (heavy distribution volume)
        - sector relative score < 50.0
        - market alpha score < 50.0
        """
        # 1. RVOL exceeds maximum 1.50 (e.g. 2.10 - abnormal distribution selling)
        tech_high_rvol = dict(self.base_pullback_tech, rvol_20d=2.10)
        res = evaluate_pullback_setup(tech_high_rvol, self.base_pullback_scores, policy=self.policy)
        self.assertFalse(res.qualified)
        self.assertFalse(res.conditions_met["volume_and_relative_strength"])
        self.assertTrue(any("Pullback RVOL 2.10 exceeds" in r for r in res.reasons))

        # 2. Sector score below 50.0
        scores_weak_sector = dict(self.base_pullback_scores, sector_relative_score=44.0)
        res = evaluate_pullback_setup(self.base_pullback_tech, scores_weak_sector, policy=self.policy)
        self.assertFalse(res.qualified)
        self.assertFalse(res.conditions_met["volume_and_relative_strength"])
        self.assertTrue(any("Sector relative score" in r for r in res.reasons))

        # 3. Market alpha score below 50.0
        scores_weak_alpha = dict(self.base_pullback_scores, market_alpha_score=42.0)
        res = evaluate_pullback_setup(self.base_pullback_tech, scores_weak_alpha, policy=self.policy)
        self.assertFalse(res.qualified)
        self.assertFalse(res.conditions_met["volume_and_relative_strength"])
        self.assertTrue(any("Market alpha score" in r for r in res.reasons))

    def test_exclusions_failures(self):
        """
        Verifies rejection when exclusions trigger:
        - Extension above anchor > 2.0 ATR
        - Earnings blackout active (days_to_earnings <= 3)
        - Extreme volatility factor > 2.50
        - Security flagged as deterministic value trap
        """
        # 1. Upward extension > 2.0 ATR (price 158, anchor 149, atr 3 -> extension = 3.0 ATR)
        tech_extended = dict(self.base_pullback_tech, current_price=158.0, ema20=149.0, atr=3.0)
        res = evaluate_pullback_setup(tech_extended, self.base_pullback_scores, policy=self.policy)
        self.assertFalse(res.qualified)
        self.assertFalse(res.conditions_met["exclusions_passed"])
        self.assertTrue(any("Extension above ema20" in r for r in res.reasons))

        # 2. Scheduled earnings in blackout window (2 days)
        tech_earnings = dict(self.base_pullback_tech, days_to_earnings=2)
        res = evaluate_pullback_setup(tech_earnings, self.base_pullback_scores, policy=self.policy)
        self.assertFalse(res.qualified)
        self.assertFalse(res.conditions_met["exclusions_passed"])
        self.assertTrue(any("Scheduled earnings in 2 day(s)" in r for r in res.reasons))

        # 3. Extreme volatility factor > 2.50 (e.g. 3.20)
        tech_high_vol = dict(self.base_pullback_tech, hv_20d=3.20)
        res = evaluate_pullback_setup(tech_high_vol, self.base_pullback_scores, policy=self.policy)
        self.assertFalse(res.qualified)
        self.assertFalse(res.conditions_met["exclusions_passed"])
        self.assertTrue(any("Volatility factor" in r for r in res.reasons))

        # 4. Security flagged as value trap
        res = evaluate_pullback_setup(
            self.base_pullback_tech,
            self.base_pullback_scores,
            policy=self.policy,
            is_value_trap=True,
        )
        self.assertFalse(res.qualified)
        self.assertFalse(res.conditions_met["exclusions_passed"])
        self.assertTrue(any("value trap" in r for r in res.reasons))

    def test_distinctness_from_catalyst_dip_setup(self):
        """
        Proves clean separation of economic setups and calibrated thresholds:
        - Catalyst Dip Buy: severe adverse headline, drop -8%, RSI 34, RVOL 1.8.
          Qualifies for CATALYST_DIP_BUY; fails PULLBACK_BUY (RSI too low, RVOL too high).
        - Trend Pullback: orderly quiet consolidation, drop -3.2%, RSI 47, RVOL 1.05, no news.
          Qualifies for PULLBACK_BUY; fails CATALYST_DIP_BUY (no news, drop not severe enough, RSI not oversold).
        """
        # Case A: Adverse News Dislocation Setup
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        news = [{"title": "AAPL supplier halts shipments", "published_at": now_str}]
        cat_tech = dict(
            self.base_pullback_tech,
            change_5d_pct=-8.0,
            rsi14=34.0,       # Oversold
            rvol_20d=1.8,      # Capitulation absorption
        )
        cat_scores = dict(
            self.base_pullback_scores,
            composite_quantitative_score=55.0, # Below standard buy floor 60.0, but clears dip floor 52.0
            valuation_history_score=68.0,
            peer_valuation_score=60.0,
        )

        cat_eval = evaluate_catalyst_setup(
            cat_tech, cat_scores, news, build_relevance_needles("AAPL")
        )
        self.assertTrue(cat_eval["qualified"])
        self.assertEqual(cat_eval["direction"], "DIP_BUY")

        # The same setup must NOT qualify as a normal trend pullback
        pb_eval = evaluate_pullback_setup(cat_tech, cat_scores, policy=self.policy)
        self.assertFalse(pb_eval.qualified)
        self.assertFalse(pb_eval.conditions_met["trend_intact"]) # Score 55.0 < 60.0
        self.assertFalse(pb_eval.conditions_met["stabilization_confirmed"]) # RSI 34 < 40
        self.assertFalse(pb_eval.conditions_met["volume_and_relative_strength"]) # RVOL 1.8 > 1.5

        # Case B: Quiet Trend Pullback Setup without any news
        clean_pb_tech = dict(self.base_pullback_tech)
        clean_pb_scores = dict(self.base_pullback_scores)

        pb_eval2 = evaluate_pullback_setup(clean_pb_tech, clean_pb_scores, policy=self.policy)
        self.assertTrue(pb_eval2.qualified)
        self.assertEqual(pb_eval2.candidate_type, "PULLBACK_BUY")

        # The trend pullback must NOT qualify as a catalyst dip buy
        cat_eval2 = evaluate_catalyst_setup(
            clean_pb_tech, clean_pb_scores, [], build_relevance_needles("AAPL")
        )
        self.assertFalse(cat_eval2["qualified"])
        self.assertEqual(cat_eval2["direction"], "NONE")

    def test_end_to_end_integration_in_main_normalizer(self):
        """
        Verifies end-to-end signal normalization in main.py:
        When a stock exhibits an intact trend pullback, it qualifies as PULLBACK_BUY
        even when LLM proposes HOLD, and records all conditions met in the signal output.
        """
        raw_output = {
            "stock": self.symbol,
            "decision": "HOLD",                  # Model suggests HOLD
            "primary_driver": "TECHNICAL",       # No news catalyst
            "confidence": 0.50,
            "buy_score": 0.30,
            "hold_score": 0.60,
            "sell_score": 0.10,
            "data_completeness": 0.95,
        }

        m_data = dict(self.base_pullback_tech)
        m_data["deterministic_5pillar_scores"] = self.base_pullback_scores

        res = normalize_master_trader_json(
            raw_output,
            self.symbol,
            m_data,
            {},
            {"news": []}, # Completely empty news feed
        )

        self.assertEqual(res["decision"], "BUY")
        self.assertEqual(res["final_decision"], "BUY")
        self.assertEqual(res["model_decision"], "HOLD")
        self.assertEqual(res["candidate_action"], "BUY")
        self.assertEqual(res["candidate_type"], "PULLBACK_BUY")

        # Signal records exact conditions met
        ev = res["setup_evidence"]
        self.assertIn("conditions_met", ev)
        cond = ev["conditions_met"]
        self.assertTrue(cond["trend_intact"])
        self.assertTrue(cond["pullback_area_tested"])
        self.assertTrue(cond["stabilization_confirmed"])
        self.assertTrue(cond["volume_and_relative_strength"])
        self.assertTrue(cond["exclusions_passed"])


if __name__ == "__main__":
    unittest.main()
