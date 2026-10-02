import datetime
import unittest
from unittest.mock import MagicMock, patch

from main import normalize_master_trader_json
from quantitative_scoring_service import QuantitativeScoringService
from gloomberb_service import GloomberbService


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
    values = {
        "current_price": 100.0,
        "atr": 3.0,
        "suggested_stop_loss": 95.0,  # stop distance = 5.0
        "suggested_target_price": 110.0,
        "rsi14": 42.0,
        "rvol_20d": 1.5,
        "ema20": 101.0,
        "ema50": 100.0,
        "change_5d_pct": -6.5,
        "high_20d": 112.0,
        "low_20d": 96.0,
        "avg_dollar_vol_20d": 50_000_000.0,
        "deterministic_5pillar_scores": {
            "composite_quantitative_score": 75.0,
            "valuation_history_score": 72.0,
            "peer_valuation_score": 70.0,
            "trend": 70.0,
            "sector": 70.0,
            "alpha": 68.0,
        },
    }
    values.update(overrides)
    return values


def make_data(**overrides):
    values = {
        "stock": "NVDA",
        "decision": "BUY",
        "primary_driver": "QUANT_STRUCTURE",
        "confidence": 0.8,
        "buy_score": 0.75,
        "hold_score": 0.15,
        "sell_score": 0.10,
        "data_completeness": 0.95,
        "horizon_days": 10,
        "quant_score": 75,
        "pillar_scores": {
            "trend": 70.0,
            "sector": 70.0,
            "alpha": 68.0,
            "valuation_history": 72.0,
            "peer_valuation": 70.0,
        },
        "bull_case": ["strong quant structure"],
        "bear_case": ["macro sensitivity"],
        "key_risks": [],
        "missing_information": [],
        "falsification_bull": ["support fails"],
        "falsification_bear": ["breakout confirms"],
    }
    values.update(overrides)
    return values


GOOD_CATALYST = [news("Nvidia slips on export quota headlines", summary="NVDA guidance")]


def run(data=None, m_data=None, gloomberb_payload=None, symbol="NVDA"):
    return normalize_master_trader_json(
        data if data is not None else make_data(),
        symbol,
        m_data if m_data is not None else make_m_data(),
        {},
        gloomberb_payload if gloomberb_payload is not None else {"news": GOOD_CATALYST},
    )


class QuantitativeExpectedMoveTests(unittest.TestCase):
    def test_expected_move_atm_straddle(self):
        # price = 100, atm_straddle = 10.0 -> expected_move = 0.85 * 10 = 8.5
        res = QuantitativeScoringService.compute_expected_move(
            price=100.0, iv=40.0, dte=10, atm_straddle=10.0
        )
        self.assertEqual(res["expected_move"], 8.5)
        self.assertEqual(res["expected_move_pct"], 8.5)
        self.assertEqual(res["calculation_method"], "ATM_STRADDLE")

    def test_expected_move_iv_formula(self):
        # price = 100, iv = 36.5%, dte = 365 -> expected_move = 100 * 0.365 * 1 = 36.5
        res = QuantitativeScoringService.compute_expected_move(
            price=100.0, iv=36.5, dte=365
        )
        self.assertEqual(res["expected_move"], 36.5)
        self.assertEqual(res["expected_move_pct"], 36.5)
        self.assertEqual(res["calculation_method"], "IV_FORMULA")

    def test_expected_move_vs_stop_violation(self):
        # entry = 100, stop = 98 (distance = 2), expected_move = 3.5 -> violation
        eval_res = QuantitativeScoringService.evaluate_expected_move_vs_stop(
            entry_price=100.0, stop_loss=98.0, expected_move=3.5
        )
        self.assertEqual(eval_res["stop_distance"], 2.0)
        self.assertEqual(eval_res["ratio"], 1.75)
        self.assertTrue(eval_res["stop_inside_expected_move"])
        self.assertEqual(eval_res["status"], "VIOLATION")

    def test_expected_move_vs_stop_adequate(self):
        # entry = 100, stop = 92 (distance = 8), expected_move = 3.5 -> adequate
        eval_res = QuantitativeScoringService.evaluate_expected_move_vs_stop(
            entry_price=100.0, stop_loss=92.0, expected_move=3.5
        )
        self.assertEqual(eval_res["stop_distance"], 8.0)
        self.assertFalse(eval_res["stop_inside_expected_move"])
        self.assertEqual(eval_res["status"], "ADEQUATE")


class VolatilityRegimeEvaluationTests(unittest.TestCase):
    def test_extreme_iv_ceiling(self):
        res = QuantitativeScoringService.evaluate_volatility_regime(
            atr=2.0, price=100.0, implied_volatility=72.0
        )
        self.assertEqual(res["vol_regime"], "EXTREME")
        self.assertTrue(res["is_extreme"])
        self.assertTrue(any("crisis ceiling" in r for r in res["reasons"]))

    def test_extreme_iv_hv_spread(self):
        # IV = 50%, HV = 20% -> ratio = 2.5x (>2.0x is extreme tail-risk expansion)
        res = QuantitativeScoringService.evaluate_volatility_regime(
            atr=2.0, price=100.0, implied_volatility=50.0, historical_volatility=20.0
        )
        self.assertEqual(res["vol_regime"], "EXTREME")
        self.assertTrue(res["is_extreme"])
        self.assertTrue(any("severe tail-risk" in r for r in res["reasons"]))

    def test_normal_volatility(self):
        res = QuantitativeScoringService.evaluate_volatility_regime(
            atr=2.0, price=100.0, implied_volatility=28.0, historical_volatility=25.0
        )
        self.assertEqual(res["vol_regime"], "NORMAL")
        self.assertFalse(res["is_extreme"])


class ScheduledCompanyEventsEvaluationTests(unittest.TestCase):
    def test_ex_dividend_imminent_blackout(self):
        res = QuantitativeScoringService.evaluate_event_risk(
            days_to_earnings=30, days_to_next_event=1, next_event_type="EX_DIVIDEND"
        )
        self.assertTrue(res["is_blackout"])
        self.assertEqual(res["reason_code"], "EVENT_RISK")
        self.assertEqual(res["event_type"], "EX_DIVIDEND")

    def test_corporate_action_imminent_blackout(self):
        res = QuantitativeScoringService.evaluate_event_risk(
            days_to_earnings=45, days_to_next_event=2, next_event_type="STOCK_SPLIT"
        )
        self.assertTrue(res["is_blackout"])
        self.assertEqual(res["reason_code"], "EVENT_RISK")

    def test_distant_event_is_not_blackout(self):
        res = QuantitativeScoringService.evaluate_event_risk(
            days_to_earnings=45, days_to_next_event=12, next_event_type="INVESTOR_DAY"
        )
        self.assertFalse(res["is_blackout"])


class MainBuyGateEnforcementTests(unittest.TestCase):
    def test_buy_gated_when_expected_move_exceeds_stop(self):
        # price=100, stop=97 (distance=3.0), expected move=4.50
        gloomberb = {
            "options": {
                "implied_volatility": "45.0%",
                "expected_move": 4.50,
                "expected_move_pct": 4.5,
            }
        }
        m_data = make_m_data(current_price=100.0, suggested_stop_loss=97.0)
        out = run(m_data=m_data, gloomberb_payload=gloomberb)

        self.assertEqual(out["decision"], "HOLD")
        self.assertEqual(out["no_trade_reason"], "EXPECTED_MOVE_EXCEEDS_STOP")
        self.assertTrue(any("exceeds stop distance" in r for r in out["key_risks"]))

    def test_buy_gated_when_scheduled_company_event_imminent(self):
        # Ex-dividend in 1 day
        gloomberb = {
            "events": {
                "days_to_next_event": 1,
                "next_event_type": "EX_DIVIDEND",
                "next_event_date": "2026-10-03",
            }
        }
        out = run(gloomberb_payload=gloomberb)
        self.assertEqual(out["decision"], "HOLD")
        self.assertEqual(out["no_trade_reason"], "EVENT_RISK")
        self.assertTrue(any("EX_DIVIDEND" in r for r in out["key_risks"]))

    def test_buy_gated_when_extreme_iv_volatility_regime(self):
        # Extreme IV = 75%
        gloomberb = {
            "options": {
                "implied_volatility": "75.0%",
                "implied_volatility_pct": 75.0,
            }
        }
        out = run(gloomberb_payload=gloomberb)
        self.assertEqual(out["decision"], "HOLD")
        self.assertEqual(out["no_trade_reason"], "VOLATILITY_REGIME")
        self.assertTrue(any("Extreme volatility" in r for r in out["key_risks"]))

    def test_headline_catalyst_cannot_bypass_expected_move_gate(self):
        # Catalyst certificate verified (DIP_BUY) + breaking headline, BUT expected move > stop distance
        data = make_data(
            decision="BUY",
            primary_driver="NEWS_CATALYST",
            buy_score=0.8,
            hold_score=0.15,
            sell_score=0.05,
        )
        m_data = make_m_data(
            current_price=100.0,
            suggested_stop_loss=98.5,  # stop distance = 1.5
            change_5d_pct=-7.4,
            rsi14=38.0,
        )
        gloomberb = {
            "news": GOOD_CATALYST,
            "options": {
                "implied_volatility": "40.0%",
                "expected_move": 3.80,  # 3.80 > 1.50
                "expected_move_pct": 3.8,
            },
        }
        out = run(data=data, m_data=m_data, gloomberb_payload=gloomberb)
        self.assertEqual(out["decision"], "HOLD")
        self.assertEqual(out["no_trade_reason"], "EXPECTED_MOVE_EXCEEDS_STOP")
        self.assertTrue(any("exceeds stop distance" in r for r in out["key_risks"]))

    def test_headline_catalyst_cannot_bypass_event_risk_gate(self):
        # Catalyst certificate verified (DIP_BUY) + headline, BUT scheduled corporate action in 1 day
        data = make_data(
            decision="BUY",
            primary_driver="NEWS_CATALYST",
        )
        m_data = make_m_data(change_5d_pct=-7.4, rsi14=38.0)
        gloomberb = {
            "news": GOOD_CATALYST,
            "events": {
                "days_to_next_event": 1,
                "next_event_type": "SHAREHOLDER_MEETING",
            },
        }
        out = run(data=data, m_data=m_data, gloomberb_payload=gloomberb)
        self.assertEqual(out["decision"], "HOLD")
        self.assertEqual(out["no_trade_reason"], "EVENT_RISK")

    def test_buy_clears_when_expected_move_is_well_covered(self):
        # stop distance = 6.0, expected move = 2.50, no event blackout, normal IV
        gloomberb = {
            "news": GOOD_CATALYST,
            "options": {
                "implied_volatility": "25.0%",
                "expected_move": 2.50,
                "expected_move_pct": 2.5,
                "days_to_expiration": 14,
            },
            "events": {
                "days_to_next_event": 25,
            },
        }
        m_data = make_m_data(current_price=100.0, suggested_stop_loss=94.0)
        out = run(m_data=m_data, gloomberb_payload=gloomberb)
        self.assertEqual(out["decision"], "BUY")
        self.assertIsNone(out["no_trade_reason"])
        self.assertEqual(out["expected_move"], 2.50)
        self.assertEqual(out["stop_distance"], 6.0)


class GloomberbServiceOptionMoveUnitTests(unittest.TestCase):
    def test_fetch_options_chain_computes_expected_move_native(self):
        service = GloomberbService()
        mock_cli_data = {
            "underlyingPrice": 150.0,
            "expirationDates": [int(TODAY.strftime("%s")) + 86400 * 7],
            "calls": [
                {"strike": 145.0, "bid": 6.0, "ask": 6.2, "impliedVolatility": 0.32, "volume": 100},
                {"strike": 150.0, "bid": 3.0, "ask": 3.2, "impliedVolatility": 0.30, "volume": 500},
                {"strike": 155.0, "bid": 1.2, "ask": 1.4, "impliedVolatility": 0.31, "volume": 200},
            ],
            "puts": [
                {"strike": 145.0, "bid": 1.1, "ask": 1.3, "impliedVolatility": 0.32, "volume": 150},
                {"strike": 150.0, "bid": 2.9, "ask": 3.1, "impliedVolatility": 0.30, "volume": 400},
                {"strike": 155.0, "bid": 5.8, "ask": 6.0, "impliedVolatility": 0.31, "volume": 100},
            ],
        }
        with patch.object(service, "run_cli", return_value=mock_cli_data):
            res = service.fetch_options_chain("AAPL")
            self.assertIsNotNone(res["expected_move"])
            self.assertGreater(res["expected_move"], 0.0)
            self.assertEqual(res["volatility_regime"], "NORMAL")
            # ATM call mid is 3.1, ATM put mid is 3.0 -> straddle 6.1 -> expected move ~ 0.85 * 6.1 = 5.18
            self.assertEqual(res["atm_straddle"], 6.1)
            self.assertAlmostEqual(res["expected_move"], 5.18, places=1)


if __name__ == "__main__":
    unittest.main()
