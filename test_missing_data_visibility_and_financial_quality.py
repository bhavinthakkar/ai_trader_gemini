import unittest
import json
import sqlite3
from unittest.mock import MagicMock, patch

from quantitative_scoring_service import QuantitativeScoringService
from main import normalize_master_trader_json, gated_confidence
from db import init_db, save_results, get_latest_signals


class MissingDataVisibilityTests(unittest.TestCase):
    def setUp(self):
        self.service = QuantitativeScoringService()

    def test_missing_sector_benchmark_is_unavailable_and_none(self):
        """When sector benchmark is absent, sector pillar must be 'unavailable' with score = None (not 50.0)."""
        tech = {"current_price": 100.0, "ema20": 95.0, "ema50": 90.0, "rsi14": 55.0, "change_5d_pct": 2.5}
        empty_sector = {}
        res = self.service.evaluate_sector_pillar(tech, empty_sector)
        self.assertEqual(res["status"], "unavailable")
        self.assertIsNone(res["score"])
        self.assertFalse(res["inputs"]["sector_5d"]["available"])

    def test_missing_peer_valuation_is_unavailable_and_none(self):
        """When peer benchmark data is absent, peer valuation pillar must be 'unavailable' with score = None (not 50.0)."""
        payload = {"financials": {"key_ratios": {"forward_pe": 20.0}}, "peer_valuation": {"direct_peer_benchmarks": []}}
        res = self.service.evaluate_peer_valuation_pillar(payload)
        self.assertEqual(res["status"], "unavailable")
        self.assertIsNone(res["score"])

    def test_missing_alpha_is_unavailable_and_none(self):
        """When alpha is absent, market alpha pillar must be 'unavailable' with score = None (not 50.0)."""
        tech = {"current_price": 100.0}
        res = self.service.evaluate_market_alpha_pillar(tech)
        self.assertEqual(res["status"], "unavailable")
        self.assertIsNone(res["score"])

    def test_measured_trend_status_when_all_inputs_present(self):
        """When all trend inputs are available, status must be 'measured'."""
        tech = {
            "current_price": 150.0,
            "ema20": 140.0,
            "ema50": 130.0,
            "rsi14": 55.0,
            "change_5d_pct": 4.0
        }
        res = self.service.evaluate_trend_pillar(tech)
        self.assertEqual(res["status"], "measured")
        self.assertIsNotNone(res["score"])
        self.assertTrue(all(v["available"] for v in res["inputs"].values()))

    def test_partial_valuation_history_status(self):
        """When only revenue growth is present without P/E pair, status is 'partial'."""
        payload = {
            "financials": {
                "key_ratios": {
                    "forward_pe": None,
                    "trailing_pe": None,
                    "revenue_growth": 0.18
                }
            }
        }
        res = self.service.evaluate_valuation_history_pillar(payload)
        self.assertEqual(res["status"], "partial")
        self.assertIsNotNone(res["score"])
        self.assertEqual(res["score"], 65.0)  # 50 base + 15 rev growth

    def test_input_tracking_metadata(self):
        """Input tracking dictionary records value, availability, and freshness."""
        tracked = QuantitativeScoringService._track_input("test_metric", 42.5, is_valid=True)
        self.assertEqual(tracked["name"], "test_metric")
        self.assertEqual(tracked["value"], 42.5)
        self.assertTrue(tracked["available"])
        self.assertEqual(tracked["freshness"], "fresh")

        missing = QuantitativeScoringService._track_input("missing_metric", None, is_valid=False)
        self.assertFalse(missing["available"])
        self.assertIsNone(missing["value"])
        self.assertEqual(missing["freshness"], "unavailable")


class DynamicWeightRenormalizationTests(unittest.TestCase):
    def setUp(self):
        self.service = QuantitativeScoringService()

    def test_composite_renormalization_missing_pillars(self):
        """
        When sector (20%) and peer (20%) are unavailable:
        Available pillars: Trend (25%), Alpha (20%), Val History (15%). Total available weight = 0.60.
        Composite must be strictly re-normalized over the 3 available pillars with zero phantom 50.0 bias.
        """
        tech = {
            "current_price": 100.0,
            "ema20": 90.0,
            "ema50": 85.0,
            "rsi14": 55.0,
            "change_5d_pct": 5.0,
            "relative_alpha_5d": 3.0,
            "atr": 2.0
        }
        # Empty sector and peer data
        gloomberb = {
            "financials": {
                "key_ratios": {
                    "forward_pe": 15.0,
                    "trailing_pe": 25.0,
                    "revenue_growth": 0.20
                }
            },
            "peer_valuation": {"direct_peer_benchmarks": []}
        }

        res = self.service.compute_5pillar_scores(tech, gloomberb, sector_bench={})
        self.assertEqual(res["pillar_status"]["sector"], "unavailable")
        self.assertIsNone(res["sector_relative_score"])
        self.assertEqual(res["pillar_status"]["peer_valuation"], "unavailable")
        self.assertIsNone(res["peer_valuation_score"])

        self.assertEqual(res["available_pillars_count"], 3)
        self.assertEqual(res["total_pillars_count"], 5)

        # Expected score:
        # Trend: 50 + 15(ema20) + 15(ema50) + 10(rsi) + 5(change) = 95.0
        # Alpha: 50 + (3.0 * 5) = 65.0
        # Val Hist: 50 + 20(pe discount) + 15(rev growth) = 85.0
        # Re-normalized raw composite: (0.25*95 + 0.20*65 + 0.15*85) / (0.25 + 0.20 + 0.15)
        # = (23.75 + 13.0 + 12.75) / 0.60 = 49.5 / 0.60 = 82.5
        self.assertAlmostEqual(res["raw_composite"], 82.5, places=1)

    def test_all_pillars_unavailable_yields_none_composite(self):
        """When no data is provided, composite is None and signal reflects insufficient data."""
        res = self.service.compute_5pillar_scores({}, {}, {})
        self.assertIsNone(res["raw_composite"])
        self.assertIsNone(res["composite_quantitative_score"])
        self.assertEqual(res["available_pillars_count"], 0)
        self.assertIn("Unavailable", res["quant_signal"])

    def test_gated_confidence_with_missing_pillars(self):
        """Gated confidence ignores None pillars and does not artificially compress confidence with 50.0."""
        # 3 high-conviction available pillars and 2 None pillars
        pillar_scores = {
            "trend": 85.0,
            "alpha": 82.0,
            "valuation_history": 88.0,
            "sector": None,
            "peer_valuation": None
        }
        conf = gated_confidence(85.0, pillar_scores, data_completeness=0.85)
        self.assertGreaterEqual(conf, 0.70)


class FinancialQualityAndBalanceSheetTests(unittest.TestCase):
    def setUp(self):
        self.service = QuantitativeScoringService()

    def test_fortress_balance_sheet_and_elite_fcf(self):
        """Fortress balance sheet (net cash, high interest coverage, elite FCF margin) yields high quality score."""
        payload = {
            "financials": {
                "key_ratios": {"forward_pe": 22.0},
                "financial_quality": {
                    "fcf_margin": 25.0,
                    "operating_margin": 28.0,
                    "total_debt": 0.0,
                    "total_cash": 5000.0,
                    "net_debt": -5000.0,
                    "net_debt_to_ebitda": 0.0,
                    "interest_coverage": 999.0,
                    "earnings_quality_ratio": 1.25,
                    "share_dilution_rate": -1.5  # buybacks
                }
            }
        }
        res = self.service.evaluate_financial_quality(payload)
        self.assertFalse(res["is_value_trap"])
        self.assertGreaterEqual(res["score"], 80.0)
        self.assertEqual(res["status"], "measured")

    def test_cash_burning_high_leverage_penalties(self):
        """Negative FCF margin, distressed leverage (>4.5x), and strained interest coverage receive heavy penalties."""
        payload = {
            "financials": {
                "key_ratios": {"forward_pe": 8.0},
                "financial_quality": {
                    "fcf_margin": -8.5,
                    "operating_margin": 4.0,
                    "total_debt": 10000.0,
                    "total_cash": 500.0,
                    "net_debt": 9500.0,
                    "net_debt_to_ebitda": 5.2,
                    "interest_coverage": 1.2,
                    "earnings_quality_ratio": 0.35,
                    "share_dilution_rate": 8.0
                }
            }
        }
        res = self.service.evaluate_financial_quality(payload)
        self.assertLess(res["score"], 40.0)
        self.assertGreaterEqual(len(res["flags"]), 3)
        self.assertTrue(res["is_value_trap"])

    def test_earnings_quality_accrual_divergence(self):
        """Poor earnings quality (OCF/Net Income < 0.4x) triggers accrual red flag."""
        payload = {
            "financials": {
                "financial_quality": {
                    "fcf_margin": 5.0,
                    "operating_margin": 15.0,
                    "net_debt_to_ebitda": 1.0,
                    "interest_coverage": 5.0,
                    "earnings_quality_ratio": 0.30  # Heavy accruals, paper earnings
                }
            }
        }
        res = self.service.evaluate_financial_quality(payload)
        self.assertTrue(any("accrual" in f for f in res["flags"]))

    def test_peer_relative_quality_comparison(self):
        """Comparing against peer FCF margins and leverage adjusts score appropriately."""
        payload = {
            "financials": {
                "financial_quality": {
                    "fcf_margin": 18.0,
                    "operating_margin": 15.0,
                    "net_debt_to_ebitda": 1.2,
                    "interest_coverage": 6.0,
                    "earnings_quality_ratio": 1.15
                }
            },
            "peer_valuation": {
                "direct_peer_benchmarks": [
                    {"forward_pe": 20.0, "fcf_margin": 8.0, "net_debt_to_ebitda": 3.0},
                    {"forward_pe": 22.0, "fcf_margin": 10.0, "net_debt_to_ebitda": 3.2}
                ]
            }
        }
        res = self.service.evaluate_financial_quality(payload)
        self.assertGreaterEqual(res["score"], 70.0)
        # Peer avg FCF is 9.0%, company is 18.0% (+9% premium vs peer group)
        self.assertEqual(res["metrics"]["peer_avg_fcf_margin"], 9.0)


class ValueTrapGuardTests(unittest.TestCase):
    def test_value_trap_detection_on_cheap_multiple_with_bad_solvency(self):
        """A forward P/E of 9.5x with Net Debt/EBITDA of 4.2x is flagged as a value trap."""
        payload = {
            "financials": {
                "key_ratios": {"forward_pe": 9.5},
                "financial_quality": {
                    "fcf_margin": -2.0,
                    "operating_margin": 5.0,
                    "net_debt_to_ebitda": 4.2,
                    "interest_coverage": 1.8,
                    "earnings_quality_ratio": 0.5
                }
            },
            "peer_valuation": {
                "direct_peer_benchmarks": [{"forward_pe": 20.0}, {"forward_pe": 22.0}]
            }
        }
        res = QuantitativeScoringService().evaluate_financial_quality(payload)
        self.assertTrue(res["is_value_trap"])
        self.assertTrue(any("Value trap" in r for r in res["value_trap_reasons"]))

    def test_value_trap_downgrades_buy_to_hold(self):
        """A BUY decision on a value trap is mechanically downgraded to HOLD with no_trade_reason = VALUE_TRAP."""
        m_data = {
            "current_price": 50.0,
            "atr": 1.5,
            "suggested_stop_loss": 48.0,
            "suggested_target_price": 55.0,
            "rsi14": 52.0,
            "rvol_20d": 1.2,
            "avg_dollar_vol_20d": 10_000_000.0,
            "deterministic_data_completeness": 0.90,
            "deterministic_5pillar_scores": {
                "composite_quantitative_score": 75.0,
                "is_value_trap": True,
                "value_trap_reasons": ["Value trap: low multiple (forward P/E 9.5x) masks structural risk: heavy leverage (Net Debt/EBITDA 4.2x)"]
            }
        }
        data = {
            "stock": "TRAP",
            "decision": "BUY",
            "buy_score": 0.70,
            "hold_score": 0.20,
            "sell_score": 0.10,
            "data_completeness": 0.90
        }
        out = normalize_master_trader_json(data, "TRAP", m_data=m_data)
        self.assertEqual(out["decision"], "HOLD")
        self.assertEqual(out["no_trade_reason"], "VALUE_TRAP")
        self.assertTrue(any("Value-Trap Alert" in r for r in out["key_risks"]))


class DatabasePersistenceTests(unittest.TestCase):
    def test_db_persistence_of_pillar_status_and_financial_quality(self):
        """New columns for pillar status, tracking, financial quality, and value trap are stored and retrieved."""
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            init_db(db_path)

            record = {
                "stock": "TEST",
                "decision": "HOLD",
                "confidence": 0.65,
                "quant_score": 68.0,
                "pillar_status": {"trend": "measured", "sector": "unavailable"},
                "pillar_input_tracking": {"current_price": {"available": True, "value": 100.0}},
                "financial_quality_score": 82.5,
                "is_value_trap": 0,
                "value_trap_reasons": []
            }

            save_results([record], model_used="TestModel", db_path=db_path)
            signals = get_latest_signals(db_path=db_path, max_age_days=None)
            self.assertEqual(len(signals), 1)
            sig = signals[0]
            self.assertEqual(sig["symbol"], "TEST")
            self.assertAlmostEqual(sig["financial_quality_score"], 82.5)
            self.assertIn("unavailable", sig["pillar_status"])


if __name__ == "__main__":
    unittest.main()
