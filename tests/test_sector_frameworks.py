#!/usr/bin/env python3
"""
Unit Tests for Sector-Specific Quantitative Frameworks & Dynamic Temperature Calibration
"""

import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from quant.sector_frameworks import (
    SECTOR_FRAMEWORKS,
    SECTOR_ALIASES,
    normalize_sector_name,
    get_sector_framework,
    get_sector_temperature,
    format_sector_system_prompt_overlay,
)
from rag_service import RAGService


class TestSectorFrameworks(unittest.TestCase):
    def test_canonical_sector_normalization(self):
        self.assertEqual(normalize_sector_name("Financials"), "Financials")
        self.assertEqual(normalize_sector_name("financials"), "Financials")
        self.assertEqual(normalize_sector_name("Information Technology"), "Information Technology")
        self.assertEqual(normalize_sector_name("Materials"), "Materials")
        self.assertEqual(normalize_sector_name("Energy"), "Energy")
        self.assertEqual(normalize_sector_name("Real Estate"), "Real Estate")
        self.assertEqual(normalize_sector_name("Health Care"), "Health Care")
        self.assertEqual(normalize_sector_name("Industrials"), "Industrials")
        self.assertEqual(normalize_sector_name("Consumer Discretionary"), "Consumer Discretionary")
        self.assertEqual(normalize_sector_name("Consumer Staples"), "Consumer Staples")
        self.assertEqual(normalize_sector_name("Communication Services"), "Communication Services")
        self.assertEqual(normalize_sector_name("Utilities"), "Utilities")

    def test_alias_normalization(self):
        self.assertEqual(normalize_sector_name("Banking"), "Financials")
        self.assertEqual(normalize_sector_name("Bank"), "Financials")
        self.assertEqual(normalize_sector_name("Technology"), "Information Technology")
        self.assertEqual(normalize_sector_name("Tech"), "Information Technology")
        self.assertEqual(normalize_sector_name("Software"), "Information Technology")
        self.assertEqual(normalize_sector_name("Semiconductors"), "Information Technology")
        self.assertEqual(normalize_sector_name("Mining"), "Materials")
        self.assertEqual(normalize_sector_name("Gold"), "Materials")
        self.assertEqual(normalize_sector_name("Basic Materials"), "Materials")
        self.assertEqual(normalize_sector_name("REIT"), "Real Estate")
        self.assertEqual(normalize_sector_name("REITs"), "Real Estate")
        self.assertEqual(normalize_sector_name("Pharma"), "Health Care")
        self.assertEqual(normalize_sector_name("Biotech"), "Health Care")
        self.assertEqual(normalize_sector_name("Oil & Gas"), "Energy")
        self.assertEqual(normalize_sector_name("Shipping"), "Energy")
        self.assertEqual(normalize_sector_name("Aerospace"), "Industrials")
        self.assertEqual(normalize_sector_name("Defense"), "Industrials")
        self.assertEqual(normalize_sector_name("Telecom"), "Communication Services")
        self.assertEqual(normalize_sector_name("Utility"), "Utilities")

    def test_unknown_sector_fallback(self):
        self.assertEqual(normalize_sector_name(None), "Equities")
        self.assertEqual(normalize_sector_name(""), "Equities")
        self.assertEqual(normalize_sector_name("Space Exploration"), "Space Exploration")

    def test_sector_calibrated_temperatures(self):
        # Strict balance-sheet / mathematical sectors: 0.10
        self.assertAlmostEqual(get_sector_temperature("Financials"), 0.10)
        self.assertAlmostEqual(get_sector_temperature("Real Estate"), 0.10)
        self.assertAlmostEqual(get_sector_temperature("Utilities"), 0.10)

        # Commodity cyclical sectors: 0.15
        self.assertAlmostEqual(get_sector_temperature("Materials"), 0.15)
        self.assertAlmostEqual(get_sector_temperature("Gold"), 0.15)
        self.assertAlmostEqual(get_sector_temperature("Energy"), 0.15)

        # Industrials & Staples: 0.18
        self.assertAlmostEqual(get_sector_temperature("Industrials"), 0.18)
        self.assertAlmostEqual(get_sector_temperature("Consumer Staples"), 0.18)

        # Discretionary & Communications: 0.22
        self.assertAlmostEqual(get_sector_temperature("Consumer Discretionary"), 0.22)
        self.assertAlmostEqual(get_sector_temperature("Communication Services"), 0.22)

        # High-growth & clinical pipeline sectors: 0.25
        self.assertAlmostEqual(get_sector_temperature("Information Technology"), 0.25)
        self.assertAlmostEqual(get_sector_temperature("Tech"), 0.25)
        self.assertAlmostEqual(get_sector_temperature("Health Care"), 0.25)
        self.assertAlmostEqual(get_sector_temperature("Biotech"), 0.25)

        # Fallback
        self.assertAlmostEqual(get_sector_temperature("Unknown Sector"), 0.20)

    def test_cli_temperature_override(self):
        # Explicit override must take precedence over sector default
        self.assertAlmostEqual(get_sector_temperature("Financials", override_temp=0.45), 0.45)
        self.assertAlmostEqual(get_sector_temperature("Technology", override_temp=0.05), 0.05)
        self.assertAlmostEqual(get_sector_temperature("Materials", override_temp=0.0), 0.0)

    def test_sector_driving_factors_content(self):
        for sector_name, fw in SECTOR_FRAMEWORKS.items():
            self.assertIn("sector", fw)
            self.assertIn("benchmark_etf", fw)
            self.assertIn("recommended_temperature", fw)
            self.assertIn("market_driving_factors", fw)
            self.assertIn("valuation_mandate", fw)
            self.assertIn("cyclical_and_event_traps", fw)

            # Every sector must have at least 4 concrete driving factors
            self.assertGreaterEqual(len(fw["market_driving_factors"]), 4)

        # Specific sector mandate assertions
        fin_fw = get_sector_framework("Financials")
        self.assertIn("NEVER use EV/EBITDA", fin_fw["valuation_mandate"])
        self.assertIn("Price-to-Tangible-Book", fin_fw["valuation_mandate"])
        self.assertIn("Yield curve", fin_fw["market_driving_factors"][0])

        mat_fw = get_sector_framework("Materials")
        self.assertIn("PETER LYNCH CYCLICAL P/E TRAP", mat_fw["valuation_mandate"])
        self.assertIn("AISC", mat_fw["valuation_mandate"])

        reit_fw = get_sector_framework("Real Estate")
        self.assertIn("NEVER evaluate standard GAAP EPS", reit_fw["valuation_mandate"])
        self.assertIn("FFO", reit_fw["valuation_mandate"])

        tech_fw = get_sector_framework("Information Technology")
        self.assertIn("Hyperscaler CapEx", tech_fw["market_driving_factors"][0])
        self.assertIn("Rule of 40", tech_fw["valuation_mandate"])

        energy_fw = get_sector_framework("Energy")
        self.assertIn("OPEC+", energy_fw["market_driving_factors"][0])
        self.assertIn("crack spreads", energy_fw["market_driving_factors"][1])

    def test_overlay_prompt_generation(self):
        full_overlay = format_sector_system_prompt_overlay("Financials", compact=False)
        self.assertIn("INSTITUTIONAL SECTOR ANALYTICAL FRAMEWORK: FINANCIALS", full_overlay)
        self.assertIn("Benchmark ETF: XLF", full_overlay)
        self.assertIn("CRITICAL SECTOR MARKET-DRIVING FACTORS", full_overlay)
        self.assertIn("SECTOR-SPECIFIC VALUATION METRIC MANDATE", full_overlay)
        self.assertIn("SECTOR CYCLICAL TRAPS", full_overlay)

        compact_overlay = format_sector_system_prompt_overlay("Materials", compact=True)
        self.assertIn("SECTOR ANALYTICAL MANDATE: MATERIALS", compact_overlay)
        self.assertIn("PETER LYNCH CYCLICAL P/E TRAP", compact_overlay)

    def test_rag_service_payload_integration(self):
        service = RAGService.__new__(RAGService)
        service.top_k = 4
        service.retrieve_knowledge_by_questions = Mock(return_value={})

        technical_data = {
            "current_price": 50.0,
            "atr": 1.5,
            "high_20d": 55.0,
            "low_20d": 45.0,
            "suggested_stop_loss": 47.0,
            "suggested_target_price": 56.0,
        }
        gloomberb_payload = {
            "symbol": "JPM",
            "profile": {"sector": "Financials"},
            "peer_valuation": {},
            "financials": {"key_ratios": {}},
            "macro_econ": {"interest_rate_outlook": {}},
            "sector_benchmark": {"symbol": "XLF", "sector_name": "Financials"},
            "options": {},
            "analyst_ratings": {},
            "earnings": {},
            "events": {},
            "insider_institutional": {},
            "data_source_status": {},
        }

        mock_scores = {
            "composite_quantitative_score": 75.0,
            "trend_score": 70.0,
            "sector_relative_score": 75.0,
            "market_alpha_score": 80.0,
            "valuation_history_score": 70.0,
            "peer_valuation_score": 75.0,
            "raw_composite": 75.0,
            "vol_factor": 1.0,
        }
        mock_rr = {
            "structural_stop": 47.0,
            "structural_target": 56.0,
            "distance_to_resistance_atr": 1.5,
            "reward_risk_ratio": 2.0,
            "breakeven_win_rate": 0.33,
        }

        with patch("rag_service.QuantitativeScoringService.compute_5pillar_scores", return_value=mock_scores), \
             patch("rag_service.QuantitativeScoringService.compute_channel_reward_risk", return_value=mock_rr), \
             patch("rag_service.QuantitativeScoringService.compute_reward_risk", return_value={"reward_risk_ratio": 2.0}):
            payload = service.get_nemotron_payload(
                gloomberb_payload,
                technical_data=technical_data,
                institutional_data=None,
                prompt_profile="full",
            )

        # Assert sector framework is incorporated into system instruction
        self.assertEqual(payload["sector"], "Financials")
        self.assertAlmostEqual(payload["recommended_temperature"], 0.10)
        self.assertIn("INSTITUTIONAL SECTOR ANALYTICAL FRAMEWORK: FINANCIALS", payload["system_instruction"])
        self.assertIn("NEVER use EV/EBITDA", payload["system_instruction"])
        self.assertIn("Yield curve", payload["system_instruction"])


if __name__ == "__main__":
    unittest.main()
