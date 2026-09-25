import unittest
from typing import Final
from unittest.mock import Mock, patch

from llamacpp_provider import LlamaCppConfig, estimate_text_tokens
from rag_service import RAGService


LOCAL_CONFIG: Final = LlamaCppConfig(
    base_url="http://127.0.0.1:11434/v1",
    model="qwen2.5",
    api_key=None,
    connect_timeout_seconds=5.0,
    read_timeout_seconds=900.0,
    write_timeout_seconds=30.0,
    pool_timeout_seconds=10.0,
    max_output_tokens=2048,
    context_tokens=8192,
    prompt_safety_margin_tokens=256,
)


class CompactRagPromptTests(unittest.TestCase):
    def _service(self) -> RAGService:
        service = RAGService.__new__(RAGService)
        service.top_k = 8
        service.retrieve_knowledge_by_questions = Mock(
            return_value={
                "Recent Material Events (Last 24h / 7d)": ["A" * 5000],
                "Short-Term Bullish & Bearish Drivers": ["B" * 5000],
                "Updated Guidance & Earnings Takeaways": ["C" * 5000],
                "Longer-Term Historical Execution Pattern": ["D" * 5000],
                "Persistent Structural & Valuation Headwinds": ["E" * 5000],
                "Historical Macro & Industry Cycles": ["F" * 5000],
            }
        )
        return service

    def test_compact_profile_fits_8k_context_with_configured_output(self):
        service = self._service()
        technical_data = {
            "current_price": 100.0,
            "atr": 3.0,
            "high_20d": 110.0,
            "low_20d": 90.0,
            "suggested_stop_loss": 95.0,
            "suggested_target_price": 110.0,
            "analyst_target_price": 120.0,
        }
        gloomberb_payload = {
            "symbol": "AAPL",
            "profile": {"sector": "Technology", "return_1y": 0.12},
            "peer_valuation": {
                "price_to_sales": 8.0,
                "price_to_free_cash_flow": 30.0,
                "direct_peer_benchmarks": [{"ticker": "MSFT", "forward_pe": 30.0}],
            },
            "financials": {
                "key_ratios": {
                    "forward_pe": 28.0,
                    "trailing_pe": 32.0,
                    "ev_to_ebitda": 24.0,
                    "revenue_growth": 0.10,
                    "earnings_growth": 0.12,
                    "profit_margins": 0.25,
                }
            },
            "macro_econ": {
                "interest_rate_outlook": {
                    "fed_funds_rate": 4.5,
                    "yield_10y_real": 1.8,
                    "yield_curve_status": "restrictive",
                }
            },
            "sector_benchmark": {"symbol": "XLK", "change_5d_pct": 0.01},
            "options": {"put_call_ratio": 0.8, "implied_volatility": 0.25},
            "analyst_ratings": {
                "mean_target_price": 120.0,
                "recommendation_rating": "buy",
                "recent_major_bank_actions": ["Bank raises target"],
            },
            "earnings": {"earnings_date": "2026-10-30", "timing": "after close"},
            "events": {"historical_earnings_surprises": []},
            "insider_institutional": {"top_institutional_holders": []},
            "data_source_status": {"news": "available"},
        }
        quant_service = Mock()
        quant_service.compute_5pillar_scores.return_value = {
            "composite_quantitative_score": 61.0,
            "trend_score": 60.0,
            "sector_relative_score": 62.0,
            "market_alpha_score": 58.0,
            "valuation_history_score": 64.0,
            "peer_valuation_score": 60.0,
            "raw_composite": 61.0,
            "vol_factor": 1.0,
        }
        quant_service.compute_channel_reward_risk.return_value = {
            "structural_stop": 95.0,
            "structural_target": 110.0,
            "distance_to_resistance_atr": 1.5,
            "reward_risk_ratio": 1.67,
            "breakeven_win_rate": 0.375,
        }
        quant_service.compute_reward_risk.return_value = {"reward_risk_ratio": 2.0}

        with (
            patch(
                "rag_service.QuantitativeScoringService.compute_5pillar_scores",
                return_value=quant_service.compute_5pillar_scores.return_value,
            ),
            patch(
                "rag_service.QuantitativeScoringService.compute_channel_reward_risk",
                return_value=quant_service.compute_channel_reward_risk.return_value,
            ),
            patch(
                "rag_service.QuantitativeScoringService.compute_reward_risk",
                return_value=quant_service.compute_reward_risk.return_value,
            ),
        ):
            payload = service.get_nemotron_payload(
                gloomberb_payload,
                technical_data=technical_data,
                institutional_data=None,
                prompt_profile="compact",
            )

        estimated_total = (
            estimate_text_tokens(payload["system_instruction"])
            + estimate_text_tokens(payload["user_prompt"])
            + LOCAL_CONFIG.max_output_tokens
            + LOCAL_CONFIG.prompt_safety_margin_tokens
        )
        self.assertEqual(payload["prompt_profile"], "compact")
        self.assertLessEqual(estimated_total, LOCAL_CONFIG.context_tokens)
        service.retrieve_knowledge_by_questions.assert_called_once_with(
            gloomberb_payload,
            technical_data,
            None,
            max_per_question=1,
        )

    def test_compact_payload_uses_instance_quantitative_scorer(self):
        service = self._service()
        technical_data = {
            "current_price": 100.0,
            "ema20": 95.0,
            "ema50": 90.0,
            "rsi14": 55.0,
            "change_5d_pct": 2.0,
            "relative_alpha_5d": 1.0,
            "atr": 3.0,
            "high_20d": 110.0,
            "low_20d": 90.0,
            "suggested_stop_loss": 95.0,
            "suggested_target_price": 110.0,
            "analyst_target_price": 120.0,
        }
        gloomberb_payload = {
            "symbol": "NVDA",
            "profile": {"sector": "Technology", "return_1y": 0.12},
            "peer_valuation": {
                "direct_peer_benchmarks": [{"ticker": "MSFT", "forward_pe": 30.0}]
            },
            "financials": {
                "key_ratios": {
                    "forward_pe": 28.0,
                    "trailing_pe": 32.0,
                    "revenue_growth": 0.10,
                }
            },
            "macro_econ": {},
            "sector_benchmark": {},
            "options": {},
            "analyst_ratings": {},
            "earnings": {},
            "insider_institutional": {},
            "data_source_status": {},
        }

        payload = service.get_nemotron_payload(
            gloomberb_payload,
            technical_data=technical_data,
            institutional_data=None,
            prompt_profile="compact",
        )

        scores = payload["technical_summary"]["deterministic_5pillar_scores"]
        self.assertIsInstance(scores["composite_quantitative_score"], float)

    def test_compact_portfolio_profile_fits_8k_context(self):
        service = self._service()
        portfolio = {
            "positions": [
                {
                    "symbol": f"TEST{i}",
                    "allocation": 0.12,
                    "sector": "Technology",
                    "notes": "N" * 2000,
                }
                for i in range(30)
            ]
        }

        payload = service.get_portfolio_analysis_payload(
            portfolio,
            analysis_horizon="next 90 days",
            prompt_profile="compact",
        )
        estimated_total = (
            estimate_text_tokens(payload["system_instruction"])
            + estimate_text_tokens(payload["user_prompt"])
            + LOCAL_CONFIG.max_output_tokens
            + LOCAL_CONFIG.prompt_safety_margin_tokens
        )

        self.assertEqual(payload["prompt_profile"], "compact")
        self.assertLessEqual(estimated_total, LOCAL_CONFIG.context_tokens)


if __name__ == "__main__":
    unittest.main()
