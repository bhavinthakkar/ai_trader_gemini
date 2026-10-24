import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from danelfin_service import DanelfinService


class TestDanelfinService(unittest.TestCase):

    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db_path = self.temp_db.name
        self.temp_db.close()
        self.service = DanelfinService(
            api_key="test_dummy_key",
            base_url="https://apirest.danelfin.com",
            db_path=self.temp_db_path,
            monthly_limit=500,
            safe_threshold=480,
        )

    def tearDown(self):
        if os.path.exists(self.temp_db_path):
            os.remove(self.temp_db_path)

    def test_database_initialization(self):
        """Verify cache and usage tables are created."""
        usage = self.service.get_usage()
        self.assertEqual(usage["call_count"], 0)
        self.assertEqual(usage["monthly_limit"], 500)
        self.assertEqual(usage["safe_threshold"], 480)
        self.assertTrue(usage["is_safe"])

    def test_usage_increment(self):
        """Verify API calls increment the monthly usage counter."""
        self.assertEqual(self.service.get_usage()["call_count"], 0)
        self.service._increment_usage()
        self.assertEqual(self.service.get_usage()["call_count"], 1)
        self.service._increment_usage()
        self.assertEqual(self.service.get_usage()["call_count"], 2)
        usage = self.service.get_usage()
        self.assertEqual(usage["remaining"], 498)
        self.assertEqual(usage["safe_remaining"], 478)

    def test_quota_safety_threshold(self):
        """Verify is_quota_available returns False when at or above safe threshold."""
        # Manually set call count to safe_threshold (480)
        with self.service._get_connection() as conn:
            cursor = conn.cursor()
            m_key = self.service.get_usage()["month"]
            cursor.execute(
                "INSERT OR REPLACE INTO danelfin_usage (month_key, call_count, updated_at) VALUES (?, ?, ?)",
                (m_key, 480, "2026-10-24 12:00:00")
            )
            conn.commit()

        self.assertFalse(self.service.is_quota_available(force_refresh=False))
        self.assertTrue(self.service.is_quota_available(force_refresh=True))

    def test_ticker_cleaning(self):
        """Verify European exchange extensions and formatting are cleaned."""
        self.assertEqual(self.service.clean_ticker_for_danelfin("AAPL"), "AAPL")
        self.assertEqual(self.service.clean_ticker_for_danelfin("aapl"), "AAPL")
        self.assertEqual(self.service.clean_ticker_for_danelfin("SAP.DE"), "SAP")
        self.assertEqual(self.service.clean_ticker_for_danelfin("MC.PA"), "MC")
        # NVD.DE resolves to NVDA via ticker_resolver
        self.assertEqual(self.service.clean_ticker_for_danelfin("NVD.DE"), "NVDA")

    @patch("requests.get")
    def test_get_scores_cache_miss_then_hit(self, mock_get):
        """Verify get_scores calls API on cache miss, caches it, and serves from cache on hit."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "2026-10-24": {
                "aiscore": 8,
                "technical": 7,
                "fundamental": 8,
                "sentiment": 9,
                "low_risk": 6
            }
        }
        mock_get.return_value = mock_response

        # 1. Cache Miss -> Calls API
        score1 = self.service.get_scores("AAPL")
        self.assertIsNotNone(score1)
        self.assertEqual(score1["aiscore"], 8)
        self.assertEqual(score1["source"], "api")
        self.assertEqual(mock_get.call_count, 1)
        self.assertEqual(self.service.get_usage()["call_count"], 1)

        # 2. Cache Hit -> Serves from SQLite, NO API call
        score2 = self.service.get_scores("AAPL")
        self.assertIsNotNone(score2)
        self.assertEqual(score2["aiscore"], 8)
        self.assertEqual(score2["source"], "cache")
        self.assertEqual(mock_get.call_count, 1)  # Still 1!
        self.assertEqual(self.service.get_usage()["call_count"], 1)  # Still 1!

    @patch("requests.get")
    def test_batch_pre_seeding_from_best_stocks(self, mock_get):
        """
        Verify get_best_stocks populates the score cache for all returned tickers,
        so subsequent get_scores calls for those tickers consume ZERO extra API calls.
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "2026-10-24": {
                "BMO": {
                    "rank": 1,
                    "aiscore": 10,
                    "fundamental": 10,
                    "technical": 7,
                    "sentiment": 10,
                    "low_risk": 6,
                    "perf_ytd": 0.30
                },
                "HSBC": {
                    "rank": 2,
                    "aiscore": 10,
                    "fundamental": 9,
                    "technical": 8,
                    "sentiment": 10,
                    "low_risk": 7,
                    "perf_ytd": 0.25
                }
            }
        }
        mock_get.return_value = mock_response

        # Fetch Best Stocks (1 API call)
        best = self.service.get_best_stocks()
        self.assertIsNotNone(best)
        self.assertEqual(mock_get.call_count, 1)
        self.assertEqual(self.service.get_usage()["call_count"], 1)

        # Now get_scores for BMO and HSBC should be instant cache hits (0 additional API calls!)
        bmo_score = self.service.get_scores("BMO")
        self.assertIsNotNone(bmo_score)
        self.assertEqual(bmo_score["aiscore"], 10)
        self.assertEqual(bmo_score["source"], "cache")

        hsbc_score = self.service.get_scores("HSBC")
        self.assertIsNotNone(hsbc_score)
        self.assertEqual(hsbc_score["aiscore"], 10)
        self.assertEqual(hsbc_score["source"], "cache")

        # mock_get should still have been called only ONCE in total!
        self.assertEqual(mock_get.call_count, 1)
        self.assertEqual(self.service.get_usage()["call_count"], 1)

    @patch("requests.get")
    def test_trading_parameters(self, mock_get):
        """Verify trading parameters endpoint retrieval and caching."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "2026-10-24": {
                "entry_price": 150.0,
                "stop_loss": 140.0,
                "stop_loss_pct": -6.67,
                "take_profit": 170.0,
                "take_profit_pct": 13.33,
                "horizon": "3M",
                "currency": "USD",
                "signal": "buy"
            }
        }
        mock_get.return_value = mock_response

        params = self.service.get_trading_parameters("AAPL")
        self.assertIsNotNone(params)
        self.assertEqual(params["entry_price"], 150.0)
        self.assertEqual(params["signal"], "buy")
        self.assertEqual(params["source"], "api")

        # Repeat call is cached
        params_cached = self.service.get_trading_parameters("AAPL")
        self.assertEqual(params_cached["source"], "cache")
        self.assertEqual(mock_get.call_count, 1)

    @patch("requests.get")
    def test_price_forecast(self, mock_get):
        """Verify price forecast quantile extraction."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "2026-10-24": {
                "signal": "buy",
                "median_1m": 0.04,
                "q05_1m": -0.08,
                "q95_1m": 0.12,
                "median_3m": 0.09,
                "q05_3m": -0.10,
                "q95_3m": 0.18,
                "take_profit_3m": 0.18,
                "stop_loss_3m": -0.10
            }
        }
        mock_get.return_value = mock_response

        fc = self.service.get_price_forecast("NVDA")
        self.assertIsNotNone(fc)
        self.assertEqual(fc["median_1m"], 0.04)
        self.assertEqual(fc["q05_1m"], -0.08)
        self.assertEqual(fc["q95_1m"], 0.12)
        self.assertEqual(fc["source"], "api")

    @patch("requests.get")
    def test_rate_limit_and_error_handling(self, mock_get):
        """Verify 429 and error codes do not increment usage or crash."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Too Many Requests"
        mock_get.return_value = mock_response

        score = self.service.get_scores("UNKNOWN_SYM")
        self.assertIsNone(score)
        # Usage counter should NOT increment on failed requests
        self.assertEqual(self.service.get_usage()["call_count"], 0)


if __name__ == "__main__":
    unittest.main()
