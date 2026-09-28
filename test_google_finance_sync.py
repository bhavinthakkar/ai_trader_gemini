#!/usr/bin/env python3
"""
Unit tests for google_finance_sync.py.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import google_finance_sync


class TestGoogleFinanceSync(unittest.TestCase):
    """Tests URL parsing, CSV parsing, and portfolio synchronization."""

    def test_convert_to_google_sheet_csv_url(self):
        url1 = "https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit?usp=sharing"
        expected1 = "https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/export?format=csv&gid=0"
        self.assertEqual(google_finance_sync.convert_to_google_sheet_csv_url(url1), expected1)

        url2 = "https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit#gid=123456"
        expected2 = "https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/export?format=csv&gid=123456"
        self.assertEqual(google_finance_sync.convert_to_google_sheet_csv_url(url2), expected2)

        self.assertIsNone(google_finance_sync.convert_to_google_sheet_csv_url("https://example.com/not_a_sheet"))

    def test_clean_ticker_symbol(self):
        self.assertEqual(google_finance_sync.clean_ticker_symbol("NASDAQ:NVDA"), "NVDA")
        self.assertEqual(google_finance_sync.clean_ticker_symbol("NYSE:PLTR"), "PLTR")
        self.assertEqual(google_finance_sync.clean_ticker_symbol("BATS:AAPL"), "AAPL")
        self.assertEqual(google_finance_sync.clean_ticker_symbol("FRA:NVD"), "NVD.DE")
        self.assertEqual(google_finance_sync.clean_ticker_symbol("ETR:SAP"), "SAP.DE")
        self.assertEqual(google_finance_sync.clean_ticker_symbol('"TSLA"'), "TSLA")
        self.assertEqual(google_finance_sync.clean_ticker_symbol("MSFT"), "MSFT")

    def test_parse_csv_content_standard(self):
        csv_data = """Symbol,Name,Shares,Purchase Price
NASDAQ:NVDA,NVIDIA Corporation,10,120.50
NASDAQ:AAPL,Apple Inc.,15,185.00
NASDAQ:TSLA,Tesla Inc.,8,220.00
"""
        holdings = google_finance_sync.parse_csv_content(csv_data)
        self.assertEqual(len(holdings), 3)
        self.assertEqual(holdings[0]["us_symbol"], "NVDA")
        self.assertEqual(holdings[0]["shares"], 10.0)
        self.assertEqual(holdings[0]["purchase_price"], 120.50)
        self.assertEqual(holdings[1]["us_symbol"], "AAPL")
        self.assertEqual(holdings[2]["us_symbol"], "TSLA")

    def test_parse_csv_content_symbols_only(self):
        csv_data = """Ticker
NVDA
AAPL
TSLA
MSFT
AMZN
"""
        holdings = google_finance_sync.parse_csv_content(csv_data)
        self.assertEqual(len(holdings), 5)
        self.assertEqual([h["us_symbol"] for h in holdings], ["NVDA", "AAPL", "TSLA", "MSFT", "AMZN"])

    @patch("requests.get")
    def test_sync_google_portfolio(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = """Symbol,Name,Shares,Purchase Price
NASDAQ:NVDA,NVIDIA Corporation,10,120.50
NASDAQ:AAPL,Apple Inc.,15,185.00
"""
        mock_get.return_value = mock_resp

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            cache_file = tf.name

        with patch("google_finance_sync.PORTFOLIO_CACHE_FILE", cache_file):
            payload = google_finance_sync.sync_google_portfolio(
                "https://docs.google.com/spreadsheets/d/fake_sheet_id/edit",
                prefer_exchange="DE",
                force_european=True,
            )
            self.assertEqual(payload["total_holdings"], 2)
            self.assertEqual(payload["holdings"][0]["symbol"], "NVD.DE")
            self.assertEqual(payload["holdings"][0]["currency"], "EUR")
            self.assertEqual(payload["holdings"][0]["currency_symbol"], "€")
            self.assertEqual(payload["holdings"][1]["symbol"], "APC.DE")

            # Check cache loading
            cached = google_finance_sync.load_cached_portfolio()
            self.assertIsNotNone(cached)
            self.assertEqual(cached["total_holdings"], 2)

        if os.path.exists(cache_file):
            os.remove(cache_file)


if __name__ == "__main__":
    unittest.main()
