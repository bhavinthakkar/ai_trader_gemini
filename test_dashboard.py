import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from dashboard import (
    format_volume,
    format_market_cap,
    parse_news_item,
    infer_trading_catalyst,
    process_single_stock,
    generate_html_dashboard,
)


class TestDashboard(unittest.TestCase):
    def test_format_volume(self):
        self.assertEqual(format_volume(None), "N/A")
        self.assertEqual(format_volume("N/A"), "N/A")
        self.assertEqual(format_volume(450), "450")
        self.assertEqual(format_volume(15000), "15K")
        self.assertEqual(format_volume(85400000), "85.4M")
        self.assertEqual(format_volume(1250000000), "1.25B")

    def test_format_market_cap(self):
        self.assertEqual(format_market_cap(None), "N/A")
        self.assertEqual(format_market_cap("N/A"), "N/A")
        self.assertEqual(format_market_cap(50000000), "$50.0M")
        self.assertEqual(format_market_cap(85000000000), "$85.00B")
        self.assertEqual(format_market_cap(2800000000000), "$2.80T")

    def test_parse_news_item_nested_content(self):
        # Newer yfinance schema with 'content' dictionary
        raw = {
            "id": "123",
            "content": {
                "title": "Nvidia Expands Next-Gen AI Chip Production",
                "summary": "Nvidia accelerates production timeline for its next architecture.",
                "provider": {"displayName": "Bloomberg"},
                "canonicalUrl": {"url": "https://bloomberg.com/news/123"},
                "pubDate": "2026-09-28T09:30:00Z"
            }
        }
        parsed = parse_news_item(raw)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["title"], "Nvidia Expands Next-Gen AI Chip Production")
        self.assertEqual(parsed["publisher"], "Bloomberg")
        self.assertEqual(parsed["url"], "https://bloomberg.com/news/123")

    def test_parse_news_item_flat_schema(self):
        # Legacy yfinance schema
        raw = {
            "id": "456",
            "title": "Apple Reports Record Quarterly Services Revenue",
            "summary": "Services revenue grew 14% year over year.",
            "publisher": "Reuters",
            "link": "https://reuters.com/news/456",
            "providerPublishTime": 1727440000
        }
        parsed = parse_news_item(raw)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["title"], "Apple Reports Record Quarterly Services Revenue")
        self.assertEqual(parsed["publisher"], "Reuters")
        self.assertEqual(parsed["url"], "https://reuters.com/news/456")

    def test_infer_trading_catalyst_earnings(self):
        news = [
            {"title": "Company Q3 Revenue and EPS Beat Estimates, Raises Full-Year Guidance", "summary": "Profit margins expanded."}
        ]
        cat, headline, summary = infer_trading_catalyst("XYZ", news, change_pct=4.5, rvol=2.1)
        self.assertEqual(cat, "EARNINGS / FINANCIALS")
        self.assertIn("Beat Estimates", headline)
        self.assertIn("gaining", summary)

    def test_infer_trading_catalyst_analyst(self):
        news = [
            {"title": "Morgan Stanley Upgrades Stock to Overweight, Raises Price Target", "summary": "Wall Street analyst sees 30% upside."}
        ]
        cat, headline, summary = infer_trading_catalyst("XYZ", news, change_pct=2.8, rvol=1.4)
        self.assertEqual(cat, "ANALYST ACTION")
        self.assertIn("Upgrades Stock", headline)

    def test_infer_trading_catalyst_tech(self):
        news = [
            {"title": "Unveils Next-Generation AI Chip Architecture at Tech Summit", "summary": "New model features high throughput."}
        ]
        cat, headline, summary = infer_trading_catalyst("XYZ", news, change_pct=1.5, rvol=1.2)
        self.assertEqual(cat, "PRODUCT / AI / TECH")

    def test_infer_trading_catalyst_volume_breakout_fallback(self):
        # When no news items match keywords, falls back to volume surge
        news = [{"title": "General market summary report", "summary": "Trading session underway."}]
        cat, headline, summary = infer_trading_catalyst("XYZ", news, change_pct=5.2, rvol=2.5)
        self.assertEqual(cat, "VOLUME SURGE")
        self.assertIn("gaining 5.20%", summary)

    def test_process_single_stock(self):
        mock_quote = {
            "symbol": "AAPL",
            "shortName": "Apple Inc.",
            "regularMarketPrice": 230.50,
            "regularMarketDayHigh": 232.00,
            "regularMarketDayLow": 228.00,
            "regularMarketOpen": 229.00,
            "regularMarketPreviousClose": 228.00,
            "regularMarketVolume": 55000000,
            "regularMarketChange": 2.50,
            "regularMarketChangePercent": 1.10,
            "marketCap": 3500000000000,
            "fiftyTwoWeekHigh": 237.00,
            "fiftyTwoWeekLow": 165.00,
        }

        mock_market_agent = MagicMock()
        mock_market_agent.analyze.return_value = {
            "rsi14": 58.4,
            "ema20": 226.0,
            "ema50": 220.0,
            "atr": 3.8,
            "vol_20d_mean": 48000000,
            "rvol_20d": 1.15,
            "relative_alpha_5d": "+1.8%",
            "forward_pe": 28.5,
        }

        with patch("dashboard.fetch_stock_news", return_value=[{"title": "Apple Unveils New AI Features", "summary": "AI updates coming to devices."}]):
            res = process_single_stock(mock_quote, mock_market_agent)

        self.assertEqual(res["symbol"], "AAPL")
        self.assertEqual(res["price"], 230.50)
        self.assertEqual(res["day_high"], 232.00)
        self.assertEqual(res["day_low"], 228.00)
        self.assertEqual(res["volume"], 55000000)
        self.assertEqual(res["rvol"], 1.15)
        self.assertEqual(res["rsi14"], 58.4)
        self.assertEqual(res["catalyst_type"], "PRODUCT / AI / TECH")
        self.assertTrue(len(res["news"]) > 0)

    def test_generate_html_dashboard(self):
        sample_data = [
            {
                "symbol": "NVDA",
                "name": "NVIDIA Corporation",
                "price": 225.0,
                "change": 4.5,
                "change_pct": 2.04,
                "day_high": 227.0,
                "day_low": 222.0,
                "volume": 85000000,
                "rvol": 1.2,
                "market_cap_str": "$3.1T",
                "intraday_range_pct": 60.0,
                "rsi14": 62.1,
                "relative_alpha_5d": "+3.2%",
                "catalyst_type": "PRODUCT / AI / TECH",
                "reason_summary": "Shares surging on new AI data center hardware release.",
                "news": [{"title": "Nvidia reveals new AI chip", "publisher": "Reuters", "url": "https://reuters.com"}]
            }
        ]

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tf:
            temp_path = tf.name

        try:
            out_file = generate_html_dashboard(sample_data, "FEAR (35)", output_path=temp_path)
            self.assertTrue(os.path.exists(out_file))
            with open(out_file, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("NVDA", content)
            self.assertIn("NVIDIA Corporation", content)
            self.assertIn("$225.00", content)
            self.assertIn("FEAR (35)", content)
            self.assertIn("PRODUCT / AI / TECH", content)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


    def test_get_nasdaq_european_universe(self):
        from ticker_resolver import get_nasdaq_european_universe
        xetra_univ = get_nasdaq_european_universe(exchange="DE")
        self.assertIn("NVD.DE", xetra_univ)
        self.assertIn("APC.DE", xetra_univ)
        self.assertIn("TL0.DE", xetra_univ)

        gettex_univ = get_nasdaq_european_universe(exchange="MU")
        self.assertIn("NVD.MU", gettex_univ)
        self.assertIn("APC.MU", gettex_univ)
        self.assertIn("TL0.MU", gettex_univ)

    @patch("dashboard.fetch_most_active_quotes")
    @patch("dashboard.load_cached_portfolio")
    def test_collect_dashboard_data_portfolio(self, mock_load, mock_fetch):
        mock_load.return_value = {
            "total_holdings": 2,
            "holdings": [
                {"symbol": "NVD.DE", "us_symbol": "NVDA", "shares": 10.0, "purchase_price": 115.0},
                {"symbol": "TL0.DE", "us_symbol": "TSLA", "shares": 5.0, "purchase_price": 210.0}
            ]
        }
        mock_fetch.return_value = [
            {"symbol": "NVD.DE", "shortName": "NVIDIA", "regularMarketPrice": 125.0, "regularMarketVolume": 500000},
            {"symbol": "TL0.DE", "shortName": "Tesla", "regularMarketPrice": 220.0, "regularMarketVolume": 300000}
        ]

        from dashboard import collect_dashboard_data
        data = collect_dashboard_data(limit=5, use_portfolio=True)
        self.assertEqual(len(data), 2)
        syms = [d["symbol"] for d in data]
        self.assertIn("NVD.DE", syms)
        self.assertIn("TL0.DE", syms)
        # Check shares and purchase price were attached
        nvd_item = next(d for d in data if d["symbol"] == "NVD.DE")
        self.assertEqual(nvd_item["shares"], 10.0)
        self.assertEqual(nvd_item["purchase_price"], 115.0)
        self.assertEqual(nvd_item["currency_symbol"], "€")


if __name__ == "__main__":
    unittest.main()
