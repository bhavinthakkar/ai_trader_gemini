#!/usr/bin/env python3
"""
Unit tests for ticker_resolver.py (European Ticker, ISIN, and WKN resolution).
"""

import unittest
from unittest.mock import MagicMock, patch

import ticker_resolver


class TestTickerResolver(unittest.TestCase):
    """Tests for ISIN detection, WKN recognition, and symbol resolution."""

    def test_is_isin_valid(self):
        self.assertTrue(ticker_resolver.is_isin("US67066G1040"))
        self.assertTrue(ticker_resolver.is_isin("DE0007164600"))
        self.assertTrue(ticker_resolver.is_isin("NL0010273215"))
        self.assertTrue(ticker_resolver.is_isin("FR0000121014"))

    def test_is_isin_invalid(self):
        self.assertFalse(ticker_resolver.is_isin("NVDA"))
        self.assertFalse(ticker_resolver.is_isin("SAP.DE"))
        self.assertFalse(ticker_resolver.is_isin("716460"))
        self.assertFalse(ticker_resolver.is_isin("US67066G104"))  # 11 chars
        self.assertFalse(ticker_resolver.is_isin("US67066G10400"))  # 13 chars

    def test_is_wkn_valid(self):
        self.assertTrue(ticker_resolver.is_wkn("918422"))
        self.assertTrue(ticker_resolver.is_wkn("716460"))
        self.assertTrue(ticker_resolver.is_wkn("A1CX3T"))
        self.assertTrue(ticker_resolver.is_wkn("A2QA4J"))

    def test_is_wkn_invalid(self):
        self.assertFalse(ticker_resolver.is_wkn("NVDA"))
        self.assertFalse(ticker_resolver.is_wkn("US67066G1040"))
        self.assertFalse(ticker_resolver.is_wkn("7164"))  # 4 chars

    def test_is_european_symbol(self):
        self.assertTrue(ticker_resolver.is_european_symbol("SAP.DE"))
        self.assertTrue(ticker_resolver.is_european_symbol("NVD.MU"))
        self.assertTrue(ticker_resolver.is_european_symbol("APC.TG"))
        self.assertTrue(ticker_resolver.is_european_symbol("BMW.F"))
        self.assertTrue(ticker_resolver.is_european_symbol("ASML.AS"))
        self.assertTrue(ticker_resolver.is_european_symbol("MC.PA"))
        self.assertFalse(ticker_resolver.is_european_symbol("NVDA"))
        self.assertFalse(ticker_resolver.is_european_symbol("AAPL"))
        self.assertFalse(ticker_resolver.is_european_symbol("000660.KS"))

    def test_get_currency_for_symbol(self):
        self.assertEqual(ticker_resolver.get_currency_for_symbol("SAP.DE"), ("EUR", "€"))
        self.assertEqual(ticker_resolver.get_currency_for_symbol("NVD.MU"), ("EUR", "€"))
        self.assertEqual(ticker_resolver.get_currency_for_symbol("ASML.AS"), ("EUR", "€"))
        self.assertEqual(ticker_resolver.get_currency_for_symbol("AZN.L"), ("GBP", "£"))
        self.assertEqual(ticker_resolver.get_currency_for_symbol("NESN.SW"), ("CHF", "CHF"))
        self.assertEqual(ticker_resolver.get_currency_for_symbol("NVDA"), ("USD", "$"))

    def test_resolve_isin_us_stock(self):
        # NVIDIA ISIN
        res = ticker_resolver.resolve_symbol("US67066G1040", prefer_exchange="DE")
        self.assertEqual(res["symbol"], "NVD.DE")
        self.assertEqual(res["underlying_symbol"], "NVDA")
        self.assertEqual(res["currency"], "EUR")
        self.assertEqual(res["currency_symbol"], "€")
        self.assertTrue(res["is_european"])

        # gettex Munich preference
        res_mu = ticker_resolver.resolve_symbol("US67066G1040", prefer_exchange="MU")
        self.assertEqual(res_mu["symbol"], "NVD.MU")
        self.assertEqual(res_mu["currency"], "EUR")

    def test_resolve_isin_german_stock(self):
        # SAP ISIN
        res = ticker_resolver.resolve_symbol("DE0007164600", prefer_exchange="DE")
        self.assertEqual(res["symbol"], "SAP.DE")
        self.assertEqual(res["company_name"], "SAP SE")
        self.assertEqual(res["currency"], "EUR")

    def test_resolve_wkn(self):
        # SAP WKN 716460
        res = ticker_resolver.resolve_symbol("716460", prefer_exchange="DE")
        self.assertEqual(res["symbol"], "SAP.DE")
        self.assertEqual(res["isin"], "DE0007164600")

    def test_resolve_us_symbol_with_european_preference(self):
        # Passing 'NVDA' with prefer_exchange="DE"
        res = ticker_resolver.resolve_symbol("NVDA", prefer_exchange="DE")
        self.assertEqual(res["symbol"], "NVD.DE")
        self.assertEqual(res["underlying_symbol"], "NVDA")
        self.assertEqual(res["currency"], "EUR")

        # Passing 'NVDA' with prefer_exchange="US"
        res_us = ticker_resolver.resolve_symbol("NVDA", prefer_exchange="US", force_european=False)
        self.assertEqual(res_us["symbol"], "NVDA")
        self.assertEqual(res_us["currency"], "USD")

    def test_resolve_mu_stock(self):
        # Default / AUTO resolution should preserve US ticker MU
        res = ticker_resolver.resolve_symbol("MU")
        self.assertEqual(res["symbol"], "MU")
        self.assertEqual(res["company_name"], "Micron Technology, Inc.")
        self.assertEqual(res["currency"], "USD")
        self.assertEqual(res["currency_symbol"], "$")
        self.assertFalse(res["is_european"])

        # Explicit European request should resolve to dual-listing
        res_eu = ticker_resolver.resolve_symbol("MU", force_european=True)
        self.assertEqual(res_eu["symbol"], "MQN.DE")
        self.assertEqual(res_eu["currency"], "EUR")
        self.assertTrue(res_eu["is_european"])

    def test_resolve_native_european_stock(self):
        # Plain ticker for German stock SAP should resolve to European exchange
        res = ticker_resolver.resolve_symbol("SAP")
        self.assertEqual(res["symbol"], "SAP.DE")
        self.assertEqual(res["currency"], "EUR")
        self.assertTrue(res["is_european"])

    def test_resolve_already_suffixed_symbol(self):
        res = ticker_resolver.resolve_symbol("RHM.DE")
        self.assertEqual(res["symbol"], "RHM.DE")
        self.assertEqual(res["currency"], "EUR")
        self.assertEqual(res["currency_symbol"], "€")

        res_mu = ticker_resolver.resolve_symbol("NVD.MU")
        self.assertEqual(res_mu["symbol"], "NVD.MU")
        self.assertEqual(res_mu["currency"], "EUR")

    def test_european_default_universe(self):
        univ_de = ticker_resolver.get_european_default_universe(exchange="DE")
        self.assertIn("SAP.DE", univ_de)
        self.assertIn("SIE.DE", univ_de)
        self.assertIn("RHM.DE", univ_de)
        self.assertIn("NVD.DE", univ_de)

        univ_mu = ticker_resolver.get_european_default_universe(exchange="MU")
        self.assertIn("SAP.MU", univ_mu)
        self.assertIn("NVD.MU", univ_mu)


if __name__ == "__main__":
    unittest.main()
