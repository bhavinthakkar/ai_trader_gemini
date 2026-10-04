#!/usr/bin/env python3
"""
European Exchange & Ticker / ISIN Resolver
==========================================
Bridges European retail exchanges (gettex / Börse München, EIX / Börse Hannover, XETRA)
with the AI Trader analysis pipeline.

Key Capabilities:
1. ISIN & WKN Resolution:
   Resolves 12-char ISINs (e.g. 'US67066G1040', 'DE0007164600', 'NL0010273215')
   or German WKNs (e.g. '716460', '918422') into active trading symbols.
2. US ↔ European Dual-Listing Mapping:
   Translates standard US tickers (e.g. 'NVDA', 'AAPL', 'TSLA', 'MSFT') into their
   European equivalents trading in EUR on XETRA ('.DE') and gettex ('.MU').
3. Multi-Currency Detection:
   Identifies trading currency ('EUR' / '€', 'USD' / '$', 'GBP' / '£', 'CHF').
4. European High-Volume gettex / XETRA Universe:
   Pre-curated universe of most liquid European blue-chips and gettex dual-listings.
"""

from __future__ import annotations

import re
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple
import requests

# Built-in reference registry of top European equities and US dual-listings traded on gettex / EIX / XETRA
SECURITIES_DIRECTORY: Dict[str, Dict[str, Any]] = {
    # ------------------ US Dual-Listings Traded on gettex / EIX / XETRA (EUR) ------------------
    "NVDA": {
        "company_name": "NVIDIA Corporation",
        "isin": "US67066G1040",
        "wkn": "918422",
        "us_symbol": "NVDA",
        "xetra_symbol": "NVD.DE",
        "gettex_symbol": "NVD.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },
    "AAPL": {
        "company_name": "Apple Inc.",
        "isin": "US0378331005",
        "wkn": "865985",
        "us_symbol": "AAPL",
        "xetra_symbol": "APC.DE",
        "gettex_symbol": "APC.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },
    "TSLA": {
        "company_name": "Tesla, Inc.",
        "isin": "US88160R1014",
        "wkn": "A1CX3T",
        "us_symbol": "TSLA",
        "xetra_symbol": "TL0.DE",
        "gettex_symbol": "TL0.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },
    "MSFT": {
        "company_name": "Microsoft Corporation",
        "isin": "US5949181045",
        "wkn": "870747",
        "us_symbol": "MSFT",
        "xetra_symbol": "MSF.DE",
        "gettex_symbol": "MSF.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },
    "AMZN": {
        "company_name": "Amazon.com, Inc.",
        "isin": "US0231351067",
        "wkn": "906866",
        "us_symbol": "AMZN",
        "xetra_symbol": "AMZ.DE",
        "gettex_symbol": "AMZ.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },
    "GOOGL": {
        "company_name": "Alphabet Inc. (Class A)",
        "isin": "US02079K3059",
        "wkn": "A14Y6F",
        "us_symbol": "GOOGL",
        "xetra_symbol": "ABEA.DE",
        "gettex_symbol": "ABEA.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },
    "GOOG": {
        "company_name": "Alphabet Inc. (Class C)",
        "isin": "US02079K1079",
        "wkn": "A14Y6H",
        "us_symbol": "GOOG",
        "xetra_symbol": "ABEC.DE",
        "gettex_symbol": "ABEC.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },
    "META": {
        "company_name": "Meta Platforms, Inc.",
        "isin": "US30303M1027",
        "wkn": "A1JWVX",
        "us_symbol": "META",
        "xetra_symbol": "FB2A.DE",
        "gettex_symbol": "FB2A.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },
    "AMD": {
        "company_name": "Advanced Micro Devices, Inc.",
        "isin": "US0079031078",
        "wkn": "863186",
        "us_symbol": "AMD",
        "xetra_symbol": "AMD.DE",
        "gettex_symbol": "AMD.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },
    "INTC": {
        "company_name": "Intel Corporation",
        "isin": "US4581401001",
        "wkn": "855681",
        "us_symbol": "INTC",
        "xetra_symbol": "INL.DE",
        "gettex_symbol": "INL.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },
    "COIN": {
        "company_name": "Coinbase Global, Inc.",
        "isin": "US19260Q1076",
        "wkn": "A2QP7J",
        "us_symbol": "COIN",
        "xetra_symbol": "1QZ.DE",
        "gettex_symbol": "1QZ.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },
    "AVGO": {
        "company_name": "Broadcom Inc.",
        "isin": "US11135F1012",
        "wkn": "A2JG9Z",
        "us_symbol": "AVGO",
        "xetra_symbol": "1YD.DE",
        "gettex_symbol": "1YD.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },
    "PLTR": {
        "company_name": "Palantir Technologies Inc.",
        "isin": "US69608A1088",
        "wkn": "A2QA4J",
        "us_symbol": "PLTR",
        "xetra_symbol": "PTX.F",
        "gettex_symbol": "PTX.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },
    "BABA": {
        "company_name": "Alibaba Group Holding",
        "isin": "US01609W1027",
        "wkn": "A117ME",
        "us_symbol": "BABA",
        "xetra_symbol": "AHLA.DE",
        "gettex_symbol": "AHLA.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },
    "BAC": {
        "company_name": "Bank of America Corp.",
        "isin": "US0605051046",
        "wkn": "858388",
        "us_symbol": "BAC",
        "xetra_symbol": "NCB.DE",
        "gettex_symbol": "NCB.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },
    "NFLX": {
        "company_name": "Netflix Inc.",
        "isin": "US64110L1061",
        "wkn": "552484",
        "us_symbol": "NFLX",
        "xetra_symbol": "NFC.DE",
        "gettex_symbol": "NFC.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },
    "SMCI": {
        "company_name": "Super Micro Computer, Inc.",
        "isin": "US86800U1043",
        "wkn": "A0MKJF",
        "us_symbol": "SMCI",
        "xetra_symbol": "4I1.DE",
        "gettex_symbol": "4I1.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },
    "MSTR": {
        "company_name": "MicroStrategy Inc.",
        "isin": "US5949724083",
        "wkn": "722713",
        "us_symbol": "MSTR",
        "xetra_symbol": "M9Y.DE",
        "gettex_symbol": "M9Y.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },
    "QCOM": {
        "company_name": "Qualcomm Inc.",
        "isin": "US7475251036",
        "wkn": "883121",
        "us_symbol": "QCOM",
        "xetra_symbol": "QCI.DE",
        "gettex_symbol": "QCI.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },
    "ARM": {
        "company_name": "Arm Holdings plc",
        "isin": "US0420682058",
        "wkn": "A3ES4P",
        "us_symbol": "ARM",
        "xetra_symbol": "09T.DE",
        "gettex_symbol": "09T.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },
    "MU": {
        "company_name": "Micron Technology, Inc.",
        "isin": "US5951121038",
        "wkn": "869020",
        "us_symbol": "MU",
        "xetra_symbol": "MQN.DE",
        "gettex_symbol": "MQN.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },
    "CRWD": {
        "company_name": "CrowdStrike Holdings",
        "isin": "US22788C1053",
        "wkn": "A2PK2R",
        "us_symbol": "CRWD",
        "xetra_symbol": "5CR.DE",
        "gettex_symbol": "5CR.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },
    "PANW": {
        "company_name": "Palo Alto Networks",
        "isin": "US6974351057",
        "wkn": "A1JZ0Q",
        "us_symbol": "PANW",
        "xetra_symbol": "5PW.DE",
        "gettex_symbol": "5PW.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },
    "CRM": {
        "company_name": "Salesforce Inc.",
        "isin": "US79466L3024",
        "wkn": "A0B87V",
        "us_symbol": "CRM",
        "xetra_symbol": "FOO.DE",
        "gettex_symbol": "FOO.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },
    "ADBE": {
        "company_name": "Adobe Inc.",
        "isin": "US00724F1012",
        "wkn": "871981",
        "us_symbol": "ADBE",
        "xetra_symbol": "ADB.DE",
        "gettex_symbol": "ADB.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },
    "ORCL": {
        "company_name": "Oracle Corporation",
        "isin": "US68389X1054",
        "wkn": "871460",
        "us_symbol": "ORCL",
        "xetra_symbol": "ORC.DE",
        "gettex_symbol": "ORC.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },
    "CSCO": {
        "company_name": "Cisco Systems Inc.",
        "isin": "US17275R1023",
        "wkn": "878841",
        "us_symbol": "CSCO",
        "xetra_symbol": "CIS.DE",
        "gettex_symbol": "CIS.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },
    "PYPL": {
        "company_name": "PayPal Holdings, Inc.",
        "isin": "US70450Y1038",
        "wkn": "A14R7U",
        "us_symbol": "PYPL",
        "xetra_symbol": "2PP.DE",
        "gettex_symbol": "2PP.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },

    # ------------------ European Native Leaders (DAX / Euro Stoxx) ------------------
    "SAP": {
        "company_name": "SAP SE",
        "isin": "DE0007164600",
        "wkn": "716460",
        "us_symbol": "SAP",
        "xetra_symbol": "SAP.DE",
        "gettex_symbol": "SAP.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },
    "SIE": {
        "company_name": "Siemens AG",
        "isin": "DE0007236101",
        "wkn": "723610",
        "us_symbol": "SIEGY",
        "xetra_symbol": "SIE.DE",
        "gettex_symbol": "SIE.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },
    "RHM": {
        "company_name": "Rheinmetall AG",
        "isin": "DE0007030009",
        "wkn": "703000",
        "us_symbol": "RNMBY",
        "xetra_symbol": "RHM.DE",
        "gettex_symbol": "RHM.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },
    "ALV": {
        "company_name": "Allianz SE",
        "isin": "DE0008404005",
        "wkn": "840400",
        "us_symbol": "ALIZY",
        "xetra_symbol": "ALV.DE",
        "gettex_symbol": "ALV.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },
    "DTE": {
        "company_name": "Deutsche Telekom AG",
        "isin": "DE0005557508",
        "wkn": "555750",
        "us_symbol": "DTEGY",
        "xetra_symbol": "DTE.DE",
        "gettex_symbol": "DTE.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },
    "MBG": {
        "company_name": "Mercedes-Benz Group AG",
        "isin": "DE0007100000",
        "wkn": "710000",
        "us_symbol": "MBGYY",
        "xetra_symbol": "MBG.DE",
        "gettex_symbol": "MBG.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },
    "BMW": {
        "company_name": "Bayerische Motoren Werke AG",
        "isin": "DE0005190003",
        "wkn": "519000",
        "us_symbol": "BMWYY",
        "xetra_symbol": "BMW.DE",
        "gettex_symbol": "BMW.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },
    "AIR": {
        "company_name": "Airbus SE",
        "isin": "NL0000235190",
        "wkn": "938914",
        "us_symbol": "EADSY",
        "xetra_symbol": "AIR.DE",
        "gettex_symbol": "AIR.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },
    "IFX": {
        "company_name": "Infineon Technologies AG",
        "isin": "DE0006231004",
        "wkn": "623100",
        "us_symbol": "IFNNY",
        "xetra_symbol": "IFX.DE",
        "gettex_symbol": "IFX.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },
    "BAS": {
        "company_name": "BASF SE",
        "isin": "DE000BASF111",
        "wkn": "BASF11",
        "us_symbol": "BASFY",
        "xetra_symbol": "BAS.DE",
        "gettex_symbol": "BAS.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },
    "BAYN": {
        "company_name": "Bayer AG",
        "isin": "DE000BAY0017",
        "wkn": "BAY001",
        "us_symbol": "BAYRY",
        "xetra_symbol": "BAYN.DE",
        "gettex_symbol": "BAYN.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },
    "VOW3": {
        "company_name": "Volkswagen AG (Vorzüge)",
        "isin": "DE0007664039",
        "wkn": "766403",
        "us_symbol": "VWAGY",
        "xetra_symbol": "VOW3.DE",
        "gettex_symbol": "VOW3.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },
    "MUV2": {
        "company_name": "Münchener Rückversicherungs-Gesellschaft",
        "isin": "DE0008430026",
        "wkn": "843002",
        "us_symbol": "MURGY",
        "xetra_symbol": "MUV2.DE",
        "gettex_symbol": "MUV2.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },
    "DBK": {
        "company_name": "Deutsche Bank AG",
        "isin": "DE0005140008",
        "wkn": "514000",
        "us_symbol": "DB",
        "xetra_symbol": "DBK.DE",
        "gettex_symbol": "DBK.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },
    "ASML": {
        "company_name": "ASML Holding N.V.",
        "isin": "NL0010273215",
        "wkn": "A1J4U4",
        "us_symbol": "ASML",
        "xetra_symbol": "ASML.AS",
        "gettex_symbol": "ASME.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },
    "MC": {
        "company_name": "LVMH Moët Hennessy Louis Vuitton",
        "isin": "FR0000121014",
        "wkn": "853292",
        "us_symbol": "LVMUY",
        "xetra_symbol": "MC.PA",
        "gettex_symbol": "MOH.MU",
        "currency": "EUR",
        "currency_symbol": "€",
    },
}

# Inverted indices for O(1) resolution
_ISIN_MAP: Dict[str, Dict[str, Any]] = {
    data["isin"]: data for data in SECURITIES_DIRECTORY.values() if data.get("isin")
}
_WKN_MAP: Dict[str, Dict[str, Any]] = {
    data["wkn"]: data for data in SECURITIES_DIRECTORY.values() if data.get("wkn")
}
_XETRA_MAP: Dict[str, Dict[str, Any]] = {
    data["xetra_symbol"]: data for data in SECURITIES_DIRECTORY.values() if data.get("xetra_symbol")
}
_GETTEX_MAP: Dict[str, Dict[str, Any]] = {
    data["gettex_symbol"]: data for data in SECURITIES_DIRECTORY.values() if data.get("gettex_symbol")
}

# European exchange suffixes recognized by Yahoo Finance
EUROPEAN_EXCHANGE_SUFFIXES = {
    ".DE": ("XETRA (Frankfurt)", "EUR", "€"),
    ".MU": ("Börse München (gettex)", "EUR", "€"),
    ".F": ("Börse Frankfurt (Floor)", "EUR", "€"),
    ".TG": ("Tradegate Exchange", "EUR", "€"),
    ".HA": ("Börse Hannover (EIX)", "EUR", "€"),
    ".HM": ("Börse Hamburg", "EUR", "€"),
    ".DU": ("Börse Düsseldorf", "EUR", "€"),
    ".BE": ("Börse Berlin", "EUR", "€"),
    ".SG": ("Börse Stuttgart", "EUR", "€"),
    ".AS": ("Euronext Amsterdam", "EUR", "€"),
    ".PA": ("Euronext Paris", "EUR", "€"),
    ".BR": ("Euronext Brussels", "EUR", "€"),
    ".LS": ("Euronext Lisbon", "EUR", "€"),
    ".MI": ("Borsa Italiana (Milan)", "EUR", "€"),
    ".MC": ("Bolsa de Madrid", "EUR", "€"),
    ".VI": ("Wiener Börse (Vienna)", "EUR", "€"),
    ".SW": ("SIX Swiss Exchange", "CHF", "CHF"),
    ".L": ("London Stock Exchange", "GBP", "£"),
}

# Cache for dynamic online ISIN search results
_SEARCH_CACHE: Dict[str, Optional[str]] = {}


def is_isin(query: str) -> bool:
    """Checks if query matches the 12-character alphanumeric ISIN standard."""
    return bool(re.match(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$", query.strip().upper()))


def is_wkn(query: str) -> bool:
    """Checks if query matches the 6-character German WKN standard."""
    return bool(re.match(r"^[A-Z0-9]{6}$", query.strip().upper()))


def is_european_symbol(symbol: str) -> bool:
    """Checks if ticker symbol contains a known European exchange suffix."""
    sym_upper = symbol.strip().upper()
    return any(sym_upper.endswith(suffix) for suffix in EUROPEAN_EXCHANGE_SUFFIXES)


def get_currency_for_symbol(symbol: str) -> Tuple[str, str]:
    """Returns (currency_code, currency_symbol) for a given stock symbol."""
    sym_upper = symbol.strip().upper()
    for suffix, (_, curr, curr_sym) in EUROPEAN_EXCHANGE_SUFFIXES.items():
        if sym_upper.endswith(suffix):
            return curr, curr_sym
    return "USD", "$"


def lookup_isin_online(isin: str) -> Optional[str]:
    """
    Queries Yahoo Finance Search API to dynamically resolve unlisted ISINs to ticker symbols.
    """
    clean_isin = isin.strip().upper()
    if clean_isin in _SEARCH_CACHE:
        return _SEARCH_CACHE[clean_isin]

    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={clean_isin}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    try:
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            quotes = data.get("quotes", [])
            for q in quotes:
                sym = q.get("symbol")
                if sym:
                    _SEARCH_CACHE[clean_isin] = sym
                    return sym
    except Exception:
        pass

    _SEARCH_CACHE[clean_isin] = None
    return None


def resolve_symbol(
    query: str,
    prefer_exchange: str = "AUTO",
    force_european: bool = False,
) -> Dict[str, Any]:
    """
    Master symbol resolver for European and US markets.
    
    Accepts:
      - ISIN: e.g. 'US67066G1040', 'DE0007164600', 'NL0010273215'
      - WKN: e.g. '716460', '918422'
      - European ticker: e.g. 'SAP.DE', 'NVD.MU', 'ASML.AS'
      - Standard US ticker: e.g. 'NVDA', 'AAPL', 'TSLA', 'MU'
      
    Args:
      query: The input identifier string.
      prefer_exchange: Preferred exchange venue: 'AUTO' (intelligent routing),
                       'DE' for XETRA (reference market), 'MU' for gettex (Börse München),
                       or 'US' for native US exchanges.
      force_european: When True, always converts US tickers to European dual-listings.
      
    Returns:
      Comprehensive dict containing resolved ticker, ISIN, company name, currency, and venues.
    """
    raw = query.strip()
    norm = raw.upper()

    pref_upper = prefer_exchange.upper().lstrip(".")
    if pref_upper in ("MU", "GETTEX"):
        target_suffix = ".MU"
        default_venue = "Börse München (gettex)"
    else:
        target_suffix = ".DE"
        default_venue = "XETRA (Deutsche Börse)"

    # 1. Exact match in ISIN directory
    if norm in _ISIN_MAP:
        rec = _ISIN_MAP[norm]
        chosen_symbol = rec["gettex_symbol"] if target_suffix == ".MU" else rec["xetra_symbol"]
        return {
            "query": raw,
            "symbol": chosen_symbol,
            "underlying_symbol": rec["us_symbol"],
            "company_name": rec["company_name"],
            "isin": rec["isin"],
            "wkn": rec["wkn"],
            "currency": rec["currency"],
            "currency_symbol": rec["currency_symbol"],
            "exchange": default_venue,
            "is_european": True,
            "xetra_symbol": rec["xetra_symbol"],
            "gettex_symbol": rec["gettex_symbol"],
        }

    # 2. Exact match in WKN directory
    if norm in _WKN_MAP:
        rec = _WKN_MAP[norm]
        chosen_symbol = rec["gettex_symbol"] if target_suffix == ".MU" else rec["xetra_symbol"]
        return {
            "query": raw,
            "symbol": chosen_symbol,
            "underlying_symbol": rec["us_symbol"],
            "company_name": rec["company_name"],
            "isin": rec["isin"],
            "wkn": rec["wkn"],
            "currency": rec["currency"],
            "currency_symbol": rec["currency_symbol"],
            "exchange": default_venue,
            "is_european": True,
            "xetra_symbol": rec["xetra_symbol"],
            "gettex_symbol": rec["gettex_symbol"],
        }

    # 3. Exact match in XETRA or gettex symbol directories
    if norm in _XETRA_MAP or norm in _GETTEX_MAP:
        rec = _XETRA_MAP.get(norm) or _GETTEX_MAP.get(norm)
        chosen_symbol = norm
        venue = "Börse München (gettex)" if norm.endswith(".MU") else "XETRA (Deutsche Börse)"
        return {
            "query": raw,
            "symbol": chosen_symbol,
            "underlying_symbol": rec["us_symbol"],
            "company_name": rec["company_name"],
            "isin": rec["isin"],
            "wkn": rec["wkn"],
            "currency": "EUR",
            "currency_symbol": "€",
            "exchange": venue,
            "is_european": True,
            "xetra_symbol": rec["xetra_symbol"],
            "gettex_symbol": rec["gettex_symbol"],
        }

    # 4. Input already has European exchange suffix (e.g. BMW.DE, SAP.MU, ASML.AS)
    if is_european_symbol(norm):
        base_tick = norm.split(".")[0]
        rec = SECURITIES_DIRECTORY.get(base_tick, {})
        curr, curr_sym = get_currency_for_symbol(norm)
        venue = "Börse München (gettex)" if norm.endswith(".MU") else "European Exchange"
        for suf, (ven, _, _) in EUROPEAN_EXCHANGE_SUFFIXES.items():
            if norm.endswith(suf):
                venue = ven
                break

        return {
            "query": raw,
            "symbol": norm,
            "underlying_symbol": rec.get("us_symbol") or base_tick,
            "company_name": rec.get("company_name", norm),
            "isin": rec.get("isin", ""),
            "wkn": rec.get("wkn", ""),
            "currency": curr,
            "currency_symbol": curr_sym,
            "exchange": venue,
            "is_european": True,
            "xetra_symbol": f"{base_tick}.DE",
            "gettex_symbol": f"{base_tick}.MU",
        }

    # 5. Standard ticker present in directory (e.g. 'NVDA', 'SAP', 'MU')
    if norm in SECURITIES_DIRECTORY:
        rec = SECURITIES_DIRECTORY[norm]
        is_us_security = rec.get("isin", "").startswith("US")

        # Determine whether to return European dual-listing or native US ticker
        wants_european = force_european or pref_upper in ("DE", "MU")
        if wants_european:
            chosen_symbol = rec["gettex_symbol"] if target_suffix == ".MU" else rec["xetra_symbol"]
            return {
                "query": raw,
                "symbol": chosen_symbol,
                "underlying_symbol": rec["us_symbol"],
                "company_name": rec["company_name"],
                "isin": rec["isin"],
                "wkn": rec["wkn"],
                "currency": "EUR",
                "currency_symbol": "€",
                "exchange": default_venue,
                "is_european": True,
                "xetra_symbol": rec["xetra_symbol"],
                "gettex_symbol": rec["gettex_symbol"],
            }
        elif is_us_security:
            # Return native US ticker (USD, $)
            return {
                "query": raw,
                "symbol": rec["us_symbol"],
                "underlying_symbol": rec["us_symbol"],
                "company_name": rec["company_name"],
                "isin": rec["isin"],
                "wkn": rec["wkn"],
                "currency": "USD",
                "currency_symbol": "$",
                "exchange": "US Exchange (NYSE/NASDAQ)",
                "is_european": False,
                "xetra_symbol": rec["xetra_symbol"],
                "gettex_symbol": rec["gettex_symbol"],
            }
        else:
            # Native European security (e.g. SAP, BMW) defaults to European exchange
            chosen_symbol = rec["gettex_symbol"] if target_suffix == ".MU" else rec["xetra_symbol"]
            return {
                "query": raw,
                "symbol": chosen_symbol,
                "underlying_symbol": rec["us_symbol"],
                "company_name": rec["company_name"],
                "isin": rec["isin"],
                "wkn": rec["wkn"],
                "currency": rec.get("currency", "EUR"),
                "currency_symbol": rec.get("currency_symbol", "€"),
                "exchange": default_venue,
                "is_european": True,
                "xetra_symbol": rec["xetra_symbol"],
                "gettex_symbol": rec["gettex_symbol"],
            }

    # 6. Dynamic online ISIN lookup fallback
    if is_isin(norm):
        online_sym = lookup_isin_online(norm)
        if online_sym:
            curr, curr_sym = get_currency_for_symbol(online_sym)
            return {
                "query": raw,
                "symbol": online_sym,
                "underlying_symbol": online_sym.split(".")[0],
                "company_name": online_sym,
                "isin": norm,
                "wkn": "",
                "currency": curr,
                "currency_symbol": curr_sym,
                "exchange": "Resolved via ISIN Search",
                "is_european": is_european_symbol(online_sym),
                "xetra_symbol": online_sym if online_sym.endswith(".DE") else f"{online_sym.split('.')[0]}.DE",
                "gettex_symbol": online_sym if online_sym.endswith(".MU") else f"{online_sym.split('.')[0]}.MU",
            }

    # 7. Default fallback: treat as symbol
    curr, curr_sym = get_currency_for_symbol(norm)
    return {
        "query": raw,
        "symbol": norm,
        "underlying_symbol": norm,
        "company_name": norm,
        "isin": "",
        "wkn": "",
        "currency": curr,
        "currency_symbol": curr_sym,
        "exchange": "US / International",
        "is_european": is_european_symbol(norm),
        "xetra_symbol": f"{norm}.DE",
        "gettex_symbol": f"{norm}.MU",
    }


def get_european_default_universe(exchange: str = "DE") -> List[str]:
    """
    Returns the default universe of highest-volume European leaders and gettex dual-listings.
    """
    tickers = []
    # Primary German & European blue chips
    native_keys = ["SAP", "SIE", "RHM", "ALV", "DTE", "MBG", "BMW", "AIR", "IFX", "BAS", "ASML", "MC"]
    for k in native_keys:
        item = SECURITIES_DIRECTORY.get(k)
        if item:
            tickers.append(item["gettex_symbol"] if exchange.upper() == "MU" else item["xetra_symbol"])

    # High-volume gettex US dual-listings
    us_keys = ["NVDA", "AAPL", "TSLA", "MSFT", "AMZN", "META", "AMD", "COIN"]
    for k in us_keys:
        item = SECURITIES_DIRECTORY.get(k)
        if item:
            tickers.append(item["gettex_symbol"] if exchange.upper() == "MU" else item["xetra_symbol"])

    return tickers


def get_nasdaq_european_universe(exchange: str = "DE") -> List[str]:
    """
    Returns the universe of top US/NASDAQ tech leaders dual-listed on European exchanges
    (gettex / XETRA) trading and settled in EUR (€).
    """
    us_keys = [
        "NVDA", "AAPL", "TSLA", "MSFT", "AMZN", "GOOGL", "META", "AMD",
        "PLTR", "COIN", "AVGO", "NFLX", "SMCI", "MSTR", "QCOM", "ARM",
        "MU", "CRWD", "PANW", "CRM", "ADBE", "ORCL", "CSCO", "INTC"
    ]
    tickers = []
    for k in us_keys:
        item = SECURITIES_DIRECTORY.get(k)
        if item:
            tickers.append(item["gettex_symbol"] if exchange.upper() == "MU" else item["xetra_symbol"])
        else:
            tickers.append(f"{k}.MU" if exchange.upper() == "MU" else f"{k}.DE")
    return tickers



if __name__ == "__main__":
    # Test suite demonstration
    test_queries = [
        "US67066G1040",   # Nvidia ISIN
        "DE0007164600",   # SAP ISIN
        "NL0010273215",   # ASML ISIN
        "716460",         # SAP WKN
        "NVDA",           # US ticker -> should resolve to NVD.DE
        "SAP.DE",         # XETRA ticker
        "NVD.MU",         # gettex ticker
        "RHM.DE",         # Rheinmetall
    ]
    print("\n🔍 TICKER / ISIN RESOLVER TEST RUN:")
    print("=" * 80)
    for tq in test_queries:
        res = resolve_symbol(tq, prefer_exchange="DE", force_european=True)
        print(f"Input: {tq:<14} -> Symbol: {res['symbol']:<10} | Company: {res['company_name']:<25} | Currency: {res['currency_symbol']} ({res['currency']}) | Venue: {res['exchange']}")
    print("=" * 80)
