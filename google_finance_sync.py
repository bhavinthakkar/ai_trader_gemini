#!/usr/bin/env python3
"""
Google Finance & Google Sheets Portfolio Synchronization Service
==================================================================
Allows users to automatically synchronize their investment portfolio / watchlist
from Google Finance (or an exported/shared Google Sheet) directly into the
AI Trader analysis pipeline.

Key Capabilities:
1. Google Sheet CSV Converter:
   Transforms any standard shared Google Sheet URL into a direct CSV export URL:
   https://docs.google.com/spreadsheets/d/{ID}/edit -> https://docs.google.com/spreadsheets/d/{ID}/export?format=csv
2. Ticker Cleaning & Normalization:
   Strips prefixes like 'NASDAQ:NVDA' -> 'NVDA', 'NYSE:PLTR' -> 'PLTR'.
3. Automatic European Gettex / XETRA Alignment:
   Resolves US/NASDAQ portfolio tickers into European listings (e.g. NVD.DE, TL0.DE)
   priced in EUR (€) for gettex / EIX European brokers.
4. Persistent Portfolio Cache:
   Saves synced portfolio to 'google_portfolio.json' for fast local access and CLI runs.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple
import requests

from ticker_resolver import resolve_symbol, is_european_symbol, get_currency_for_symbol

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PORTFOLIO_CACHE_FILE = os.path.join(BASE_DIR, "google_portfolio.json")
CONFIG_FILE = os.path.join(BASE_DIR, "google_portfolio_config.json")


def convert_to_google_sheet_csv_url(url: str) -> Optional[str]:
    """
    Extracts spreadsheet ID and gid from a Google Sheets URL and converts it
    to a direct CSV download link.
    Supported inputs:
    - https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit?usp=sharing
    - https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit#gid=0
    - https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/export?format=csv
    """
    if not url:
        return None

    url = url.strip()
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    if not match:
        return None

    sheet_id = match.group(1)

    # Extract gid (tab id) if present
    gid = "0"
    gid_match = re.search(r"[?#&]gid=([0-9]+)", url)
    if gid_match:
        gid = gid_match.group(1)

    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"


def clean_ticker_symbol(raw_sym: str) -> str:
    """
    Cleans Google Finance ticker symbols by stripping exchange prefixes and quotes.
    Examples:
    - 'NASDAQ:NVDA' -> 'NVDA'
    - 'NYSE:PLTR' -> 'PLTR'
    - 'FRA:NVD' -> 'NVD.DE'
    - 'ETR:SAP' -> 'SAP.DE'
    - ' \"AAPL\" ' -> 'AAPL'
    """
    if not raw_sym:
        return ""
    s = str(raw_sym).strip().strip("\"'").upper()

    # Handle Google Finance exchange prefixes
    if ":" in s:
        parts = s.split(":", 1)
        prefix, symbol = parts[0].strip(), parts[1].strip()
        if prefix in ["NASDAQ", "NYSE", "BATS", "AMEX", "INDEXSP", "INDEXNASDAQ"]:
            return symbol
        elif prefix in ["FRA", "ETR", "GER"]:
            return f"{symbol}.DE" if not symbol.endswith(".DE") else symbol
        elif prefix == "LON":
            return f"{symbol}.L" if not symbol.endswith(".L") else symbol
        return symbol

    return s


def parse_csv_content(csv_text: str) -> List[Dict[str, Any]]:
    """
    Parses CSV text into a structured list of portfolio holdings.
    Intelligently identifies symbol, shares, and purchase price columns.
    """
    lines = [line for line in csv_text.strip().splitlines() if line.strip()]
    if not lines:
        return []

    # Try DictReader first
    reader = csv.reader(lines)
    header = [h.strip().lower() for h in next(reader, [])]

    symbol_col_idx = -1
    shares_col_idx = -1
    price_col_idx = -1
    name_col_idx = -1

    for idx, col in enumerate(header):
        if col in ["symbol", "ticker", "code", "isin", "stock", "instrument"]:
            symbol_col_idx = idx
        elif col in ["shares", "qty", "quantity", "holding", "units"]:
            shares_col_idx = idx
        elif col in ["purchase price", "cost per share", "avg price", "buy price", "purchase_price", "price"]:
            price_col_idx = idx
        elif col in ["name", "company", "description", "security name"]:
            name_col_idx = idx

    # If header didn't specify symbol, check first column
    if symbol_col_idx == -1:
        symbol_col_idx = 0

    holdings = []
    for row in reader:
        if not row or len(row) <= symbol_col_idx:
            continue
        raw_sym = row[symbol_col_idx].strip()
        if not raw_sym or raw_sym.lower() in ["symbol", "ticker", "total", "cash"]:
            continue

        sym = clean_ticker_symbol(raw_sym)
        if not sym or len(sym) > 16:
            continue

        shares = None
        if shares_col_idx != -1 and len(row) > shares_col_idx:
            try:
                shares = float(row[shares_col_idx].replace(",", "").replace("$", "").replace("€", "").strip() or 0)
            except ValueError:
                shares = None

        cost_price = None
        if price_col_idx != -1 and len(row) > price_col_idx:
            try:
                cost_price = float(row[price_col_idx].replace(",", "").replace("$", "").replace("€", "").strip() or 0)
            except ValueError:
                cost_price = None

        company_name = ""
        if name_col_idx != -1 and len(row) > name_col_idx:
            company_name = row[name_col_idx].strip()

        holdings.append({
            "raw_symbol": raw_sym,
            "us_symbol": sym,
            "shares": shares,
            "purchase_price": cost_price,
            "name": company_name,
        })

    return holdings


def fetch_portfolio_from_google_sheet(sheet_url: str) -> List[Dict[str, Any]]:
    """
    Downloads CSV from a public/shared Google Sheet and parses portfolio holdings.
    """
    csv_url = convert_to_google_sheet_csv_url(sheet_url)
    if not csv_url:
        raise ValueError(f"Invalid Google Sheet URL format: '{sheet_url}'")

    print(f"[GoogleFinanceSync] Fetching Google Sheet CSV from: {csv_url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    resp = requests.get(csv_url, headers=headers, timeout=15)
    if resp.status_code != 200:
        raise ConnectionError(
            f"Failed to fetch Google Sheet (HTTP {resp.status_code}). "
            "Please ensure the sheet is set to 'Anyone with the link can view'."
        )

    return parse_csv_content(resp.text)


def fetch_portfolio_from_google_finance_html(gf_url: str) -> List[Dict[str, Any]]:
    """
    Fetches and extracts tickers from a public Google Finance watchlist/portfolio page.
    """
    print(f"[GoogleFinanceSync] Fetching Google Finance page from: {gf_url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    resp = requests.get(gf_url, headers=headers, timeout=15)
    if resp.status_code != 200:
        raise ConnectionError(f"Failed to fetch Google Finance URL (HTTP {resp.status_code})")

    html = resp.text
    # Extract ticker patterns like data-symbol="NASDAQ:NVDA" or class="..."><div class="...">NVDA</div>
    matches = re.findall(r'data-symbol="([^"]+)"', html)
    if not matches:
        matches = re.findall(r'/quote/([A-Z0-9_\-\.]+:[A-Z0-9_\-\.]+)', html)

    found_symbols = []
    seen = set()
    for m in matches:
        cleaned = clean_ticker_symbol(m)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            found_symbols.append({
                "raw_symbol": m,
                "us_symbol": cleaned,
                "shares": None,
                "purchase_price": None,
                "name": cleaned,
            })

    return found_symbols


def sync_google_portfolio(
    url: str,
    prefer_exchange: str = "DE",
    force_european: bool = True
) -> Dict[str, Any]:
    """
    Main sync engine:
    1. Fetches holdings from Google Sheet or Google Finance URL.
    2. Automatically resolves US/NASDAQ tickers to European gettex/XETRA listings in EUR (€).
    3. Caches payload locally in 'google_portfolio.json'.
    4. Saves configuration in 'google_portfolio_config.json'.
    """
    url = url.strip()
    if not url:
        raise ValueError("URL cannot be empty.")

    if "spreadsheets" in url:
        raw_holdings = fetch_portfolio_from_google_sheet(url)
    elif "finance" in url:
        raw_holdings = fetch_portfolio_from_google_finance_html(url)
    else:
        # Fallback: try Google Sheet CSV conversion first, then raw request
        raw_holdings = fetch_portfolio_from_google_sheet(url)

    if not raw_holdings:
        raise ValueError("No stock holdings or ticker symbols could be extracted from the provided URL.")

    resolved_items = []
    for h in raw_holdings:
        us_sym = h["us_symbol"]
        res = resolve_symbol(us_sym, prefer_exchange=prefer_exchange, force_european=force_european)
        resolved_items.append({
            "symbol": res["symbol"],
            "us_symbol": res.get("underlying_symbol", us_sym),
            "company_name": res.get("company_name", h.get("name") or us_sym),
            "currency": res.get("currency", "EUR"),
            "currency_symbol": res.get("currency_symbol", "€"),
            "exchange": res.get("exchange", "gettex / XETRA"),
            "isin": res.get("isin", ""),
            "shares": h.get("shares"),
            "purchase_price": h.get("purchase_price"),
            "is_european": res.get("is_european", True),
        })

    payload = {
        "url": url,
        "prefer_exchange": prefer_exchange,
        "synced_at": __import__("datetime").datetime.now().isoformat(),
        "total_holdings": len(resolved_items),
        "holdings": resolved_items,
    }

    # Save to local cache
    try:
        with open(PORTFOLIO_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump({"url": url, "prefer_exchange": prefer_exchange}, f, indent=2)
    except Exception as e:
        print(f"[GoogleFinanceSync] Notice: could not save cache: {e}")

    return payload


def save_custom_portfolio_symbols(
    symbols: List[str],
    prefer_exchange: str = "DE",
    force_european: bool = True
) -> Dict[str, Any]:
    """
    Saves an explicit list of ticker symbols or ISINs directly to google_portfolio.json,
    resolving each into EUR (€) on gettex / XETRA.
    """
    resolved_items = []
    seen = set()
    for s in symbols:
        sym = clean_ticker_symbol(s)
        if not sym or sym in seen:
            continue
        seen.add(sym)
        res = resolve_symbol(sym, prefer_exchange=prefer_exchange, force_european=force_european)
        resolved_items.append({
            "symbol": res["symbol"],
            "us_symbol": res.get("underlying_symbol", sym),
            "company_name": res.get("company_name", sym),
            "currency": res.get("currency", "EUR"),
            "currency_symbol": res.get("currency_symbol", "€"),
            "exchange": res.get("exchange", "gettex / XETRA"),
            "isin": res.get("isin", ""),
            "shares": None,
            "purchase_price": None,
            "is_european": res.get("is_european", True),
        })

    payload = {
        "url": "Direct Watchlist Input",
        "prefer_exchange": prefer_exchange,
        "synced_at": __import__("datetime").datetime.now().isoformat(),
        "total_holdings": len(resolved_items),
        "holdings": resolved_items,
    }

    try:
        with open(PORTFOLIO_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump({"url": "Direct Watchlist Input", "prefer_exchange": prefer_exchange}, f, indent=2)
        print(f"[GoogleFinanceSync] Successfully saved {len(resolved_items)} portfolio holdings to '{PORTFOLIO_CACHE_FILE}'.")
    except Exception as e:
        print(f"[GoogleFinanceSync] Notice: could not save cache: {e}")

    return payload



def load_cached_portfolio() -> Optional[Dict[str, Any]]:
    """Loads the cached Google Finance portfolio if it exists."""
    if os.path.exists(PORTFOLIO_CACHE_FILE):
        try:
            with open(PORTFOLIO_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def get_saved_portfolio_config() -> Optional[Dict[str, str]]:
    """Loads the saved portfolio URL configuration."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def get_portfolio_symbols(prefer_exchange: str = "DE") -> List[str]:
    """
    Returns list of active portfolio symbols (e.g. ['NVD.DE', 'TL0.DE', 'APC.DE']).
    Falls back to a core US Tech / gettex universe if no portfolio is synced.
    """
    cached = load_cached_portfolio()
    if cached and cached.get("holdings"):
        return [h["symbol"] for h in cached["holdings"] if h.get("symbol")]

    # Default fallback: Top NASDAQ leaders on gettex / XETRA
    default_us_tech = ["NVDA", "AAPL", "TSLA", "MSFT", "AMZN", "META", "GOOGL", "AMD", "PLTR", "COIN"]
    return [resolve_symbol(s, prefer_exchange=prefer_exchange, force_european=True)["symbol"] for s in default_us_tech]


if __name__ == "__main__":
    # Test sample parser
    sample_csv = """Symbol,Name,Shares,Purchase Price
NASDAQ:NVDA,NVIDIA Corporation,10,120.50
NASDAQ:AAPL,Apple Inc.,15,185.00
NASDAQ:TSLA,Tesla Inc.,8,220.00
NASDAQ:PLTR,Palantir Technologies,50,28.00
NASDAQ:AMD,Advanced Micro Devices,12,145.00
"""
    print("Testing Google Finance CSV parser...")
    parsed = parse_csv_content(sample_csv)
    print(f"Extracted {len(parsed)} holdings:")
    for item in parsed:
        res = resolve_symbol(item["us_symbol"], prefer_exchange="DE", force_european=True)
        print(f"  • {item['raw_symbol']} -> {res['symbol']} ({res['currency_symbol']}{res['currency']}) | {res['company_name']}")
