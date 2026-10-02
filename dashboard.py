#!/usr/bin/env python3
"""
Gloomberb Market Intelligence Dashboard (Model-Free)
====================================================
Collects real-time market data for the most traded stocks of the day,
analyzes price structure (price, day high/low, open, volume, RVOL, 52w range, technicals),
and identifies the news and catalysts explaining why they are experiencing heavy volume.

Outputs:
  - Terminal interactive dashboard (via Rich)
  - Standalone modern HTML dashboard (dashboard.html)
  - Structured JSON export (optional via --json)
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import requests
import yfinance as yf
from dotenv import load_dotenv

from market_agent import MarketAgent
from news_agent import NewsAgent
from gloomberb_service import GloomberbService
from ticker_resolver import (
    resolve_symbol,
    get_european_default_universe,
    get_nasdaq_european_universe,
    get_currency_for_symbol,
    is_european_symbol
)
from google_finance_sync import (
    sync_google_portfolio,
    load_cached_portfolio,
    get_portfolio_symbols
)

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

load_dotenv()

# Default fallback universe of highly liquid active market leaders
DEFAULT_ACTIVE_SYMBOLS = [
    "NVDA", "TSLA", "AAPL", "AMD", "PLTR",
    "AMZN", "MSFT", "INTC", "GOOGL", "META",
    "SOFI", "MARA", "COIN", "BAC", "BABA"
]

CATALYST_KEYWORDS: Dict[str, List[str]] = {
    "EARNINGS / FINANCIALS": [
        "earnings", "eps", "revenue", "quarter", "q1", "q2", "q3", "q4", "fiscal",
        "profit", "guidance", "sales", "margin", "top-line", "bottom-line", "beat",
        "miss", "ebitda", "cash flow", "dividend"
    ],
    "ANALYST ACTION": [
        "upgrade", "downgrade", "price target", "analyst", "rating", "overweight",
        "underweight", "outperform", "underperform", "wall street", "initiate",
        "reiterate", "bullish call", "bearish call"
    ],
    "M&A / CONTRACTS": [
        "merger", "acquisition", "acquire", "deal", "buyout", "takeover", "contract",
        "partnership", "partner", "joint venture", "award", "offering", "restructur",
        "stake", "investment"
    ],
    "PRODUCT / AI / TECH": [
        "ai", "chip", "launch", "unveil", "gpu", "model", "patent", "fda", "drug",
        "trial", "phase", "breakthrough", "release", "feature", "platform", "cloud"
    ],
    "REGULATORY / LEGAL": [
        "investigat", "lawsuit", "sec", "doj", "probe", "antitrust", "ban", "tariff",
        "sanction", "fine", "penalty", "subpoena", "ruling", "court", "patent dispute"
    ],
    "MANAGEMENT / INSIDER": [
        "ceo", "cfo", "executive", "resign", "appoint", "step down", "hire", "board",
        "insider", "form 4", "layoff", "job cut", "insider buying", "insider sale"
    ],
    "MACRO / GEOPOLITICAL": [
        "fed", "inflation", "interest rate", "powell", "treasury", "yield", "recession",
        "war", "tension", "middle east", "china", "stimulus", "oil", "energy price"
    ],
}


def format_volume(num: Any) -> str:
    """Format volume numbers with K, M, B suffixes."""
    if num is None or num == "N/A":
        return "N/A"
    try:
        val = float(num)
        if val >= 1e9:
            return f"{val / 1e9:.2f}B"
        elif val >= 1e6:
            return f"{val / 1e6:.1f}M"
        elif val >= 1e3:
            return f"{val / 1e3:.0f}K"
        return f"{int(val)}"
    except (ValueError, TypeError):
        return str(num)


def format_market_cap(num: Any, curr_sym: str = "$") -> str:
    """Format market cap numbers with currency, M, B, T suffixes."""
    if num is None or num == "N/A":
        return "N/A"
    try:
        val = float(num)
        if val >= 1e12:
            return f"{curr_sym}{val / 1e12:.2f}T"
        elif val >= 1e9:
            return f"{curr_sym}{val / 1e9:.2f}B"
        elif val >= 1e6:
            return f"{curr_sym}{val / 1e6:.1f}M"
        elif val >= 1e3:
            return f"{curr_sym}{val / 1e3:.0f}K"
        return f"{curr_sym}{val:.2f}"
    except (ValueError, TypeError):
        return str(num)


def fetch_most_active_quotes(
    limit: int = 10,
    symbols: Optional[List[str]] = None,
    market: str = "US",
    prefer_exchange: str = "DE",
) -> List[Dict[str, Any]]:
    """
    Fetches real-time market data for the most traded stocks of the day.
    Supports US and European (gettex / XETRA) markets and ISIN resolution.
    Priority 1: Custom symbols or ISINs (if supplied)
    Priority 2: European gettex/XETRA universe (if market == 'EU')
    Priority 3: yfinance screener for 'most_actives' (US)
    Priority 4: Gloomberb CLI movers
    Priority 5: Default active universe
    """
    if symbols:
        print(f"[Dashboard] Sourcing data for {len(symbols)} requested tickers/ISINs...")
        items = []
        for raw_sym in symbols[:limit]:
            resolved = resolve_symbol(raw_sym, prefer_exchange=prefer_exchange, force_european=(market.upper() == "EU"))
            sym = resolved["symbol"]
            curr = resolved.get("currency", "USD")
            curr_sym = resolved.get("currency_symbol", "$")
            try:
                t = yf.Ticker(sym)
                info = t.info or {}
                fast_info = getattr(t, "fast_info", {})
                price = (
                    info.get("regularMarketPrice")
                    or fast_info.get("last_price")
                    or info.get("currentPrice")
                )
                if price is not None:
                    raw_prev_close = (
                        info.get("regularMarketPreviousClose")
                        or fast_info.get("previous_close")
                    )
                    raw_change = info.get("regularMarketChange")
                    raw_change_pct = info.get("regularMarketChangePercent")

                    # If previous close is missing or invalid but we have change:
                    if (raw_prev_close is None or raw_prev_close <= 0) and raw_change is not None:
                        calc_prev = price - float(raw_change)
                        if calc_prev > 0:
                            raw_prev_close = calc_prev

                    prev_close = raw_prev_close if (raw_prev_close is not None and raw_prev_close > 0) else price

                    # Compute change if missing
                    if raw_change is not None:
                        change_val = float(raw_change)
                    elif prev_close > 0:
                        change_val = price - prev_close
                    else:
                        change_val = 0.0

                    # Compute mathematical percentage change
                    if prev_close > 0 and abs(price - prev_close) > 1e-6:
                        math_pct = ((price - prev_close) / prev_close) * 100.0
                    else:
                        math_pct = 0.0

                    # Authoritative percentage change (eliminate flawed < 1.0 heuristic multiplier)
                    if raw_change_pct is not None:
                        raw_pct = float(raw_change_pct)
                        if abs(raw_pct - math_pct) < 0.2:
                            change_pct_val = raw_pct
                        elif abs(raw_pct * 100.0 - math_pct) < 0.2:
                            change_pct_val = raw_pct * 100.0
                        elif prev_close > 0 and raw_prev_close is not None and float(raw_prev_close) > 0:
                            change_pct_val = math_pct
                        else:
                            change_pct_val = raw_pct
                    else:
                        change_pct_val = math_pct

                    items.append({
                        "symbol": sym,
                        "shortName": info.get("shortName") or info.get("longName") or resolved.get("company_name") or sym,
                        "regularMarketPrice": price,
                        "regularMarketDayHigh": info.get("regularMarketDayHigh") or fast_info.get("day_high") or price,
                        "regularMarketDayLow": info.get("regularMarketDayLow") or fast_info.get("day_low") or price,
                        "regularMarketOpen": info.get("regularMarketOpen") or fast_info.get("open") or price,
                        "regularMarketPreviousClose": prev_close,
                        "regularMarketVolume": info.get("regularMarketVolume") or fast_info.get("last_volume") or 0,
                        "regularMarketChange": change_val,
                        "regularMarketChangePercent": change_pct_val,
                        "marketCap": info.get("marketCap") or fast_info.get("market_cap"),
                        "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh") or fast_info.get("year_high"),
                        "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow") or fast_info.get("year_low"),
                        "averageDailyVolume3Month": info.get("averageDailyVolume3Month") or fast_info.get("three_month_average_volume"),
                        "currency": curr,
                        "currency_symbol": curr_sym,
                        "isin": resolved.get("isin", ""),
                    })
            except Exception as e:
                print(f"[Dashboard] Error fetching quote for {sym}: {e}")
        if items:
            return items

    # European market mode: query US/NASDAQ tech leaders on gettex / XETRA
    if market.upper() == "EU":
        eu_universe = get_nasdaq_european_universe(exchange=prefer_exchange)
        print(f"[Dashboard] Sourcing European NASDAQ dual-listings (gettex / XETRA: {len(eu_universe)} symbols in EUR)...")
        return fetch_most_active_quotes(limit=limit, symbols=eu_universe, market=market, prefer_exchange=prefer_exchange)

    print(f"[Dashboard] Fetching top {limit} most active stocks via Yahoo Finance screener...")
    try:
        screen_res = yf.screen("most_actives")
        if isinstance(screen_res, dict) and "quotes" in screen_res:
            quotes = screen_res.get("quotes", [])
            valid_quotes = [q for q in quotes if q.get("symbol") and q.get("regularMarketPrice") is not None]
            if valid_quotes:
                print(f"[Dashboard] Successfully retrieved {len(valid_quotes)} active tickers from screener.")
                return valid_quotes[:limit]
    except Exception as e:
        print(f"[Dashboard] Notice: yfinance screener encountered error ({e}); trying fallbacks...")

    # Fallback to Gloomberb CLI movers
    try:
        gloom = GloomberbService()
        movers = gloom.fetch_market_movers("active")
        if movers:
            symbols = [m["symbol"] for m in movers if m.get("symbol")]
            print(f"[Dashboard] Retrieved {len(symbols)} movers from Gloomberb CLI: {symbols}")
            return fetch_most_active_quotes(limit=limit, symbols=symbols)
    except Exception as e:
        print(f"[Dashboard] Gloomberb movers fallback error: {e}")

    # Fallback to default universe
    print(f"[Dashboard] Using standard high-volume active universe fallback ({len(DEFAULT_ACTIVE_SYMBOLS)} symbols)...")
    return fetch_most_active_quotes(limit=limit, symbols=DEFAULT_ACTIVE_SYMBOLS[:limit])


def parse_news_item(item: Any) -> Optional[Dict[str, str]]:
    """Normalizes news article structures across various yfinance schemas."""
    if not isinstance(item, dict):
        return None
    content = item.get("content") if isinstance(item.get("content"), dict) else {}
    title = content.get("title") or item.get("title") or ""
    summary = content.get("summary") or content.get("description") or item.get("summary") or ""
    provider = ""
    if isinstance(content.get("provider"), dict):
        provider = content["provider"].get("displayName") or ""
    if not provider:
        provider = item.get("publisher") or item.get("provider") or ""
    url = ""
    if isinstance(content.get("canonicalUrl"), dict):
        url = content["canonicalUrl"].get("url") or ""
    elif isinstance(content.get("clickThroughUrl"), dict):
        url = content["clickThroughUrl"].get("url") or ""
    if not url:
        url = item.get("link") or ""
    pub_time = content.get("pubDate") or item.get("providerPublishTime") or ""
    if not title:
        return None
    return {
        "title": str(title).strip(),
        "summary": str(summary).strip(),
        "publisher": str(provider).strip() or "Market News",
        "url": str(url).strip(),
        "pub_time": str(pub_time).strip()
    }


def fetch_stock_news(symbol: str, limit: int = 5) -> List[Dict[str, str]]:
    """Fetches real-time news articles from yfinance and NewsAgent RSS/API fallbacks."""
    articles: List[Dict[str, str]] = []
    seen_titles = set()

    # 1. Primary: yfinance structured news
    try:
        t = yf.Ticker(symbol)
        raw_news = t.news or []
        for raw in raw_news:
            parsed = parse_news_item(raw)
            if parsed and parsed["title"] not in seen_titles:
                articles.append(parsed)
                seen_titles.add(parsed["title"])
                if len(articles) >= limit:
                    break
    except Exception as e:
        print(f"[Dashboard] yfinance news fetch notice for {symbol}: {e}")

    # 2. Secondary fallback: NewsAgent RSS feed
    if len(articles) < limit:
        try:
            na = NewsAgent()
            rss_headlines = na.fetch_from_rss(symbol)
            for hl in rss_headlines:
                title = hl.replace("[RSS Feed] ", "").strip()
                if title not in seen_titles:
                    articles.append({
                        "title": title,
                        "summary": "",
                        "publisher": "Yahoo Finance RSS",
                        "url": f"https://finance.yahoo.com/quote/{symbol}/news",
                        "pub_time": ""
                    })
                    seen_titles.add(title)
                    if len(articles) >= limit:
                        break
        except Exception as e:
            print(f"[Dashboard] NewsAgent RSS notice for {symbol}: {e}")

    return articles[:limit]


def infer_trading_catalyst(
    symbol: str,
    news_list: List[Dict[str, str]],
    change_pct: float = 0.0,
    rvol: float = 1.0
) -> Tuple[str, str, str]:
    """
    Infers the primary catalyst category and generates a crisp trading reason
    explaining why this stock is experiencing abnormal volume today.
    Returns: (catalyst_type, primary_headline, reason_summary)
    """
    best_cat: Optional[str] = None
    best_score = 0
    best_headline = ""

    # Keyword frequency scoring across news headlines and summaries
    for item in news_list:
        title = item.get("title", "")
        summary = item.get("summary", "")
        text = f"{title} {summary}".lower()

        for cat, kw_list in CATALYST_KEYWORDS.items():
            score = sum(1 for kw in kw_list if kw in text)
            if score > best_score:
                best_score = score
                best_cat = cat
                best_headline = title

    direction = "gaining" if change_pct >= 0 else "declining"
    abs_chg = abs(change_pct)

    if best_cat and best_headline:
        reason_summary = (
            f"{best_cat.title()}: {best_headline}. "
            f"Shares are {direction} {abs_chg:.2f}% on {rvol:.1f}x relative volume."
        )
        return best_cat, best_headline, reason_summary

    # Fallback to volume & momentum dynamics if no specific news keywords match
    if rvol >= 1.8:
        cat = "VOLUME SURGE"
        headline = f"Heavy institutional volume surge ({rvol:.1f}x normal average)"
        reason_summary = (
            f"Exceptional volume surge of {rvol:.1f}x 20-day average. "
            f"Stock is {direction} {abs_chg:.2f}% on heavy institutional flow."
        )
    elif abs_chg >= 3.5:
        cat = "MOMENTUM EXPANSION"
        headline = f"Strong price velocity expansion ({change_pct:+.2f}%)"
        reason_summary = (
            f"Sharp intraday price expansion ({change_pct:+.2f}%) "
            f"driving increased retail and algorithmic participation."
        )
    else:
        cat = "ACTIVE MARKET LIQUIDITY"
        headline = "Broad market liquidity and institutional turnover"
        reason_summary = (
            f"Trading actively among market leaders on high baseline liquidity "
            f"with {rvol:.1f}x relative volume."
        )

    return cat, headline, reason_summary


def process_single_stock(quote: Dict[str, Any], market_agent: MarketAgent) -> Dict[str, Any]:
    """Processes quote data, technical analysis, and news for a single active stock."""
    symbol = quote.get("symbol", "").upper()
    name = quote.get("shortName") or quote.get("longName") or symbol
    curr = quote.get("currency")
    curr_sym = quote.get("currency_symbol")
    if not curr_sym:
        curr, curr_sym = get_currency_for_symbol(symbol)

    current_price = round(float(quote.get("regularMarketPrice") or 0.0), 2)
    day_high = round(float(quote.get("regularMarketDayHigh") or current_price), 2)
    day_low = round(float(quote.get("regularMarketDayLow") or current_price), 2)
    day_open = round(float(quote.get("regularMarketOpen") or current_price), 2)
    raw_prev_close = quote.get("regularMarketPreviousClose")
    prev_close = round(float(raw_prev_close), 2) if (raw_prev_close is not None and float(raw_prev_close) > 0) else current_price

    # Mathematical price change from current_price and prev_close
    calc_change = round(current_price - prev_close, 2)
    raw_change = quote.get("regularMarketChange")
    if raw_change is not None:
        raw_change_val = round(float(raw_change), 2)
        if abs(raw_change_val - calc_change) < 0.05 or raw_prev_close is None:
            change = raw_change_val
        else:
            change = calc_change
    elif prev_close > 0:
        change = calc_change
    else:
        change = 0.0

    # Mathematical percent change: ((current_price - prev_close) / prev_close) * 100.0
    if prev_close > 0 and abs(current_price - prev_close) > 1e-6:
        calc_pct = ((current_price - prev_close) / prev_close) * 100.0
    else:
        calc_pct = 0.0

    raw_change_pct = quote.get("regularMarketChangePercent")
    if raw_change_pct is not None:
        raw_val = float(raw_change_pct)
        # Check if raw matches calc_pct (e.g., 0.32% or 5.2%)
        if abs(raw_val - calc_pct) < 0.2:
            change_pct = round(raw_val, 2)
        # Check if raw was passed as decimal fraction (e.g., 0.035 for 3.5%)
        elif abs(raw_val * 100.0 - calc_pct) < 0.2:
            change_pct = round(raw_val * 100.0, 2)
        # Authoritative mathematical calculation when prev_close is reliable
        elif prev_close > 0 and raw_prev_close is not None and float(raw_prev_close) > 0:
            change_pct = round(calc_pct, 2)
        else:
            change_pct = round(raw_val, 2)
    elif prev_close > 0:
        change_pct = round(calc_pct, 2)
    else:
        change_pct = 0.0

    volume = int(quote.get("regularMarketVolume") or 0)
    avg_vol_3m = quote.get("averageDailyVolume3Month")
    avg_vol_3m = int(avg_vol_3m) if avg_vol_3m else None
    market_cap = quote.get("marketCap")
    high_52w = round(float(quote.get("fiftyTwoWeekHigh") or 0.0), 2) if quote.get("fiftyTwoWeekHigh") else None
    low_52w = round(float(quote.get("fiftyTwoWeekLow") or 0.0), 2) if quote.get("fiftyTwoWeekLow") else None

    # Compute technical indicators via MarketAgent
    tech_data: Dict[str, Any] = {}
    try:
        tech_data = market_agent.analyze(symbol) or {}
    except Exception as e:
        print(f"[Dashboard] MarketAgent analysis notice for {symbol}: {e}")

    if tech_data.get("currency_symbol"):
        curr_sym = tech_data.get("currency_symbol")
        curr = tech_data.get("currency", curr)

    rsi14 = tech_data.get("rsi14")
    ema20 = tech_data.get("ema20")
    ema50 = tech_data.get("ema50")
    atr = tech_data.get("atr")
    vol_20d_mean = tech_data.get("vol_20d_mean") or avg_vol_3m
    rvol = tech_data.get("rvol_20d")
    if rvol is None and vol_20d_mean and vol_20d_mean > 0:
        rvol = round(volume / vol_20d_mean, 2)
    elif rvol is None:
        rvol = 1.0

    relative_alpha_5d = tech_data.get("relative_alpha_5d", "N/A")
    forward_pe = tech_data.get("forward_pe", "N/A")
    days_to_earnings = tech_data.get("days_to_earnings")
    suggested_stop = tech_data.get("suggested_stop_loss")
    suggested_target = tech_data.get("suggested_target_price")

    # Fetch news articles
    news_items = fetch_stock_news(symbol, limit=4)

    # Infer trading catalyst & reason
    catalyst_type, primary_headline, reason_summary = infer_trading_catalyst(
        symbol, news_items, change_pct=change_pct, rvol=rvol
    )

    # Intraday range position percentage (0% = at low, 100% = at high)
    range_span = day_high - day_low
    if range_span > 0:
        intraday_pos_pct = round(((current_price - day_low) / range_span) * 100.0, 1)
    else:
        intraday_pos_pct = 50.0

    # Look up portfolio metadata if available
    portfolio_meta: Dict[str, Any] = {}
    try:
        cached_pf = load_cached_portfolio()
        if cached_pf and cached_pf.get("holdings"):
            for h in cached_pf["holdings"]:
                h_sym = h.get("symbol", "").upper()
                h_us = h.get("us_symbol", "").upper()
                if h_sym == symbol or h_us == symbol or h_sym.split(".")[0] == symbol.split(".")[0]:
                    portfolio_meta = h
                    break
    except Exception:
        pass

    shares = portfolio_meta.get("shares")
    purchase_price = portfolio_meta.get("purchase_price")
    pos_val = round(current_price * shares, 2) if (shares is not None and shares > 0) else None
    unrealized_pl = round((current_price - purchase_price) * shares, 2) if (shares is not None and purchase_price is not None) else None
    unrealized_pl_pct = round(((current_price - purchase_price) / purchase_price) * 100.0, 2) if (purchase_price is not None and purchase_price > 0) else None

    return {
        "symbol": symbol,
        "name": name,
        "price": current_price,
        "currency": curr or "USD",
        "currency_symbol": curr_sym or "$",
        "isin": quote.get("isin") or tech_data.get("isin", ""),
        "change": change,
        "change_pct": change_pct,
        "day_high": day_high,
        "day_low": day_low,
        "day_open": day_open,
        "prev_close": prev_close,
        "intraday_range_pct": intraday_pos_pct,
        "volume": volume,
        "avg_volume": vol_20d_mean,
        "rvol": rvol,
        "market_cap": market_cap,
        "market_cap_str": format_market_cap(market_cap, curr_sym=curr_sym or "$"),
        "high_52w": high_52w,
        "low_52w": low_52w,
        "rsi14": rsi14,
        "ema20": ema20,
        "ema50": ema50,
        "atr": atr,
        "relative_alpha_5d": relative_alpha_5d,
        "forward_pe": forward_pe,
        "days_to_earnings": days_to_earnings,
        "suggested_stop": suggested_stop,
        "suggested_target": suggested_target,
        "catalyst_type": catalyst_type,
        "primary_headline": primary_headline,
        "reason_summary": reason_summary,
        "news": news_items,
        "shares": shares,
        "purchase_price": purchase_price,
        "position_val": pos_val,
        "unrealized_pl": unrealized_pl,
        "unrealized_pl_pct": unrealized_pl_pct,
    }


def collect_dashboard_data(
    limit: int = 10,
    symbols: Optional[List[str]] = None,
    market: str = "US",
    prefer_exchange: str = "DE",
    use_portfolio: bool = False,
    portfolio_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Concurrently collects quotes, technicals, and news for all selected active stocks."""
    if portfolio_url:
        try:
            print(f"[Dashboard] Syncing Google portfolio from URL: {portfolio_url}...")
            sync_google_portfolio(portfolio_url, prefer_exchange=prefer_exchange, force_european=True)
            use_portfolio = True
        except Exception as e:
            print(f"[Dashboard] Error syncing Google portfolio from URL: {e}")

    if use_portfolio and not symbols:
        portfolio_symbols = get_portfolio_symbols(prefer_exchange=prefer_exchange)
        print(f"[Dashboard] Sourcing {len(portfolio_symbols)} holdings from Google Finance portfolio...")
        symbols = portfolio_symbols
        market = "EU"

    raw_quotes = fetch_most_active_quotes(limit=limit, symbols=symbols, market=market, prefer_exchange=prefer_exchange)
    if not raw_quotes:
        print("[Dashboard] No quotes available.")
        return []

    print(f"[Dashboard] Analyzing {len(raw_quotes)} active tickers concurrently...")
    market_agent = MarketAgent()
    results: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=min(len(raw_quotes), 6)) as executor:
        future_map = {
            executor.submit(process_single_stock, q, market_agent): q.get("symbol")
            for q in raw_quotes
        }
        for future in as_completed(future_map):
            sym = future_map[future]
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                print(f"[Dashboard] Error processing stock {sym}: {e}")

    # Sort results by trading volume descending (most traded first)
    results.sort(key=lambda x: x.get("volume", 0), reverse=True)
    return results


def render_terminal_dashboard(data: List[Dict[str, Any]], fear_greed: str) -> None:
    """Renders a formatted interactive dashboard directly to the terminal."""
    if not data:
        print("\n[Dashboard] No stock data to display.")
        return

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not RICH_AVAILABLE:
        # Clean plain text fallback
        print("\n" + "=" * 80)
        print(f"📊 GLOOMBERB MOST TRADED STOCKS DASHBOARD ({now_str})")
        print(f"Sentiment: {fear_greed}")
        print("=" * 80)
        for item in data:
            sym = item['symbol']
            curr_sym = item.get('currency_symbol', '$')
            price = item['price']
            chg = item['change_pct']
            vol = format_volume(item['volume'])
            rvol = item['rvol']
            high = item['day_high']
            low = item['day_low']
            cat = item['catalyst_type']
            reason = item['reason_summary']
            sign = "+" if chg >= 0 else ""
            print(f"\n[{sym}] {curr_sym}{price:.2f} ({sign}{chg:.2f}%) | Vol: {vol} ({rvol:.1f}x) | Range: {curr_sym}{low:.2f} - {curr_sym}{high:.2f}")
            print(f"  Catalyst: [{cat}]")
            print(f"  Reason:   {reason}")
        print("\n" + "=" * 80)
        return

    console = Console()

    # Header Panel
    total_vol = sum(d.get("volume", 0) for d in data)
    top_volume_stock = data[0]["symbol"] if data else "N/A"
    top_gainer = max(data, key=lambda x: x.get("change_pct", 0.0))
    top_decliner = min(data, key=lambda x: x.get("change_pct", 0.0))

    summary_text = Text()
    summary_text.append("⚡ GLOOMBERB REAL-TIME MOST TRADED STOCKS DASHBOARD ⚡\n", style="bold cyan")
    summary_text.append(f"Timestamp: {now_str}  |  Market Sentiment: ", style="dim")
    fg_style = "bold green" if "GREED" in fear_greed.upper() else ("bold red" if "FEAR" in fear_greed.upper() else "bold yellow")
    summary_text.append(f"{fear_greed}\n", style=fg_style)
    summary_text.append(
        f"Analyzed {len(data)} Stocks  |  Combined Volume: {format_volume(total_vol)}  |  "
        f"Volume Leader: {top_volume_stock}  |  "
        f"Top Gainer: {top_gainer['symbol']} ({top_gainer['change_pct']:+.2f}%)  |  "
        f"Top Decliner: {top_decliner['symbol']} ({top_decliner['change_pct']:+.2f}%)",
        style="white"
    )
    console.print(Panel(summary_text, box=box.ROUNDED, border_style="cyan"))

    # Main Metrics Table
    table = Table(
        title="Most Active Stocks Today (Price, Range, Volume & Trading Drivers)",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold magenta",
        title_style="bold bold"
    )

    table.add_column("Symbol", style="bold white", width=8)
    table.add_column("Price", justify="right", style="bold", width=9)
    table.add_column("Day Change", justify="right", width=12)
    table.add_column("Day Range (Low - High)", justify="center", width=22)
    table.add_column("Volume (RVOL)", justify="right", width=16)
    table.add_column("52W Range", justify="center", width=16)
    table.add_column("RSI(14)", justify="center", width=8)
    table.add_column("Catalyst Driver", style="bold yellow", width=22)

    for item in data:
        sym = item["symbol"]
        curr_sym = item.get("currency_symbol", "$")
        price_str = f"{curr_sym}{item['price']:.2f}"
        chg = item["change_pct"]
        chg_style = "green" if chg >= 0 else "red"
        chg_str = f"[{chg_style}]{chg:+.2f}% ({curr_sym}{item['change']:+.2f})[/{chg_style}]"

        range_str = f"{curr_sym}{item['day_low']:.2f} - {curr_sym}{item['day_high']:.2f}"
        vol_str = f"{format_volume(item['volume'])} ({item['rvol']:.1f}x)"
        if item["rvol"] >= 1.5:
            vol_str = f"[bold yellow]{vol_str}[/bold yellow]"

        range_52w = "N/A"
        if item.get("low_52w") and item.get("high_52w"):
            range_52w = f"{curr_sym}{item['low_52w']:.1f} - {curr_sym}{item['high_52w']:.1f}"

        rsi_val = item.get("rsi14")
        if rsi_val is not None:
            if rsi_val >= 70:
                rsi_str = f"[bold red]{rsi_val:.1f}[/bold red]"
            elif rsi_val <= 30:
                rsi_str = f"[bold green]{rsi_val:.1f}[/bold green]"
            else:
                rsi_str = f"{rsi_val:.1f}"
        else:
            rsi_str = "N/A"

        cat = item["catalyst_type"]
        table.add_row(sym, price_str, chg_str, range_str, vol_str, range_52w, rsi_str, cat)

    console.print(table)

    # Detailed News & Reasons Breakdown Panel
    console.print("\n[bold cyan]📌 Why Are They Trading So Much Today? (Catalyst Deep Dive)[/bold cyan]")
    for item in data:
        sym = item["symbol"]
        curr_sym = item.get("currency_symbol", "$")
        name = item["name"]
        chg = item["change_pct"]
        chg_color = "green" if chg >= 0 else "red"
        vol_str = format_volume(item["volume"])
        rvol = item["rvol"]

        detail_text = Text()
        detail_text.append(f"• [{sym}] {name} ", style="bold white")
        detail_text.append(f"{curr_sym}{item['price']:.2f} ({chg:+.2f}%) ", style=f"bold {chg_color}")
        detail_text.append(f"| Vol: {vol_str} ({rvol:.1f}x 20d avg) | Range: {curr_sym}{item['day_low']:.2f} - {curr_sym}{item['day_high']:.2f}\n", style="dim")
        detail_text.append(f"  ➤ Core Driver: ", style="bold yellow")
        detail_text.append(f"{item['reason_summary']}\n", style="white")

        news_items = item.get("news", [])
        if news_items:
            detail_text.append("  ➤ Top News Headlines:\n", style="dim cyan")
            for idx, n in enumerate(news_items[:2]):
                pub = f"[{n['publisher']}] " if n.get('publisher') else ""
                detail_text.append(f"     {idx+1}. {pub}{n['title']}\n", style="dim")

        console.print(Panel(detail_text, box=box.ROUNDED, border_style="dim"))


def generate_html_dashboard(data: List[Dict[str, Any]], fear_greed: str, output_path: str = "dashboard.html") -> str:
    """Generates a responsive modern dark-mode HTML dashboard file."""
    now_str = datetime.datetime.now().strftime("%B %d, %Y - %H:%M:%S UTC")
    total_vol = sum(d.get("volume", 0) for d in data)
    top_volume_stock = data[0]["symbol"] if data else "N/A"
    top_gainer = max(data, key=lambda x: x.get("change_pct", 0.0)) if data else {}
    top_decliner = min(data, key=lambda x: x.get("change_pct", 0.0)) if data else {}

    rows_html = []
    cards_html = []

    for item in data:
        sym = item["symbol"]
        curr_sym = item.get("currency_symbol", "$")
        name = item["name"]
        price = item["price"]
        chg = item["change_pct"]
        chg_val = item["change"]
        day_low = item["day_low"]
        day_high = item["day_high"]
        vol_str = format_volume(item["volume"])
        rvol = item["rvol"]
        cap_str = item["market_cap_str"]
        cat = item["catalyst_type"]
        pos_pct = item["intraday_range_pct"]
        rsi_str = f"{item['rsi14']:.1f}" if item.get("rsi14") else "N/A"
        alpha_str = str(item.get("relative_alpha_5d", "N/A"))

        is_pos = chg >= 0
        badge_class = "badge-green" if is_pos else "badge-red"
        sign = "+" if is_pos else ""

        # Table Row
        rows_html.append(f"""
        <tr>
            <td class="font-bold text-white">{sym}<br><span class="text-xs text-gray-400 font-normal">{name[:20]}</span></td>
            <td class="text-right font-mono font-bold">{curr_sym}{price:.2f}</td>
            <td class="text-right"><span class="badge {badge_class}">{sign}{chg:.2f}% ({sign}{curr_sym}{abs(chg_val):.2f})</span></td>
            <td>
                <div class="range-container">
                    <span class="range-val">{curr_sym}{day_low:.2f}</span>
                    <div class="range-bar"><div class="range-fill" style="width: {pos_pct}%;"></div></div>
                    <span class="range-val">{curr_sym}{day_high:.2f}</span>
                </div>
            </td>
            <td class="text-right font-mono">{vol_str} <span class="text-xs {'text-yellow-400 font-bold' if rvol >= 1.5 else 'text-gray-400'}">({rvol:.1f}x)</span></td>
            <td class="text-center font-mono text-sm">{rsi_str}</td>
            <td><span class="catalyst-tag">{cat}</span></td>
            <td class="text-sm text-gray-300 max-w-xs">{item['reason_summary']}</td>
        </tr>
        """)

        # Stock News & Detail Card
        news_html_list = []
        for n in item.get("news", [])[:3]:
            title = n.get("title", "")
            pub = n.get("publisher", "News")
            url = n.get("url", "#")
            news_html_list.append(f"""
                <li class="mb-2">
                    <a href="{url}" target="_blank" rel="noopener noreferrer" class="news-link">
                        <span class="text-xs text-cyan-400 font-semibold">[{pub}]</span> {title}
                    </a>
                </li>
            """)
        news_block = "".join(news_html_list) if news_html_list else "<li class='text-gray-500 text-sm'>No news headlines retrieved.</li>"

        cards_html.append(f"""
        <div class="stock-card">
            <div class="flex justify-between items-start mb-3">
                <div>
                    <h3 class="text-xl font-bold text-white">{sym} <span class="text-sm font-normal text-gray-400">({name})</span></h3>
                    <p class="text-xs text-gray-400">Market Cap: {cap_str} | Alpha vs SPY: {alpha_str}</p>
                </div>
                <div class="text-right">
                    <div class="text-2xl font-mono font-bold text-white">{curr_sym}{price:.2f}</div>
                    <span class="badge {badge_class}">{sign}{chg:.2f}%</span>
                </div>
            </div>

            <div class="grid grid-cols-4 gap-2 py-2 mb-3 bg-gray-900 rounded p-2 text-xs font-mono">
                <div><span class="text-gray-400">Low:</span> {curr_sym}{day_low:.2f}</div>
                <div><span class="text-gray-400">High:</span> {curr_sym}{day_high:.2f}</div>
                <div><span class="text-gray-400">Volume:</span> {vol_str}</div>
                <div><span class="text-gray-400">RVOL:</span> {rvol:.1f}x</div>
            </div>

            <div class="mb-3">
                <span class="catalyst-tag mb-1 inline-block">{cat}</span>
                <p class="text-sm text-gray-200 mt-1"><strong>Trading Reason:</strong> {item['reason_summary']}</p>
            </div>

            <div class="border-t border-gray-800 pt-3">
                <h4 class="text-xs uppercase tracking-wider text-gray-400 font-bold mb-2">Latest News & Catalysts</h4>
                <ul class="list-disc list-inside text-sm text-gray-300">
                    {news_block}
                </ul>
            </div>
        </div>
        """)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gloomberb - Most Traded Stocks Dashboard</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ background-color: #0b0f19; color: #e2e8f0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; padding: 24px; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1e293b; padding-bottom: 20px; margin-bottom: 24px; }}
        .title {{ font-size: 26px; font-weight: 800; color: #38bdf8; letter-spacing: -0.5px; }}
        .subtitle {{ font-size: 13px; color: #94a3b8; margin-top: 4px; }}
        .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 28px; }}
        .kpi-card {{ background: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 16px; }}
        .kpi-label {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; color: #9ca3af; }}
        .kpi-val {{ font-size: 22px; font-weight: 700; margin-top: 6px; font-family: ui-monospace, monospace; }}
        .table-card {{ background: #111827; border: 1px solid #1f2937; border-radius: 8px; overflow: hidden; margin-bottom: 32px; }}
        table {{ width: 100%; border-collapse: collapse; text-align: left; }}
        th {{ background: #1f2937; color: #cbd5e1; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; padding: 12px 16px; font-weight: 600; }}
        td {{ padding: 14px 16px; border-bottom: 1px solid #1f2937; font-size: 14px; vertical-align: middle; }}
        tr:hover {{ background-color: #1a2234; }}
        .badge {{ display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: 700; font-family: ui-monospace, monospace; }}
        .badge-green {{ background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }}
        .badge-red {{ background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }}
        .catalyst-tag {{ background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 4px; padding: 3px 8px; font-size: 11px; font-weight: 700; }}
        .range-container {{ display: flex; align-items: center; gap: 8px; width: 180px; font-family: ui-monospace, monospace; font-size: 11px; color: #94a3b8; }}
        .range-bar {{ flex: 1; height: 6px; background: #374151; border-radius: 3px; position: relative; overflow: hidden; }}
        .range-fill {{ height: 100%; background: #38bdf8; border-radius: 3px; }}
        .cards-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 20px; }}
        .stock-card {{ background: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 20px; }}
        .news-link {{ color: #cbd5e1; text-decoration: none; transition: color 0.2s; }}
        .news-link:hover {{ color: #38bdf8; text-decoration: underline; }}
        .text-cyan-400 {{ color: #22d3ee; }}
        .text-yellow-400 {{ color: #facc15; }}
        .text-gray-400 {{ color: #9ca3af; }}
        .text-xs {{ font-size: 11px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <div class="title">⚡ Gloomberb Most Traded Stocks Dashboard</div>
                <div class="subtitle">High-Volume Intraday Movers, Price Geometry & Catalyst Drivers • {now_str}</div>
            </div>
            <div>
                <span class="badge" style="background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); padding: 8px 12px; font-size: 13px;">
                    {fear_greed}
                </span>
            </div>
        </div>

        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Active Stocks Tracked</div>
                <div class="kpi-val text-cyan-400">{len(data)} Stocks</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Combined Volume</div>
                <div class="kpi-val text-white">{format_volume(total_vol)}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Volume Leader</div>
                <div class="kpi-val text-yellow-400">{top_volume_stock}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Top Active Gainer</div>
                <div class="kpi-val" style="color: #34d399;">{top_gainer.get('symbol', 'N/A')} ({top_gainer.get('change_pct', 0.0):+.2f}%)</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Top Active Decliner</div>
                <div class="kpi-val" style="color: #f87171;">{top_decliner.get('symbol', 'N/A')} ({top_decliner.get('change_pct', 0.0):+.2f}%)</div>
            </div>
        </div>

        <div class="table-card">
            <table>
                <thead>
                    <tr>
                        <th>Symbol & Name</th>
                        <th style="text-align: right;">Price</th>
                        <th style="text-align: right;">Change</th>
                        <th style="text-align: center;">Day Range</th>
                        <th style="text-align: right;">Volume (RVOL)</th>
                        <th style="text-align: center;">RSI(14)</th>
                        <th>Catalyst</th>
                        <th>Trading Driver Summary</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(rows_html)}
                </tbody>
            </table>
        </div>

        <h2 style="font-size: 18px; font-weight: 700; color: #38bdf8; margin-bottom: 16px;">📰 Catalyst & News Deep Dive</h2>
        <div class="cards-grid">
            {"".join(cards_html)}
        </div>
    </div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n[Dashboard] Standalone HTML dashboard saved successfully to '{output_path}'.")
    return output_path





def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gloomberb Most Traded Stocks & News Catalyst Dashboard (Model-Free)"
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=10,
        help="Number of active stocks to analyze (default: 10)"
    )
    parser.add_argument(
        "--tickers", "-t",
        type=str,
        default=None,
        help="Optional comma-separated list of symbols (e.g. NVDA,TSLA,INTC,AAPL or ISINs)"
    )
    parser.add_argument(
        "--market", "-m",
        choices=["US", "EU"],
        default="US",
        help="Target market: US or EU (Europe - gettex / XETRA) (default: US)"
    )
    parser.add_argument(
        "--eu",
        action="store_true",
        help="Shorthand for European market (--market EU)"
    )
    parser.add_argument(
        "--prefer-exchange",
        choices=["DE", "MU", "F", "HA", "TG"],
        default="DE",
        help="Preferred European exchange suffix (DE=XETRA, MU=gettex, HA=Hannover/EIX) (default: DE)"
    )
    parser.add_argument(
        "--html",
        type=str,
        default="dashboard.html",
        help="Path to generate standalone HTML dashboard (default: dashboard.html)"
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Disable HTML dashboard generation"
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Optional path to export raw dashboard dataset as JSON"
    )
    parser.add_argument(
        "--portfolio",
        action="store_true",
        help="Analyze Google Finance / Google Sheet portfolio holdings (dual-listed in EUR)"
    )
    parser.add_argument(
        "--portfolio-url",
        type=str,
        default=None,
        help="URL of shared Google Sheet or Google Finance portfolio to sync before analyzing"
    )
    parser.add_argument(
        "--streamlit",
        action="store_true",
        help="Ensure Streamlit web dashboard server (app.py) is running"
    )
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.tickers.split(",") if s.strip()] if args.tickers else None
    use_portfolio = args.portfolio or bool(args.portfolio_url)
    market = "EU" if (args.eu or use_portfolio) else args.market.upper()

    print("\n" + "=" * 65)
    mode_tag = "PORTFOLIO" if use_portfolio else market
    print(f"⚡ GLOOMBERB MOST TRADED STOCKS & CATALYST DASHBOARD [{mode_tag}] ⚡")
    print("=" * 65)

    # 1. Fetch market sentiment (CNN Fear & Greed Index)
    try:
        fear_greed = NewsAgent().fetch_cnn_fear_and_greed()
    except Exception:
        fear_greed = "Neutral (50)"

    # 2. Concurrently collect stock quotes, technicals & news
    data = collect_dashboard_data(
        limit=args.limit,
        symbols=symbols,
        market=market,
        prefer_exchange=args.prefer_exchange,
        use_portfolio=use_portfolio,
        portfolio_url=args.portfolio_url,
    )
    if not data:
        print("\n❌ Failed to gather market data for active stocks.")
        sys.exit(1)

    # 3. Render terminal dashboard
    render_terminal_dashboard(data, fear_greed)

    # 4. Generate HTML dashboard
    if not args.no_html:
        generate_html_dashboard(data, fear_greed, output_path=args.html)

    # 5. Export JSON if requested
    if args.json:
        try:
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump({"timestamp": datetime.datetime.now().isoformat(), "fear_greed": fear_greed, "stocks": data}, f, indent=2)
            print(f"[Dashboard] JSON dataset exported to '{args.json}'.")
        except Exception as e:
            print(f"[Dashboard] Error exporting JSON: {e}")


    # 7. Start Streamlit web dashboard if requested
    if args.streamlit:
        try:
            from streamlit_server import ensure_streamlit_running
            ensure_streamlit_running()
        except Exception as e:
            print(f"[Dashboard] Notice: Could not start Streamlit server: {e}")


if __name__ == "__main__":
    main()
