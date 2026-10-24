"""Danelfin AI Financial Analytics Service.

Provides programmatic access to Danelfin AI-powered scores, probabilistic price forecasts,
trading parameters, and curated strategy picks with a strict quota-protection layer
designed for Free Tier (500 calls/month) and paid plans.

Key Features:
- 24-Hour SQLite Caching: Scores update once daily at market close; re-queries cost 0 calls.
- Monthly Usage Tracker: Tracks and enforces safe monthly budget limits (e.g. 480/500 calls).
- Batch Pre-seeding: Best Stocks (/v3/beststocks) pre-populates the score cache for all 25 tickers
  in a single API call, saving up to 25 individual API requests.
- European Ticker Normalization: Automatically strips exchange suffixes (e.g. SAP.DE -> SAP)
  and resolves dual-listed symbols via ticker_resolver.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("DanelfinService")

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trader.db")
DEFAULT_BASE_URL = os.getenv("DANELFIN_BASE_URL", "https://apirest.danelfin.com").rstrip("/")
DEFAULT_MONTHLY_LIMIT = int(os.getenv("DANELFIN_MONTHLY_LIMIT", "500"))
DEFAULT_SAFE_THRESHOLD = int(os.getenv("DANELFIN_SAFE_THRESHOLD", "480"))


def _today_utc() -> str:
    """Returns today's date in YYYY-MM-DD UTC."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")


def _month_key_utc() -> str:
    """Returns current month key in YYYY-MM UTC."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m")


class DanelfinService:
    """
    Danelfin REST API Client with SQLite Caching and Quota Management.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        db_path: str = DEFAULT_DB_PATH,
        monthly_limit: int = DEFAULT_MONTHLY_LIMIT,
        safe_threshold: int = DEFAULT_SAFE_THRESHOLD,
    ):
        self.api_key = (api_key or os.getenv("DANELFIN_API_KEY", "")).strip()
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.db_path = db_path
        self.monthly_limit = monthly_limit
        self.safe_threshold = safe_threshold
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initializes cache and usage tracking tables if they do not exist."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS danelfin_cache (
                        ticker TEXT NOT NULL,
                        endpoint TEXT NOT NULL,
                        fetch_date TEXT NOT NULL,
                        data_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY (ticker, endpoint, fetch_date)
                    );
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS danelfin_usage (
                        month_key TEXT PRIMARY KEY,
                        call_count INTEGER DEFAULT 0,
                        updated_at TEXT NOT NULL
                    );
                    """
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"Error initializing Danelfin tables: {e}")

    # -------------------------------------------------------------------------
    # Quota & Usage Management
    # -------------------------------------------------------------------------

    def get_usage(self, month_key: Optional[str] = None) -> Dict[str, Any]:
        """Returns current monthly API call usage and remaining budget."""
        m_key = month_key or _month_key_utc()
        call_count = 0
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT call_count FROM danelfin_usage WHERE month_key = ?",
                    (m_key,),
                )
                row = cursor.fetchone()
                if row:
                    call_count = int(row["call_count"])
        except Exception as e:
            logger.warning(f"Error reading Danelfin usage: {e}")

        remaining = max(0, self.monthly_limit - call_count)
        safe_remaining = max(0, self.safe_threshold - call_count)
        is_safe = call_count < self.safe_threshold

        return {
            "month": m_key,
            "call_count": call_count,
            "monthly_limit": self.monthly_limit,
            "safe_threshold": self.safe_threshold,
            "remaining": remaining,
            "safe_remaining": safe_remaining,
            "is_safe": is_safe,
        }

    def _increment_usage(self) -> int:
        """Atomically increments API usage counter for the current month."""
        m_key = _month_key_utc()
        now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO danelfin_usage (month_key, call_count, updated_at)
                    VALUES (?, 1, ?)
                    ON CONFLICT(month_key) DO UPDATE SET
                        call_count = call_count + 1,
                        updated_at = excluded.updated_at;
                    """,
                    (m_key, now_str),
                )
                conn.commit()
                cursor.execute(
                    "SELECT call_count FROM danelfin_usage WHERE month_key = ?",
                    (m_key,),
                )
                row = cursor.fetchone()
                return int(row["call_count"]) if row else 1
        except Exception as e:
            logger.warning(f"Error incrementing Danelfin usage: {e}")
            return 1

    def is_quota_available(self, force_refresh: bool = False) -> bool:
        """Checks if current usage is below safe threshold, unless forced."""
        if force_refresh:
            return True
        usage = self.get_usage()
        return bool(usage.get("is_safe", True))

    # -------------------------------------------------------------------------
    # Caching Helpers
    # -------------------------------------------------------------------------

    def _get_from_cache(
        self,
        ticker: str,
        endpoint: str,
        date_str: Optional[str] = None,
        allow_stale: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Retrieves cached JSON payload if available."""
        target_date = date_str or _today_utc()
        ticker_clean = ticker.upper().strip()

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if allow_stale:
                    cursor.execute(
                        """
                        SELECT data_json, fetch_date FROM danelfin_cache
                        WHERE ticker = ? AND endpoint = ?
                        ORDER BY fetch_date DESC
                        LIMIT 1
                        """,
                        (ticker_clean, endpoint),
                    )
                else:
                    cursor.execute(
                        """
                        SELECT data_json, fetch_date FROM danelfin_cache
                        WHERE ticker = ? AND endpoint = ? AND fetch_date = ?
                        """,
                        (ticker_clean, endpoint, target_date),
                    )
                row = cursor.fetchone()
                if row:
                    data = json.loads(row["data_json"])
                    return data
        except Exception as e:
            logger.warning(f"Cache read error for {ticker_clean}/{endpoint}: {e}")
        return None

    def _save_to_cache(
        self,
        ticker: str,
        endpoint: str,
        data: Dict[str, Any],
        date_str: Optional[str] = None,
    ) -> None:
        """Stores response payload into SQLite cache."""
        target_date = date_str or _today_utc()
        ticker_clean = ticker.upper().strip()
        now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO danelfin_cache (ticker, endpoint, fetch_date, data_json, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(ticker, endpoint, fetch_date) DO UPDATE SET
                        data_json = excluded.data_json,
                        created_at = excluded.created_at;
                    """,
                    (ticker_clean, endpoint, target_date, json.dumps(data), now_str),
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"Cache write error for {ticker_clean}/{endpoint}: {e}")

    # -------------------------------------------------------------------------
    # Symbol Resolution Helper
    # -------------------------------------------------------------------------

    @staticmethod
    def clean_ticker_for_danelfin(symbol: str) -> str:
        """
        Normalizes ticker for Danelfin API.
        - US securities traded on European exchanges (e.g. NVD.DE, TL0.DE with US ISIN)
          resolve to their US underlying ticker (NVDA, TSLA).
        - Native European stocks (e.g. SAP.DE, MC.PA) strip exchange suffixes (SAP, MC).
        - Standard US tickers remain unchanged.
        """
        if not symbol:
            return ""
        s = symbol.strip().upper()
        try:
            from ticker_resolver import resolve_symbol
            info = resolve_symbol(s)
            isin = str(info.get("isin") or "")
            underlying = info.get("underlying_symbol")
            if isin.startswith("US") and underlying:
                return underlying.strip().upper()
            if "." in s:
                parts = s.split(".")
                if len(parts) == 2 and len(parts[1]) in (2, 3) and parts[1].isalpha():
                    return parts[0]
            if underlying and not underlying.endswith("Y"):
                return underlying.strip().upper()
        except Exception:
            pass

        if "." in s:
            parts = s.split(".")
            if len(parts) == 2 and len(parts[1]) in (2, 3) and parts[1].isalpha():
                return parts[0]
        return s

    # -------------------------------------------------------------------------
    # Core API Methods
    # -------------------------------------------------------------------------

    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: int = 10,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Makes an authenticated request to Danelfin REST API."""
        if not self.api_key:
            return None, "DANELFIN_API_KEY is not set."

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {
            "x-api-key": self.api_key,
            "Accept": "application/json",
        }

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=timeout)
            if resp.status_code == 200:
                self._increment_usage()
                return resp.json(), None
            elif resp.status_code == 429:
                return None, f"Danelfin API Rate Limit Exceeded (HTTP 429)."
            elif resp.status_code == 404:
                return None, f"Ticker not found in Danelfin universe (HTTP 404)."
            else:
                return None, f"Danelfin API returned HTTP {resp.status_code}: {resp.text[:150]}"
        except requests.RequestException as e:
            return None, f"Danelfin request error: {e}"

    def get_scores(
        self,
        symbol: str,
        force_refresh: bool = False,
        allow_stale: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieves Danelfin AI Score and 4 sub-scores (1–10 scale).
        Endpoint: GET /ranking?ticker={ticker}
        
        Checks SQLite cache first for today's date.
        If cache miss and quota is within limit, calls API and stores result.
        """
        ticker = self.clean_ticker_for_danelfin(symbol)
        if not ticker:
            return None

        today = _today_utc()

        # 1. Cache Check
        if not force_refresh:
            cached = self._get_from_cache(ticker, "ranking", today, allow_stale=False)
            if cached:
                cached["source"] = "cache"
                return cached

        # 2. Quota Check
        if not self.is_quota_available(force_refresh):
            if allow_stale:
                stale = self._get_from_cache(ticker, "ranking", allow_stale=True)
                if stale:
                    stale["source"] = "stale_cache_quota_limit"
                    return stale
            logger.warning(f"Danelfin quota threshold reached. Skipping API call for {ticker}.")
            return None

        # 3. API Request
        data, err = self._make_request("ranking", params={"ticker": ticker})
        if err or not isinstance(data, dict):
            if allow_stale:
                stale = self._get_from_cache(ticker, "ranking", allow_stale=True)
                if stale:
                    stale["source"] = "stale_cache_on_error"
                    return stale
            logger.warning(f"Failed to fetch Danelfin scores for {ticker}: {err}")
            return None

        # Danelfin /ranking returns a dict keyed by date: {"YYYY-MM-DD": {"aiscore": 7, ...}}
        latest_date = next(iter(data)) if data else today
        scores = data.get(latest_date, {}) if isinstance(data, dict) else {}

        result = {
            "ticker": ticker,
            "original_symbol": symbol,
            "date": latest_date,
            "aiscore": scores.get("aiscore"),
            "technical": scores.get("technical"),
            "fundamental": scores.get("fundamental"),
            "sentiment": scores.get("sentiment"),
            "low_risk": scores.get("low_risk"),
            "source": "api",
        }

        # 4. Save to Cache
        self._save_to_cache(ticker, "ranking", result, today)
        return result

    def get_best_stocks(self, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """
        Retrieves Danelfin's Top 25 Best Stocks Strategy.
        Endpoint: GET /v3/beststocks
        
        Batch Pre-seeding: Automatically populates the individual score cache
        for all 25 stocks in a single API call!
        """
        today = _today_utc()
        cache_key = "__GLOBAL__"

        if not force_refresh:
            cached = self._get_from_cache(cache_key, "beststocks", today)
            if cached:
                return cached

        if not self.is_quota_available(force_refresh):
            stale = self._get_from_cache(cache_key, "beststocks", allow_stale=True)
            if stale:
                return stale
            return None

        data, err = self._make_request("v3/beststocks")
        if err or not isinstance(data, dict):
            logger.warning(f"Failed to fetch Danelfin best stocks: {err}")
            return None

        latest_date = next(iter(data)) if data else today
        picks = data.get(latest_date, {}) if isinstance(data, dict) else {}

        # Pre-seed individual scores for all 25 tickers
        for sym, vals in picks.items():
            if isinstance(vals, dict):
                single_score = {
                    "ticker": sym,
                    "original_symbol": sym,
                    "date": latest_date,
                    "aiscore": vals.get("aiscore"),
                    "technical": vals.get("technical"),
                    "fundamental": vals.get("fundamental"),
                    "sentiment": vals.get("sentiment"),
                    "low_risk": vals.get("low_risk"),
                    "rank": vals.get("rank"),
                    "perf_ytd": vals.get("perf_ytd"),
                    "sector": vals.get("sector"),
                    "country": vals.get("country"),
                    "source": "api_preseeded",
                }
                self._save_to_cache(sym, "ranking", single_score, today)

        result = {
            "date": latest_date,
            "count": len(picks),
            "picks": picks,
            "source": "api",
        }

        self._save_to_cache(cache_key, "beststocks", result, today)
        return result

    def get_daily_picks_tickers(self, limit: int = 25, force_refresh: bool = False) -> List[str]:
        """
        Returns the ordered list of top tickers from Danelfin's daily Best Stocks strategy.
        Pre-seeds the individual score cache for all tickers in 1 single API call.
        """
        best = self.get_best_stocks(force_refresh=force_refresh)
        if not best or not best.get("picks"):
            return []
        picks = best["picks"]
        return list(picks.keys())[:limit]

    def get_trade_ideas(
        self,
        market: str = "us_stocks",
        direction: str = "long",
        min_win_rate_3m: float = 0.60,
        min_ai_score: int = 7,
        limit: int = 25,
        force_refresh: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieves Danelfin's filtered trade ideas with proven track records.
        Endpoint: GET /v3/trade-ideas
        """
        today = _today_utc()
        cache_key = f"ideas_{market}_{direction}_{min_ai_score}_{min_win_rate_3m}"

        if not force_refresh:
            cached = self._get_from_cache(cache_key, "trade-ideas", today)
            if cached:
                return cached

        if not self.is_quota_available(force_refresh):
            return self._get_from_cache(cache_key, "trade-ideas", allow_stale=True)

        params: Dict[str, Any] = {
            "direction": direction,
            "min_ai_score": min_ai_score,
            "min_win_rate_3m": min_win_rate_3m,
            "limit": limit,
        }
        if market == "europe":
            params["market"] = "europe"

        data, err = self._make_request("v3/trade-ideas", params=params)
        if err or not isinstance(data, dict):
            logger.warning(f"Failed to fetch Danelfin trade ideas: {err}")
            return None

        # Filter out metadata keys like total, limit, offset
        date_keys = [k for k in data.keys() if k not in ("total", "limit", "offset")]
        latest_date = date_keys[0] if date_keys else today
        ideas = data.get(latest_date, {}) if isinstance(data, dict) else {}

        # Pre-seed individual scores for all returned trade ideas
        if isinstance(ideas, dict):
            for sym, vals in ideas.items():
                if isinstance(vals, dict):
                    single_score = {
                        "ticker": sym,
                        "original_symbol": sym,
                        "date": latest_date,
                        "aiscore": vals.get("aiscore"),
                        "technical": vals.get("technical"),
                        "fundamental": vals.get("fundamental"),
                        "sentiment": vals.get("sentiment"),
                        "low_risk": vals.get("low_risk"),
                        "win_rate_3m": vals.get("win_rate_3m"),
                        "win_rate_1m": vals.get("win_rate_1m"),
                        "source": "api_preseeded",
                    }
                    self._save_to_cache(sym, "ranking", single_score, today)

        result = {
            "date": latest_date,
            "total": data.get("total", len(ideas)),
            "count": len(ideas),
            "ideas": ideas,
            "source": "api",
        }

        self._save_to_cache(cache_key, "trade-ideas", result, today)
        return result

    def get_trading_parameters(
        self,
        symbol: str,
        force_refresh: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieves suggested entry price, stop-loss, and take-profit levels.
        Endpoint: GET /v3/trading-parameters?ticker={ticker}
        """
        ticker = self.clean_ticker_for_danelfin(symbol)
        if not ticker:
            return None

        today = _today_utc()

        if not force_refresh:
            cached = self._get_from_cache(ticker, "trading-parameters", today)
            if cached:
                cached["source"] = "cache"
                return cached

        if not self.is_quota_available(force_refresh):
            stale = self._get_from_cache(ticker, "trading-parameters", allow_stale=True)
            if stale:
                stale["source"] = "stale_cache_quota_limit"
                return stale
            return None

        data, err = self._make_request("v3/trading-parameters", params={"ticker": ticker})
        if err or not isinstance(data, dict):
            logger.warning(f"Failed to fetch trading parameters for {ticker}: {err}")
            return None

        latest_date = next(iter(data)) if data else today
        params = data.get(latest_date, {}) if isinstance(data, dict) else {}

        result = {
            "ticker": ticker,
            "original_symbol": symbol,
            "date": latest_date,
            "entry_price": params.get("entry_price"),
            "stop_loss": params.get("stop_loss"),
            "stop_loss_pct": params.get("stop_loss_pct"),
            "take_profit": params.get("take_profit"),
            "take_profit_pct": params.get("take_profit_pct"),
            "horizon": params.get("horizon", "3M"),
            "currency": params.get("currency", "USD"),
            "signal": params.get("signal"),
            "source": "api",
        }

        self._save_to_cache(ticker, "trading-parameters", result, today)
        return result

    def get_price_forecast(
        self,
        symbol: str,
        horizon: Optional[str] = None,
        force_refresh: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieves probabilistic price forecast quantiles (median, q05, q16, q84, q95).
        Endpoint: GET /v3/price-forecast?ticker={ticker}
        """
        ticker = self.clean_ticker_for_danelfin(symbol)
        if not ticker:
            return None

        today = _today_utc()

        if not force_refresh:
            cached = self._get_from_cache(ticker, "price-forecast", today)
            if cached:
                cached["source"] = "cache"
                return cached

        if not self.is_quota_available(force_refresh):
            stale = self._get_from_cache(ticker, "price-forecast", allow_stale=True)
            if stale:
                stale["source"] = "stale_cache_quota_limit"
                return stale
            return None

        req_params: Dict[str, Any] = {"ticker": ticker}
        if horizon:
            req_params["horizon"] = horizon

        data, err = self._make_request("v3/price-forecast", params=req_params)
        if err or not isinstance(data, dict):
            logger.warning(f"Failed to fetch price forecast for {ticker}: {err}")
            return None

        latest_date = next(iter(data)) if data else today
        fc = data.get(latest_date, {}) if isinstance(data, dict) else {}

        result = {
            "ticker": ticker,
            "original_symbol": symbol,
            "date": latest_date,
            "signal": fc.get("signal"),
            "median_1m": fc.get("median_1m"),
            "q05_1m": fc.get("q05_1m"),
            "q16_1m": fc.get("q16_1m"),
            "q84_1m": fc.get("q84_1m"),
            "q95_1m": fc.get("q95_1m"),
            "median_3m": fc.get("median_3m"),
            "q05_3m": fc.get("q05_3m"),
            "q95_3m": fc.get("q95_3m"),
            "take_profit_3m": fc.get("take_profit_3m"),
            "stop_loss_3m": fc.get("stop_loss_3m"),
            "source": "api",
        }

        self._save_to_cache(ticker, "price-forecast", result, today)
        return result

    def get_performance(
        self,
        symbol: str,
        signal: str = "buy",
        force_refresh: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieves historical win rate and alpha track record for this ticker's signals.
        Endpoint: GET /v3/performance?ticker={ticker}&signal={signal}
        """
        ticker = self.clean_ticker_for_danelfin(symbol)
        if not ticker:
            return None

        today = _today_utc()
        cache_key = f"performance_{signal}"

        if not force_refresh:
            cached = self._get_from_cache(ticker, cache_key, today)
            if cached:
                cached["source"] = "cache"
                return cached

        if not self.is_quota_available(force_refresh):
            stale = self._get_from_cache(ticker, cache_key, allow_stale=True)
            if stale:
                stale["source"] = "stale_cache_quota_limit"
                return stale
            return None

        data, err = self._make_request("v3/performance", params={"ticker": ticker, "signal": signal})
        if err or not isinstance(data, dict):
            logger.warning(f"Failed to fetch performance for {ticker}: {err}")
            return None

        latest_date = next(iter(data)) if data else today
        perf = data.get(latest_date, {}) if isinstance(data, dict) else {}

        result = {
            "ticker": ticker,
            "original_symbol": symbol,
            "date": latest_date,
            "signal": perf.get("signal", signal),
            "win_rate_1m": perf.get("win_rate_1m"),
            "alpha_win_rate_1m": perf.get("alpha_win_rate_1m"),
            "avg_perf_1m": perf.get("avg_perf_1m"),
            "avg_alpha_1m": perf.get("avg_alpha_1m"),
            "win_rate_3m": perf.get("win_rate_3m"),
            "alpha_win_rate_3m": perf.get("alpha_win_rate_3m"),
            "avg_perf_3m": perf.get("avg_perf_3m"),
            "avg_alpha_3m": perf.get("avg_alpha_3m"),
            "source": "api",
        }

        self._save_to_cache(ticker, cache_key, result, today)
        return result

    def get_comprehensive_intel(
        self,
        symbol: str,
        include_deep: bool = False,
    ) -> Dict[str, Any]:
        """
        Aggregates Danelfin intelligence for a given symbol.
        By default, calls only get_scores (1 API call or 0 if cached) to strictly
        preserve Free Tier quota. If include_deep is True, also fetches trading
        parameters and price forecast.
        """
        scores = self.get_scores(symbol)
        trading_params = None
        forecast = None

        if include_deep:
            trading_params = self.get_trading_parameters(symbol)
            forecast = self.get_price_forecast(symbol)

        return {
            "symbol": symbol,
            "ticker": self.clean_ticker_for_danelfin(symbol),
            "scores": scores,
            "trading_parameters": trading_params,
            "price_forecast": forecast,
            "usage": self.get_usage(),
        }
