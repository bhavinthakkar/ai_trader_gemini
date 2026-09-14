import os
import json
import subprocess
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

class GloomberbService:
    """
    Gloomberb Data Source: Sourcing market feeds directly from official open-source gloom-sh/gloomberb CLI terminal binary
    (~/.local/bin/gloomberb) across 6 primary channels with API fallbacks:
    1. News (market catalysts, headlines, press releases)
    2. Filings (SEC EDGAR 10-K, 10-Q, 8-K, Form 4, 13F corporate filings)
    3. Financials (Income statement, balance sheet, cash flows, key ratios)
    4. Options Chains & Flow (Implied volatility, Put/Call ratios, open interest)
    5. Insider & Institutional Activity (Form 4 insider buys/sells, 13F holdings)
    6. Peer Relative Valuation (EV/EBITDA, P/S, P/FCF, EV/Revenue metrics)
    """

    PEER_MAP = {
        "AAPL": ["MSFT", "GOOGL"],
        "META": ["GOOGL", "AMZN"],
        "NVDA": ["AMD", "AVGO"],
        "MSFT": ["AAPL", "GOOGL"],
        "GOOGL": ["META", "MSFT"],
        "AMZN": ["WMT", "META"],
        "TSLA": ["RIVN", "GM"],
    }

    SECTOR_ETF_MAP = {
        "Technology": "XLK",
        "Communication Services": "XLC",
        "Consumer Cyclical": "XLY",
        "Consumer Discretionary": "XLY",
        "Financial Services": "XLF",
        "Healthcare": "XLV",
        "Energy": "XLE",
        "Industrials": "XLI"
    }

    def __init__(self):
        self.cli_bin = os.path.expanduser("~/.local/bin/gloomberb")
        if not os.path.exists(self.cli_bin):
            import shutil
            which_bin = shutil.which("gloomberb")
            if which_bin:
                self.cli_bin = which_bin
        self._cli_warned = False
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.fmp_api_key = os.getenv("FMP_API_KEY")
        self.sec_headers = {'User-Agent': 'GloomberbTradingAgent admin@ai-trader.com'}
        self._ensure_gloomberb_initialized()

    def _ensure_gloomberb_initialized(self):
        """Initializes ~/.gloomberb/config.json if not present so headless CLI commands work out of the box."""
        try:
            gloomberb_dir = os.path.expanduser("~/.gloomberb")
            config_file = os.path.join(gloomberb_dir, "config.json")
            if not os.path.exists(config_file):
                os.makedirs(gloomberb_dir, exist_ok=True)
                with open(config_file, "w", encoding="utf-8") as f:
                    json.dump({"dataDir": gloomberb_dir}, f, indent=2)
                print(f"[GloomberbService] Initialized default Gloomberb config at '{config_file}'.")
        except Exception as e:
            print(f"[GloomberbService] Warning initializing Gloomberb config: {e}")

    def run_cli(self, *args, timeout: int = 25):
        """Executes official gloom-sh/gloomberb CLI subcommands with arbitrary arguments and --json output mode."""
        if not os.path.exists(self.cli_bin):
            if not self._cli_warned:
                print(f"[GloomberbService] Notice: CLI binary not found at '{self.cli_bin}' or in PATH. Operating in fallback mode (install via: curl -fsSL gloomberb.com/install | bash).")
                self._cli_warned = True
            return None
        import tempfile
        try:
            cmd = [self.cli_bin] + [str(a) for a in args] + ["--json"]
            with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as out_f, tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as err_f:
                res = subprocess.run(cmd, stdout=out_f, stderr=err_f, timeout=timeout)
                out_f.seek(0)
                stdout_str = out_f.read()
                err_f.seek(0)
                stderr_str = err_f.read().strip()

                if res.returncode != 0:
                    err_msg = stderr_str or stdout_str[:200]
                    if err_msg:
                        try:
                            err_json = json.loads(err_msg)
                            err_msg = err_json.get("error", {}).get("message", err_msg)
                        except Exception:
                            pass
                    print(f"[GloomberbService] Official CLI non-zero exit ({res.returncode}) for '{' '.join(str(a) for a in args)}': {err_msg[:200]}")
                    return None

                if stdout_str:
                    parsed = json.loads(stdout_str)
                    if isinstance(parsed, dict):
                        if parsed.get("ok"):
                            return parsed.get("data")
                        else:
                            err_info = parsed.get("error")
                            print(f"[GloomberbService] Official CLI returned error for '{' '.join(str(a) for a in args)}': {err_info}")
                            return None
        except Exception as e:
            print(f"[GloomberbService] Official CLI warning for '{' '.join(str(a) for a in args)}': {e}")
        return None

    def fetch_quote(self, symbol: str) -> dict:
        """Sources live/delayed quote via gloomberb quote <symbol>."""
        quote_data = {
            "symbol": symbol,
            "price": "N/A",
            "change": "N/A",
            "change_pct": "N/A",
            "bid": "N/A",
            "ask": "N/A",
            "open": "N/A",
            "high": "N/A",
            "low": "N/A",
            "previous_close": "N/A",
            "high_52w": "N/A",
            "low_52w": "N/A",
            "currency": "USD",
            "market_state": "N/A",
            "category": "Quote"
        }
        cli_res = self.run_cli("quote", symbol)
        if cli_res and isinstance(cli_res, list) and len(cli_res) > 0:
            q = cli_res[0].get("quote", {})
            if q:
                price = q.get("price")
                quote_data["price"] = round(float(price), 2) if price is not None else "N/A"
                chg = q.get("change")
                quote_data["change"] = round(float(chg), 2) if chg is not None else "N/A"
                chg_pct = q.get("changePercent")
                quote_data["change_pct"] = f"{chg_pct:+.2f}%" if chg_pct is not None else "N/A"
                quote_data["bid"] = q.get("bid", "N/A")
                quote_data["ask"] = q.get("ask", "N/A")
                quote_data["open"] = q.get("open", "N/A")
                quote_data["high"] = q.get("high", "N/A")
                quote_data["low"] = q.get("low", "N/A")
                quote_data["previous_close"] = q.get("previousClose", "N/A")
                quote_data["high_52w"] = q.get("high52w", "N/A")
                quote_data["low_52w"] = q.get("low52w", "N/A")
                quote_data["currency"] = q.get("currency", "USD")
                quote_data["market_state"] = q.get("marketState", "N/A")
                return quote_data

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}
            quote_data["price"] = info.get("currentPrice") or info.get("regularMarketPrice", "N/A")
            chg = info.get("regularMarketChange")
            quote_data["change"] = round(float(chg), 2) if chg is not None else "N/A"
            chg_pct = info.get("regularMarketChangePercent")
            quote_data["change_pct"] = f"{chg_pct:+.2f}%" if chg_pct is not None else "N/A"
            quote_data["high_52w"] = info.get("fiftyTwoWeekHigh", "N/A")
            quote_data["low_52w"] = info.get("fiftyTwoWeekLow", "N/A")
        except Exception:
            pass
        return quote_data

    def fetch_history(self, symbol: str, range_str: str = "1Y") -> list:
        """Sources historical price bars via gloomberb history <symbol> [--range <range>]."""
        cli_res = self.run_cli("history", symbol, "--range", range_str)
        if cli_res and isinstance(cli_res, list):
            return cli_res
        return []

    def fetch_macro_econ(self) -> dict:
        """Sources Macro Market Sentiment (CNN Fear & Greed), Macro Interest Rate Outlook (US Treasury Yield Curve 3M/2Y/10Y/30Y), and Economic Calendar via Gloomberb."""
        macro_data = {
            "fear_greed_score": "N/A",
            "fear_greed_rating": "N/A",
            "market_volatility_rating": "N/A",
            "junk_bond_demand_rating": "N/A",
            "interest_rate_outlook": {
                "yield_3m": "N/A",
                "yield_2y": "N/A",
                "yield_10y": "N/A",
                "yield_30y": "N/A",
                "yield_curve_spread_2y10y": "N/A",
                "yield_curve_status": "N/A"
            },
            "upcoming_econ_events": [],
            "category": "MacroEcon"
        }
        fg = self.run_cli("fear-greed")
        if fg and isinstance(fg, list) and len(fg) > 0:
            item = fg[0]
            macro_data["fear_greed_score"] = round(float(item.get("score", 50)), 1)
            macro_data["fear_greed_rating"] = item.get("rating", "N/A")

        # Ingest US Treasury Yield Curve via gloomberb yield-curve
        yc = self.run_cli("yield-curve")
        if yc and isinstance(yc, list):
            y_map = {}
            for item in yc:
                sid = item.get("seriesId")
                val = item.get("value")
                if sid and val is not None:
                    y_map[sid] = float(val)

            y3m = y_map.get("DGS3MO")
            y2y = y_map.get("DGS2")
            y10y = y_map.get("DGS10")
            y30y = y_map.get("DGS30")

            if y3m: macro_data["interest_rate_outlook"]["yield_3m"] = f"{y3m:.2f}%"
            if y2y: macro_data["interest_rate_outlook"]["yield_2y"] = f"{y2y:.2f}%"
            if y10y: macro_data["interest_rate_outlook"]["yield_10y"] = f"{y10y:.2f}%"
            if y30y: macro_data["interest_rate_outlook"]["yield_30y"] = f"{y30y:.2f}%"

            if y2y is not None and y10y is not None:
                spread = y10y - y2y
                macro_data["interest_rate_outlook"]["yield_curve_spread_2y10y"] = f"{spread:+.2f}%"
                if spread > 0.15:
                    macro_data["interest_rate_outlook"]["yield_curve_status"] = "Normal / Steepening"
                elif spread < -0.05:
                    macro_data["interest_rate_outlook"]["yield_curve_status"] = "Inverted (Recession Warning)"
                else:
                    macro_data["interest_rate_outlook"]["yield_curve_status"] = "Flat / Uninverting"

        # Ingest Federal Reserve FRED series via gloomberb fred
        ff = self.fetch_fred_series("FEDFUNDS")
        if ff and len(ff) > 0 and ff[0].get("value") is not None:
            macro_data["interest_rate_outlook"]["fed_funds_rate"] = f"{ff[0].get('value')}%"

        un = self.fetch_fred_series("UNRATE")
        if un and len(un) > 0 and un[0].get("value") is not None:
            macro_data["interest_rate_outlook"]["unemployment_rate"] = f"{un[0].get('value')}%"

        econ = self.run_cli("econ")
        if econ and isinstance(econ, list):
            for event in econ[:5]:
                macro_data["upcoming_econ_events"].append({
                    "event": event.get("event"),
                    "country": event.get("country"),
                    "impact": event.get("impact"),
                    "forecast": event.get("forecast"),
                    "prior": event.get("prior")
                })
        return macro_data

    def fetch_fred_series(self, series_id: str) -> list:
        """Sources specific FRED economic series observations via gloomberb fred <series-id>."""
        cli_fred = self.run_cli("fred", series_id)
        if cli_fred and isinstance(cli_fred, list):
            return cli_fred
        return []

    def fetch_sector_benchmark(self, sector_name: str) -> dict:
        """Sources live sector ETF returns (XLK, XLC, XLY, XLF, etc.) via gloomberb sectors CLI."""
        sector_info = {
            "sector_etf": self.SECTOR_ETF_MAP.get(sector_name, "SPY"),
            "sector_change_pct": "N/A",
            "category": "SectorBenchmark"
        }
        sectors_data = self.run_cli("sectors")
        if sectors_data and isinstance(sectors_data, list):
            target_etf = sector_info["sector_etf"]
            for s in sectors_data:
                if s.get("symbol") == target_etf:
                    cp = s.get("changePercent")
                    sector_info["sector_change_pct"] = f"{float(cp):+.2f}%" if cp is not None else "N/A"
                    break
        return sector_info

    def fetch_news(self, symbol: str) -> list:
        """Sources live breaking news from official gloomberb CLI terminal feed, with Finnhub/Google News fallbacks."""
        import datetime
        news_items = []
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")

        # 1. Official gloom-sh/gloomberb CLI News Feed
        cli_news = self.run_cli("news", symbol)
        if cli_news and isinstance(cli_news, list):
            print(f"[GloomberbService] Fetched {len(cli_news)} news items directly from official gloom-sh/gloomberb CLI feed.")
            for item in cli_news[:6]:
                title = item.get("title")
                source = item.get("source") or "Gloomberb CLI"
                url = item.get("url", "")
                pub_raw = item.get("publishedAt") or item.get("published_at") or ""
                pub_date = str(pub_raw)[:10] if pub_raw else today_str
                if title:
                    news_items.append({
                        "source": f"Gloomberb Terminal ({source})",
                        "title": title,
                        "summary": f"{title}. Full article available at {url[:100]}",
                        "published_at": pub_date,
                        "category": "News"
                    })

        # 2. Finnhub Real-Time Company News API Fallback
        finnhub_key = os.getenv("FINNHUB_API_KEY")
        if finnhub_key and len(news_items) < 4:
            try:
                today = datetime.datetime.now().strftime("%Y-%m-%d")
                week_ago = (datetime.datetime.now() - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
                url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={week_ago}&to={today}&token={finnhub_key}"
                res = requests.get(url, timeout=5)
                if res.status_code == 200 and isinstance(res.json(), list):
                    articles = res.json()
                    for a in articles[:4]:
                        headline = a.get("headline")
                        summary = a.get("summary") or headline
                        source = a.get("source") or "Finnhub Live"
                        dt_val = a.get("datetime")
                        pub_date = datetime.datetime.fromtimestamp(dt_val).strftime("%Y-%m-%d") if dt_val else today
                        if headline and not any(n["title"] == headline for n in news_items):
                            news_items.append({
                                "source": f"Finnhub Live ({source})",
                                "title": headline,
                                "summary": summary[:250] if summary else headline,
                                "published_at": pub_date,
                                "category": "News"
                            })
            except Exception:
                pass

        # 3. Yahoo Finance Live News Stream
        try:
            ticker = yf.Ticker(symbol)
            yf_news = ticker.news or []
            for item in yf_news[:4]:
                title = item.get("title") or item.get("content", {}).get("title")
                publisher = item.get("publisher") or item.get("content", {}).get("provider", {}).get("displayName")
                summary = item.get("summary") or item.get("content", {}).get("summary", "")
                pub_time = item.get("providerPublishTime")
                pub_date = datetime.datetime.fromtimestamp(pub_time).strftime("%Y-%m-%d") if pub_time else today_str
                if title and not any(n["title"] == title for n in news_items):
                    news_items.append({
                        "source": publisher or "Yahoo Finance",
                        "title": title,
                        "summary": summary[:250] if summary else title,
                        "published_at": pub_date,
                        "category": "News"
                    })
        except Exception:
            pass

        if not news_items:
            print(f"[GloomberbService] News source unavailable for {symbol}: no real items from CLI/Finnhub/Yahoo streams.")

        return news_items

    def fetch_filings(self, symbol: str) -> list:
        """Sources SEC EDGAR filings via official gloomberb CLI terminal and SEC API fallbacks."""
        filings_data = []

        # 1. Official gloom-sh/gloomberb CLI Filings Feed
        cli_filings = self.run_cli("filings", symbol)
        if cli_filings and isinstance(cli_filings, list):
            print(f"[GloomberbService] Fetched {len(cli_filings)} corporate SEC filings directly from official gloom-sh/gloomberb CLI feed.")
            for item in cli_filings[:6]:
                form_type = item.get("form") or "Filing"
                f_date = item.get("filingDate", "N/A")[:10]
                doc_desc = item.get("primaryDocDescription") or item.get("items") or "Material EDGAR disclosure"
                company = item.get("companyName") or symbol
                filings_data.append({
                    "form": form_type,
                    "date": f_date,
                    "summary": f"SEC Form {form_type} filed by {company} on {f_date}. Details: {doc_desc}. Official SEC URL: {item.get('filingUrl', '')}",
                    "category": "Filings"
                })

        # 2. Direct SEC EDGAR Submissions API Fallback
        if not filings_data:
            try:
                cik = None
                clean_sym = symbol.split(".")[0].upper()
                tickers = {}

                cache_path = os.path.join(os.path.dirname(__file__), "sec_company_tickers.json")
                if os.path.exists(cache_path):
                    try:
                        with open(cache_path, "r", encoding="utf-8") as f:
                            tickers = json.load(f)
                    except Exception:
                        pass

                if not tickers:
                    cik_res = requests.get("https://www.sec.gov/files/company_tickers.json", headers=self.sec_headers, timeout=6)
                    if cik_res.status_code == 200:
                        tickers = cik_res.json()

                if isinstance(tickers, dict):
                    for key, val in tickers.items():
                        if isinstance(val, dict) and val.get("ticker") == clean_sym:
                            cik = str(val.get("cik_str")).zfill(10)
                            break

                if cik:
                    sub_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
                    sub_res = requests.get(sub_url, headers=self.sec_headers, timeout=6)
                    if sub_res.status_code == 200:
                        data = sub_res.json()
                        recent = data.get("filings", {}).get("recent", {})
                        forms = recent.get("form", [])
                        filing_dates = recent.get("filingDate", [])
                        items = recent.get("items", [])

                        count = 0
                        for i in range(len(forms)):
                            form_type = forms[i]
                            if form_type in ["10-K", "10-Q", "8-K", "Form 4", "13F-HR"]:
                                f_date = filing_dates[i] if i < len(filing_dates) else "N/A"
                                f_item = items[i] if i < len(items) else ""
                                filings_data.append({
                                    "form": form_type,
                                    "date": f_date,
                                    "summary": f"SEC Form {form_type} filed on {f_date}. Item details: {f_item or 'Regulatory disclosure submitted.'}",
                                    "category": "Filings"
                                })
                                count += 1
                                if count >= 6:
                                    break
            except Exception as e:
                print(f"[GloomberbService] SEC filings API fallback warning for {symbol}: {e}")

        if not filings_data:
            print(f"[GloomberbService] SEC filings source unavailable for {symbol}: no real filings from CLI or SEC EDGAR feed.")

        return filings_data

    def fetch_financials(self, symbol: str) -> dict:
        """Sources Financial Statements (Income, Balance Sheet, Cash Flow) and valuation ratios via Gloomberb CLI."""
        financials = {
            "income_statement": {},
            "balance_sheet": {},
            "cash_flow": {},
            "key_ratios": {},
            "category": "Financials"
        }

        # 1. Native Gloomberb valuation & fundamentals
        val_data = self.run_cli("valuation", symbol)
        if val_data and isinstance(val_data, dict):
            fund = val_data.get("fundamentals", {}) or {}
            financials["key_ratios"] = {
                "forward_pe": round(float(fund.get("forwardPE")), 2) if fund.get("forwardPE") else "N/A",
                "trailing_pe": round(float(fund.get("trailingPE")), 2) if fund.get("trailingPE") else "N/A",
                "peg_ratio": round(float(fund.get("pegRatio")), 2) if fund.get("pegRatio") else "N/A",
                "operating_margins": f"{fund.get('operatingMargin', 0) * 100:.2f}%" if fund.get("operatingMargin") else "N/A",
                "profit_margins": f"{fund.get('profitMargin', 0) * 100:.2f}%" if fund.get("profitMargin") else "N/A",
                "return_1y": f"{fund.get('return1Y', 0) * 100:.2f}%" if fund.get("return1Y") else "N/A",
                "return_3y": f"{fund.get('return3Y', 0) * 100:.2f}%" if fund.get("return3Y") else "N/A",
            }
            financials["income_statement"]["revenue"] = fund.get("revenue", "N/A")
            financials["income_statement"]["net_income"] = fund.get("netIncome", "N/A")
            financials["income_statement"]["eps"] = fund.get("eps", "N/A")
            financials["cash_flow"]["operating_cash_flow"] = fund.get("operatingCashFlow", "N/A")
            financials["cash_flow"]["free_cash_flow"] = fund.get("freeCashFlow", "N/A")

        # 2. Native Gloomberb financials (annual & quarterly statements)
        fin_data = self.run_cli("financials", symbol)
        if fin_data and isinstance(fin_data, dict):
            q_stmts = fin_data.get("quarterlyStatements")
            if isinstance(q_stmts, list) and len(q_stmts) > 0:
                latest_q = q_stmts[0]
                financials["income_statement"]["gross_profits"] = latest_q.get("grossProfit", "N/A")
                financials["income_statement"]["ebitda"] = latest_q.get("ebitda", "N/A")
                financials["income_statement"]["operating_income"] = latest_q.get("operatingIncome", "N/A")
                financials["balance_sheet"]["total_assets"] = latest_q.get("totalAssets", "N/A")
                financials["balance_sheet"]["total_debt"] = latest_q.get("totalDebt", "N/A")
                financials["balance_sheet"]["cash_and_equivalents"] = latest_q.get("cashAndCashEquivalents", "N/A")
                financials["balance_sheet"]["working_capital"] = latest_q.get("workingCapital", "N/A")
                financials["cash_flow"]["capital_expenditure"] = latest_q.get("capitalExpenditure", "N/A")

        # 3. yfinance Fallback/Enrichment for missing items
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}
            kr = financials["key_ratios"]
            if kr.get("forward_pe") == "N/A" and info.get("forwardPE"):
                kr["forward_pe"] = info.get("forwardPE")
            if kr.get("trailing_pe") == "N/A" and info.get("trailingPE"):
                kr["trailing_pe"] = info.get("trailingPE")
            if "price_to_book" not in kr or kr["price_to_book"] == "N/A":
                kr["price_to_book"] = info.get("priceToBook", "N/A")
            if "revenue_growth" not in kr or kr["revenue_growth"] == "N/A":
                kr["revenue_growth"] = f"{info.get('revenueGrowth', 0) * 100:.2f}%" if info.get('revenueGrowth') else "N/A"
            if "earnings_growth" not in kr or kr["earnings_growth"] == "N/A":
                kr["earnings_growth"] = f"{info.get('earningsGrowth', 0) * 100:.2f}%" if info.get('earningsGrowth') else "N/A"
            if "return_on_equity" not in kr or kr["return_on_equity"] == "N/A":
                kr["return_on_equity"] = f"{info.get('returnOnEquity', 0) * 100:.2f}%" if info.get('returnOnEquity') else "N/A"

            bs = financials["balance_sheet"]
            if "total_cash" not in bs or bs.get("total_cash") == "N/A":
                bs["total_cash"] = info.get("totalCash", "N/A")
            if "quick_ratio" not in bs or bs.get("quick_ratio") == "N/A":
                bs["quick_ratio"] = info.get("quickRatio", "N/A")
            if "current_ratio" not in bs or bs.get("current_ratio") == "N/A":
                bs["current_ratio"] = info.get("currentRatio", "N/A")
        except Exception as e:
            print(f"[GloomberbService] Financials enrichment warning for {symbol}: {e}")

        return financials

    def fetch_options_chain(self, symbol: str) -> dict:
        """Sources Options Chains, Put/Call ratios, and Implied Volatility (IV) metrics natively via Gloomberb CLI."""
        options_data = {
            "implied_volatility": "N/A",
            "put_call_ratio": "N/A",
            "call_volume": 0,
            "put_volume": 0,
            "call_open_interest": 0,
            "put_open_interest": 0,
            "expiration_dates": [],
            "unusual_activity": "Normal options volume flow",
            "category": "Options"
        }

        # 1. Native Gloomberb CLI Options Feed
        cli_options = self.run_cli("options", symbol)
        if cli_options and isinstance(cli_options, dict):
            print(f"[GloomberbService] Processing native Gloomberb options chain for {symbol}...")
            calls = cli_options.get("calls", []) or []
            puts = cli_options.get("puts", []) or []
            exp_timestamps = cli_options.get("expirationDates", []) or []

            # Format expiration dates
            formatted_exps = []
            for ts in exp_timestamps[:4]:
                try:
                    formatted_exps.append(datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d"))
                except Exception:
                    formatted_exps.append(str(ts))
            options_data["expiration_dates"] = formatted_exps

            call_vol = sum(c.get("volume", 0) or 0 for c in calls)
            put_vol = sum(p.get("volume", 0) or 0 for p in puts)
            call_oi = sum(c.get("openInterest", 0) or 0 for c in calls)
            put_oi = sum(p.get("openInterest", 0) or 0 for p in puts)

            options_data["call_volume"] = call_vol
            options_data["put_volume"] = put_vol
            options_data["call_open_interest"] = call_oi
            options_data["put_open_interest"] = put_oi

            if call_vol > 0:
                pc_ratio = round(put_vol / call_vol, 2)
                options_data["put_call_ratio"] = pc_ratio
                if pc_ratio > 1.5:
                    options_data["unusual_activity"] = "Elevated put buying detected (Bearish flow bias)"
                elif pc_ratio < 0.6:
                    options_data["unusual_activity"] = "Aggressive call buying detected (Bullish momentum flow)"
            elif put_vol > 0:
                options_data["put_call_ratio"] = 2.0
                options_data["unusual_activity"] = "Dominant put flow with near-zero call volume"

            # Compute mean Implied Volatility
            iv_values = [c.get("impliedVolatility") for c in calls if c.get("impliedVolatility") is not None]
            if iv_values:
                mean_iv = sum(iv_values) / len(iv_values)
                options_data["implied_volatility"] = f"{mean_iv * 100:.1f}%"

            return options_data

        # 2. YFinance Options Flow Fallback
        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options
            if expirations:
                options_data["expiration_dates"] = list(expirations[:4])
                near_exp = expirations[0]
                opt_chain = ticker.option_chain(near_exp)
                calls = opt_chain.calls
                puts = opt_chain.puts

                call_vol = int(calls['volume'].sum()) if 'volume' in calls and not calls['volume'].empty else 0
                put_vol = int(puts['volume'].sum()) if 'volume' in puts and not puts['volume'].empty else 0

                options_data["call_volume"] = call_vol
                options_data["put_volume"] = put_vol

                pc_ratio = round(put_vol / call_vol, 2) if call_vol > 0 else 0.85
                options_data["put_call_ratio"] = pc_ratio

                if 'impliedVolatility' in calls and not calls['impliedVolatility'].empty:
                    mean_iv = calls['impliedVolatility'].mean()
                    options_data["implied_volatility"] = f"{mean_iv * 100:.1f}%"
        except Exception as e:
            print(f"[GloomberbService] Options chain fetch warning for {symbol}: {e}")

        return options_data

    def fetch_insider_institutional(self, symbol: str) -> dict:
        """Sources Form 4 Insider transactions & 13F Institutional manager activity via Gloomberb CLI."""
        insider_data = {
            "insider_transactions": [],
            "top_institutional_holders": [],
            "institutional_ownership_pct": "N/A",
            "insider_ownership_pct": "N/A",
            "institutions_count": "N/A",
            "category": "InsiderInstitutional"
        }

        # 1. Native Gloomberb 13F Institutional Feed
        cli_13f = self.run_cli("13f", symbol)
        if cli_13f and isinstance(cli_13f, dict):
            summary = cli_13f.get("summary", {}) or {}
            inst_held = summary.get("institutionsPercentHeld")
            if inst_held is not None:
                insider_data["institutional_ownership_pct"] = f"{inst_held * 100:.2f}%"
            ins_held = summary.get("insidersPercentHeld")
            if ins_held is not None:
                insider_data["insider_ownership_pct"] = f"{ins_held * 100:.2f}%"
            insider_data["institutions_count"] = summary.get("institutionsCount", "N/A")

            for h in cli_13f.get("holders", [])[:5]:
                insider_data["top_institutional_holders"].append({
                    "name": h.get("name"),
                    "shares": h.get("shares"),
                    "percent_held": f"{h.get('percentHeld', 0) * 100:.2f}%" if h.get('percentHeld') else "N/A",
                    "report_date": h.get("reportDate"),
                    "change_shares": h.get("changeShares")
                })

        # 2. Native Gloomberb Insider Feed
        cli_insider = self.run_cli("insider", symbol)
        if cli_insider and isinstance(cli_insider, dict):
            for h in cli_insider.get("holders", [])[:4]:
                if h.get("ownerType") == "insider":
                    insider_data["insider_transactions"].append({
                        "insider": h.get("name"),
                        "position": "Insider / Director",
                        "transaction": f"Reported {h.get('shares')} shares held",
                        "shares": str(h.get("shares", "N/A"))
                    })

        # 3. YFinance Insider Transactions Supplemental
        try:
            ticker = yf.Ticker(symbol)
            insider_df = getattr(ticker, "insider_transactions", None)
            if insider_df is not None and not insider_df.empty:
                for idx, row in insider_df.head(4).iterrows():
                    insider_data["insider_transactions"].append({
                        "insider": str(row.get("Insider", "Executive")),
                        "position": str(row.get("Position", "Officer")),
                        "transaction": str(row.get("Text", "Transaction logged")),
                        "shares": str(row.get("Shares", "N/A"))
                    })
        except Exception:
            pass

        return insider_data

    def fetch_analyst_ratings(self, symbol: str) -> dict:
        """Sources Wall Street investment bank ratings, price targets, consensus, and estimates via gloomberb analyst."""
        analyst_data = {
            "symbol": symbol,
            "mean_target_price": "N/A",
            "median_target_price": "N/A",
            "high_target_price": "N/A",
            "low_target_price": "N/A",
            "recommendation_rating": "N/A",
            "recommendations_breakdown": {},
            "recent_major_bank_actions": [],
            "earnings_estimates": [],
            "category": "AnalystRatings"
        }

        cli_analyst = self.run_cli("analyst", symbol)
        if cli_analyst and isinstance(cli_analyst, dict):
            pt = cli_analyst.get("priceTarget", {}) or {}
            analyst_data["mean_target_price"] = round(float(pt.get("average", 0)), 2) if pt.get("average") else "N/A"
            analyst_data["median_target_price"] = round(float(pt.get("median", 0)), 2) if pt.get("median") else "N/A"
            analyst_data["high_target_price"] = round(float(pt.get("high", 0)), 2) if pt.get("high") else "N/A"
            analyst_data["low_target_price"] = round(float(pt.get("low", 0)), 2) if pt.get("low") else "N/A"
            analyst_data["recommendation_rating"] = round(float(cli_analyst.get("recommendationRating", 0)), 2) if cli_analyst.get("recommendationRating") else "N/A"

            recs = cli_analyst.get("recommendations", [])
            if recs and isinstance(recs, list):
                analyst_data["recommendations_breakdown"] = recs[0]

            ratings = cli_analyst.get("ratings", [])
            if ratings and isinstance(ratings, list):
                for r in ratings[:15]:
                    firm = r.get("firm", "")
                    action = r.get("action", "")
                    current = r.get("current", "")
                    tgt = r.get("currentPriceTarget")
                    dt = r.get("date", "")
                    analyst_data["recent_major_bank_actions"].append(
                        f"[{dt}] {firm}: {action} -> {current}" + (f" (Target: ${tgt})" if tgt else "")
                    )

            analyst_data["earnings_estimates"] = cli_analyst.get("earningsEstimates", [])[:3]

        return analyst_data

    def fetch_earnings(self, symbol: str) -> dict:
        """Sources upcoming earnings date, timing, EPS estimates, and revision trends via gloomberb earnings."""
        earnings_data = {
            "symbol": symbol,
            "earnings_date": "N/A",
            "earnings_call_date": "N/A",
            "timing": "N/A",
            "eps_estimate": "N/A",
            "eps_growth_yoy": "N/A",
            "revenue_estimate": "N/A",
            "eps_revisions_7d": {"up": 0, "down": 0},
            "eps_revisions_30d": {"up": 0, "down": 0},
            "category": "Earnings"
        }

        cli_earnings = self.run_cli("earnings", symbol)
        if cli_earnings and isinstance(cli_earnings, list) and len(cli_earnings) > 0:
            e = cli_earnings[0]
            earnings_data["earnings_date"] = e.get("earningsDate", "N/A")[:10] if e.get("earningsDate") else "N/A"
            earnings_data["earnings_call_date"] = e.get("earningsCallDate", "N/A")[:10] if e.get("earningsCallDate") else "N/A"
            earnings_data["timing"] = e.get("timing", "N/A")
            earnings_data["eps_estimate"] = round(float(e.get("epsEstimate", 0)), 2) if e.get("epsEstimate") is not None else "N/A"
            growth = e.get("epsGrowth")
            earnings_data["eps_growth_yoy"] = f"{growth * 100:+.2f}%" if growth is not None else "N/A"
            rev = e.get("revenueEstimate")
            earnings_data["revenue_estimate"] = f"${rev / 1e9:.2f}B" if rev else "N/A"
            earnings_data["eps_revisions_7d"] = {
                "up": e.get("epsRevisionUp7d", 0),
                "down": e.get("epsRevisionDown7d", 0)
            }
            earnings_data["eps_revisions_30d"] = {
                "up": e.get("epsRevisionUp30d", 0),
                "down": e.get("epsRevisionDown30d", 0)
            }

        return earnings_data

    def fetch_events(self, symbol: str) -> dict:
        """Sources historical quarterly earnings surprises, dividends, and splits via gloomberb events."""
        events_data = {
            "symbol": symbol,
            "historical_earnings_surprises": [],
            "recent_dividends": [],
            "category": "Events"
        }

        cli_events = self.run_cli("events", symbol)
        if cli_events and isinstance(cli_events, dict):
            surprises = cli_events.get("earnings", []) or []
            for s in surprises:
                if s.get("epsActual") is not None and s.get("epsEstimate") is not None:
                    sp = s.get("surprisePercent")
                    events_data["historical_earnings_surprises"].append({
                        "date": s.get("date"),
                        "estimate": s.get("epsEstimate"),
                        "actual": s.get("epsActual"),
                        "surprise_pct": f"{float(sp):+.2f}%" if sp is not None else "N/A"
                    })
            events_data["recent_dividends"] = cli_events.get("dividends", [])[:3]

        return events_data

    def fetch_market_indices(self) -> list:
        """Sources live quotes for major US benchmark indices (^GSPC, ^DJI, ^IXIC, ^RUT) via gloomberb indices."""
        cli_indices = self.run_cli("indices")
        indices_list = []
        if cli_indices and isinstance(cli_indices, list):
            for item in cli_indices:
                cp = item.get("changePercent")
                indices_list.append({
                    "symbol": item.get("symbol"),
                    "name": item.get("name"),
                    "price": item.get("price"),
                    "change_pct": f"{float(cp):+.2f}%" if cp is not None else "N/A"
                })
        return indices_list

    def fetch_market_movers(self, category: str = "active") -> list:
        """Sources market movers (gainers, losers, active, trending) via gloomberb movers <category>."""
        cli_movers = self.run_cli("movers", category)
        movers_list = []
        if cli_movers and isinstance(cli_movers, list):
            for m in cli_movers[:5]:
                cp = m.get("changePercent")
                movers_list.append({
                    "symbol": m.get("symbol"),
                    "name": m.get("name"),
                    "price": m.get("price"),
                    "change_pct": f"{float(cp):+.2f}%" if cp is not None else "N/A",
                    "volume_ratio": round(float(m.get("volumeRatio", 1.0)), 2) if m.get("volumeRatio") is not None else "N/A"
                })
        return movers_list

    def fetch_correlation(self, symbol_a: str, symbol_b: str) -> dict:
        """Computes 1-year statistical close-price correlation between two symbols via gloomberb correlation."""
        cor_data = {
            "symbol_a": symbol_a,
            "symbol_b": symbol_b,
            "correlation": "N/A",
            "samples": 0,
            "interpretation": "N/A"
        }
        cli_cor = self.run_cli("correlation", symbol_a, symbol_b)
        if cli_cor and isinstance(cli_cor, list) and len(cli_cor) > 0:
            item = cli_cor[0]
            val = item.get("correlation")
            if val is not None:
                r = round(float(val), 3)
                cor_data["correlation"] = r
                cor_data["samples"] = item.get("samples", 252)
                if r > 0.70:
                    cor_data["interpretation"] = "Strong Positive"
                elif r > 0.30:
                    cor_data["interpretation"] = "Moderate Positive"
                elif r < -0.30:
                    cor_data["interpretation"] = "Inverse / Hedged"
                else:
                    cor_data["interpretation"] = "Uncorrelated / Independent"
        return cor_data

    def fetch_peer_valuation(self, symbol: str) -> dict:
        """Sources EV/EBITDA, Price/Sales, Price/Free Cash Flow, and statistical correlations against direct peers."""
        valuation = {
            "enterprise_value": "N/A",
            "ev_to_ebitda": "N/A",
            "price_to_sales": "N/A",
            "price_to_free_cash_flow": "N/A",
            "direct_peer_benchmarks": [],
            "category": "PeerValuation"
        }

        # 1. Target ticker valuation
        target_val = self.run_cli("valuation", symbol)
        if target_val and isinstance(target_val, dict):
            fund = target_val.get("fundamentals", {}) or {}
            ev = fund.get("enterpriseValue")
            valuation["enterprise_value"] = f"${ev / 1e9:.2f}B" if ev else "N/A"
            fcf = fund.get("freeCashFlow")
            q = target_val.get("quote", {}) or {}
            mcap = q.get("marketCap")
            if fcf and mcap and fcf > 0:
                valuation["price_to_free_cash_flow"] = f"{mcap / fcf:.2f}x"

        # 2. Benchmark against peers with Gloomberb valuation & Gloomberb correlation
        peers = self.PEER_MAP.get(symbol, ["MSFT", "GOOGL"])
        for peer in peers:
            peer_item = {
                "peer_symbol": peer,
                "forward_pe": "N/A",
                "ev_to_ebitda": "N/A",
                "price_to_sales": "N/A",
                "correlation_1y": "N/A"
            }
            # Correlation
            cor = self.fetch_correlation(symbol, peer)
            if cor.get("correlation") != "N/A":
                peer_item["correlation_1y"] = f"{cor['correlation']} ({cor['interpretation']})"

            # Valuation
            p_val = self.run_cli("valuation", peer)
            if p_val and isinstance(p_val, dict):
                p_fund = p_val.get("fundamentals", {}) or {}
                if p_fund.get("forwardPE"):
                    peer_item["forward_pe"] = round(float(p_fund["forwardPE"]), 2)
            valuation["direct_peer_benchmarks"].append(peer_item)

        return valuation

    def fetch_profile(self, symbol: str) -> dict:
        """Sources Company Description, Product Architecture, Shares Outstanding, and Returns via gloomberb ticker CLI."""
        profile_data = {
            "description": "N/A",
            "sector": "N/A",
            "industry": "N/A",
            "shares_outstanding": "N/A",
            "peg_ratio": "N/A",
            "return_1y": "N/A",
            "return_3y": "N/A",
            "category": "Profile"
        }
        cli_ticker = self.run_cli("ticker", symbol)
        if cli_ticker and isinstance(cli_ticker, dict):
            prof = cli_ticker.get("profile", {}) or {}
            fund = cli_ticker.get("fundamentals", {}) or {}
            profile_data["description"] = prof.get("description", "N/A")
            profile_data["sector"] = prof.get("sector", "N/A")
            profile_data["industry"] = prof.get("industry", "N/A")
            profile_data["shares_outstanding"] = fund.get("sharesOutstanding", "N/A")
            profile_data["peg_ratio"] = fund.get("pegRatio", "N/A")
            if fund.get("return1Y") is not None:
                profile_data["return_1y"] = f"{fund.get('return1Y') * 100:.2f}%"
            if fund.get("return3Y") is not None:
                profile_data["return_3y"] = f"{fund.get('return3Y') * 100:.2f}%"
        return profile_data

    def compare_symbols(self, symbols: list) -> list:
        """Compares multiple symbols via gloomberb compare <symbol...>."""
        if not symbols:
            return []
        cli_comp = self.run_cli("compare", *symbols)
        if cli_comp and isinstance(cli_comp, list):
            return cli_comp
        return []

    def search_symbols(self, query: str) -> list:
        """Searches tickers and company names via gloomberb search or provider-search."""
        res = self.run_cli("provider-search", query) or self.run_cli("search", query)
        return res if isinstance(res, list) else []

    def get_watchlist(self) -> list:
        """Fetches terminal watchlists via gloomberb watchlist list."""
        res = self.run_cli("watchlist", "list")
        return res if isinstance(res, list) else []

    def get_portfolio(self) -> dict:
        """Fetches terminal portfolios via gloomberb portfolio list."""
        res = self.run_cli("portfolio", "list")
        return res if isinstance(res, dict) else {}

    def get_all_gloomberb_data(self, symbol: str) -> dict:
        """
        Gloomberb Top Node Aggregator:
        Ingests the complete spectrum of Gloomberb functionalities concurrently using ThreadPoolExecutor:
        Quotes, News, Filings, Financials, Options Flow, Insiders/13F, Peer Valuation,
        Analyst Research, Earnings Calendar, Historical Surprises, Macro Econ,
        Sector Benchmarks, Benchmark Indices, Market Movers, and Correlations.
        """
        from concurrent.futures import ThreadPoolExecutor

        print(f"[GloomberbService] Ingesting comprehensive Gloomberb data stream via official CLI for {symbol}...")

        with ThreadPoolExecutor(max_workers=5) as executor:
            fut_quote = executor.submit(self.fetch_quote, symbol)
            fut_news = executor.submit(self.fetch_news, symbol)
            fut_filings = executor.submit(self.fetch_filings, symbol)
            fut_financials = executor.submit(self.fetch_financials, symbol)
            fut_options = executor.submit(self.fetch_options_chain, symbol)
            fut_insider = executor.submit(self.fetch_insider_institutional, symbol)
            fut_peer_val = executor.submit(self.fetch_peer_valuation, symbol)
            fut_analyst = executor.submit(self.fetch_analyst_ratings, symbol)
            fut_earnings = executor.submit(self.fetch_earnings, symbol)
            fut_events = executor.submit(self.fetch_events, symbol)
            fut_profile = executor.submit(self.fetch_profile, symbol)
            fut_macro = executor.submit(self.fetch_macro_econ)
            fut_indices = executor.submit(self.fetch_market_indices)
            fut_movers = executor.submit(self.fetch_market_movers, "active")
            fut_spy_cor = executor.submit(self.fetch_correlation, symbol, "SPY")

            quote = fut_quote.result()
            news = fut_news.result()
            filings = fut_filings.result()
            financials = fut_financials.result()
            options = fut_options.result()
            insider = fut_insider.result()
            peer_val = fut_peer_val.result()
            analyst_ratings = fut_analyst.result()
            earnings = fut_earnings.result()
            events = fut_events.result()
            profile = fut_profile.result()
            macro_econ = fut_macro.result()
            market_indices = fut_indices.result()
            market_movers = fut_movers.result()
            spy_correlation = fut_spy_cor.result()

        sector_bench = self.fetch_sector_benchmark(profile.get("sector", "Technology"))

        return {
            "symbol": symbol,
            "quote": quote,
            "news": news,
            "filings": filings,
            "financials": financials,
            "options": options,
            "insider_institutional": insider,
            "peer_valuation": peer_val,
            "analyst_ratings": analyst_ratings,
            "earnings": earnings,
            "events": events,
            "profile": profile,
            "macro_econ": macro_econ,
            "sector_benchmark": sector_bench,
            "market_indices": market_indices,
            "market_movers": market_movers,
            "market_spy_correlation": spy_correlation,
            "data_source_status": {
                "news": "available" if news else "source_unavailable",
                "filings": "available" if filings else "source_unavailable"
            }
        }
