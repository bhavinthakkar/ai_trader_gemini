import os
import json
import requests
import xml.etree.ElementTree as ET
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

class InstitutionalDataService:
    """
    Institutional Multi-Source Data Service:
    Sourcing high-conviction financial documents across 5 specialized channels:
    1. Investor-Relations (IR) Website announcements
    2. SEC EDGAR Direct Submissions (10-K, 10-Q, 8-K, Form 4)
    3. Earnings Call Transcript Highlights & Guidance
    4. Official Corporate Press Releases (PRNewswire, BusinessWire)
    5. Reputable Financial News (Reuters, Bloomberg, MarketWatch, CNBC)
    """

    def __init__(self):
        self.sec_headers = {'User-Agent': 'GloomberbTradingAgent admin@ai-trader.com'}
        self.finnhub_key = os.getenv("FINNHUB_API_KEY")
        self._ticker_map = None

    def _get_cik_from_symbol(self, symbol: str) -> str:
        """Resolves stock ticker symbol to official 10-digit SEC CIK string."""
        if self._ticker_map is None:
            mapping = {}
            cache_path = os.path.join(os.path.dirname(__file__), "sec_company_tickers.json")
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        raw = json.load(f)
                        for item in raw.values():
                            t = str(item.get("ticker", "")).strip().upper()
                            c = item.get("cik_str")
                            if t and c is not None:
                                mapping[t] = str(c).zfill(10)
                except Exception as e:
                    print(f"[InstitutionalDataService] Error reading local SEC tickers: {e}")

            if not mapping:
                try:
                    res = requests.get("https://www.sec.gov/files/company_tickers.json", headers=self.sec_headers, timeout=5)
                    if res.status_code == 200:
                        raw = res.json()
                        for item in raw.values():
                            t = str(item.get("ticker", "")).strip().upper()
                            c = item.get("cik_str")
                            if t and c is not None:
                                mapping[t] = str(c).zfill(10)
                except Exception as e:
                    print(f"[InstitutionalDataService] Error querying SEC company tickers: {e}")

            self._ticker_map = mapping

        clean_sym = symbol.strip().upper().split(".")[0]
        return self._ticker_map.get(clean_sym)

    def fetch_ir_press_releases(self, symbol: str) -> list:
        """Sources official Investor-Relations (IR) press releases & announcements."""
        import datetime
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        items = []
        try:
            url = f"https://feed.businesswire.com/rss/home/?rss=G1QFDFhJTEB2Wl5YWA=="
            res = requests.get(url, timeout=5)
            if res.status_code == 200:
                root = ET.fromstring(res.content)
                for item in root.findall('.//item')[:3]:
                    title = item.find('title').text if item.find('title') is not None else ""
                    desc = item.find('description').text if item.find('description') is not None else ""
                    if title and symbol.upper() in title.upper():
                        items.append({
                            "title": title,
                            "summary": desc[:300] if desc else title,
                            "source": "BusinessWire IR",
                            "category": "InvestorRelations",
                            "document_type": "IRRelease",
                            "reliability": 0.95,
                            "importance": 0.85,
                            "published_at": today_str
                        })
        except Exception:
            pass

        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news or []
            for n in news:
                title = n.get("title") or n.get("content", {}).get("title")
                pub = n.get("publisher") or n.get("content", {}).get("provider", {}).get("displayName", "")
                if title and ("PR" in pub or "Business Wire" in pub or "GlobeNewswire" in pub or "Accesswire" in pub):
                    items.append({
                        "title": title,
                        "summary": f"Official IR Announcement: {title}",
                        "source": f"Investor Relations ({pub})",
                        "category": "InvestorRelations",
                        "document_type": "IRRelease",
                        "reliability": 0.95,
                        "importance": 0.85,
                        "published_at": today_str
                    })
        except Exception:
            pass

        if not items:
            print(f"[InstitutionalDataService] IR press release source unavailable for {symbol}.")
        return items[:4]

    def fetch_sec_edgar_direct(self, symbol: str) -> list:
        """Queries SEC EDGAR API directly to retrieve itemized 10-K, 10-Q, 8-K, and Form 4 filings."""
        import datetime
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        filings = []
        cik = self._get_cik_from_symbol(symbol)
        if not cik:
            print(f"[InstitutionalDataService] SEC CIK not found for symbol {symbol} (non-US or unmapped).")
            return filings

        try:
            cik_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            res = requests.get(cik_url, headers=self.sec_headers, timeout=5)
            if res.status_code == 200:
                data = res.json()
                recent = data.get("filings", {}).get("recent", {})
                forms = recent.get("form", [])
                dates = recent.get("filingDate", [])
                items = recent.get("items", [])
                descs = recent.get("primaryDocDescription", [])
                for i in range(min(6, len(forms))):
                    form_name = forms[i]
                    doc_item = items[i] if i < len(items) and items[i] else ""
                    doc_desc = descs[i] if i < len(descs) and descs[i] else f"Form {form_name}"
                    f_date = dates[i] if i < len(dates) else today_str
                    filings.append({
                        "form": form_name,
                        "date": f_date,
                        "items": doc_item,
                        "summary": f"SEC EDGAR Direct Submission Form {form_name} filed on {f_date}. Description: {doc_desc}. Items: {doc_item}",
                        "source": "SEC EDGAR Direct",
                        "category": "SECFilings",
                        "document_type": f"Form-{form_name}",
                        "reliability": 1.00,
                        "importance": 0.95,
                        "published_at": f_date
                    })
        except Exception as e:
            print(f"[InstitutionalDataService] SEC EDGAR direct warning for {symbol}: {e}")

        return filings[:5]

    def fetch_earnings_call_highlights(self, symbol: str) -> list:
        """Sources recent Earnings Call Transcript highlights, CEO/CFO opening remarks, and Q&A takeaways."""
        highlights = []
        # NOTE: No fabricated "transcripts". yfinance `info` metrics (revenue/earnings growth)
        # are real metadata but are NOT an earnings-call transcript; they must never be labeled
        # as one. Return no document unless a genuine transcript source is available.
        print(f"[InstitutionalDataService] Earnings call transcript source unavailable for {symbol} (no genuine transcript provider configured).")
        return highlights

    def fetch_official_press_releases(self, symbol: str) -> list:
        """Sources official corporate press releases via PRNewswire and GlobeNewswire channels."""
        import datetime
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        press_releases = []
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news or []
            for item in news:
                title = item.get("title") or item.get("content", {}).get("title")
                pub = item.get("publisher") or item.get("content", {}).get("provider", {}).get("displayName", "")
                if title and ("Newswire" in pub or "PR" in pub or "Business Wire" in pub):
                    press_releases.append({
                        "title": title,
                        "summary": f"Official Corporate Press Release via {pub}: {title}",
                        "source": f"Press Release ({pub})",
                        "category": "OfficialPressRelease",
                        "document_type": "PressRelease",
                        "reliability": 0.95,
                        "importance": 0.80,
                        "published_at": today_str
                    })
        except Exception:
            pass

        if not press_releases:
            print(f"[InstitutionalDataService] Official press release source unavailable for {symbol}.")
        return press_releases[:4]

    def fetch_reputable_financial_news(self, symbol: str) -> list:
        """Sources Tier-1 financial media news coverage (Reuters, Bloomberg, MarketWatch, CNBC, WSJ)."""
        import datetime
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        reputable_news = []
        try:
            if self.finnhub_key:
                week_ago = (datetime.datetime.now() - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
                url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={week_ago}&to={today_str}&token={self.finnhub_key}"
                res = requests.get(url, timeout=5)
                if res.status_code == 200 and isinstance(res.json(), list):
                    for a in res.json()[:5]:
                        source = a.get("source", "Financial Media")
                        headline = a.get("headline", "")
                        summary = a.get("summary") or headline
                        dt = a.get("datetime")
                        pub_date = datetime.datetime.fromtimestamp(dt).strftime("%Y-%m-%d") if dt else today_str
                        # Dynamic reliability lookup
                        rel_score = 0.90 if any(t in source.lower() for t in ["reuters", "bloomberg", "wsj"]) else 0.70
                        if headline:
                            reputable_news.append({
                                "title": headline,
                                "summary": summary[:300],
                                "source": f"Reputable News ({source})",
                                "category": "ReputableNews",
                                "document_type": "ReputableNews",
                                "reliability": rel_score,
                                "importance": 0.75,
                                "published_at": pub_date
                            })
        except Exception:
            pass

        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news or []
            for item in news[:5]:
                title = item.get("title") or item.get("content", {}).get("title")
                publisher = item.get("publisher") or item.get("content", {}).get("provider", {}).get("displayName", "Financial Media")
                summary = item.get("summary") or item.get("content", {}).get("summary", title)
                rel_score = 0.90 if any(t in publisher.lower() for t in ["reuters", "bloomberg", "wsj"]) else 0.70
                if title and not any(r["title"] == title for r in reputable_news):
                    reputable_news.append({
                        "title": title,
                        "summary": summary[:300] if summary else title,
                        "source": f"Reputable Media ({publisher})",
                        "category": "ReputableNews",
                        "document_type": "ReputableNews",
                        "reliability": rel_score,
                        "importance": 0.75,
                        "published_at": today_str
                    })
        except Exception:
            pass

        return reputable_news[:6]

    def get_all_institutional_data(self, symbol: str) -> dict:
        """
        Aggregates documents across all 5 institutional channels:
        IR Website, SEC EDGAR Direct, Earnings Calls, Press Releases, and Reputable Financial News.
        """
        print(f"[InstitutionalDataService] Sourcing multi-channel institutional data for {symbol}...")
        ir_prs = self.fetch_ir_press_releases(symbol)
        sec_direct = self.fetch_sec_edgar_direct(symbol)
        earnings_calls = self.fetch_earnings_call_highlights(symbol)
        official_prs = self.fetch_official_press_releases(symbol)
        reputable_news = self.fetch_reputable_financial_news(symbol)

        return {
            "symbol": symbol,
            "investor_relations": ir_prs,
            "sec_edgar_direct": sec_direct,
            "earnings_calls": earnings_calls,
            "official_press_releases": official_prs,
            "reputable_news": reputable_news,
            "source_status": {
                "investor_relations": "available" if ir_prs else "source_unavailable",
                "sec_edgar_direct": "available" if sec_direct else "source_unavailable",
                "earnings_calls": "available" if earnings_calls else "source_unavailable",
                "official_press_releases": "available" if official_prs else "source_unavailable",
                "reputable_news": "available" if reputable_news else "source_unavailable"
            }
        }
