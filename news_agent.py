import os
import json
import requests
import urllib.request
import xml.etree.ElementTree as ET
import yfinance as yf
from dotenv import load_dotenv
from llm_service import query_llm, extract_json

load_dotenv()

COMPANY_NAME_MAP = {
    "MU": "Micron",
    "SKHY": "SK Hynix",
    "AMAT": "Applied Materials",
    "STXH": "Seagate",
    "WDC": "Western Digital",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "Nvidia",
    "AMZN": "Amazon",
    "META": "Meta",
    "GOOGL": "Google",
    "TSLA": "Tesla",
    "AMD": "AMD",
    "MRVL": "Marvell",
    "PLTR": "Palantir"
}

class NewsAgent:
    """
    News Agent: Fetches real-time news articles using newsapi.ai (Event Registry)
    or newsapi.org (with an automatic yfinance fallback), synthesizes key developments,
    and estimates short-term market impact via selected LLM (gemma4:12b or Gemini 3.6 Flash).
    """

    def __init__(self, model_choice="gemini"):
        self.model_choice = model_choice
        self.news_api_key = os.getenv("NEWS_API_KEY")

    def fetch_from_newsapi_ai(self, symbol: str) -> list:
        if not self.news_api_key:
            return []

        search_query = COMPANY_NAME_MAP.get(symbol, symbol)
        url = "https://newsapi.ai/api/v1/article/getArticles"
        params = {
            "keyword": search_query,
            "keywordLoc": "title",
            "articlesCount": 5,
            "articlesSortBy": "date",
            "lang": "eng",
            "resultType": "articles",
            "apiKey": self.news_api_key
        }

        headlines = []
        try:
            res = requests.get(url, params=params, timeout=6)
            if res.status_code == 200:
                data = res.json()
                # newsapi.ai stores articles in articles.results or articles
                articles = data.get("articles", {}).get("results", [])
                if not isinstance(articles, list):
                    articles = data.get("articles", [])

                for a in articles[:4]:
                    title = a.get("title")
                    source = a.get("source", {}).get("title") or "newsapi.ai"
                    body = a.get("body", "")
                    snippet = body[:120] + "..." if body else ""
                    if title:
                        text = f"[{source}] {title}"
                        if snippet:
                            text += f" - {snippet}"
                        headlines.append(text)

                if headlines:
                    print(f"[NewsAgent] Successfully fetched {len(headlines)} articles from newsapi.ai for {symbol}.")
            else:
                print(f"[NewsAgent] newsapi.ai returned status {res.status_code} for {symbol}.")
        except Exception as e:
            print(f"[NewsAgent] newsapi.ai request error for {symbol}: {e}")

        return headlines

    def fetch_from_newsapi_org(self, symbol: str) -> list:
        if not self.news_api_key:
            return []

        search_query = COMPANY_NAME_MAP.get(symbol, symbol)
        url = f"https://newsapi.org/v2/everything?q={search_query}&sortBy=publishedAt&language=en&pageSize=5&apiKey={self.news_api_key}"

        headlines = []
        try:
            res = requests.get(url, timeout=6)
            if res.status_code == 200:
                articles = res.json().get("articles", [])
                for a in articles[:4]:
                    title = a.get("title")
                    source = a.get("source", {}).get("name") or "NewsAPI"
                    description = a.get("description")
                    if title:
                        text = f"[{source}] {title}"
                        if description:
                            text += f" - {description}"
                        headlines.append(text)
                if headlines:
                    print(f"[NewsAgent] Successfully fetched {len(headlines)} articles from newsapi.org for {symbol}.")
            else:
                print(f"[NewsAgent] newsapi.org returned status {res.status_code} for {symbol}.")
        except Exception as e:
            print(f"[NewsAgent] newsapi.org request error for {symbol}: {e}")

        return headlines

    def fetch_from_yfinance(self, symbol: str) -> list:
        headlines = []
        try:
            ticker = yf.Ticker(symbol)
            raw_news = ticker.news or []
            for item in raw_news[:4]:
                title = item.get("title") or item.get("content", {}).get("title")
                publisher = item.get("publisher") or item.get("content", {}).get("provider", {}).get("displayName")
                if title:
                    headlines.append(f"[{publisher}] {title}" if publisher else title)
            if headlines:
                print(f"[NewsAgent] Fetched {len(headlines)} news items from yfinance fallback for {symbol}.")
        except Exception as e:
            print(f"[NewsAgent] yfinance news fetch error for {symbol}: {e}")

        return headlines

    def fetch_from_rss(self, symbol: str) -> list:
        print(f"[NewsAgent] Fetching RSS feed for {symbol}...")
        url = f"https://finance.yahoo.com/rss/headline?s={symbol}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        headlines = []
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                xml_data = response.read()
                root = ET.fromstring(xml_data)
                for item in root.findall(".//item"):
                    title = item.find("title")
                    pubDate = item.find("pubDate")
                    if title is not None:
                        pub_str = f" ({pubDate.text})" if pubDate is not None else ""
                        headlines.append(f"[RSS Feed] {title.text}{pub_str}")
                        if len(headlines) >= 5:
                            break
                if headlines:
                    print(f"[NewsAgent] Successfully fetched {len(headlines)} articles from RSS feed for {symbol}.")
        except Exception as e:
            print(f"[NewsAgent] RSS news fetch error for {symbol}: {e}")
        return headlines

    def fetch_13f_filings(self, symbol: str) -> str:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.institutional_holders
            if df is None or df.empty:
                return "No recent 13F filings data found."
            
            lines = []
            for _, row in df.head(8).iterrows():
                holder = row.get("Holder", "Unknown")
                date_reported = row.get("Date Reported")
                if hasattr(date_reported, "strftime"):
                    date_reported = date_reported.strftime("%Y-%m-%d")
                shares = row.get("Shares", 0)
                value = row.get("Value", 0)
                pct_change = row.get("pctChange", 0.0)
                
                change_str = f"{pct_change*100:+.2f}%" if pct_change is not None else "0.00%"
                lines.append(f"- {holder}: {shares:,} shares (${value:,}) as of {date_reported} (Change: {change_str})")
            
            return "\n".join(lines)
        except Exception as e:
            print(f"[NewsAgent] yfinance 13F filings fetch error for {symbol}: {e}")
            return f"Error retrieving 13F data: {e}"

    def fetch_cnn_fear_and_greed(self) -> str:
        print("[NewsAgent] Fetching CNN Fear & Greed Index...")
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://edition.cnn.com/",
        }
        try:
            res = requests.get(url, headers=headers, timeout=10)
            if res.status_code == 200:
                data = res.json()
                if "fear_and_greed" in data:
                    fg = data["fear_and_greed"]
                    score = fg.get("score")
                    rating = fg.get("rating")
                    if score is not None and rating is not None:
                        return f"CNN Fear & Greed Index: {score:.1f} ({rating.upper()})"
        except Exception as e:
            print(f"[NewsAgent] Error fetching Fear & Greed: {e}")
        return "CNN Fear & Greed Index: N/A"

    def fetch_sector_sentiment(self, symbol: str) -> str:
        print(f"[NewsAgent] Fetching sector sentiment for {symbol}...")
        sector_etfs = {
            "Technology": "XLK",
            "Financials": "XLF",
            "Healthcare": "XLV",
            "Energy": "XLE",
            "Industrials": "XLI",
            "Consumer Discretionary": "XLY",
            "Consumer Staples": "XLP",
            "Utilities": "XLU",
            "Materials": "XLB",
            "Real Estate": "XLRE",
            "Communication Services": "XLC",
            "Financial Services": "XLF",
            "Healthcare Services": "XLV",
            "Consumer Cyclical": "XLY",
            "Consumer Defensive": "XLP",
            "Basic Materials": "XLB"
        }
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}
            sector = info.get("sector")
            if not sector:
                return "Sector Sentiment: Unknown sector"
            
            etf = sector_etfs.get(sector)
            if not etf:
                for key, val in sector_etfs.items():
                    if key.lower() in sector.lower() or sector.lower() in key.lower():
                        etf = val
                        break
                        
            if not etf:
                return f"Sector Sentiment: No ETF mapping for sector '{sector}'"
                
            etf_ticker = yf.Ticker(etf)
            spy_ticker = yf.Ticker("SPY")
            
            etf_hist = etf_ticker.history(period="10d")
            spy_hist = spy_ticker.history(period="10d")
            
            if etf_hist.empty or spy_hist.empty:
                return f"Sector Sentiment: Missing history for {etf} or SPY"
                
            etf_close = etf_hist["Close"]
            spy_close = spy_hist["Close"]
            
            if len(etf_close) < 6 or len(spy_close) < 6:
                return f"Sector Sentiment: Insufficient data for {etf} or SPY"
                
            etf_perf = ((etf_close.iloc[-1] - etf_close.iloc[-6]) / etf_close.iloc[-6]) * 100
            spy_perf = ((spy_close.iloc[-1] - spy_close.iloc[-6]) / spy_close.iloc[-6]) * 100
            relative_perf = etf_perf - spy_perf
            
            sentiment = "NEUTRAL"
            if relative_perf > 1.0:
                sentiment = "BULLISH (Outperforming S&P 500)"
            elif relative_perf < -1.0:
                sentiment = "BEARISH (Underperforming S&P 500)"
                
            return f"Sector: {sector} ({etf})\nSector 5D Return: {etf_perf:.2f}%\nS&P 500 5D Return: {spy_perf:.2f}%\nSector Relative Performance: {relative_perf:+.2f}%\nSector Sentiment: {sentiment}"
            
        except Exception as e:
            print(f"[NewsAgent] Error fetching sector sentiment: {e}")
            return "Sector Sentiment: Error retrieving"

    def analyze(self, symbol: str, analyst_data: dict = None) -> dict:
        print(f"[NewsAgent] Evaluating market news, 13F filings, bank ratings, SEC filings, FRED macro data & sentiment for {symbol} ...")
        headlines = []
        source_used = "yfinance"

        # 1. Try newsapi.ai (Event Registry)
        if self.news_api_key:
            headlines = self.fetch_from_newsapi_ai(symbol)
            if headlines:
                source_used = "newsapi.ai"

            # 2. Try newsapi.org if newsapi.ai returned nothing
            if not headlines:
                headlines = self.fetch_from_newsapi_org(symbol)
                if headlines:
                    source_used = "newsapi.org"

        # 3. Fallback: yfinance
        if not headlines:
            headlines = self.fetch_from_yfinance(symbol)
            source_used = "yfinance"

        # 4. Supplement with RSS Feed
        rss_headlines = self.fetch_from_rss(symbol)
        if rss_headlines:
            headlines = rss_headlines + headlines
            source_used += " + RSS Feed"

        filings_data = self.fetch_13f_filings(symbol)

        # Process Analyst/Bank Ratings, SEC Filings & Macro Data
        bank_ratings_str = "No recent major bank rating actions available."
        sec_filings_str = "No recent SEC filings available."
        macro_str = "No macroeconomic data available."
        
        if analyst_data:
            consensus = analyst_data.get("wall_street_consensus", "N/A")
            target_price = analyst_data.get("mean_target_price", "N/A")
            actions = analyst_data.get("recent_major_bank_actions", [])
            actions_lines = "\n".join(f"- {act}" for act in actions) if actions else "- No recent major bank actions"
            bank_ratings_str = f"Consensus: {consensus}\nMean Target Price: {target_price}\nRecent Bank Actions:\n{actions_lines}"

            filings_list = analyst_data.get("recent_sec_filings", [])
            sec_filings_str = "\n".join(filings_list) if filings_list else "- No recent SEC filings"
            
            macro_items = analyst_data.get("macro_data", {})
            if macro_items:
                macro_str = "\n".join(f"- {name}: {val}" for name, val in macro_items.items())

        # Fetch market-wide and sector-specific sentiment indicators
        fg_sentiment = self.fetch_cnn_fear_and_greed()
        sector_sentiment = self.fetch_sector_sentiment(symbol)

        if not headlines:
            return {
                "news_summary": "No major catalyst news reported today.",
                "news_sentiment": "NEUTRAL",
                "estimated_impact": "LOW",
                "catalyst_note": "No fresh catalysts found.",
                "raw_headlines": [],
                "source": "None",
                "13f_filings": filings_data,
                "analyst_ratings": bank_ratings_str,
                "recent_sec_filings": sec_filings_str,
                "macro_data": macro_str,
                "market_sentiment": fg_sentiment,
                "sector_sentiment": sector_sentiment
            }

        prompt = f"""Target Stock: {symbol}
Recent Live News Articles ({source_used}):
""" + "\n".join(f"- {h}" for h in headlines) + f"""

Recent 13F Filings (Top Institutional Holders & Position Changes):
{filings_data}

Wall Street Analyst Ratings & Targets:
{bank_ratings_str}

Recent SEC EDGAR Filings (10-K, 10-Q, 8-K):
{sec_filings_str}

Macroeconomic Indicators (FRED):
{macro_str}

Market Sentiment Indicators:
- {fg_sentiment}

Sector Sentiment & Relative Performance:
{sector_sentiment}

Summarize today's news, 13F holdings trends, analyst ratings/actions, recent SEC EDGAR filings, macroeconomic indicators, market sentiment, and sector sentiment, and estimate their combined short-term market impact for swing trading (1 to 10 trading days).
Return a JSON object matching this schema:
{{
  "news_summary": "1-2 sentence summary of today's key news, 13F filings, analyst rating, SEC filing, and macroeconomic/sentiment developments",
  "news_sentiment": "BULLISH|BEARISH|NEUTRAL",
  "estimated_impact": "HIGH|MEDIUM|LOW",
  "catalyst_note": "Specific impact of this news, institutional movement, analyst actions, SEC filings, macro environment, and sentiment on short-term price momentum"
}}
"""

        try:
            # For news summarization, use single-stage fast extraction (qwen) if twostage/local is selected
            effective_model = "qwen" if self.model_choice in ["twostage", "local"] else self.model_choice
            res_content = query_llm(
                system_instruction="You are a financial news, institutional holdings, analyst ratings, SEC corporate filings, macroeconomics, and market/sector sentiment intelligence agent. Analyze news, 13F data, bank ratings, SEC EDGAR filings, FRED macro data, and CNN/Sector sentiment, then output JSON.",
                user_prompt=prompt,
                model_choice=effective_model
            )
            parsed = extract_json(res_content)
            if isinstance(parsed, dict):
                parsed["raw_headlines"] = headlines
                parsed["source"] = source_used
                parsed["13f_filings"] = filings_data
                parsed["analyst_ratings"] = bank_ratings_str
                parsed["recent_sec_filings"] = sec_filings_str
                parsed["macro_data"] = macro_str
                parsed["market_sentiment"] = fg_sentiment
                parsed["sector_sentiment"] = sector_sentiment
                return parsed
        except Exception as e:
            print(f"[NewsAgent] LLM news summary error for {symbol}: {e}")

        return {
            "news_summary": f"Recent developments: {'; '.join(headlines[:2])}",
            "news_sentiment": "NEUTRAL",
            "estimated_impact": "MEDIUM",
            "catalyst_note": "Standard news, analyst, filings, macro, and sentiment flow",
            "raw_headlines": headlines,
            "source": source_used,
            "13f_filings": filings_data,
            "analyst_ratings": bank_ratings_str,
            "recent_sec_filings": sec_filings_str,
            "macro_data": macro_str,
            "market_sentiment": fg_sentiment,
            "sector_sentiment": sector_sentiment
        }
