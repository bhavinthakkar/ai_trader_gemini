import os
import json
import requests
import yfinance as yf
from dotenv import load_dotenv
from ollama import chat

load_dotenv()

COMPANY_NAME_MAP = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "Nvidia",
    "AMZN": "Amazon",
    "META": "Meta",
    "GOOGL": "Google",
    "TSLA": "Tesla",
    "AMD": "AMD",
    "NFLX": "Netflix",
    "PLTR": "Palantir"
}

class NewsAgent:
    """
    News Agent: Fetches real-time news articles using newsapi.ai (Event Registry)
    or newsapi.org (with an automatic yfinance fallback), synthesizes key developments,
    and estimates short-term market impact via local LLM (gemma3:4b).
    """

    def __init__(self, model_name=None):
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "gemma3:4b")
        self.news_api_key = os.getenv("NEWS_API_KEY")

    def fetch_from_newsapi_ai(self, symbol: str) -> list:
        if not self.news_api_key:
            return []

        search_query = COMPANY_NAME_MAP.get(symbol, symbol)
        url = f"https://newsapi.ai/api/v1/article/getArticles?keyword={search_query}&articlesCount=5&articlesSortBy=date&lang=eng&resultType=articles&apiKey={self.news_api_key}"

        headlines = []
        try:
            res = requests.get(url, timeout=6)
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

    def analyze(self, symbol: str) -> dict:
        print(f"[NewsAgent] Evaluating market news & impact for {symbol}...")
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

        if not headlines:
            return {
                "news_summary": "No major catalyst news reported today.",
                "news_sentiment": "NEUTRAL",
                "estimated_impact": "LOW",
                "catalyst_note": "No fresh catalysts found.",
                "raw_headlines": [],
                "source": "None"
            }

        prompt = f"""Target Stock: {symbol}
Recent Live News Articles ({source_used}):
""" + "\n".join(f"- {h}" for h in headlines) + """

Summarize today's news and estimate its short-term market impact for swing trading (1 to 10 trading days).
Return a JSON object matching this schema:
{
  "news_summary": "1-2 sentence summary of today's key news developments",
  "news_sentiment": "BULLISH|BEARISH|NEUTRAL",
  "estimated_impact": "HIGH|MEDIUM|LOW",
  "catalyst_note": "Specific impact of this news on short-term price momentum"
}
"""

        try:
            response_obj = chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a financial news intelligence agent. Analyze news articles and output JSON."},
                    {"role": "user", "content": prompt}
                ],
                format="json"
            )
            parsed = json.loads(response_obj["message"]["content"])
            if isinstance(parsed, dict):
                parsed["raw_headlines"] = headlines
                parsed["source"] = source_used
                return parsed
        except Exception as e:
            print(f"[NewsAgent] LLM news summary error for {symbol}: {e}")

        return {
            "news_summary": f"Recent developments: {'; '.join(headlines[:2])}",
            "news_sentiment": "NEUTRAL",
            "estimated_impact": "MEDIUM",
            "catalyst_note": "Standard news flow",
            "raw_headlines": headlines,
            "source": source_used
        }
