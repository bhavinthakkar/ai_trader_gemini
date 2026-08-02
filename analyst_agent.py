import os
import requests
import finnhub
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

class AnalystAgent:
    """
    Analyst Agent: Extracts Wall Street investment bank rating actions, upgrades/downgrades,
    consensus recommendations, and target prices using Finnhub or Financial Modeling Prep (FMP),
    with an automatic yfinance fallback.
    """

    MAJOR_BANKS = [
        'Bank of America', 'BofA', 'Barclays', 'Credit Suisse', 'Deutsche Bank',
        'Evercore', 'Goldman Sachs', 'JPMorgan', 'JP Morgan', 'Morgan Stanley',
        'UBS', 'Citigroup', 'Citi', 'Jefferies', 'Bernstein', 'Wells Fargo'
    ]

    def __init__(self):
        self.finnhub_key = os.getenv("FINNHUB_API_KEY", "").strip()
        self.fmp_key = os.getenv("FMP_API_KEY", "").strip()
        self.finnhub_client = finnhub.Client(api_key=self.finnhub_key) if self.finnhub_key else None

    def fetch_from_fmp(self, symbol: str) -> dict:
        if not self.fmp_key:
            return {}

        print(f"[AnalystAgent] Querying Financial Modeling Prep (FMP) for {symbol}...")
        recent_actions = []
        target_price = "N/A"
        consensus = "N/A"

        try:
            # 1. Price Target Summary (FMP Stable API)
            pt_url = f"https://financialmodelingprep.com/stable/price-target-summary?symbol={symbol}&apikey={self.fmp_key}"
            res_pt = requests.get(pt_url, timeout=5)
            if res_pt.status_code == 200 and res_pt.json():
                data = res_pt.json()
                if isinstance(data, list) and len(data) > 0:
                    target_price = round(float(data[0].get("lastMonthAvgPriceTarget", 0)), 2)
            else:
                print(f"[AnalystAgent] FMP price target API returned status {res_pt.status_code} for {symbol}.")

            # 2. Upgrades & Downgrades / Bank Grades (FMP Stable API)
            ud_url = f"https://financialmodelingprep.com/stable/grades?symbol={symbol}&apikey={self.fmp_key}"
            res_ud = requests.get(ud_url, timeout=5)
            if res_ud.status_code == 200 and res_ud.json():
                data = res_ud.json()
                if isinstance(data, list):
                    for item in data[:20]:
                        company = item.get("gradingCompany", "")
                        if any(b.lower() in company.lower() for b in self.MAJOR_BANKS):
                            date_str = str(item.get("date", ""))[:10]
                            to_grade = item.get("newGrade", "")
                            action = item.get("action", "")
                            recent_actions.append(f"[{date_str}] {company}: {to_grade} ({action})")
            else:
                print(f"[AnalystAgent] FMP upgrades API returned status {res_ud.status_code} for {symbol}.")

            if recent_actions or target_price != "N/A":
                print(f"[AnalystAgent] Successfully fetched analyst data from FMP for {symbol}.")
                return {
                    "wall_street_consensus": consensus,
                    "mean_target_price": target_price,
                    "recent_major_bank_actions": recent_actions[:5],
                    "source": "FinancialModelingPrep"
                }

        except Exception as e:
            print(f"[AnalystAgent] FMP request error for {symbol}: {e}")

        return {}

    def fetch_from_finnhub(self, symbol: str) -> dict:
        if not self.finnhub_key or not self.finnhub_client:
            return {}

        print(f"[AnalystAgent] Querying Finnhub for {symbol}...")
        recent_actions = []
        target_price = "N/A"
        consensus = "N/A"

        # 1. Price Target (Free Finnhub Endpoint)
        try:
            pt_res = self.finnhub_client.price_target(symbol)
            if pt_res and isinstance(pt_res, dict) and pt_res.get("targetMean"):
                target_price = round(float(pt_res["targetMean"]), 2)
        except Exception:
            pass

        # 2. Recommendation Trends (Free Finnhub Endpoint)
        try:
            rec_res = self.finnhub_client.recommendation_trends(symbol)
            if rec_res and isinstance(rec_res, list) and len(rec_res) > 0:
                latest_rec = rec_res[0]
                strong_buy = latest_rec.get("strongBuy", 0)
                buy = latest_rec.get("buy", 0)
                hold = latest_rec.get("hold", 0)
                sell = latest_rec.get("sell", 0)
                if buy + strong_buy > hold + sell:
                    consensus = "BUY / STRONG BUY"
                elif sell > buy:
                    consensus = "SELL"
                else:
                    consensus = "HOLD"
        except Exception:
            pass

        # 3. Upgrades & Downgrades (Finnhub Premium Endpoint - handle 403 silently)
        try:
            from_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
            to_date = datetime.now().strftime("%Y-%m-%d")

            ud_res = self.finnhub_client.upgrade_downgrade(symbol=symbol, _from=from_date, to=to_date)
            if ud_res and isinstance(ud_res, list):
                for item in ud_res[:20]:
                    company = item.get("company", "")
                    if any(b.lower() in company.lower() for b in self.MAJOR_BANKS):
                        raw_date = item.get("actionTime") or item.get("gradeTime") or ""
                        date_str = str(raw_date)[:10]
                        to_grade = item.get("toGrade", "")
                        action = item.get("action", "")
                        recent_actions.append(f"[{date_str}] {company}: {to_grade} ({action})")
        except Exception:
            pass

        if recent_actions or target_price != "N/A" or consensus != "N/A":
            print(f"[AnalystAgent] Successfully fetched analyst data from Finnhub for {symbol}.")
            return {
                "wall_street_consensus": consensus,
                "mean_target_price": target_price,
                "recent_major_bank_actions": recent_actions[:5] if recent_actions else [],
                "source": "Finnhub"
            }

        return {}

    def fetch_from_yfinance(self, symbol: str, log_prefix: str = "Fallback") -> dict:
        recent_actions = []
        target_mean_price = None
        recommendation_key = None

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}
            target_mean_price = info.get("targetMeanPrice")
            recommendation_key = info.get("recommendationKey")

            ud = ticker.upgrades_downgrades
            if ud is not None and not ud.empty:
                ud_reset = ud.reset_index()
                for idx, row in ud_reset.head(20).iterrows():
                    firm = str(row.get("Firm", ""))
                    if any(b.lower() in firm.lower() for b in self.MAJOR_BANKS):
                        date_str = str(row.get("GradeDate", ""))[:10]
                        to_grade = row.get("ToGrade", "")
                        action = row.get("Action", "main")
                        recent_actions.append(f"[{date_str}] {firm}: {to_grade} ({action})")
            print(f"[AnalystAgent] [{log_prefix}] Extracted analyst ratings from yfinance for {symbol}.")
        except Exception as e:
            print(f"[AnalystAgent] yfinance analyst fetch error for {symbol}: {e}")

        return {
            "wall_street_consensus": recommendation_key.upper() if recommendation_key else "N/A",
            "mean_target_price": round(float(target_mean_price), 2) if target_mean_price else "N/A",
            "recent_major_bank_actions": recent_actions[:5] if recent_actions else ["No recent major bank rating changes recorded"],
            "source": "yfinance"
        }

    def analyze(self, symbol: str) -> dict:
        print(f"[AnalystAgent] Extracting investment bank ratings & targets for {symbol}...")

        # 1. Try Financial Modeling Prep (FMP) if configured
        if self.fmp_key:
            fmp_res = self.fetch_from_fmp(symbol)
            if fmp_res and (fmp_res.get("recent_major_bank_actions") or fmp_res.get("mean_target_price") != "N/A"):
                return fmp_res

        # 2. Try Finnhub if configured
        if self.finnhub_key:
            fh_res = self.fetch_from_finnhub(symbol)
            if fh_res:
                # If Finnhub free tier blocked the detailed bank action list, supplement from yfinance
                if not fh_res.get("recent_major_bank_actions"):
                    print(f"[AnalystAgent] Finnhub free tier missing bank list. Supplementing bank actions from yfinance for {symbol}...")
                    yf_res = self.fetch_from_yfinance(symbol, log_prefix="Supplemented")
                    fh_res["recent_major_bank_actions"] = yf_res.get("recent_major_bank_actions", [])
                return fh_res

        # 3. Fallback: yfinance
        return self.fetch_from_yfinance(symbol, log_prefix="Primary")
