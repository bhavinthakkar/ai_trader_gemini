import os
import json
import requests
import finnhub
import subprocess
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
                            try:
                                action_date = datetime.strptime(date_str, "%Y-%m-%d")
                                if datetime.now() - action_date > timedelta(days=30):
                                    continue
                            except Exception:
                                pass
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
            from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            to_date = datetime.now().strftime("%Y-%m-%d")

            ud_res = self.finnhub_client.upgrade_downgrade(symbol=symbol, _from=from_date, to=to_date)
            if ud_res and isinstance(ud_res, list):
                for item in ud_res[:20]:
                    company = item.get("company", "")
                    if any(b.lower() in company.lower() for b in self.MAJOR_BANKS):
                        raw_date = item.get("actionTime") or item.get("gradeTime") or ""
                        date_str = str(raw_date)[:10]
                        try:
                            action_date = datetime.strptime(date_str, "%Y-%m-%d")
                            if datetime.now() - action_date > timedelta(days=30):
                                continue
                        except Exception:
                            pass
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
                        try:
                            action_date = datetime.strptime(date_str, "%Y-%m-%d")
                            if datetime.now() - action_date > timedelta(days=30):
                                continue
                        except Exception:
                            pass
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

    def get_cik_from_ticker(self, symbol: str) -> str:
        symbol = symbol.upper()
        # Look for local cached mapping first to avoid hitting SEC rate limit
        cache_path = os.path.join(os.path.dirname(__file__), "sec_company_tickers.json")
        data = None
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    data = json.load(f)
            except Exception:
                pass
        
        if not data:
            try:
                headers = {"User-Agent": "antigravity.coder@gmail.com"}
                res = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers, timeout=5)
                if res.status_code == 200:
                    data = res.json()
                    with open(cache_path, "w") as f:
                        json.dump(data, f)
            except Exception as e:
                print(f"[AnalystAgent] SEC company_tickers fetch error: {e}")
                return None
        
        if data:
            for entry in data.values():
                if entry.get("ticker") == symbol:
                    return str(entry.get("cik_str")).zfill(10)
        return None

    def fetch_sec_filings(self, symbol: str) -> list:
        print(f"[AnalystAgent] Fetching SEC EDGAR filings for {symbol}...")
        cik = self.get_cik_from_ticker(symbol)
        if not cik:
            print(f"[AnalystAgent] CIK not found for {symbol}.")
            return ["No CIK found; unable to retrieve SEC filings."]
        
        headers = {"User-Agent": "antigravity.coder@gmail.com"}
        try:
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            res = requests.get(url, headers=headers, timeout=5)
            if res.status_code == 200:
                filings = res.json()
                recent = filings.get("filings", {}).get("recent", {})
                
                forms = recent.get("form", [])
                filing_dates = recent.get("filingDate", [])
                primary_docs = recent.get("primaryDocument", [])
                descriptions = recent.get("primaryDocDescription", [])
                accession_numbers = recent.get("accessionNumber", [])
                
                recent_filings = []
                count = 0
                for idx, form in enumerate(forms):
                    if form in ["10-K", "10-Q", "8-K"]:
                        acc_num = accession_numbers[idx].replace("-", "")
                        primary_doc = primary_docs[idx]
                        doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_num}/{primary_doc}"
                        recent_filings.append(f"[{filing_dates[idx]}] Form {form} - {descriptions[idx]} - {doc_url}")
                        count += 1
                        if count >= 5:
                            break
                return recent_filings if recent_filings else ["No recent 10-K, 10-Q, or 8-K filings found."]
            else:
                print(f"[AnalystAgent] SEC submissions API returned status {res.status_code} for CIK {cik}.")
        except Exception as e:
            print(f"[AnalystAgent] SEC EDGAR request error for {symbol}: {e}")
            
        return ["Error retrieving SEC filings."]

    def fetch_fred_macro_data(self) -> dict:
        print("[AnalystAgent] Fetching macroeconomic indicators from FRED...")
        indicators = {
            "UNRATE": "Unemployment Rate",
            "CPIAUCSL": "Consumer Price Index (CPI)",
            "FEDFUNDS": "Federal Funds Effective Rate",
            "GDP": "Gross Domestic Product (GDP)"
        }
        
        macro_results = {}
        
        for series_id, name in indicators.items():
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            try:
                cmd = ["curl", "-s", "-L", "--max-time", "15", url]
                res = subprocess.run(cmd, capture_output=True, text=True, check=True)
                if res.returncode == 0 and res.stdout:
                    lines = res.stdout.strip().split("\n")
                    if lines and not lines[0].startswith("<!DOCTYPE"):
                        date_val, val = self._parse_fred_csv(lines)
                        
                        if series_id == "CPIAUCSL":
                            yoy_val = self._calculate_fred_yoy(lines)
                            macro_results[name] = f"{val} (YoY Inflation: {yoy_val})"
                        elif series_id == "GDP":
                            yoy_val = self._calculate_fred_yoy(lines)
                            macro_results[name] = f"${float(val):,.2f} Billion (YoY Growth: {yoy_val})"
                        elif series_id == "UNRATE":
                            macro_results[name] = f"{val}% (as of {date_val})"
                        elif series_id == "FEDFUNDS":
                            macro_results[name] = f"{val}% (as of {date_val})"
                        continue
            except Exception as e:
                print(f"[AnalystAgent] FRED fetch error for {series_id}: {e}")
                
            macro_results[name] = "N/A"
            
        return macro_results

    def _parse_fred_csv(self, lines):
        if not lines or len(lines) < 2:
            return "N/A", "N/A"
        for line in reversed(lines[1:]):
            parts = line.strip().split(",")
            if len(parts) == 2:
                date_val, str_val = parts[0], parts[1]
                if str_val and str_val != "." and str_val.strip() != "":
                    try:
                        float(str_val)
                        return date_val, str_val
                    except ValueError:
                        pass
        return "N/A", "N/A"

    def _calculate_fred_yoy(self, lines):
        if not lines or len(lines) < 6:
            return "N/A"
        data_map = {}
        for line in lines[1:]:
            parts = line.strip().split(",")
            if len(parts) == 2:
                date_val, str_val = parts[0], parts[1]
                if str_val and str_val != "." and str_val.strip() != "":
                    try:
                        data_map[date_val] = float(str_val)
                    except ValueError:
                        pass
        sorted_dates = sorted(data_map.keys())
        if not sorted_dates:
            return "N/A"
        latest_date = sorted_dates[-1]
        latest_val = data_map[latest_date]
        try:
            latest_dt = datetime.strptime(latest_date, "%Y-%m-%d")
            one_year_ago_dt = latest_dt - timedelta(days=365)
            closest_date = min(sorted_dates, key=lambda d: abs(datetime.strptime(d, "%Y-%m-%d") - one_year_ago_dt))
            one_year_ago_val = data_map[closest_date]
            yoy_change = ((latest_val - one_year_ago_val) / one_year_ago_val) * 100
            return f"{yoy_change:.2f}%"
        except Exception:
            return "N/A"

    def analyze(self, symbol: str) -> dict:
        print(f"[AnalystAgent] Extracting investment bank ratings & targets for {symbol}...")

        res = None
        # 1. Try Financial Modeling Prep (FMP) if configured
        if self.fmp_key:
            fmp_res = self.fetch_from_fmp(symbol)
            if fmp_res and (fmp_res.get("recent_major_bank_actions") or fmp_res.get("mean_target_price") != "N/A"):
                res = fmp_res

        # 2. Try Finnhub if configured
        if not res and self.finnhub_key:
            fh_res = self.fetch_from_finnhub(symbol)
            if fh_res:
                if not fh_res.get("recent_major_bank_actions"):
                    print(f"[AnalystAgent] Finnhub free tier missing bank list. Supplementing bank actions from yfinance for {symbol}...")
                    yf_res = self.fetch_from_yfinance(symbol, log_prefix="Supplemented")
                    fh_res["recent_major_bank_actions"] = yf_res.get("recent_major_bank_actions", [])
                res = fh_res

        # 3. Fallback: yfinance
        if not res:
            res = self.fetch_from_yfinance(symbol, log_prefix="Primary")

        # 4. Fetch SEC EDGAR filings
        sec_filings = self.fetch_sec_filings(symbol)
        res["recent_sec_filings"] = sec_filings

        # 5. Fetch FRED macroeconomic indicators
        macro_data = self.fetch_fred_macro_data()
        res["macro_data"] = macro_data

        return res
