import requests
from typing import Dict, Any, Optional

class InstitutionalDataAgent:
    """
    Institutional Data Agent: Queries official SEC EDGAR APIs for US public companies.
    Fetches company CIK, recent regulatory filings (Form 4 insider transactions, 
    10-K/10-Q financial statements, 8-K material events, 13D/13G/13F institutional ownership),
    and summarizes institutional & insider activity.
    """

    TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    SUBMISSIONS_URL_FMT = "https://data.sec.gov/submissions/CIK{cik}.json"

    def __init__(self, user_agent: str = "AITrader/1.0 (admin@aitrader.com)"):
        self.headers = {"User-Agent": user_agent}
        self._ticker_map: Optional[Dict[str, Dict[str, Any]]] = None

    def _load_ticker_map(self) -> Dict[str, Dict[str, Any]]:
        if self._ticker_map is not None:
            return self._ticker_map

        print("[InstitutionalDataAgent] Loading SEC EDGAR company ticker map...")
        try:
            res = requests.get(self.TICKERS_URL, headers=self.headers, timeout=10)
            if res.status_code == 200:
                data = res.json()
                mapping = {}
                for entry in data.values():
                    ticker = str(entry.get("ticker", "")).upper()
                    cik_val = entry.get("cik_str")
                    title = entry.get("title", "")
                    if ticker and cik_val is not None:
                        mapping[ticker] = {
                            "cik": str(cik_val).zfill(10),
                            "title": title
                        }
                self._ticker_map = mapping
                print(f"[InstitutionalDataAgent] Loaded {len(mapping)} SEC tickers.")
                return self._ticker_map
            else:
                print(f"[InstitutionalDataAgent] Failed to fetch SEC tickers: HTTP {res.status_code}")
        except Exception as e:
            print(f"[InstitutionalDataAgent] Error loading SEC tickers: {e}")

        self._ticker_map = {}
        return self._ticker_map

    def _get_cik_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        ticker_map = self._load_ticker_map()
        clean_symbol = symbol.strip().upper()
        # Handle ticker symbols with dots/hyphens if applicable
        if clean_symbol in ticker_map:
            return ticker_map[clean_symbol]
        base_symbol = clean_symbol.split(".")[0]
        return ticker_map.get(base_symbol)

    def analyze(self, symbol: str) -> Dict[str, Any]:
        print(f"[InstitutionalDataAgent] Fetching SEC EDGAR data for {symbol}...")
        cik_info = self._get_cik_info(symbol)

        if not cik_info:
            print(f"[InstitutionalDataAgent] Ticker {symbol} not found in SEC EDGAR directory (Non-US or foreign stock).")
            return {
                "symbol": symbol,
                "cik": "N/A",
                "company_name": "N/A",
                "sic_description": "N/A",
                "insider_transactions_exist": False,
                "recent_filings": [],
                "summary": f"No SEC EDGAR filings available for {symbol} (Non-US or unmapped ticker)."
            }

        cik = cik_info["cik"]
        title = cik_info["title"]
        url = self.SUBMISSIONS_URL_FMT.format(cik=cik)

        try:
            res = requests.get(url, headers=self.headers, timeout=10)
            if res.status_code != 200:
                print(f"[InstitutionalDataAgent] HTTP {res.status_code} fetching SEC submissions for {symbol} (CIK: {cik}).")
                return {
                    "symbol": symbol,
                    "cik": cik,
                    "company_name": title,
                    "sic_description": "N/A",
                    "insider_transactions_exist": False,
                    "recent_filings": [],
                    "summary": f"SEC EDGAR submission data temporarily unavailable for {symbol} (CIK {cik})."
                }

            sec_data = res.json()
            sic_desc = sec_data.get("sicDescription", "N/A")
            company_name = sec_data.get("name", title)
            insider_owner = bool(sec_data.get("insiderTransactionForOwnerExists", 0))
            insider_issuer = bool(sec_data.get("insiderTransactionForIssuerExists", 0))
            insider_flag = insider_owner or insider_issuer

            filings_recent = sec_data.get("filings", {}).get("recent", {})
            forms = filings_recent.get("form", [])
            filing_dates = filings_recent.get("filingDate", [])
            descriptions = filings_recent.get("primaryDocDescription", [])

            recent_filings_list = []
            form_counts = {}

            for idx in range(min(15, len(forms))):
                form_type = forms[idx]
                f_date = filing_dates[idx] if idx < len(filing_dates) else "N/A"
                desc = descriptions[idx] if idx < len(descriptions) else ""

                recent_filings_list.append({
                    "form": form_type,
                    "filing_date": f_date,
                    "description": desc
                })
                form_counts[form_type] = form_counts.get(form_type, 0) + 1

            # Format concise summary
            top_forms_str = ", ".join([f"{fmt} ({cnt})" for fmt, cnt in list(form_counts.items())[:4]])
            latest_date = filing_dates[0] if filing_dates else "N/A"
            insider_note = "Insider transaction filings recorded." if insider_flag else "No recent insider transaction flags."

            summary = (
                f"CIK {cik} ({company_name} - {sic_desc}). "
                f"Latest filing on {latest_date}. "
                f"Recent SEC filings: {top_forms_str if top_forms_str else 'None'}. "
                f"{insider_note}"
            )

            print(f"[InstitutionalDataAgent] Successfully fetched SEC data for {symbol} (CIK: {cik}).")
            return {
                "symbol": symbol,
                "cik": cik,
                "company_name": company_name,
                "sic_description": sic_desc,
                "insider_transactions_exist": insider_flag,
                "recent_filings": recent_filings_list[:8],
                "summary": summary
            }

        except Exception as e:
            print(f"[InstitutionalDataAgent] Error fetching SEC submissions for {symbol}: {e}")
            return {
                "symbol": symbol,
                "cik": cik,
                "company_name": title,
                "sic_description": "N/A",
                "insider_transactions_exist": False,
                "recent_filings": [],
                "summary": f"SEC EDGAR fetch error for {symbol}: {e}"
            }
