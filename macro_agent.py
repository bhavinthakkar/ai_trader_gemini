import os
import io
import time
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional

class MacroDataAgent:
    """
    Macro Data Agent (marco_data / macro_data): Collects macroeconomic context from FRED 
    (Federal Reserve Economic Data) and speculative/commercial futures positioning 
    from CFTC Commitment of Traders (COT) reports.
    """

    FRED_SERIES = {
        "DGS10": "10-Year Treasury Yield",
        "DGS2": "2-Year Treasury Yield",
        "T10Y2Y": "10Y-2Y Yield Curve Spread",
        "DFII10": "10-Year Real Interest Rate (TIPS)",
        "FEDFUNDS": "Federal Funds Effective Rate",
        "UNRATE": "Unemployment Rate"
    }

    CFTC_COT_URL = "https://www.cftc.gov/dea/newcot/FinFutWk.txt"
    _cache: Optional[Dict[str, Any]] = None
    _cache_timestamp: float = 0
    CACHE_DURATION: float = 3600  # 1 hour cache for macro indicators

    def __init__(self):
        self.fred_api_key = os.getenv("FRED_API_KEY", "").strip()

    def fetch_single_fred_series(self, series_id: str) -> tuple:
        """
        Fetches the latest value, date, and 5-day historical delta for a single FRED series.
        """
        # 1. Primary: Native Gloomberb FRED CLI terminal engine
        try:
            from gloomberb_service import GloomberbService
            gb = GloomberbService()
            obs = gb.fetch_fred_series(series_id)
            if obs and isinstance(obs, list) and len(obs) > 0:
                valid_obs = [item for item in obs if item.get("value") is not None and str(item.get("value")) != "."]
                if valid_obs:
                    latest = valid_obs[0]
                    val = float(latest["value"])
                    delta_5d = None
                    if len(valid_obs) >= 5:
                        val_5d = float(valid_obs[4]["value"])
                        delta_5d = val - val_5d
                    return series_id, {"date": str(latest.get("date")), "value": val, "change_5d": delta_5d}
        except Exception as e:
            print(f"[MacroDataAgent] Gloomberb FRED series warning for {series_id}: {e}")

        # 2. Try official FRED API if key exists
        if self.fred_api_key:
            try:
                url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={self.fred_api_key}&file_type=json&sort_order=desc&limit=10"
                res = requests.get(url, timeout=5)
                if res.status_code == 200:
                    obs = res.json().get("observations", [])
                    valid_obs = [item for item in obs if item.get("value") and item.get("value") != "."]
                    if valid_obs:
                        latest = valid_obs[0]
                        val = float(latest["value"])
                        delta_5d = None
                        if len(valid_obs) >= 5:
                            delta_5d = val - float(valid_obs[4]["value"])
                        return series_id, {"date": latest.get("date"), "value": val, "change_5d": delta_5d}
            except Exception as e:
                print(f"[MacroDataAgent] FRED API error for {series_id}: {e}")

        # 3. Direct public CSV fallback
        try:
            csv_url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            res = requests.get(csv_url, timeout=6, headers={"User-Agent": "AITrader/1.0 (contact@aitrader.com)"})
            if res.status_code == 200:
                df = pd.read_csv(io.StringIO(res.text))
                df = df.dropna()
                df = df[df[series_id] != "."]
                if not df.empty:
                    last_row = df.iloc[-1]
                    val = float(last_row[series_id])
                    delta_5d = None
                    if len(df) >= 6:
                        val_5d = float(df.iloc[-6][series_id])
                        delta_5d = val - val_5d
                    return series_id, {
                        "date": str(last_row["observation_date"]),
                        "value": val,
                        "change_5d": delta_5d
                    }
        except Exception as e:
            print(f"[MacroDataAgent] Public FRED CSV fetch error for {series_id}: {e}")

        return series_id, None

    def fetch_cftc_cot(self) -> Dict[str, Any]:
        """
        Fetches and parses weekly CFTC Commitment of Traders (COT) financial futures report.
        """
        print("[MacroDataAgent] Fetching CFTC COT financial futures report...")
        try:
            res = requests.get(self.CFTC_COT_URL, headers={"User-Agent": "AITrader/1.0 (contact@aitrader.com)"}, timeout=6)
            if res.status_code == 200:
                df = pd.read_csv(io.StringIO(res.text), header=None)
                if df.empty or df.shape[1] < 8:
                    return {"status": "Empty report", "summary": "CFTC report structure unexpected."}

                relevant_contracts = {}
                for idx, row in df.iterrows():
                    contract_name = str(row[0]).strip()
                    report_date = str(row[2]).strip()
                    open_interest = row[7] if len(row) > 7 else "N/A"

                    if "E-MINI S&P 500" in contract_name.upper():
                        relevant_contracts["sp500"] = f"E-MINI S&P 500 (Open Interest: {open_interest}, Date: {report_date})"
                    elif "NASDAQ MINI" in contract_name.upper() or "NASDAQ-100" in contract_name.upper():
                        relevant_contracts["nasdaq"] = f"NASDAQ 100 Mini (Open Interest: {open_interest}, Date: {report_date})"
                    elif "VIX FUTURES" in contract_name.upper():
                        relevant_contracts["vix"] = f"VIX Futures (Open Interest: {open_interest}, Date: {report_date})"

                summary_parts = []
                if "sp500" in relevant_contracts:
                    summary_parts.append(relevant_contracts["sp500"])
                if "nasdaq" in relevant_contracts:
                    summary_parts.append(relevant_contracts["nasdaq"])
                if "vix" in relevant_contracts:
                    summary_parts.append(relevant_contracts["vix"])

                cot_summary = " | ".join(summary_parts) if summary_parts else "CFTC financial futures report active."

                return {
                    "status": "Success",
                    "contracts": relevant_contracts,
                    "summary": cot_summary
                }
        except Exception as e:
            print(f"[MacroDataAgent] CFTC COT fetch error: {e}")

        return {
            "status": "Unavailable",
            "summary": "CFTC COT financial futures report parsed."
        }

    def analyze(self, symbol: str = None) -> Dict[str, Any]:
        """
        Gathers FRED macro metrics and CFTC COT positioning data. Uses 1-hour memory cache.
        """
        now = time.time()
        if MacroDataAgent._cache and (now - MacroDataAgent._cache_timestamp) < MacroDataAgent.CACHE_DURATION:
            print("[MacroDataAgent] Returning cached macro & CFTC data.")
            return MacroDataAgent._cache

        print("[MacroDataAgent] Gathering FRED & CFTC macro intelligence...")

        # 1. Fetch FRED Macro Series in parallel using ThreadPoolExecutor
        fred_data = {}
        with ThreadPoolExecutor(max_workers=len(self.FRED_SERIES)) as executor:
            future_to_series = {
                executor.submit(self.fetch_single_fred_series, s_id): s_id
                for s_id in self.FRED_SERIES
            }
            for future in as_completed(future_to_series):
                series_id, info = future.result()
                if info:
                    fred_data[series_id] = info

        dgs10_info = fred_data.get("DGS10", {})
        dgs2_info = fred_data.get("DGS2", {})
        t10y2y_info = fred_data.get("T10Y2Y", {})
        dfii10_info = fred_data.get("DFII10", {})
        fedfunds_info = fred_data.get("FEDFUNDS", {})
        unrate_info = fred_data.get("UNRATE", {})

        dgs10_val = dgs10_info.get("value")
        dgs2_val = dgs2_info.get("value")
        t10y2y_val = t10y2y_info.get("value")
        dfii10_val = dfii10_info.get("value")
        fedfunds_val = fedfunds_info.get("value")
        unrate_val = unrate_info.get("value")

        dgs10_d5 = dgs10_info.get("change_5d")

        # Fallback spread computation if T10Y2Y was missing but 10Y and 2Y exist
        if t10y2y_val is None and dgs10_val is not None and dgs2_val is not None:
            t10y2y_val = round(dgs10_val - dgs2_val, 2)

        dgs10_str = f"{dgs10_val:.2f}%" if dgs10_val is not None else "N/A"
        dgs2_str = f"{dgs2_val:.2f}%" if dgs2_val is not None else "N/A"
        t10y2y_str = f"{t10y2y_val:+.2f}%" if t10y2y_val is not None else "N/A"
        dfii10_str = f"{dfii10_val:.2f}%" if dfii10_val is not None else "N/A"
        fedfunds_str = f"{fedfunds_val:.2f}%" if fedfunds_val is not None else "N/A"
        unrate_str = f"{unrate_val:.1f}%" if unrate_val is not None else "N/A"

        if dgs10_d5 is not None:
            dgs10_d5_bps = dgs10_d5 * 100
            dgs10_d5_str = f"{dgs10_d5:+.2f}% ({dgs10_d5_bps:+.1f} bps)"
        else:
            dgs10_d5_str = "N/A"

        if t10y2y_val is not None:
            if t10y2y_val > 0.15:
                yield_curve_status = "Normal / Steepening"
            elif t10y2y_val < -0.05:
                yield_curve_status = "Inverted (Recession Flag)"
            else:
                yield_curve_status = "Flat / Uninverting"
        else:
            yield_curve_status = "Unknown"

        # 2. Fetch CFTC COT Data
        cftc_data = self.fetch_cftc_cot()
        cftc_summary = cftc_data.get("summary", "")

        # 3. Formulate Summary
        summary = (
            f"FRED Macro: 10Y Yield {dgs10_str} (5d Velocity: {dgs10_d5_str}), 2Y Yield {dgs2_str}, "
            f"10Y-2Y Spread {t10y2y_str} ({yield_curve_status}), 10Y Real Yield (TIPS) {dfii10_str}, "
            f"Fed Funds Rate {fedfunds_str}, Unemployment {unrate_str}. "
            f"CFTC COT: {cftc_summary}"
        )

        result = {
            "us_10y_yield": dgs10_str,
            "us_2y_yield": dgs2_str,
            "yield_curve_spread_10y2y": t10y2y_str,
            "yield_curve_status": yield_curve_status,
            "us_10y_real_yield": dfii10_str,
            "us_10y_yield_5d_change": dgs10_d5_str,
            "fed_funds_rate": fedfunds_str,
            "unemployment_rate": unrate_str,
            "cftc_cot_summary": cftc_summary,
            "summary": summary
        }

        MacroDataAgent._cache = result
        MacroDataAgent._cache_timestamp = now
        print("[MacroDataAgent] Macro data successfully collected.")

        return result

# Alias for compatibility if referenced as MarcoDataAgent
MarcoDataAgent = MacroDataAgent
