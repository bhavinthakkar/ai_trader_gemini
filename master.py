#!/usr/bin/env python3
"""
Gloomberb Master Orchestration Engine
======================================
1. Executes the Gloomberb Market Intelligence Dashboard (dashboard.py)
   to collect real-time data, compute technicals/price geometry, and infer
   volume catalysts for the most traded stocks of the day.
2. Identifies the ranked list of most active volume leaders.
3. Sequentially executes the 6-agent AI trading analysis pipeline (main.py)
   on each stock, recording signals, updating outcome tracking, and sending alerts.
4. Outputs an aggregated execution report and summary table.

Usage:
  python master.py [model] [options]
  python master.py free
  python master.py gemini --limit 5
  python master.py qwen -t NVDA,TSLA,INTC
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from streamlit_server import ensure_streamlit_running, is_streamlit_running, get_lan_ip

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_SCRIPT = os.path.join(BASE_DIR, "dashboard.py")
MAIN_SCRIPT = os.path.join(BASE_DIR, "main.py")
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "trader.db")

VALID_MODELS = [
    "nemotron", "nvidia", "ultra", "nemotron-ultra", "550b",
    "kimi", "kimi-k3", "k3", "moonshot",
    "super", "nemotron-super", "120b",
    "gemini", "openrouter", "free", "openrouter/free",
    "minimax", "minimax-m3", "minimax_m3", "m3",
    "qwen", "llamacpp", "qwen-llamacpp",
]


def get_latest_signal(symbol: str, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """Fetches the most recent trade signal record from SQLite database."""
    if not os.path.exists(db_path):
        return None
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        with conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, symbol, decision, confidence, buy_score, hold_score, sell_score,
                       quant_score, horizon_days, timestamp, model_used
                FROM signals
                WHERE symbol = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (symbol.upper(),)
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
    except Exception:
        return None
    return None


def run_dashboard_script(
    limit: int = 10,
    tickers: Optional[str] = None,
    market: str = "US",
    prefer_exchange: str = "DE",
    portfolio: bool = False,
    portfolio_url: Optional[str] = None,
    html_path: Optional[str] = "dashboard.html",
    no_html: bool = False,
    json_path: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[int, Optional[Dict[str, Any]]]:
    """
    Executes dashboard.py as a subprocess, inherits terminal stdio,
    and returns (returncode, parsed_json_data).
    """
    cmd = [sys.executable, DASHBOARD_SCRIPT, "--limit", str(limit)]

    if market.upper() == "EU" or portfolio or portfolio_url:
        cmd.extend(["--market", "EU"])
    if prefer_exchange:
        cmd.extend(["--prefer-exchange", prefer_exchange])
    if portfolio:
        cmd.append("--portfolio")
    if portfolio_url:
        cmd.extend(["--portfolio-url", portfolio_url])
    if tickers:
        cmd.extend(["--tickers", tickers])
    if no_html:
        cmd.append("--no-html")
    elif html_path:
        cmd.extend(["--html", html_path])
    if json_path:
        cmd.extend(["--json", json_path])

    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    print(f"\n[Master] Launching dashboard script: {' '.join(cmd)}")
    sys.stdout.flush()

    try:
        res = subprocess.run(cmd, cwd=BASE_DIR, env=run_env)
        retcode = res.returncode
    except Exception as e:
        print(f"[Master] Error executing dashboard script: {e}")
        return 1, None

    data = None
    if json_path and os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[Master] Warning: Failed to read dashboard JSON output '{json_path}': {e}")

    return retcode, data


def extract_most_traded_stocks(dashboard_data: Optional[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    """Extracts top active stock records from dashboard dataset."""
    if not dashboard_data or not isinstance(dashboard_data, dict):
        return []
    stocks = dashboard_data.get("stocks", [])
    if not isinstance(stocks, list):
        return []
    return stocks[:limit]


def run_main_analysis(
    symbol: str,
    model: str = "free",
    market: str = "US",
    prefer_exchange: str = "DE",
    temperature: Optional[float] = None,
    reasoning_budget: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[int, float]:
    """
    Executes main.py for a single symbol with the specified model and options.
    Inherits terminal stdout/stderr for real-time interactive logging.
    Returns (returncode, elapsed_seconds).
    """
    cmd = [sys.executable, MAIN_SCRIPT, model, symbol]

    if market.upper() == "EU":
        cmd.extend(["--market", "EU"])
    if prefer_exchange:
        cmd.extend(["--prefer-exchange", prefer_exchange])
    if temperature is not None:
        cmd.extend(["--temperature", str(temperature)])
    if reasoning_budget is not None:
        cmd.extend(["--reasoning-budget", str(reasoning_budget)])
    if reasoning_effort is not None:
        cmd.extend(["--reasoning-effort", str(reasoning_effort)])

    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    start_time = time.time()
    try:
        res = subprocess.run(cmd, cwd=BASE_DIR, env=run_env)
        retcode = res.returncode
    except KeyboardInterrupt:
        print(f"\n[Master] User interrupted analysis for {symbol}.")
        raise
    except Exception as e:
        print(f"[Master] Error executing main.py for {symbol}: {e}")
        retcode = 1

    elapsed = time.time() - start_time
    return retcode, elapsed


def print_active_stocks_table(stocks: List[Dict[str, Any]], model: str) -> None:
    """Displays a clean summary of the active stocks identified by dashboard.py."""
    if RICH_AVAILABLE:
        console = Console()
        table = Table(
            title=f"🎯 Identified {len(stocks)} Most Traded Stocks to Analyze (Target Model: {model.upper()})",
            box=box.ROUNDED,
            header_style="bold cyan",
            title_style="bold yellow",
        )
        table.add_column("Rank", justify="center", style="bold white", width=6)
        table.add_column("Symbol", justify="center", style="bold yellow", width=10)
        table.add_column("Company Name", style="white", min_width=20)
        table.add_column("Price", justify="right", style="bold", width=10)
        table.add_column("Change", justify="right", width=10)
        table.add_column("Volume", justify="right", style="cyan", width=12)
        table.add_column("RVOL", justify="right", style="magenta", width=8)
        table.add_column("Primary Catalyst Driver", style="green", min_width=25)

        for idx, s in enumerate(stocks, 1):
            curr_sym = s.get("currency_symbol", "$")
            chg = s.get("change_pct", 0.0)
            chg_style = "bold green" if chg >= 0 else "bold red"
            chg_str = f"{chg:+.2f}%"
            vol = s.get("volume", 0)
            vol_str = f"{vol / 1e6:.1f}M" if vol >= 1e6 else f"{vol:,}"
            rvol = s.get("rvol", 1.0)
            cat = s.get("catalyst_type", "GENERAL MARKET")

            table.add_row(
                str(idx),
                s.get("symbol", "N/A"),
                s.get("name", s.get("symbol", "N/A")),
                f"{curr_sym}{s.get('price', 0.0):.2f}",
                Text(chg_str, style=chg_style),
                vol_str,
                f"{rvol:.1f}x",
                cat,
            )
        console.print(table)
    else:
        print(f"\n🎯 Identified {len(stocks)} Most Traded Stocks to Analyze (Target Model: {model.upper()}):")
        print("-" * 80)
        for idx, s in enumerate(stocks, 1):
            sym = s.get("symbol", "N/A")
            curr_sym = s.get("currency_symbol", "$")
            price = s.get("price", 0.0)
            chg = s.get("change_pct", 0.0)
            vol = s.get("volume", 0)
            vol_str = f"{vol / 1e6:.1f}M" if vol >= 1e6 else f"{vol:,}"
            rvol = s.get("rvol", 1.0)
            cat = s.get("catalyst_type", "GENERAL MARKET")
            print(f" {idx:2d}. {sym:<6} | {curr_sym}{price:>7.2f} ({chg:>+6.2f}%) | Vol: {vol_str:>7} ({rvol:4.1f}x) | Catalyst: {cat}")
        print("-" * 80)


def render_master_summary(
    results: List[Dict[str, Any]],
    total_elapsed: float,
    model: str,
    dry_run: bool = False,
) -> None:
    """Renders the final post-execution status and signal outcomes table."""
    success_count = sum(1 for r in results if r.get("returncode") == 0)
    failed_count = len(results) - success_count

    if RICH_AVAILABLE:
        console = Console()
        title_text = f"🏁 Master Orchestration Summary (Model: {model.upper()}) | Total Elapsed: {total_elapsed:.1f}s"
        table = Table(
            title=title_text,
            box=box.HEAVY_EDGE,
            header_style="bold white on blue",
            title_style="bold yellow",
        )
        table.add_column("#", justify="center", style="bold white", width=4)
        table.add_column("Symbol", justify="center", style="bold yellow", width=10)
        table.add_column("Price / Chg", justify="center", width=18)
        table.add_column("Catalyst Driver", style="white", min_width=20)
        table.add_column("Execution", justify="center", width=12)
        table.add_column("Duration", justify="right", width=10)
        table.add_column("Signal", justify="center", width=10)
        table.add_column("Conf", justify="right", width=8)
        table.add_column("Quant", justify="right", width=8)

        for idx, r in enumerate(results, 1):
            sym = r.get("symbol", "N/A")
            stock_info = r.get("stock_info", {})
            curr_sym = stock_info.get("currency_symbol", "$")
            price = stock_info.get("price", 0.0)
            chg = stock_info.get("change_pct", 0.0)
            chg_style = "green" if chg >= 0 else "red"
            price_text = Text(f"{curr_sym}{price:.2f} ({chg:+.2f}%)", style=chg_style)

            cat = stock_info.get("catalyst_type", "N/A")
            dur = f"{r.get('elapsed', 0.0):.1f}s"

            if dry_run:
                exec_text = Text("DRY-RUN", style="bold cyan")
                sig_text = Text("-", style="dim")
                conf_str = "-"
                quant_str = "-"
            elif r.get("returncode") == 0:
                exec_text = Text("SUCCESS", style="bold green")
                sig = r.get("latest_signal") or {}
                decision = sig.get("decision", "N/A")
                if decision == "BUY":
                    sig_text = Text("BUY", style="bold green")
                elif decision == "SELL":
                    sig_text = Text("SELL", style="bold red")
                else:
                    sig_text = Text(decision, style="bold yellow")

                conf = sig.get("confidence")
                conf_str = f"{conf:.2f}" if conf is not None else "N/A"
                quant = sig.get("quant_score")
                quant_str = f"{quant:.1f}" if quant is not None else "N/A"
            else:
                exec_text = Text(f"FAILED ({r.get('returncode')})", style="bold red")
                sig_text = Text("ERROR", style="bold red")
                conf_str = "-"
                quant_str = "-"

            table.add_row(
                str(idx),
                sym,
                price_text,
                cat,
                exec_text,
                dur,
                sig_text,
                conf_str,
                quant_str,
            )

        console.print("\n")
        console.print(table)
        status_panel = Panel(
            f"[bold]Sequential Execution Complete:[/bold] [green]{success_count} Passed[/green], "
            f"[red]{failed_count} Failed[/red], [cyan]{len(results)} Total[/cyan] | "
            f"Total Runtime: [bold]{total_elapsed:.1f}s[/bold]",
            style="bold cyan",
            box=box.ROUNDED,
        )
        console.print(status_panel)
    else:
        print("\n" + "=" * 80)
        print(f"🏁 MASTER ORCHESTRATION SUMMARY (Model: {model.upper()}) | Total Elapsed: {total_elapsed:.1f}s")
        print("=" * 80)
        for idx, r in enumerate(results, 1):
            sym = r.get("symbol", "N/A")
            stock_info = r.get("stock_info", {})
            curr_sym = stock_info.get("currency_symbol", "$")
            price = stock_info.get("price", 0.0)
            chg = stock_info.get("change_pct", 0.0)
            dur = f"{r.get('elapsed', 0.0):.1f}s"
            status = "DRY-RUN" if dry_run else ("SUCCESS" if r.get("returncode") == 0 else f"FAILED ({r.get('returncode')})")
            sig = r.get("latest_signal") or {}
            dec = sig.get("decision", "-") if r.get("returncode") == 0 else "ERROR"
            conf = sig.get("confidence", "-")
            print(f" {idx:2d}. {sym:<6} | {curr_sym}{price:>7.2f} ({chg:>+6.2f}%) | Status: {status:<12} | Time: {dur:<7} | Decision: {dec} (Conf: {conf})")
        print("=" * 80)
        print(f"Complete: {success_count} Passed, {failed_count} Failed. Total Runtime: {total_elapsed:.1f}s")


def run_master(
    model: str = "free",
    limit: int = 5,
    dashboard_limit: Optional[int] = None,
    tickers: Optional[str] = None,
    market: str = "US",
    prefer_exchange: str = "DE",
    portfolio: bool = False,
    portfolio_url: Optional[str] = None,
    html_path: str = "dashboard.html",
    no_html: bool = False,
    temperature: Optional[float] = None,
    reasoning_budget: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
    delay: float = 2.0,
    dry_run: bool = False,
    stop_on_error: bool = False,
    output_json: Optional[str] = None,
    skip_dashboard: bool = False,
    dashboard_json: Optional[str] = None,
    no_streamlit: bool = False,
    streamlit_port: int = 8501,
) -> Dict[str, Any]:
    """
    Executes the full master orchestration workflow:
    0. Guarantees Streamlit dashboard server is active for live web access.
    1. Runs dashboard.py to collect market data and infer news catalysts.
    2. Identifies the top N most traded stocks.
    3. Sequentially executes main.py on each stock.
    4. Gathers outcome metrics and prints summary table.
    """
    master_start_time = time.time()
    env = os.environ.copy()

    # -------------------------------------------------------------
    # Step 0: Ensure Streamlit Dashboard Server is Running
    # -------------------------------------------------------------
    if not no_streamlit:
        ensure_streamlit_running(port=streamlit_port, verbose=True)

    effective_dash_limit = dashboard_limit if dashboard_limit is not None else max(limit, 10)

    # -------------------------------------------------------------
    # Step 1: Execute Dashboard Script (or load cached JSON)
    # -------------------------------------------------------------
    dashboard_data: Optional[Dict[str, Any]] = None
    tmp_json_path = None

    if skip_dashboard and dashboard_json and os.path.exists(dashboard_json):
        print(f"[Master] Skipping dashboard run; loading data from '{dashboard_json}'...")
        try:
            with open(dashboard_json, "r", encoding="utf-8") as f:
                dashboard_data = json.load(f)
        except Exception as e:
            print(f"[Master] Error loading '{dashboard_json}': {e}")

    if dashboard_data is None:
        if dashboard_json:
            tmp_json_path = dashboard_json
        else:
            with tempfile.NamedTemporaryFile(suffix=".json", prefix="gloomberb_dash_", delete=False) as tf:
                tmp_json_path = tf.name

        dash_code, dashboard_data = run_dashboard_script(
            limit=effective_dash_limit,
            tickers=tickers,
            market=market,
            prefer_exchange=prefer_exchange,
            portfolio=portfolio,
            portfolio_url=portfolio_url,
            html_path=html_path,
            no_html=no_html,
            json_path=tmp_json_path,
            env=env,
        )

        if dash_code != 0 and not dashboard_data:
            print(f"\n❌ [Master] Dashboard execution failed (exit code: {dash_code}). Aborting master pipeline.")
            return {"status": "failed", "step": "dashboard", "exit_code": dash_code}

    # -------------------------------------------------------------
    # Step 2: Find the Most Traded Stocks
    # -------------------------------------------------------------
    active_stocks = extract_most_traded_stocks(dashboard_data, limit=limit)
    if not active_stocks:
        print("\n❌ [Master] No active stock records found from dashboard. Aborting.")
        return {"status": "failed", "step": "extract_stocks", "stocks": []}

    print_active_stocks_table(active_stocks, model=model)

    # -------------------------------------------------------------
    # Step 3: Sequentially Execute main.py on Each Stock
    # -------------------------------------------------------------
    results: List[Dict[str, Any]] = []

    if dry_run:
        print("\n[Master] --dry-run specified: Skipping sequential main.py execution.")
        for s in active_stocks:
            results.append({
                "symbol": s.get("symbol"),
                "stock_info": s,
                "returncode": 0,
                "elapsed": 0.0,
                "latest_signal": None,
                "dry_run": True,
            })
    else:
        total_tickers = len(active_stocks)
        print(f"\n{'='*75}")
        print(f"🚀 COMMENCING SEQUENTIAL 6-AGENT AI SYNTHESIS ON {total_tickers} STOCKS")
        print(f"   Model: {model.upper()} | Sequential Mode: Clean Subprocesses")
        print(f"{'='*75}")

        for idx, stock_item in enumerate(active_stocks, 1):
            symbol = stock_item.get("symbol", "").upper()
            if not symbol:
                continue

            print(f"\n{'─'*75}")
            print(f"▶ [{idx}/{total_tickers}] Launching main.py for '{symbol}' | Model: {model.upper()}")
            print(f"   Catalyst: {stock_item.get('catalyst_type', 'N/A')}")
            print(f"   Headline: {stock_item.get('primary_headline', 'N/A')}")
            print(f"{'─'*75}\n")
            sys.stdout.flush()

            retcode, elapsed = run_main_analysis(
                symbol=symbol,
                model=model,
                market=market,
                prefer_exchange=prefer_exchange,
                temperature=temperature,
                reasoning_budget=reasoning_budget,
                reasoning_effort=reasoning_effort,
                env=env,
            )

            latest_sig = get_latest_signal(symbol)

            res_record = {
                "symbol": symbol,
                "stock_info": stock_item,
                "returncode": retcode,
                "elapsed": elapsed,
                "latest_signal": latest_sig,
                "dry_run": False,
            }
            results.append(res_record)

            if retcode == 0:
                dec = latest_sig.get("decision", "RECORDED") if latest_sig else "RECORDED"
                conf = latest_sig.get("confidence", "N/A") if latest_sig else "N/A"
                print(f"\n✅ [{idx}/{total_tickers}] Completed {symbol} in {elapsed:.1f}s -> Decision: {dec} (Conf: {conf})")
            else:
                print(f"\n❌ [{idx}/{total_tickers}] Error analyzing {symbol} (Exit code: {retcode}, Time: {elapsed:.1f}s)")
                if stop_on_error:
                    print("[Master] --stop-on-error flag set; stopping sequential pipeline early.")
                    break

            # Rate-limit cooldown delay between sequential calls
            if idx < total_tickers and delay > 0:
                print(f"[Master] Pausing {delay:.1f}s before next analysis...")
                time.sleep(delay)

    # -------------------------------------------------------------
    # Step 4: Summary & Export
    # -------------------------------------------------------------
    total_elapsed = time.time() - master_start_time
    render_master_summary(results, total_elapsed=total_elapsed, model=model, dry_run=dry_run)

    # Export combined JSON output if requested
    master_report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model": model,
        "market": market,
        "prefer_exchange": prefer_exchange,
        "total_elapsed_seconds": round(total_elapsed, 2),
        "stocks_analyzed": len(results),
        "results": results,
        "dashboard_summary": {
            "fear_greed": dashboard_data.get("fear_greed") if dashboard_data else None,
            "total_active_stocks_sited": len(dashboard_data.get("stocks", [])) if dashboard_data else 0,
        }
    }

    if output_json:
        try:
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(master_report, f, indent=2)
            print(f"[Master] Full orchestration report saved to '{output_json}'.")
        except Exception as e:
            print(f"[Master] Warning: Failed to save JSON summary: {e}")

    # Clean up temporary json file if generated
    if tmp_json_path and not dashboard_json and os.path.exists(tmp_json_path):
        try:
            os.remove(tmp_json_path)
        except Exception:
            pass

    return master_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gloomberb Master Orchestration: Dashboard Discovery -> Sequential 6-Agent AI Trader Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run dashboard, find top 5 most traded stocks, and analyze each with the 'free' model:
  python master.py free

  # Run on top 3 most traded stocks with Gemini:
  python master.py gemini --limit 3

  # Run local Qwen 2.5 14B via llama.cpp:
  python master.py qwen -n 3

  # Dry-run mode: View dashboard and selected stocks without calling main.py:
  python master.py free --limit 5 --dry-run

  # Explicit tickers override (run dashboard & main.py for NVDA, TSLA, INTC):
  python master.py free --tickers NVDA,TSLA,INTC
        """
    )
    parser.add_argument(
        "model_arg",
        nargs="?",
        default=None,
        help="LLM model choice for main.py (default: 'free'). Choices: 'free', 'gemini', 'nemotron', 'ultra', 'super', 'kimi', 'openrouter', 'qwen', 'llamacpp', etc."
    )
    parser.add_argument(
        "--model", "-m",
        dest="model_opt",
        default=None,
        help="LLM model choice (alternative to positional argument)"
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=5,
        help="Number of most traded stocks to analyze with main.py (default: 5)"
    )
    parser.add_argument(
        "--tickers", "-t",
        type=str,
        default=None,
        help="Optional comma-separated list of symbols to force (e.g. NVDA,TSLA,INTC)"
    )
    parser.add_argument(
        "--dashboard-limit",
        type=int,
        default=None,
        help="Number of stocks to gather on the dashboard (default: max(limit, 10))"
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
        "--temperature",
        type=float,
        default=None,
        help="Model sampling temperature to pass to main.py"
    )
    parser.add_argument(
        "--reasoning-budget",
        type=int,
        default=None,
        help="Internal reasoning token budget for reasoning models"
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        choices=["low", "medium", "high", "max"],
        default=None,
        help="Reasoning effort level ('low', 'medium', 'high', 'max')"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Cooldown delay in seconds between sequential main.py executions (default: 2.0)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run dashboard and extract most traded stocks without executing main.py"
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop sequential execution immediately if any ticker fails"
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Optional path to export full master orchestration report as JSON"
    )
    parser.add_argument(
        "--skip-dashboard",
        action="store_true",
        help="Skip executing dashboard.py and use existing JSON from --dashboard-json"
    )
    parser.add_argument(
        "--dashboard-json",
        type=str,
        default=None,
        help="Path to pre-existing dashboard JSON dataset"
    )
    parser.add_argument(
        "--market",
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
        "--no-streamlit",
        action="store_true",
        help="Do not automatically start the Streamlit dashboard server"
    )
    parser.add_argument(
        "--streamlit-port",
        type=int,
        default=8501,
        help="Port for Streamlit dashboard server (default: 8501)"
    )

    args = parser.parse_args()

    # Determine model choice
    raw_model = args.model_opt or args.model_arg or "free"
    model_choice = str(raw_model).strip().lower()
    if model_choice in ["bunny", "space-bunny", "space-bunny-alpha", "stealth/space-bunny-alpha", "sb"]:
        print("[master] Notice: Space Bunny Alpha ('bunny') is no longer available on OpenRouter; automatically redirecting to 'free' preset.")
        model_choice = "free"

    if model_choice not in VALID_MODELS:
        print(f"\n❌ ERROR: Invalid model choice '{raw_model}'!")
        print(f"Supported choices: {', '.join(sorted(set(VALID_MODELS)))}\n")
        sys.exit(1)

    use_portfolio = args.portfolio or bool(args.portfolio_url)
    market = "EU" if (args.eu or use_portfolio) else args.market.upper()

    print("\n" + "=" * 70)
    print("⚡ GLOOMBERB MASTER ORCHESTRATION PIPELINE ⚡")
    print("   1. Discover Most Traded Stocks (dashboard.py)")
    print("   2. Sequential 6-Agent AI Deep Synthesis (main.py)")
    print("=" * 70)
    print(f"• LLM Model:           {model_choice.upper()}")
    target_tag = "Google Finance Portfolio (EUR)" if use_portfolio else f"{market} (Preferred Exchange: {args.prefer_exchange})"
    print(f"• Target Market:       {target_tag}")
    print(f"• Stocks Limit:        {args.limit}")
    print(f"• HTML Dashboard:      {'Disabled' if args.no_html else args.html}")
    print(f"• Mode:                {'DRY-RUN (Discovery only)' if args.dry_run else 'Full Execution'}")
    print(f"• Inter-run Delay:     {args.delay:.1f}s")
    if args.tickers:
        print(f"• Explicit Tickers:    {args.tickers}")
    if args.portfolio_url:
        print(f"• Portfolio URL:       {args.portfolio_url}")
    print("-" * 70)

    report = run_master(
        model=model_choice,
        limit=args.limit,
        dashboard_limit=args.dashboard_limit,
        tickers=args.tickers,
        market=market,
        prefer_exchange=args.prefer_exchange,
        portfolio=args.portfolio,
        portfolio_url=args.portfolio_url,
        html_path=args.html,
        no_html=args.no_html,
        temperature=args.temperature,
        reasoning_budget=args.reasoning_budget,
        reasoning_effort=args.reasoning_effort,
        delay=args.delay,
        dry_run=args.dry_run,
        stop_on_error=args.stop_on_error,
        output_json=args.json,
        skip_dashboard=args.skip_dashboard,
        dashboard_json=args.dashboard_json,
        no_streamlit=args.no_streamlit,
        streamlit_port=args.streamlit_port,
    )

    if report.get("status") == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
