#!/usr/bin/env python3
"""
Danelfin Daily Batch Runner
===========================
Automates the daily execution of the 6-agent AI trading analysis pipeline on the
top 25 stocks selected by Danelfin's flagship Best Stocks Strategy (/v3/beststocks).

Features:
- Quota Optimal: Consumes only 1 Danelfin API call per day for all 25 stocks.
- Idempotent: Uses --skip-existing to skip tickers already analyzed today in trader.db.
- Resilient: Continues through individual stock failures and generates an aggregated report.
- Schedulable: Can be executed via systemd timer or cron once daily after market close.

Usage:
  python danelfin_daily_runner.py [model] [--limit 25] [--delay 2.0] [--dry-run]
  python danelfin_daily_runner.py flint --limit 25
  python danelfin_daily_runner.py free --limit 10
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from danelfin_service import DanelfinService
from master import run_master, VALID_MODELS

load_dotenv()


def run_daily_danelfin_pipeline(
    model: str = "flint",
    limit: int = 25,
    delay: float = 2.0,
    dry_run: bool = False,
    force_refresh_picks: bool = False,
    skip_existing: bool = True,
    no_streamlit: bool = True,
) -> dict:
    """
    Executes the end-to-end daily analysis pipeline on Danelfin's top 25 picks.
    """
    print("\n" + "=" * 75)
    print("🌟 DANELFIN AI DAILY 25-TICKER ANALYSIS PIPELINE 🌟")
    print(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {model.upper()} | Limit: {limit} Stocks | Skip Existing Today: {skip_existing}")
    print("=" * 75)

    dan_svc = DanelfinService()
    usage_before = dan_svc.get_usage()
    print(f"[Danelfin] Free Tier Quota: {usage_before['call_count']}/{usage_before['monthly_limit']} calls used ({usage_before['safe_remaining']} safe calls remaining).")

    # Step 1: Sourcing Top Tickers (1 API call pre-seeds scores for all 25 tickers)
    print(f"\n[Danelfin] Sourcing daily Best Stocks picks from Danelfin AI...")
    tickers = dan_svc.get_daily_picks_tickers(limit=limit, force_refresh=force_refresh_picks)

    if not tickers:
        print("❌ Error: Could not retrieve Danelfin daily picks. Check API key, network, or quota limit.")
        return {"status": "error", "message": "Failed to retrieve Danelfin picks"}

    tickers_str = ",".join(tickers)
    print(f"✅ Sourced {len(tickers)} tickers (1 API call consumed): {', '.join(tickers)}")

    # Step 2: Run Master Orchestration Pipeline
    print(f"\n🚀 Launching 6-Agent AI Deep Synthesis on {len(tickers)} Danelfin picks...")
    report = run_master(
        model=model,
        limit=len(tickers),
        dashboard_limit=len(tickers),
        tickers=tickers_str,
        market="US",
        delay=delay,
        dry_run=dry_run,
        skip_existing=skip_existing,
        no_streamlit=no_streamlit,
        html_path="danelfin_daily_report.html",
    )

    usage_after = dan_svc.get_usage()
    print("\n" + "=" * 75)
    print(f"🏁 DANELFIN DAILY RUN COMPLETE")
    print(f"Danelfin API Calls Made This Run: {usage_after['call_count'] - usage_before['call_count']} call(s)")
    print(f"Remaining Monthly Quota: {usage_after['remaining']}/500 calls")
    print("=" * 75 + "\n")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Automate 6-agent analysis pipeline on Danelfin's 25 daily Best Stocks"
    )
    parser.add_argument(
        "model_arg",
        nargs="?",
        default=None,
        help="AI Model (e.g. flint, free, nemotron, super, qwen, gemini)"
    )
    parser.add_argument(
        "-m", "--model",
        dest="model_opt",
        default=None,
        help="AI Model option"
    )
    parser.add_argument(
        "-l", "--limit",
        type=int,
        default=25,
        help="Number of Danelfin picks to analyze (default: 25, max: 25)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Cooldown delay in seconds between sequential ticker analyses (default: 2.0s)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch picks, compute technicals/catalysts, but do not execute LLM analysis"
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Do not skip tickers already analyzed today; re-analyze all tickers"
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Bypass local cache and force fresh API fetch from Danelfin"
    )
    parser.add_argument(
        "--with-streamlit",
        action="store_true",
        help="Start or keep Streamlit dashboard server running"
    )

    args = parser.parse_args()

    raw_model = args.model_opt or args.model_arg or "flint"
    model_choice = str(raw_model).strip().lower()
    if model_choice in ["haiku", "haiku-5.5", "haike", "haike-5.5", "claude", "claude-haiku", "claude-haiku-5-5", "claude-5-5-haiku", "anthropic"]:
        print("[Danelfin] Notice: Claude LLM has been replaced; automatically redirecting to 'flint' (Recursive: Flint (free) from OpenRouter).")
        model_choice = "flint"

    if model_choice not in VALID_MODELS:
        print(f"\n❌ ERROR: Invalid model choice '{raw_model}'!")
        print(f"Supported choices: {', '.join(sorted(set(VALID_MODELS)))}\n")
        sys.exit(1)

    limit = min(max(args.limit, 1), 25)

    run_daily_danelfin_pipeline(
        model=model_choice,
        limit=limit,
        delay=args.delay,
        dry_run=args.dry_run,
        force_refresh_picks=args.force_refresh,
        skip_existing=not args.no_skip,
        no_streamlit=not args.with_streamlit,
    )


if __name__ == "__main__":
    main()
