#!/usr/bin/env python3
"""
AI Trader Mobile Backend REST API
=================================
FastAPI backend providing REST endpoints for mobile applications:
- Live and historical swing trade signals
- Screener of most active / high-volume movers (US & EU gettex/XETRA)
- Background execution of 6-agent AI stock synthesis & master runner
- Model outcome tracking & portfolio risk memos
- ISIN / European symbol resolution
"""

from __future__ import annotations

import os
import sys
import json
import asyncio
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from db import (
    init_db,
    get_latest_signals,
    get_signal_history,
    get_summary_stats,
    get_outcome_performance_stats,
    get_latest_portfolio_reviews,
    delete_signals_older_than,
    update_signal_outcomes,
    get_connection,
)
from dashboard import collect_dashboard_data
from ticker_resolver import resolve_symbol, get_currency_for_symbol

BASE_DIR = Path(__file__).resolve().parent

# Initialize FastAPI app
app = FastAPI(
    title="AI Trader API",
    description="High-performance REST API powering the AI Trader mobile and web applications",
    version="1.0.0",
)

# Enable CORS for cross-origin mobile and web requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory background task registry
RUNNING_JOBS: Dict[str, Dict[str, Any]] = {}


# ==========================================
# Pydantic Request / Response Models
# ==========================================

class AnalyzeRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol or European ISIN (e.g. NVDA, US67066G1040, NVD.DE)")
    model: str = Field(default="ling", description="AI Model to run (e.g. ling, free, gemini, nemotron, kimi, qwen)")
    is_eu: bool = Field(default=False, description="Flag for European gettex/XETRA tickers")


class MasterRunRequest(BaseModel):
    model: str = Field(default="ling", description="AI Model to run across top active stocks")
    limit: int = Field(default=5, ge=1, le=25, description="Number of top active stocks to analyze")
    dashboard_limit: int = Field(default=10, ge=1, le=50, description="Number of active stocks to screen")
    market: str = Field(default="US", description="Market to scan: 'US' or 'EU'")
    prefer_exchange: str = Field(default="DE", description="Preferred German exchange suffix (e.g. DE, MU)")


class PurgeRequest(BaseModel):
    days: int = Field(default=7, ge=1, le=365, description="Purge signals older than this number of days")


# ==========================================
# Lifecycle & Health Endpoints
# ==========================================

@app.on_event("startup")
def startup_event():
    init_db()


@app.get("/api/health", tags=["System"])
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "app": "AI Trader API",
        "version": "1.0.0",
    }


# ==========================================
# Signals & Reporting Endpoints
# ==========================================

@app.get("/api/signals/latest", tags=["Signals"])
def fetch_latest_signals(
    days: Optional[int] = Query(default=7, ge=1, le=365, description="Filter signals within last N days (null for all-time)"),
    decision: Optional[str] = Query(default=None, description="Filter by decision: BUY, SELL, or HOLD"),
    search: Optional[str] = Query(default=None, description="Search ticker symbol or company name"),
    sort_by: Optional[str] = Query(default="date", description="Sort by 'date' (analysis execution timestamp) or 'symbol'"),
    order: Optional[str] = Query(default="desc", description="Sort order: 'desc' (newest first) or 'asc' (oldest first)"),
):
    """Returns the newest signal for each stock within the recency window."""
    signals = get_latest_signals(max_age_days=days)

    # Ensure company_name is populated
    for s in signals:
        if not s.get("company_name"):
            info = resolve_symbol(s.get("symbol", ""))
            s["company_name"] = info.get("company_name") or s.get("symbol")

    if decision:
        dec_filter = decision.strip().upper()
        signals = [s for s in signals if s.get("decision") == dec_filter]

    if search:
        q = search.strip().upper()
        signals = [
            s for s in signals
            if q in s.get("symbol", "").upper() or q in (s.get("company_name") or "").upper()
        ]

    # Sort signals
    if sort_by == "date":
        signals.sort(key=lambda s: str(s.get("timestamp") or ""), reverse=(order.lower() != "asc"))
    elif sort_by == "symbol":
        signals.sort(key=lambda s: str(s.get("symbol") or "").upper(), reverse=(order.lower() == "desc"))

    return {
        "count": len(signals),
        "days_window": days,
        "signals": signals,
    }


@app.get("/api/signals/summary", tags=["Signals"])
def fetch_summary_metrics(
    days: Optional[int] = Query(default=7, ge=1, le=365, description="Recency window in days for metrics calculation")
):
    """Returns summary counts (Tracked Stocks, Buy, Sell, Hold counts, Model Win Rate, Last Run)."""
    stats = get_summary_stats(max_age_days=days)
    outcome_stats = get_outcome_performance_stats()

    return {
        "recency_days": days,
        "summary": stats,
        "performance": outcome_stats,
    }


@app.get("/api/signals/history", tags=["Signals"])
def fetch_signal_history(
    symbol: Optional[str] = Query(default=None, description="Filter by ticker symbol"),
    limit: int = Query(default=100, ge=1, le=500, description="Maximum number of historical records"),
):
    """Returns chronologically sorted historical analysis signals."""
    records = get_signal_history(symbol=symbol, limit=limit)
    return {
        "count": len(records),
        "symbol": symbol.strip().upper() if symbol else None,
        "records": records,
    }


@app.delete("/api/signals/purge", tags=["Signals"])
def purge_old_signals(req: PurgeRequest):
    """Permanently purges records older than N days from SQLite database."""
    deleted_count = delete_signals_older_than(days=req.days)
    return {
        "purged_records": deleted_count,
        "cutoff_days": req.days,
        "message": f"Successfully purged {deleted_count} signals older than {req.days} days.",
    }


# ==========================================
# Market Screener (Most Traded Movers)
# ==========================================

@app.get("/api/stocks/active", tags=["Market Screener"])
def fetch_active_movers(
    limit: int = Query(default=10, ge=1, le=50, description="Number of active stocks to return"),
    market: str = Query(default="US", description="Market: 'US' or 'EU'"),
    prefer_exchange: str = Query(default="DE", description="German exchange suffix for EU stocks (e.g. DE, MU)"),
):
    """Returns real-time high-volume active stocks and news catalysts."""
    try:
        active = collect_dashboard_data(limit=limit, market=market.upper(), prefer_exchange=prefer_exchange.upper())
        return {
            "market": market.upper(),
            "count": len(active),
            "stocks": active,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch market mover data: {str(e)}",
        )


# ==========================================
# Ticker & ISIN Resolver
# ==========================================

@app.get("/api/resolve", tags=["Market Screener"])
def resolve_ticker(
    query: str = Query(..., description="ISIN, WKN, or ticker symbol (e.g. US67066G1040, NVD.DE, AAPL, MU)"),
    prefer_exchange: str = Query(default="AUTO", description="Preferred exchange (AUTO, DE, MU, US)"),
    force_european: bool = Query(default=False, description="Map US tickers to German gettex/XETRA equivalents"),
):
    """Resolves ISIN or ticker to exchange symbol and currency."""
    info = resolve_symbol(query, prefer_exchange=prefer_exchange, force_european=force_european)
    curr_code, curr_sym = get_currency_for_symbol(info["symbol"])
    return {
        "query": query,
        "resolved_symbol": info["symbol"],
        "company_name": info.get("company_name", info["symbol"]),
        "currency_code": curr_code,
        "currency_symbol": curr_sym,
        "exchange": info.get("exchange", "Unknown"),
        "isin": info.get("isin"),
    }


# ==========================================
# Background Execution (AI Analysis & Master)
# ==========================================

def _run_single_analysis_job(job_id: str, symbol: str, model: str, is_eu: bool):
    """Worker task executing main.py in a background process."""
    RUNNING_JOBS[job_id]["status"] = "running"
    RUNNING_JOBS[job_id]["started_at"] = datetime.now(timezone.utc).isoformat()

    # Gracefully redirect retired bunny model to ling (inclusionAI: Ling 3.1 Flash)
    effective_model = model.strip().lower()
    if effective_model in ["bunny", "space-bunny", "space-bunny-alpha", "stealth/space-bunny-alpha", "sb"]:
        effective_model = "ling"

    main_script = BASE_DIR / "main.py"
    cmd = [sys.executable, str(main_script), effective_model, symbol]
    if is_eu:
        cmd.append("--eu")

    try:
        proc = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True, timeout=300)
        RUNNING_JOBS[job_id]["exit_code"] = proc.returncode
        RUNNING_JOBS[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        if proc.returncode == 0:
            if "No analysis results generated." in (proc.stdout or ""):
                RUNNING_JOBS[job_id]["status"] = "failed"
                RUNNING_JOBS[job_id]["error"] = f"No analysis results generated for {symbol}. Insufficient price data or ticker not found."
            else:
                RUNNING_JOBS[job_id]["status"] = "completed"
        else:
            RUNNING_JOBS[job_id]["status"] = "failed"
            RUNNING_JOBS[job_id]["error"] = proc.stderr[-1000:] if proc.stderr else proc.stdout[-1000:]
    except subprocess.TimeoutExpired:
        RUNNING_JOBS[job_id]["status"] = "timeout"
        RUNNING_JOBS[job_id]["error"] = "Analysis exceeded timeout limit (300s)."
    except Exception as e:
        RUNNING_JOBS[job_id]["status"] = "error"
        RUNNING_JOBS[job_id]["error"] = str(e)


@app.post("/api/analyze", tags=["Execution"])
def trigger_analysis(req: AnalyzeRequest, background_tasks: BackgroundTasks):
    """Triggers a 6-agent AI analysis on a specific stock symbol or ISIN."""
    clean_sym = req.symbol.strip().upper()
    # If explicitly EU or an ISIN / WKN format, resolve to German ticker
    if req.is_eu or len(clean_sym) == 12 or clean_sym.endswith(".DE") or clean_sym.endswith(".MU"):
        resolved = resolve_symbol(clean_sym, prefer_exchange="DE", force_european=True)
        target_symbol = resolved["symbol"]
    else:
        resolved = resolve_symbol(clean_sym, prefer_exchange="AUTO", force_european=False)
        target_symbol = resolved["symbol"]

    req_model = req.model.strip().lower()
    if req_model in ["bunny", "space-bunny", "space-bunny-alpha", "stealth/space-bunny-alpha", "sb"]:
        req_model = "ling"

    job_id = f"job_{target_symbol}_{int(datetime.now().timestamp())}"

    RUNNING_JOBS[job_id] = {
        "job_id": job_id,
        "target_symbol": target_symbol,
        "model": req_model,
        "status": "queued",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    background_tasks.add_task(
        _run_single_analysis_job,
        job_id=job_id,
        symbol=target_symbol,
        model=req_model,
        is_eu=req.is_eu,
    )

    return {
        "job_id": job_id,
        "target_symbol": target_symbol,
        "model": req_model,
        "status": "queued",
        "message": f"Queued 6-agent analysis for '{target_symbol}' using model '{req_model}'.",
    }


@app.get("/api/jobs/{job_id}", tags=["Execution"])
def get_job_status(job_id: str):
    """Checks the status of an ongoing or completed background job."""
    if job_id not in RUNNING_JOBS:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return RUNNING_JOBS[job_id]


# ==========================================
# Outcome Evaluation & Portfolio Reviews
# ==========================================

@app.post("/api/outcomes/evaluate", tags=["Evaluation"])
def evaluate_outcomes():
    """Evaluates pending forward trade outcomes against actual historical market price action."""
    evaluated_count = update_signal_outcomes()
    performance = get_outcome_performance_stats()
    return {
        "evaluated_count": evaluated_count,
        "performance": performance,
    }


@app.get("/api/portfolio/reviews", tags=["Evaluation"])
def fetch_portfolio_reviews(
    limit: int = Query(default=10, ge=1, le=50, description="Max reviews to return")
):
    """Returns stored investment-committee risk memos."""
    reviews = get_latest_portfolio_reviews(limit=limit)
    formatted = []
    for r in reviews:
        try:
            rev_dict = json.loads(r.get("review_json") or "{}")
        except Exception:
            rev_dict = {}
        formatted.append({
            "id": r.get("id"),
            "timestamp": r.get("timestamp"),
            "model_used": r.get("model_used"),
            "review": rev_dict,
        })
    return {
        "count": len(formatted),
        "reviews": formatted,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
