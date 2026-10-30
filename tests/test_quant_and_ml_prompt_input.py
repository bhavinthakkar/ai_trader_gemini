"""
Unit tests for Deterministic Quant Score and ML 20-Session Score prompt formatting and LLM input injection.
"""

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest
from quant.composite_score import (
    blend_quant_and_ml_scores,
    format_quant_and_ml_context,
)
from rag_service import RAGService


def test_format_quant_and_ml_context_in_scope():
    """Verify that in-scope tickers receive formatted Deterministic Quant, ML Alpha, and Blended Composite."""
    det_scores = {
        "composite_quantitative_score": 46.41,
        "raw_composite": 45.78,
        "vol_factor": 0.85,
        "trend_score": 27.24,
        "sector_relative_score": 52.70,
        "market_alpha_score": 60.80,
        "valuation_history_score": 85.00,
        "peer_valuation_score": 17.58,
        "financial_quality_score": 76.00,
        "is_value_trap": False,
        "value_trap_reasons": [],
    }
    ml_context = {
        "ticker": "INTC",
        "model_status": "IN_SCOPE",
        "in_scope": True,
        "score_0_100": 83.33,
        "ml_score": 83.33,
        "ml_weight": 0.20,
        "rank": 5,
        "universe_size": 25,
        "rank_percentile": 0.833,
        "predicted_20d_excess_return": 0.0503,
        "score_date": "2026-10-14",
        "model_name": "xgb_focused_ranker_latest",
        "stance": "BULLISH",
        "signal": "STRONG OUTPERFORM",
    }
    m_data = {
        "symbol": "INTC",
        "current_price": 93.74,
        "atr": 5.54,
    }

    prompt_text = format_quant_and_ml_context(
        deterministic_scores=det_scores,
        ml_context=ml_context,
        m_data=m_data,
        symbol="INTC",
    )

    # 1. Deterministic Quant Score checks
    assert "DETERMINISTIC QUANT SCORE: 46.41 / 100" in prompt_text
    assert "Historical Trend (25% weight):          27.24 / 100" in prompt_text
    assert "Sector Relative Strength (20% weight):  52.70 / 100" in prompt_text
    assert "Relative Market Alpha (20% weight):     60.80 / 100" in prompt_text
    assert "Valuation History (15% weight):         85.00 / 100" in prompt_text
    assert "Peer Valuation (20% weight):            17.58 / 100" in prompt_text
    assert "Financial Quality: 76.00 / 100 (Value-Trap Guard: CLEAR" in prompt_text
    assert "Vol Factor: 0.85" in prompt_text

    # 2. ML 20-Session Score checks
    assert "20-SESSION ML ALPHA SCORE: 83.33 / 100" in prompt_text
    assert "Model Status: IN_SCOPE (Model: xgb_focused_ranker_latest" in prompt_text
    assert "Cross-Sectional Rank: #5 of 25 (83.3th percentile, Stance: BULLISH" in prompt_text
    assert "Predicted 20-Session Excess Return: +5.03%" in prompt_text
    assert "Configured Blending Weight: 20%" in prompt_text

    # 3. Blended Composite checks (0.80 * 46.41 + 0.20 * 83.33 = 53.79)
    assert "BLENDED DECISION COMPOSITE: 53.79 / 100" in prompt_text
    assert "(0.80 * 46.41 Det. Quant) + (0.20 * 83.33 ML Score) = 53.79 / 100" in prompt_text

    # 4. Directive checks (no boundary mandates, clear summary for Master Trader)
    assert "QUANTITATIVE SUMMARY FOR MASTER TRADER:" in prompt_text
    assert "Deterministic Quant Score: 46.41 / 100" in prompt_text
    assert "ML 20-Session Predictive Score: 83.33 / 100" in prompt_text
    assert "Blended Decision Composite: 53.79 / 100" in prompt_text
    assert "Weigh these quantitative scores alongside all qualitative evidence" in prompt_text


def test_format_quant_and_ml_context_out_of_scope():
    """Verify that out-of-scope tickers receive 100% Deterministic Quant Score and 0% ML Weight."""
    det_scores = {
        "composite_quantitative_score": 73.16,
        "raw_composite": 77.25,
        "vol_factor": 0.85,
        "trend_score": 34.65,
        "sector_relative_score": 89.75,
        "market_alpha_score": 97.85,
        "valuation_history_score": 85.00,
        "peer_valuation_score": 76.60,
        "financial_quality_score": 100.00,
        "is_value_trap": False,
        "value_trap_reasons": [],
    }
    ml_context = {
        "ticker": "000660.KS",
        "model_status": "OUT_OF_SCOPE",
        "in_scope": False,
        "score_0_100": None,
        "ml_score": None,
        "ml_weight": 0.0,
        "message": "'000660.KS' is outside the active ML model universe.",
    }
    m_data = {
        "symbol": "000660.KS",
        "current_price": 1341000.0,
        "atr": 93700.0,
    }

    prompt_text = format_quant_and_ml_context(
        deterministic_scores=det_scores,
        ml_context=ml_context,
        m_data=m_data,
        symbol="000660.KS",
    )

    assert "DETERMINISTIC QUANT SCORE: 73.16 / 100" in prompt_text
    assert "20-SESSION ML ALPHA SCORE: None (OUT_OF_SCOPE)" in prompt_text
    assert "Configured Blending Weight: 0.0%" in prompt_text
    assert "BLENDED DECISION COMPOSITE: 73.16 / 100" in prompt_text
    assert "100% Deterministic Quant Score = 73.16 / 100 (ML weight = 0.0%)" in prompt_text
    assert "QUANTITATIVE SUMMARY FOR MASTER TRADER:" in prompt_text
    assert "Deterministic Quant Score: 73.16 / 100" in prompt_text
    assert "ML 20-Session Predictive Score: N/A / 100" in prompt_text
    assert "Blended Decision Composite: 73.16 / 100" in prompt_text


def test_format_quant_and_ml_context_insufficient_pillars():
    """Verify graceful handling when fewer than 3 pillars are available."""
    det_scores = {
        "composite_quantitative_score": None,
        "raw_composite": None,
        "vol_factor": 1.0,
        "trend_score": None,
        "sector_relative_score": None,
        "market_alpha_score": None,
        "valuation_history_score": None,
        "peer_valuation_score": None,
    }

    prompt_text = format_quant_and_ml_context(
        deterministic_scores=det_scores,
        ml_context=None,
        m_data={},
        symbol="UNKNOWN",
    )

    assert "DETERMINISTIC QUANT SCORE: N/A / 100" in prompt_text
    assert "BLENDED DECISION COMPOSITE: None (INSUFFICIENT_PILLARS)" in prompt_text
    assert "QUANTITATIVE SUMMARY FOR MASTER TRADER:" in prompt_text


def test_rag_service_benchmark_summary_includes_quant_and_ml_scores():
    """Verify that RAGService includes deterministic_quant_score and ml_20session_score in structured summary."""
    rag = RAGService()
    gloomberb_payload = {
        "symbol": "NVDA",
        "profile": {"sector": "Technology"},
        "news": [],
        "quote": {"price": 140.0},
        "financials": {"key_ratios": {"forward_pe": 35.0, "trailing_pe": 45.0}},
        "peer_valuation": {"direct_peer_benchmarks": []},
        "sector_benchmark": {"sector_name": "Technology"},
        "options": {},
        "analyst_ratings": {"mean_target_price": 160.0},
    }
    technical_data = {
        "current_price": 140.0,
        "change_5d_pct": 2.5,
        "market_spy_5d_pct": 1.0,
        "relative_alpha_5d": 1.5,
        "rsi14": 55.0,
        "ema20": 138.0,
        "ema50": 130.0,
        "atr": 4.5,
        "high_20d": 145.0,
        "low_20d": 130.0,
        "suggested_stop_loss": 132.0,
        "suggested_target_price": 155.0,
        "analyst_target_price": 160.0,
        "ml_context": {
            "ticker": "NVDA",
            "model_status": "IN_SCOPE",
            "in_scope": True,
            "score_0_100": 90.0,
            "ml_score": 90.0,
            "ml_weight": 0.20,
        },
    }

    payload = rag.get_nemotron_payload(
        gloomberb_payload=gloomberb_payload,
        technical_data=technical_data,
        prompt_profile="full",
    )

    summary = payload["technical_summary"]
    assert "deterministic_quant_score" in summary
    assert "ml_20session_score" in summary
    assert "blended_composite_score" in summary
    assert summary["ml_weight"] == 0.20
    assert summary["ml_20session_score"] == 90.0
    assert summary["deterministic_quant_score"] is not None
    assert summary["blended_composite_score"] is not None
