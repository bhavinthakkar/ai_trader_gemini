"""
Unit tests for Step 4: Feeding Candidates into Existing Compliance Gates.

Verifies:
1. Decoupled setup generation allows deterministic candidates (Pullback, Catalyst Dip)
   to reach compliance gates even when the model proposes HOLD.
2. Separate gate behavior is preserved:
   - Catalyst Dip Buy requires composite score >= 52.0 (dip_buy_minimum) with dislocation checks.
   - Pullback Buy strictly requires composite score >= 60.0 (buy_minimum); standard floor is NOT lowered.
3. Every rejected candidate documents a specific failed gate and reason code
   (e.g., BUY_FLOOR, EVENT_RISK, LIQUIDITY, REWARD_TO_RISK).
4. Safety invariants hold: MACRO_EVENT never unlocks, opposing model intents are respected.
"""

import datetime
import pytest

from main import normalize_master_trader_json
from quant.policy_config import load_policy_config
from quant.structured_decision import (
    StandardGateName,
    StandardReasonCode,
    build_structured_decision,
    qualify_candidate_setup,
)

TODAY = datetime.date.today()


def _make_news(title: str, days_ago: int = 0, summary: str = ""):
    pub = TODAY - datetime.timedelta(days=days_ago)
    return {
        "title": title,
        "summary": summary,
        "published_at": pub.isoformat(),
        "category": "News",
    }


def _make_valid_market_data(**overrides):
    base = {
        "current_price": 100.0,
        "atr": 2.5,
        "suggested_stop_loss": 96.0,
        "suggested_target_price": 108.0,
        "rsi14": 42.0,
        "rvol_20d": 1.5,
        "ema20": 99.0,
        "ema50": 95.0,
        "ema200": 90.0,
        "high_20d": 105.0,
        "low_20d": 95.0,
        "change_5d_pct": -6.5,
        "avg_dollar_vol_20d": 50_000_000.0,
        "deterministic_5pillar_scores": {
            "composite_quantitative_score": 64.0,
            "valuation_history_score": 70.0,
            "peer_valuation_score": 65.0,
            "trend": 65.0,
            "sector": 62.0,
            "alpha": 60.0,
        },
        "days_to_earnings": 30,
        "is_value_trap": False,
    }
    base.update(overrides)
    return base


def _make_model_data(**overrides):
    base = {
        "stock": "AAPL",
        "decision": "HOLD",
        "primary_driver": "QUANT_STRUCTURE",
        "confidence": 0.55,
        "buy_score": 0.30,
        "hold_score": 0.60,
        "sell_score": 0.10,
        "data_completeness": 0.95,
        "horizon_days": 10,
        "quant_score": 64,
        "bull_case": ["Support holding nicely"],
        "bear_case": ["Short-term consolidation"],
        "key_risks": [],
        "missing_information": [],
    }
    base.update(overrides)
    return base


class TestCandidateFeedingIntoGates:
    """Tests that candidates reach gates when model says HOLD and receive explicit gate outcomes."""

    def test_pullback_candidate_reaches_gates_and_certifies_when_model_says_hold(self):
        """A valid pullback candidate with composite score 64.0 (>= 60.0) reaches gates and certifies BUY even when model says HOLD."""
        policy = load_policy_config()
        m_data = _make_valid_market_data(
            current_price=100.0,
            ema20=99.5,
            ema50=95.0,
            ema200=90.0,
            change_5d_pct=-4.0,
            deterministic_5pillar_scores={
                "composite_quantitative_score": 64.0,
                "trend": 65.0,
                "valuation_history_score": 70.0,
                "peer_valuation_score": 65.0,
                "sector": 60.0,
                "alpha": 60.0,
            },
        )
        data = _make_model_data(decision="HOLD", primary_driver="TECHNICAL_PULLBACK")

        out = normalize_master_trader_json(
            data=data,
            symbol="AAPL",
            m_data=m_data,
            macro_data={},
            gloomberb_payload={},
            policy=policy,
        )

        assert out["model_decision"] == "HOLD"
        assert out["candidate_action"] == "BUY"
        assert out["candidate_type"] == "PULLBACK_BUY"
        assert out["final_decision"] == "BUY"
        assert out["decision"] == "BUY"
        assert StandardGateName.BUY_FLOOR in out["gate_results"]["passed_gates"]
        assert StandardGateName.REWARD_TO_RISK in out["gate_results"]["passed_gates"]
        assert StandardReasonCode.BUY_ORDER_CERTIFIED in out["gate_results"]["reason_codes"]

    def test_pullback_candidate_rejected_by_buy_floor_when_score_below_standard_minimum(self):
        """Pullback setup with score 56.0 reaches gates but is rejected by BUY_FLOOR because 56.0 < 60.0 (does NOT lower standard floor)."""
        policy = load_policy_config()
        m_data = _make_valid_market_data(
            current_price=100.0,
            ema20=99.5,
            ema50=95.0,
            ema200=90.0,
            change_5d_pct=-4.0,
            deterministic_5pillar_scores={
                "composite_quantitative_score": 56.0,  # Below 60.0 buy_minimum, even though above 52.0 dip floor
                "trend": 60.0,
                "valuation_history_score": 65.0,
                "peer_valuation_score": 60.0,
                "sector": 55.0,
                "alpha": 50.0,
            },
        )
        data = _make_model_data(decision="HOLD", primary_driver="TECHNICAL_PULLBACK")

        out = normalize_master_trader_json(
            data=data,
            symbol="AAPL",
            m_data=m_data,
            macro_data={},
            gloomberb_payload={},
            policy=policy,
        )

        assert out["model_decision"] == "HOLD"
        assert out["candidate_action"] == "BUY"
        assert out["candidate_type"] == "PULLBACK_BUY"
        assert out["final_decision"] == "HOLD"
        assert StandardGateName.BUY_FLOOR in out["gate_results"]["failed_gates"]
        assert StandardReasonCode.SCORE_BELOW_BUY_THRESHOLD in out["gate_results"]["reason_codes"]
        assert StandardReasonCode.SCORE_BELOW_BUY_THRESHOLD in out["rejection_reasons"]

    def test_catalyst_dip_candidate_reaches_gates_and_certifies_when_model_says_hold(self):
        """Certified catalyst dip buy with score 54.0 (>= 52.0 dip floor) reaches gates and certifies BUY when model says HOLD."""
        policy = load_policy_config()
        m_data = _make_valid_market_data(
            current_price=100.0,
            atr=3.0,
            suggested_stop_loss=98.0,
            suggested_target_price=108.0,
            rsi14=38.0,
            change_5d_pct=-7.5,
            high_20d=110.0,
            low_20d=99.0,
            deterministic_5pillar_scores={
                "composite_quantitative_score": 54.0,  # Clears 52.0 dip floor
                "valuation_history_score": 71.0,
                "peer_valuation_score": 66.0,
                "trend": 35.0,
                "sector": 60.0,
                "alpha": 45.0,
            },
        )
        data = _make_model_data(decision="HOLD", primary_driver="NEWS_CATALYST")
        news_payload = [_make_news("Apple slips as analyst trims price target after supply chain delay", summary="AAPL guidance")]

        out = normalize_master_trader_json(
            data=data,
            symbol="AAPL",
            m_data=m_data,
            macro_data={},
            gloomberb_payload={"news": news_payload},
            policy=policy,
        )

        assert out["model_decision"] == "HOLD"
        assert out["candidate_action"] == "BUY"
        assert out["candidate_type"] == "CATALYST_DIP_BUY"
        assert out["final_decision"] == "BUY"
        assert StandardGateName.BUY_FLOOR in out["gate_results"]["passed_gates"]
        assert StandardReasonCode.BUY_ORDER_CERTIFIED in out["gate_results"]["reason_codes"]

    def test_catalyst_dip_rejected_by_buy_floor_when_score_below_dip_floor(self):
        """Catalyst dip candidate with score 50.0 (< 52.0 dip floor) reaches gates and is rejected by BUY_FLOOR."""
        policy = load_policy_config()
        m_data = _make_valid_market_data(
            current_price=100.0,
            atr=3.0,
            suggested_stop_loss=98.0,
            suggested_target_price=108.0,
            rsi14=38.0,
            change_5d_pct=-7.5,
            high_20d=110.0,
            low_20d=99.0,
            deterministic_5pillar_scores={
                "composite_quantitative_score": 50.0,  # Below 52.0 dip floor
                "valuation_history_score": 71.0,
                "peer_valuation_score": 66.0,
                "trend": 30.0,
                "sector": 55.0,
                "alpha": 40.0,
            },
        )
        data = _make_model_data(decision="HOLD", primary_driver="NEWS_CATALYST")
        news_payload = [_make_news("Apple slips as analyst trims price target after supply chain delay", summary="AAPL guidance")]

        out = normalize_master_trader_json(
            data=data,
            symbol="AAPL",
            m_data=m_data,
            macro_data={},
            gloomberb_payload={"news": news_payload},
            policy=policy,
        )

        assert out["model_decision"] == "HOLD"
        assert out["candidate_action"] == "BUY"
        assert out["candidate_type"] == "CATALYST_DIP_BUY"
        assert out["final_decision"] == "HOLD"
        assert StandardGateName.BUY_FLOOR in out["gate_results"]["failed_gates"]
        assert StandardReasonCode.SCORE_BELOW_BUY_THRESHOLD in out["gate_results"]["reason_codes"]
        assert StandardReasonCode.SCORE_BELOW_BUY_THRESHOLD in out["rejection_reasons"]

    def test_candidate_rejected_by_earnings_blackout(self):
        """A qualifying candidate (score 65.0) within earnings blackout reaches gates and is rejected by EVENT_RISK."""
        policy = load_policy_config()
        m_data = _make_valid_market_data(
            current_price=100.0,
            ema20=99.5,
            ema50=95.0,
            ema200=90.0,
            change_5d_pct=-4.0,
            days_to_earnings=2,  # Within 3 session blackout window
            deterministic_5pillar_scores={
                "composite_quantitative_score": 65.0,
                "trend": 65.0,
                "valuation_history_score": 70.0,
                "peer_valuation_score": 65.0,
                "sector": 62.0,
                "alpha": 60.0,
            },
        )
        data = _make_model_data(decision="HOLD", primary_driver="TECHNICAL_PULLBACK")

        out = normalize_master_trader_json(
            data=data,
            symbol="AAPL",
            m_data=m_data,
            macro_data={},
            gloomberb_payload={},
            policy=policy,
        )

        assert out["candidate_action"] == "BUY"
        assert out["final_decision"] == "HOLD"
        assert StandardGateName.EVENT_RISK in out["gate_results"]["failed_gates"]
        assert any(
            rc in out["gate_results"]["reason_codes"]
            for rc in (StandardReasonCode.EARNINGS_BLACKOUT_ACTIVE, StandardReasonCode.EVENT_RISK_BLACKOUT)
        )
        assert any(
            rc in out["rejection_reasons"]
            for rc in (StandardReasonCode.EARNINGS_BLACKOUT_ACTIVE, StandardReasonCode.EVENT_RISK_BLACKOUT)
        )

    def test_candidate_rejected_by_liquidity_gate(self):
        """A qualifying candidate with insufficient dollar volume reaches gates and is rejected by LIQUIDITY."""
        policy = load_policy_config()
        m_data = _make_valid_market_data(
            current_price=100.0,
            ema20=99.5,
            ema50=95.0,
            ema200=90.0,
            change_5d_pct=-4.0,
            avg_dollar_vol_20d=1_000_000.0,  # Below $5M minimum requirement
            deterministic_5pillar_scores={
                "composite_quantitative_score": 65.0,
                "trend": 65.0,
                "valuation_history_score": 70.0,
                "peer_valuation_score": 65.0,
                "sector": 62.0,
                "alpha": 60.0,
            },
        )
        data = _make_model_data(decision="HOLD", primary_driver="TECHNICAL_PULLBACK")

        out = normalize_master_trader_json(
            data=data,
            symbol="AAPL",
            m_data=m_data,
            macro_data={},
            gloomberb_payload={},
            policy=policy,
        )

        assert out["candidate_action"] == "BUY"
        assert out["final_decision"] == "HOLD"
        assert StandardGateName.LIQUIDITY in out["gate_results"]["failed_gates"]
        assert StandardReasonCode.LIQUIDITY_INSUFFICIENT in out["gate_results"]["reason_codes"]

    def test_candidate_rejected_by_reward_to_risk_gate(self):
        """A candidate with inadequate reward-to-risk geometry (< 1.5) reaches gates and is rejected by REWARD_TO_RISK."""
        policy = load_policy_config()
        m_data = _make_valid_market_data(
            current_price=100.0,
            suggested_stop_loss=96.0,       # Risk = $4.00
            suggested_target_price=102.0,   # Reward = $2.00 -> RR = 0.5 < 1.5
            ema20=99.5,
            ema50=95.0,
            ema200=90.0,
            change_5d_pct=-4.0,
            deterministic_5pillar_scores={
                "composite_quantitative_score": 65.0,
                "trend": 65.0,
                "valuation_history_score": 70.0,
                "peer_valuation_score": 65.0,
                "sector": 62.0,
                "alpha": 60.0,
            },
        )
        data = _make_model_data(decision="HOLD", primary_driver="TECHNICAL_PULLBACK")

        out = normalize_master_trader_json(
            data=data,
            symbol="AAPL",
            m_data=m_data,
            macro_data={},
            gloomberb_payload={},
            policy=policy,
        )

        assert out["candidate_action"] == "BUY"
        assert out["final_decision"] == "HOLD"
        assert StandardGateName.REWARD_TO_RISK in out["gate_results"]["failed_gates"]
        assert StandardReasonCode.REWARD_TO_RISK_INSUFFICIENT in out["gate_results"]["reason_codes"]

    def test_candidate_rejected_by_value_trap(self):
        """A candidate with value trap detection reaches gates and is rejected by BUY_FLOOR."""
        policy = load_policy_config()
        m_data = _make_valid_market_data(
            current_price=100.0,
            ema20=99.5,
            ema50=95.0,
            ema200=90.0,
            change_5d_pct=-4.0,
            is_value_trap=True,
            deterministic_5pillar_scores={
                "composite_quantitative_score": 65.0,
                "trend": 65.0,
                "valuation_history_score": 70.0,
                "peer_valuation_score": 65.0,
                "sector": 62.0,
                "alpha": 60.0,
                "is_value_trap": True,
            },
        )
        data = _make_model_data(decision="HOLD", primary_driver="TECHNICAL_PULLBACK")

        out = normalize_master_trader_json(
            data=data,
            symbol="AAPL",
            m_data=m_data,
            macro_data={},
            gloomberb_payload={},
            policy=policy,
        )

        assert out["candidate_action"] == "BUY"
        assert out["final_decision"] == "HOLD"
        assert StandardGateName.BUY_FLOOR in out["gate_results"]["failed_gates"]
        assert StandardReasonCode.VALUE_TRAP_DETECTED in out["gate_results"]["reason_codes"]

    def test_macro_event_driver_never_unlocks_candidate(self):
        """Primary driver MACRO_EVENT is refused regardless of news headlines or catalyst certificates."""
        policy = load_policy_config()
        m_data = _make_valid_market_data(
            change_5d_pct=-7.5,
            deterministic_5pillar_scores={
                "composite_quantitative_score": 55.0,
                "valuation_history_score": 70.0,
                "peer_valuation_score": 65.0,
                "trend": 40.0,
                "sector": 60.0,
                "alpha": 45.0,
            },
        )
        news_payload = [_make_news("Fed hikes interest rates unexpectedly", summary="Macro monetary shock")]
        data = _make_model_data(decision="HOLD", primary_driver="MACRO_EVENT")

        out = normalize_master_trader_json(
            data=data,
            symbol="AAPL",
            m_data=m_data,
            macro_data={},
            gloomberb_payload={"news": news_payload},
            policy=policy,
        )

        assert out["candidate_action"] == "HOLD"
        assert out["candidate_type"] == "NO_SETUP"
        assert out["final_decision"] == "HOLD"
        assert out["no_trade_reason"] == "INSUFFICIENT_EVIDENCE"

    def test_opposing_sell_intent_is_not_inverted_by_dip_certificate(self):
        """When model specifically expresses SELL intent, an adverse dip certificate does NOT unlock a BUY."""
        policy = load_policy_config()
        m_data = _make_valid_market_data(
            current_price=100.0,
            atr=3.0,
            suggested_stop_loss=98.0,
            suggested_target_price=108.0,
            rsi14=38.0,
            change_5d_pct=-7.5,
            high_20d=110.0,
            low_20d=99.0,
            deterministic_5pillar_scores={
                "composite_quantitative_score": 54.0,
                "valuation_history_score": 71.0,
                "peer_valuation_score": 66.0,
                "trend": 35.0,
                "sector": 60.0,
                "alpha": 45.0,
            },
        )
        data = _make_model_data(decision="SELL", primary_driver="NEWS_CATALYST")
        news_payload = [_make_news("Apple slips as analyst trims price target after supply chain delay", summary="AAPL guidance")]

        out = normalize_master_trader_json(
            data=data,
            symbol="AAPL",
            m_data=m_data,
            macro_data={},
            gloomberb_payload={"news": news_payload},
            policy=policy,
        )

        assert out["final_decision"] == "HOLD"
        assert out["candidate_action"] != "BUY"

    def test_candidate_rejected_by_extreme_volatility_regime(self):
        """When extreme volatility regime is detected (IV > 100%), candidate is rejected by EVENT_RISK."""
        policy = load_policy_config()
        m_data = _make_valid_market_data(
            current_price=100.0,
            ema20=99.5,
            ema50=95.0,
            ema200=90.0,
            change_5d_pct=-4.0,
            iv_pct=125.0,  # Extreme volatility regime
            deterministic_5pillar_scores={
                "composite_quantitative_score": 65.0,
                "trend": 65.0,
                "valuation_history_score": 70.0,
                "peer_valuation_score": 65.0,
                "sector": 62.0,
                "alpha": 60.0,
            },
        )
        data = _make_model_data(decision="HOLD", primary_driver="TECHNICAL_PULLBACK")

        out = normalize_master_trader_json(
            data=data,
            symbol="AAPL",
            m_data=m_data,
            macro_data={},
            gloomberb_payload={},
            policy=policy,
        )

        assert out["candidate_action"] == "BUY"
        assert out["final_decision"] == "HOLD"
        assert StandardGateName.EVENT_RISK in out["gate_results"]["failed_gates"]
        assert StandardReasonCode.VOLATILITY_REGIME in out["gate_results"]["reason_codes"]

    def test_no_setup_qualified_documents_no_setup_origin_and_reason(self):
        """When neither pullback nor catalyst nor composite score qualifies, decision is HOLD with NO_SETUP."""
        policy = load_policy_config()
        m_data = _make_valid_market_data(
            current_price=100.0,
            ema20=105.0,
            ema50=110.0,
            ema200=115.0,
            change_5d_pct=-1.0,  # No pullback pattern
            deterministic_5pillar_scores={
                "composite_quantitative_score": 48.0,  # Neutral zone: > 45.0 (no short) and < 52.0 (no dip)
                "trend": 48.0,
                "valuation_history_score": 48.0,
                "peer_valuation_score": 48.0,
                "sector": 48.0,
                "alpha": 48.0,
            },
        )
        data = _make_model_data(decision="HOLD", primary_driver="QUANT_STRUCTURE")

        out = normalize_master_trader_json(
            data=data,
            symbol="AAPL",
            m_data=m_data,
            macro_data={},
            gloomberb_payload={},
            policy=policy,
        )

        assert out["candidate_action"] == "HOLD"
        assert out["candidate_type"] == "NO_SETUP"
        assert out["final_decision"] == "HOLD"
        assert out["decision_origin"] == "NO_SETUP"
        assert any(
            r in out["rejection_reasons"]
            for r in (StandardReasonCode.NO_QUALIFIED_SETUP, StandardReasonCode.NO_TRADE_DEFAULT)
        )
