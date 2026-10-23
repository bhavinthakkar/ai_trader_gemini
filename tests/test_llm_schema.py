"""Unit Tests for Strict LLM Evidence Analysis Schema and Validation (Step 4.2).

Validates:
1. Strict schema compliance for LLMEvidenceAnalysis with full 10+ field contract.
2. Rejection of unsupported/extra fields (extra="forbid").
3. Rejection of prohibited broker commands (order_type, quantity, execute_order, etc.).
4. Rejection of unknown/mismatched tickers.
5. Rejection of invalid ISO 8601 timestamps.
6. Enforcement of limited advisory action enumeration (CANDIDATE_BUY, CANDIDATE_HOLD, CANDIDATE_SELL, NO_ACTION).
7. Validation of catalyst and risk flag provenance (document IDs, timestamp, event horizon, passage).
8. Grounding verification against retrieved RAG evidence document IDs.
9. Separation of explanatory prose (summary) from deterministic/executable fields.
10. Bridge translation from LLMEvidenceAnalysis to normalized signal dictionary.
11. Deterministic evidence veto rule execution in normalize_master_trader_json.
12. Backward compatibility with traditional signal format in validate_signal_json.
"""

from __future__ import annotations

import unittest

try:
    import pydantic
except ImportError:
    pydantic = None

from llm_evidence_schema import (
    AnalysisStatus,
    AdvisoryAction,
    CatalystPolarity,
    RiskSeverity,
    CatalystEvidenceItem,
    RiskFlagEvidenceItem,
    EvidenceContradiction,
    Forecast20dEvidence,
    RelativeOutlook,
    LLMEvidenceAnalysis,
    FORBIDDEN_BROKER_FIELDS,
    validate_llm_evidence_response,
    evidence_analysis_to_signal_dict,
)
from signal_schema import validate_signal_json
try:
    from main import normalize_master_trader_json
except ImportError:
    normalize_master_trader_json = None


@unittest.skipIf(
    pydantic is None or normalize_master_trader_json is None,
    "pydantic and root dependencies (e.g. httpx2) required for LLM evidence schema tests",
)
class TestLLMEvidenceSchema(unittest.TestCase):
    """Test suite for Step 4.2 Strict Evidence Schema."""

    def setUp(self):
        self.valid_payload = {
            "ticker": "NVDA",
            "as_of": "2026-10-15T16:00:00Z",
            "analysis_status": "COMPLETE",
            "advisory_action": "CANDIDATE_BUY",
            "advisory_confidence": 0.75,
            "catalysts": [
                {
                    "catalyst_id": "cat_nvda_01",
                    "description": "Next-gen Blackwell B200 GPU ramp exceeding hyperscaler demand forecasts",
                    "category": "PRODUCT_LAUNCH",
                    "polarity": "BULLISH",
                    "event_horizon": "MEDIUM_TERM_20D",
                    "timestamp": "2026-10-15T10:00:00Z",
                    "supporting_document_ids": ["doc_nvda_sec_001"],
                    "relevant_passage": "Management confirmed initial shipments of Blackwell architecture have commenced with robust hyperscaler backlog.",
                    "is_priced_in": False,
                }
            ],
            "risk_flags": [
                {
                    "flag_id": "rf_nvda_01",
                    "risk_name": "Export controls and geopolitical supply chain concentration",
                    "category": "REGULATION",
                    "severity": "MEDIUM",
                    "event_horizon": "MULTI_YEAR_FUNDAMENTAL",
                    "supporting_document_ids": ["doc_nvda_10k_002"],
                    "relevant_passage": "Changes in semiconductor export license regulations may limit addressable market velocity in selected regions.",
                    "veto_trigger": False,
                }
            ],
            "supporting_document_ids": ["doc_nvda_sec_001", "doc_nvda_10k_002"],
            "contradictions": [
                {
                    "contradiction_id": "ct_01",
                    "pillar_or_source_a": "Technical Trend",
                    "pillar_or_source_b": "Historical Valuation",
                    "description": "Strong price momentum near 52-week highs diverges from 3-year historical P/E multiple median.",
                    "severity": "LOW",
                    "supporting_document_ids": ["doc_nvda_sec_001"],
                }
            ],
            "missing_information": [
                "Full-year custom silicon margins not disclosed in latest 10-Q filing"
            ],
            "forecast_20d": {
                "relative_outlook": "OUTPERFORM",
                "confidence": 0.75,
                "rationale": "Sustained data center accelerator momentum with confirmed backlog support.",
            },
            "summary": "Nvidia demonstrates continued high-margin data center dominance with expanding Blackwell shipments. Technical momentum is supported by balance sheet fortress cash flow.",
        }

    def test_valid_llm_evidence_analysis_schema(self):
        """Verify that a complete, well-formed evidence object validates successfully."""
        model, errors = validate_llm_evidence_response(self.valid_payload, expected_ticker="NVDA")
        self.assertEqual(errors, [])
        self.assertIsNotNone(model)
        self.assertEqual(model.ticker, "NVDA")
        self.assertEqual(model.analysis_status, AnalysisStatus.COMPLETE)
        self.assertEqual(model.advisory_action, AdvisoryAction.CANDIDATE_BUY)
        self.assertEqual(len(model.catalysts), 1)
        self.assertEqual(len(model.risk_flags), 1)
        self.assertEqual(len(model.contradictions), 1)
        self.assertIn("doc_nvda_sec_001", model.supporting_document_ids)

    def test_rejection_of_unsupported_fields(self):
        """Verify that any extraneous unsupported field triggers an immediate schema rejection (extra='forbid')."""
        payload = dict(self.valid_payload)
        payload["unsupported_custom_parameter"] = "exploit_or_hallucination"
        model, errors = validate_llm_evidence_response(payload, expected_ticker="NVDA")
        self.assertIsNone(model)
        self.assertTrue(any("unsupported_custom_parameter" in e.lower() or "extra" in e.lower() for e in errors))

    def test_rejection_of_prohibited_broker_commands(self):
        """Verify that any broker command or order execution instruction is strictly forbidden."""
        for forbidden_field in ("order_type", "shares", "quantity", "limit_price", "execute_order", "execute_now", "broker_routing"):
            payload = dict(self.valid_payload)
            payload[forbidden_field] = "MARKET" if "type" in forbidden_field else 100
            model, errors = validate_llm_evidence_response(payload, expected_ticker="NVDA")
            self.assertIsNone(model)
            self.assertTrue(
                any("prohibited broker command" in e.lower() or forbidden_field in e.lower() for e in errors),
                f"Failed to reject prohibited broker command field: {forbidden_field}"
            )

    def test_rejection_of_unknown_or_mismatched_ticker(self):
        """Verify that unknown tickers or mismatched ticker symbols are rejected."""
        # 1. Ticker mismatch against expected instrument
        model, errors = validate_llm_evidence_response(self.valid_payload, expected_ticker="AAPL")
        self.assertIsNone(model)
        self.assertTrue(any("ticker mismatch" in e.lower() for e in errors))

        # 2. Empty ticker
        payload = dict(self.valid_payload)
        payload["ticker"] = ""
        model, errors = validate_llm_evidence_response(payload)
        self.assertIsNone(model)
        self.assertTrue(any("ticker" in e.lower() for e in errors))

        # 3. Invalid characters in ticker
        payload = dict(self.valid_payload)
        payload["ticker"] = "NVDA@#!"
        model, errors = validate_llm_evidence_response(payload)
        self.assertIsNone(model)
        self.assertTrue(any("invalid ticker" in e.lower() for e in errors))

    def test_rejection_of_invalid_timestamp(self):
        """Verify that malformed or non-ISO timestamps in 'as_of' are rejected."""
        for bad_ts in ("not-a-timestamp", "2026/10/15", "yesterday", "16:00:00", ""):
            payload = dict(self.valid_payload)
            payload["as_of"] = bad_ts
            model, errors = validate_llm_evidence_response(payload, expected_ticker="NVDA")
            self.assertIsNone(model)
            self.assertTrue(any("timestamp" in e.lower() or "as_of" in e.lower() for e in errors))

    def test_advisory_action_limited_enumeration(self):
        """Verify that only the closed advisory enumeration is accepted and does not authorize execution."""
        # Valid actions
        for action in ("CANDIDATE_BUY", "CANDIDATE_HOLD", "CANDIDATE_SELL", "NO_ACTION"):
            payload = dict(self.valid_payload)
            payload["advisory_action"] = action
            model, errors = validate_llm_evidence_response(payload, expected_ticker="NVDA")
            self.assertEqual(errors, [])
            self.assertEqual(model.advisory_action.value, action)

        # Invalid action / execution directive
        for bad_action in ("EXECUTE_BUY", "BUY_NOW", "STRONG_BUY", "ALL_IN"):
            payload = dict(self.valid_payload)
            payload["advisory_action"] = bad_action
            model, errors = validate_llm_evidence_response(payload, expected_ticker="NVDA")
            self.assertIsNone(model)
            self.assertTrue(any("advisory_action" in e.lower() for e in errors))

    def test_catalyst_and_risk_flag_provenance_validation(self):
        """Verify that catalysts and risk flags must include supporting document IDs, timestamp, horizon, and passages."""
        # Missing supporting document IDs in catalyst
        bad_catalyst_payload = dict(self.valid_payload)
        bad_catalyst_payload["catalysts"] = [
            {
                "catalyst_id": "cat_01",
                "description": "Product launch",
                "category": "PRODUCT_LAUNCH",
                "polarity": "BULLISH",
                "event_horizon": "MEDIUM_TERM_20D",
                "timestamp": "2026-10-15T10:00:00Z",
                "supporting_document_ids": [],  # Empty!
                "relevant_passage": "Management confirmed product rollout.",
            }
        ]
        model, errors = validate_llm_evidence_response(bad_catalyst_payload, expected_ticker="NVDA")
        self.assertIsNone(model)
        self.assertTrue(any("supporting_document_ids" in e.lower() or "at least 1" in e.lower() for e in errors))

        # Missing relevant passage in risk flag
        bad_risk_payload = dict(self.valid_payload)
        bad_risk_payload["risk_flags"] = [
            {
                "flag_id": "rf_01",
                "risk_name": "Regulatory delay",
                "category": "REGULATION",
                "severity": "HIGH",
                "event_horizon": "MEDIUM_TERM_20D",
                "supporting_document_ids": ["doc_01"],
                "relevant_passage": "",  # Empty!
            }
        ]
        model, errors = validate_llm_evidence_response(bad_risk_payload, expected_ticker="NVDA")
        self.assertIsNone(model)
        self.assertTrue(any("relevant_passage" in e.lower() or "at least" in e.lower() for e in errors))

    def test_grounding_validation_against_retrieved_evidence(self):
        """Verify that citing document IDs not in the retrieved evidence set fails grounding validation."""
        valid_docs = {"doc_nvda_sec_001", "doc_nvda_10k_002"}
        # All cited IDs present in retrieved evidence
        model, errors = validate_llm_evidence_response(
            self.valid_payload,
            expected_ticker="NVDA",
            valid_document_ids=valid_docs,
        )
        self.assertEqual(errors, [])
        self.assertIsNotNone(model)

        # Hallucinated document ID cited
        bad_payload = dict(self.valid_payload)
        bad_payload["catalysts"] = [
            {
                "catalyst_id": "cat_nvda_01",
                "description": "Hallucinated catalyst",
                "category": "PRODUCT_LAUNCH",
                "polarity": "BULLISH",
                "event_horizon": "MEDIUM_TERM_20D",
                "timestamp": "2026-10-15T10:00:00Z",
                "supporting_document_ids": ["doc_hallucinated_unretrieved_999"],
                "relevant_passage": "Unverified claim from nonexistent document.",
            }
        ]
        model, errors = validate_llm_evidence_response(
            bad_payload,
            expected_ticker="NVDA",
            valid_document_ids=valid_docs,
        )
        self.assertIsNone(model)
        self.assertTrue(any("ungrounded" in e.lower() and "doc_hallucinated_unretrieved_999" in e for e in errors))

    def test_separation_of_explanatory_text_from_executable_fields(self):
        """Verify that explanatory text in 'summary' is preserved but cannot alter deterministic decision fields."""
        payload = dict(self.valid_payload)
        payload["advisory_action"] = "NO_ACTION"
        payload["summary"] = "BUY NOW IMMEDIATELY 1000 SHARES AT MARKET! Strongest buy ever seen."
        model, errors = validate_llm_evidence_response(payload, expected_ticker="NVDA")
        self.assertEqual(errors, [])
        self.assertIsNotNone(model)
        self.assertEqual(model.advisory_action, AdvisoryAction.NO_ACTION)

        # Bridge to signal dict
        signal = evidence_analysis_to_signal_dict(model)
        self.assertEqual(signal["decision"], "HOLD")
        self.assertEqual(signal["summary"], payload["summary"])

    def test_evidence_to_signal_dict_bridge(self):
        """Verify that evidence_analysis_to_signal_dict generates a normalized payload compatible with decision engine."""
        model, errors = validate_llm_evidence_response(self.valid_payload, expected_ticker="NVDA")
        self.assertEqual(errors, [])
        signal = evidence_analysis_to_signal_dict(model)

        self.assertEqual(signal["stock"], "NVDA")
        self.assertEqual(signal["decision"], "BUY")
        self.assertEqual(signal["decision_origin"], "EVIDENCE_SERVICE")
        self.assertEqual(signal["advisory_action"], "CANDIDATE_BUY")
        self.assertEqual(len(signal["catalysts"]), 1)
        self.assertEqual(len(signal["risk_flags"]), 1)
        self.assertEqual(len(signal["contradictions"]), 1)
        self.assertIn("Blackwell", signal["bull_case"][0])

    def test_deterministic_evidence_veto_in_normalize(self):
        """Verify that a critical evidence risk flag or veto trigger causes a deterministic downgrade to HOLD."""
        # 1. Setup a candidate BUY signal with high composite
        payload = dict(self.valid_payload)
        payload["advisory_action"] = "CANDIDATE_BUY"
        payload["risk_flags"] = [
            {
                "flag_id": "rf_fatal_01",
                "risk_name": "DOJ Antitrust Subpoena and Customer Exclusivity Investigation",
                "category": "REGULATION",
                "severity": "CRITICAL",  # CRITICAL severity!
                "event_horizon": "MEDIUM_TERM_20D",
                "supporting_document_ids": ["doc_nvda_sec_001"],
                "relevant_passage": "DOJ served formal subpoena investigating bundled AI server exclusivity contracts.",
                "veto_trigger": True,     # Deterministic veto trigger!
            }
        ]
        model, errors = validate_llm_evidence_response(payload, expected_ticker="NVDA")
        self.assertEqual(errors, [])
        signal = evidence_analysis_to_signal_dict(model)

        # Market data simulating high quantitative score
        m_data = {
            "current_price": 120.0,
            "suggested_stop_loss": 115.0,
            "suggested_target_price": 140.0,
            "high_20d": 130.0,
            "low_20d": 110.0,
            "deterministic_data_completeness": 0.90,
            "deterministic_5pillar_scores": {
                "composite_quantitative_score": 78.5,
                "raw_composite": 78.5,
                "pillar_status": {},
                "pillar_input_tracking": {},
            },
            "atr": 3.0,
            "rvol_20d": 1.2,
            "avg_dollar_vol_20d": 50_000_000.0,
            "days_to_earnings": 45,
            "is_value_trap": False,
        }

        # Run through deterministic normalization
        normalized = normalize_master_trader_json(signal, "NVDA", m_data=m_data)

        # The quantitative score would qualify for BUY, but the critical evidence risk flag forces HOLD!
        self.assertEqual(normalized["decision"], "HOLD")
        self.assertEqual(normalized["no_trade_reason"], "EVIDENCE_VETO")
        self.assertTrue(any("Critical Evidence Risk Flag" in r or "EVIDENCE_VETO" in r for r in normalized["key_risks"]))

    def test_validate_signal_json_backward_compatibility(self):
        """Verify that validate_signal_json supports both new strict evidence schema and traditional signals."""
        # A. Strict evidence analysis payload
        norm_ev, errors_ev = validate_signal_json(self.valid_payload, expected_ticker="NVDA")
        self.assertEqual(errors_ev, [])
        self.assertEqual(norm_ev["stock"], "NVDA")
        self.assertEqual(norm_ev["decision"], "BUY")

        # B. Traditional signal payload
        trad_payload = {
            "stock": "MSFT",
            "decision": "BUY",
            "primary_driver": "QUANT_STRUCTURE",
            "confidence": 0.80,
            "buy_score": 0.70,
            "hold_score": 0.20,
            "sell_score": 0.10,
            "horizon_days": 20,
            "horizon_sessions": 20,
            "quant_score": 75.0,
            "forecast_20d": {
                "relative_outlook": "OUTPERFORM",
                "confidence": 0.80,
                "rationale": "Strong Azure cloud momentum",
            },
        }
        norm_tr, errors_tr = validate_signal_json(
            trad_payload,
            required_horizon_sessions=20,
            require_forecast_20d=True,
            expected_ticker="MSFT",
        )
        self.assertEqual(errors_tr, [])
        self.assertEqual(norm_tr["decision"], "BUY")

        # C. Forbidden broker command in traditional payload is also rejected
        trad_forbidden = dict(trad_payload)
        trad_forbidden["execute_order"] = True
        norm_fb, errors_fb = validate_signal_json(trad_forbidden)
        self.assertEqual(norm_fb, {})
        self.assertTrue(any("prohibited broker command" in e.lower() for e in errors_fb))

        # D. Traditional signal containing catalysts, risk_flags, advisory_action, and analysis_status does not falsely trigger evidence schema
        trad_with_catalysts = dict(trad_payload)
        trad_with_catalysts["catalysts"] = [{"rank": 1, "catalyst": "Strong cloud backlog"}]
        trad_with_catalysts["risk_flags"] = [{"risk": "Foreign exchange headwind"}]
        trad_with_catalysts["advisory_action"] = "CANDIDATE_BUY"
        trad_with_catalysts["analysis_status"] = "COMPLETE"
        norm_cat, errors_cat = validate_signal_json(
            trad_with_catalysts,
            required_horizon_sessions=20,
            require_forecast_20d=True,
            expected_ticker="MSFT",
        )
        self.assertEqual(errors_cat, [])
        self.assertEqual(norm_cat["decision"], "BUY")

        # E. Traditional signal with NEUTRAL decision normalizes safely to HOLD
        trad_neutral = dict(trad_payload)
        trad_neutral["decision"] = "NEUTRAL"
        norm_neu, errors_neu = validate_signal_json(
            trad_neutral,
            required_horizon_sessions=20,
            require_forecast_20d=True,
            expected_ticker="MSFT",
        )
        self.assertEqual(errors_neu, [])
        self.assertEqual(norm_neu["decision"], "HOLD")

        # F. Traditional signal with 0-100 percentage probability scale normalizes to [0, 1]
        trad_pct = dict(trad_payload)
        trad_pct["buy_score"] = 70.0
        trad_pct["hold_score"] = 20.0
        trad_pct["sell_score"] = 10.0
        norm_pct, errors_pct = validate_signal_json(
            trad_pct,
            required_horizon_sessions=20,
            require_forecast_20d=True,
            expected_ticker="MSFT",
        )
        self.assertEqual(errors_pct, [])
        self.assertAlmostEqual(norm_pct["buy_score"] + norm_pct["hold_score"] + norm_pct["sell_score"], 1.0, places=3)
        self.assertAlmostEqual(norm_pct["buy_score"], 0.70, places=2)


if __name__ == "__main__":
    unittest.main()
