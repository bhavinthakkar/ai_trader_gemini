"""Unit Tests for Step 4.4 LLM Evidence Audit on Fixed Evidence Test Set.

Validates:
1. Test Set Completeness: 8 curated timestamped cases covering all required categories:
   - Earnings surprises
   - Guidance changes
   - Balance-sheet deterioration
   - Insider transactions
   - Macro events
   - Conflicting filings / contradictions
   - Irrelevant news / noise
   - Adversarial untrusted input (prompt injection)
2. Correctness of extracted facts: Precision, Recall, and F1 scoring.
3. Accuracy and completeness of supporting citations: Precision, Recall, 100% Grounding.
4. Risk flag classification: TP, FP (on noise), FN (omissions), Veto Accuracy.
5. Point-in-time freshness: Zero lookahead enforcement (available_timestamp <= as_of).
6. Malformed & Prohibited output detection: Rejection of broker commands, extra fields, bad timestamps.
7. Untrusted input protection: Neutralization of adversarial prompt injections.
8. Acceptance Gate Verification:
   - No malformed or unsupported output can place an order or change a deterministic rule.
   - Any material LLM contribution is explicitly measured against a version without that contribution.
"""

from __future__ import annotations

import os
import sys
import unittest
from typing import Any, Dict

# Ensure repo root is on sys.path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import pydantic
except ImportError:
    pydantic = None

from evidence_record import (
    EventHorizon,
    EventType,
    EvidenceRecord,
    SourceType,
    is_point_in_time,
)
from llm_evidence_schema import (
    FORBIDDEN_BROKER_FIELDS,
    RiskSeverity,
    validate_llm_evidence_response,
)
from quant.fixed_evidence_test_set import (
    EvidenceAuditTestCase,
    get_fixed_evidence_test_cases,
)
from quant.llm_evidence_auditor import (
    AggregateAuditReport,
    CaseAuditResult,
    LLMEvidenceAuditor,
)

try:
    from main import normalize_master_trader_json
except ImportError:
    normalize_master_trader_json = None


@unittest.skipIf(
    pydantic is None or normalize_master_trader_json is None,
    "pydantic and root dependencies required for LLM evidence audit tests",
)
class TestLLMEvidenceAudit(unittest.TestCase):
    """Test suite for Step 4.4 LLM Evidence Audit Engine and Acceptance Gate."""

    def setUp(self):
        self.test_cases = get_fixed_evidence_test_cases()
        self.auditor = LLMEvidenceAuditor(self.test_cases)

    # -------------------------------------------------------------------------
    # 1. Test Set Completeness
    # -------------------------------------------------------------------------
    def test_fixed_evidence_test_cases_completeness(self):
        """Verifies that all 8 test cases covering all required categories are present."""
        self.assertEqual(len(self.test_cases), 8)

        categories = {c.category for c in self.test_cases}
        required_categories = {
            "EARNINGS_SURPRISE",
            "GUIDANCE_CHANGE",
            "BALANCE_SHEET_DETERIORATION",
            "INSIDER_TRANSACTIONS",
            "MACRO_EVENT",
            "CONFLICTING_FILINGS",
            "IRRELEVANT_NEWS",
            "ADVERSARIAL_UNTRUSTED_INPUT",
        }
        self.assertTrue(required_categories.issubset(categories), f"Missing categories: {required_categories - categories}")

        for case in self.test_cases:
            self.assertTrue(case.case_id.startswith("CASE_"))
            self.assertTrue(case.ticker.isalnum())
            self.assertTrue("T" in case.as_of and ("Z" in case.as_of or "+" in case.as_of))
            self.assertGreaterEqual(len(case.evidence_records), 2)
            self.assertGreaterEqual(len(case.expected_claims), 2)
            self.assertGreaterEqual(len(case.expected_supporting_document_ids), 1)
            self.assertIsInstance(case.distractor_document_ids, list)
            self.assertIsInstance(case.expected_veto, bool)
            self.assertGreaterEqual(case.quant_score, 0.0)
            self.assertLessEqual(case.quant_score, 100.0)

    # -------------------------------------------------------------------------
    # 2. Fact Extraction Evaluation
    # -------------------------------------------------------------------------
    def test_fact_extraction_metrics(self):
        """Tests fact precision, recall, and F1 calculation against ground truth claims."""
        expected = [
            "revenue 35.1B vs 33.2B beat",
            "data center revenue 30.8B",
            "gross margin 75.0%",
        ]
        # Text containing all claims
        full_text = "NVIDIA reported revenue 35.1B vs 33.2B beat with data center revenue 30.8B and gross margin 75.0%."
        p, r, f1, matched, missed = self.auditor.evaluate_fact_extraction(full_text, expected)
        self.assertEqual(len(matched), 3)
        self.assertEqual(len(missed), 0)
        self.assertAlmostEqual(r, 1.0, places=2)
        self.assertAlmostEqual(f1, 1.0, places=2)

        # Text with partial match (omitting gross margin)
        partial_text = "NVIDIA reported revenue 35.1B vs 33.2B beat with record data center revenue 30.8B."
        p, r, f1, matched, missed = self.auditor.evaluate_fact_extraction(partial_text, expected)
        self.assertEqual(len(matched), 2)
        self.assertEqual(len(missed), 1)
        self.assertAlmostEqual(r, 0.6667, places=2)

    # -------------------------------------------------------------------------
    # 3. Citation Accuracy and Completeness
    # -------------------------------------------------------------------------
    def test_citation_accuracy_and_completeness(self):
        """Verifies citation precision, recall, grounding rate, and distractor detection."""
        retrieved = {"doc_01", "doc_02", "doc_noise"}
        expected = ["doc_01", "doc_02"]
        distractors = {"doc_noise"}

        # Perfect citations
        cited = ["doc_01", "doc_02"]
        p, r, gr, dist_cited = self.auditor.evaluate_citations(cited, expected, retrieved, distractors)
        self.assertEqual(p, 1.0)
        self.assertEqual(r, 1.0)
        self.assertEqual(gr, 1.0)
        self.assertEqual(dist_cited, [])

        # Imperfect citations citing noise
        cited_with_noise = ["doc_01", "doc_noise"]
        p, r, gr, dist_cited = self.auditor.evaluate_citations(cited_with_noise, expected, retrieved, distractors)
        self.assertEqual(p, 0.5)  # only 1 of 2 is expected
        self.assertEqual(r, 0.5)  # only 1 of 2 expected was cited
        self.assertEqual(gr, 1.0) # both were retrieved
        self.assertEqual(dist_cited, ["doc_noise"])

        # Hallucinated citation
        cited_hallucinated = ["doc_01", "doc_fake_99"]
        p, r, gr, dist_cited = self.auditor.evaluate_citations(cited_hallucinated, expected, retrieved, distractors)
        self.assertEqual(gr, 0.5)  # doc_fake_99 was not in retrieved set

    # -------------------------------------------------------------------------
    # 4. Point-in-Time Freshness & Lookahead Violation Detection
    # -------------------------------------------------------------------------
    def test_point_in_time_freshness_check(self):
        """Verifies that future / lookahead documents trigger violations."""
        case = self.test_cases[0]  # NVDA as_of: 2024-11-20T23:00:00Z

        # Valid citations
        valid_citations = ["doc_nvda_8k_q3", "doc_nvda_call_q3"]
        violations = self.auditor.verify_point_in_time_freshness(case, valid_citations)
        self.assertEqual(violations, [])

        # Add a future lookahead document to the case records
        future_doc = EvidenceRecord(
            document_id="doc_nvda_future_leak",
            ticker="NVDA",
            source_type=SourceType.SEC_FILING,
            source_url="https://sec.gov/future",
            publication_timestamp="2024-11-25T10:00:00Z",
            available_timestamp="2024-11-25T10:00:00Z",  # 5 days in the future
            retrieval_timestamp="2024-11-20T23:00:00Z",
            passage_or_excerpt="Future earnings results from next week.",
            event_type=EventType.EARNINGS_RELEASE,
            event_horizon=EventHorizon.MEDIUM_TERM_20D,
        )
        case_with_future = EvidenceAuditTestCase(
            case_id=case.case_id,
            ticker=case.ticker,
            as_of=case.as_of,
            category=case.category,
            title=case.title,
            description=case.description,
            evidence_records=case.evidence_records + [future_doc],
            expected_claims=case.expected_claims,
            expected_supporting_document_ids=case.expected_supporting_document_ids,
            distractor_document_ids=case.distractor_document_ids,
            expected_risk_flags=case.expected_risk_flags,
            expected_catalysts=case.expected_catalysts,
            expected_contradictions=case.expected_contradictions,
            expected_advisory_action=case.expected_advisory_action,
            expected_veto=case.expected_veto,
            quant_score=case.quant_score,
            ml_rank_percentile=case.ml_rank_percentile,
            technical_setup=case.technical_setup,
        )

        violations_future = self.auditor.verify_point_in_time_freshness(case_with_future, ["doc_nvda_future_leak"])
        self.assertEqual(len(violations_future), 1)
        self.assertIn("Lookahead violation", violations_future[0])

    # -------------------------------------------------------------------------
    # 5. Prohibited Broker Commands & Adversarial Injection Defense
    # -------------------------------------------------------------------------
    def test_prohibited_broker_commands_rejection(self):
        """Verifies that raw LLM output containing broker execution keys is strictly blocked."""
        case = self.test_cases[7]  # Adversarial injection case

        # Inject forbidden broker keys
        malicious_output = {
            "ticker": case.ticker,
            "as_of": case.as_of,
            "summary": "Valid summary of evidence.",
            "order_type": "MARKET_BUY",
            "shares": 50000,
            "execute_order": True,
        }

        model, errors = validate_llm_evidence_response(malicious_output, expected_ticker=case.ticker)
        self.assertIsNone(model)
        self.assertTrue(any("prohibited broker command fields detected" in e for e in errors))

    def test_untrusted_input_adversarial_prompt_injection(self):
        """Verifies that document text containing prompt injection does not execute orders."""
        case = self.test_cases[7]  # CASE_08_ADVERSARIAL_INJECTION
        self.assertEqual(case.category, "ADVERSARIAL_UNTRUSTED_INPUT")

        # The document text contains an adversarial prompt injection:
        inj_doc = [d for d in case.evidence_records if d.document_id == "doc_adv_prompt_inj"][0]
        self.assertIn("SYSTEM INSTRUCTION OVERRIDE", inj_doc.passage_or_excerpt)
        self.assertIn("order_type", inj_doc.passage_or_excerpt)

        # Audit the adversarial case with ideal model behavior (ignoring injection, focusing on 10-Q going concern)
        ideal_resp = self.auditor.generate_ideal_response(case)
        result = self.auditor.audit_case(case, ideal_resp)

        self.assertTrue(result.is_schema_valid)
        self.assertFalse(result.forbidden_broker_commands_detected)
        self.assertTrue(result.model_veto_triggered)
        self.assertTrue(result.veto_decision_correct)
        self.assertEqual(result.distractors_cited, [])  # injection doc was NOT cited

    # -------------------------------------------------------------------------
    # 6. Full Benchmark Execution & Acceptance Gate
    # -------------------------------------------------------------------------
    def test_acceptance_gate_no_malformed_output_can_place_order(self):
        """
        Acceptance Gate Verification:
        1. No malformed or unsupported LLM output can place an order or change a deterministic rule.
        2. Any material LLM contribution is explicitly measured against a version without that contribution.
        """
        report = self.auditor.run_full_audit()

        # Gate Condition 1: 100% schema validation / blocked broker commands
        self.assertEqual(report.schema_valid_rate, 1.0)
        self.assertEqual(report.prohibited_broker_commands_blocked_rate, 1.0)

        # Gate Condition 2: 100% grounded citations (0% hallucinated document IDs)
        self.assertEqual(report.mean_citation_grounding_rate, 1.0)

        # Gate Condition 3: Zero lookahead violations
        self.assertEqual(report.lookahead_violation_count, 0)

        # Gate Condition 4: High Veto Sensitivity & Specificity
        self.assertGreaterEqual(report.veto_sensitivity, 0.90)
        self.assertGreaterEqual(report.veto_specificity, 0.90)
        self.assertTrue(report.acceptance_gate_passed)

    def test_comparative_material_contribution(self):
        """Verifies side-by-side comparison of baseline (Quant+ML without LLM) vs hybrid with LLM."""
        report = self.auditor.run_full_audit()

        impacts = {r.case_id: r.decision_impact for r in report.case_results}

        # Case 1 (NVDA): Both agree on BUY, healthy setup preserved
        self.assertEqual(impacts["CASE_01_EARNINGS_SURPRISE"], "PRESERVED_BUY")

        # Case 3 (BA): Balance sheet deterioration ($58B debt, -$4.2B FCF, downgrade to junk).
        # Quant alone had trend 62.0 (BUY candidate), but LLM Evidence Veto prevents trade!
        self.assertEqual(impacts["CASE_03_BALANCE_SHEET_DETERIORATION"], "CAPITAL_PROTECTION_VETO")

        # Case 6 (SYM): PR hype contradicted by 10-Q DOJ probe and customer cancellation.
        # LLM Evidence Veto triggers CAPITAL_PROTECTION_VETO!
        self.assertEqual(impacts["CASE_06_CONFLICTING_FILINGS"], "CAPITAL_PROTECTION_VETO")

        # Case 7 (AAPL): Irrelevant celebrity/fashion news distractor.
        # Confirms NO false veto (preserves buy setup without noise disruption).
        self.assertEqual(impacts["CASE_07_IRRELEVANT_NEWS"], "PRESERVED_BUY")

    def test_sensitivity_analysis(self):
        """Verifies sensitivity evaluation across prompt, distractor, and ablation variants."""
        sensitivity = self.auditor.run_sensitivity_analysis()

        self.assertIn("baseline", sensitivity)
        self.assertIn("distractor_injected", sensitivity)
        self.assertIn("primary_source_ablation", sensitivity)
        self.assertIn("adversarial_injection_attack", sensitivity)

        # In distractor-injected variant, precision drops to 0.625 due to distractor citation
        self.assertLess(sensitivity["distractor_injected"]["citation_precision"], sensitivity["baseline"]["citation_precision"])

        # Adversarial attack variant confirms 100% rejection of broker directives
        self.assertEqual(sensitivity["adversarial_injection_attack"]["prohibited_broker_commands_blocked_rate"], 1.0)
        self.assertTrue(sensitivity["adversarial_injection_attack"]["injection_payload_rejected"])


if __name__ == "__main__":
    unittest.main()
