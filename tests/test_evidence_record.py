"""Unit Tests for Auditable Evidence Record and Zero-Lookahead RAG (Step 4.1).

Validates:
1. Explicit 10-field evidence record schema completeness.
2. Primary source prioritization hierarchy (SEC Filings, Macro Releases, IR, Transcripts).
3. Deterministic collision-resistant document ID generation.
4. Point-in-time zero-lookahead enforcement: publication_timestamp <= available_timestamp <= as_of_timestamp.
5. Automated inference of SourceType, EventType, and EventHorizon.
6. SQLite persistence and foreign-key linking to trading signals in `rag_evidence_records`.
7. Zero-lookahead `query_evidence_as_of` database filtering.
8. Metadata-hybrid RAG integration: `RAGService.get_nemotron_payload` embeds complete EvidenceRecord objects.
9. Strict point-in-time document pruning when as_of_timestamp is passed to RAG retrieval.
"""

from __future__ import annotations

import datetime
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from evidence_record import (
    EvidenceRecord,
    SourceType,
    EventType,
    EventHorizon,
    SOURCE_RELIABILITY_WEIGHTS,
    format_iso_timestamp,
    generate_document_id,
    is_point_in_time,
    infer_source_type,
    infer_event_type,
    infer_event_horizon,
)
from db import (
    init_db,
    save_results,
    save_evidence_records,
    get_evidence_records_for_signal,
    query_evidence_as_of,
)
from rag_service import RAGService


class TestEvidenceRecord(unittest.TestCase):
    """Test suite for Step 4.1 Auditable Evidence Records."""

    def test_evidence_record_10_field_schema(self):
        """Verify that all 10 required fields are present, normalized, and correctly typed."""
        rec = EvidenceRecord(
            document_id="doc_nvda_001",
            ticker="nvda",
            source_type=SourceType.SEC_FILING.value,
            source_url="https://www.sec.gov/edgar/data/1045810/000104581024000029/nvda-20240128.htm",
            publication_timestamp="2024-02-21T21:05:00Z",
            available_timestamp="2024-02-21T21:05:00Z",
            retrieval_timestamp="2026-10-16T12:00:00Z",
            passage_or_excerpt="Full year fiscal 2024 revenue rose 126% to a record $60.9 billion.",
            event_type=EventType.EARNINGS_RELEASE.value,
            event_horizon=EventHorizon.MULTI_YEAR_FUNDAMENTAL.value,
            reliability=1.00,
            importance=1.00,
            hybrid_score=0.92,
            sub_query="Updated Guidance & Earnings Takeaways",
        )

        data = rec.to_dict()
        required_fields = [
            "document_id",
            "ticker",
            "source_type",
            "source_url",
            "publication_timestamp",
            "available_timestamp",
            "retrieval_timestamp",
            "passage_or_excerpt",
            "event_type",
            "event_horizon",
        ]
        for f in required_fields:
            self.assertIn(f, data, f"Missing required field: {f}")
            self.assertTrue(data[f], f"Field {f} must not be empty")

        self.assertEqual(data["ticker"], "NVDA")
        self.assertEqual(data["source_type"], "SEC_FILING")
        self.assertEqual(data["event_type"], "EARNINGS_RELEASE")
        self.assertEqual(data["event_horizon"], "MULTI_YEAR_FUNDAMENTAL")
        self.assertEqual(data["publication_timestamp"], "2024-02-21T21:05:00Z")

    def test_primary_source_hierarchy_and_weights(self):
        """Verify primary source prioritization: SEC, Macro, IR, and Transcripts rank highest."""
        self.assertEqual(SOURCE_RELIABILITY_WEIGHTS[SourceType.SEC_FILING.value], 1.00)
        self.assertEqual(SOURCE_RELIABILITY_WEIGHTS[SourceType.MACRO_RELEASE.value], 1.00)
        self.assertEqual(SOURCE_RELIABILITY_WEIGHTS[SourceType.COMPANY_IR.value], 0.95)
        self.assertEqual(SOURCE_RELIABILITY_WEIGHTS[SourceType.EARNINGS_CALL_TRANSCRIPT.value], 0.90)

        # Secondary / media sources have lower reliability
        self.assertEqual(SOURCE_RELIABILITY_WEIGHTS[SourceType.ANALYST_RESEARCH.value], 0.75)
        self.assertEqual(SOURCE_RELIABILITY_WEIGHTS[SourceType.REPUTABLE_NEWS.value], 0.70)
        self.assertEqual(SOURCE_RELIABILITY_WEIGHTS[SourceType.GENERAL_NEWS.value], 0.60)

        # Primary sources strictly outweigh secondary sources
        self.assertGreater(
            SOURCE_RELIABILITY_WEIGHTS[SourceType.SEC_FILING.value],
            SOURCE_RELIABILITY_WEIGHTS[SourceType.REPUTABLE_NEWS.value],
        )

    def test_deterministic_document_id(self):
        """Verify deterministic, collision-resistant document ID generation."""
        id1 = generate_document_id("AAPL", "SEC_FILING", "2025-01-15T00:00:00Z", "Record revenue growth.")
        id2 = generate_document_id("AAPL", "SEC_FILING", "2025-01-15T00:00:00Z", "Record revenue growth.")
        id3 = generate_document_id("AAPL", "SEC_FILING", "2025-01-15T00:00:00Z", "Different passage.")

        self.assertTrue(id1.startswith("doc_aapl_"))
        self.assertEqual(id1, id2)
        self.assertNotEqual(id1, id3)

    def test_point_in_time_filtering(self):
        """Verify strict point-in-time check: publication <= available <= as_of."""
        as_of = "2025-06-01T16:00:00Z"

        # 1. Past document: valid
        self.assertTrue(is_point_in_time("2025-05-15T10:00:00Z", "2025-05-15T10:00:00Z", as_of))

        # 2. Document published at the exact same minute: valid
        self.assertTrue(is_point_in_time("2025-06-01T16:00:00Z", "2025-06-01T16:00:00Z", as_of))

        # 3. Document published after as_of: lookahead violation, must reject
        self.assertFalse(is_point_in_time("2025-06-01T16:01:00Z", "2025-06-01T16:01:00Z", as_of))
        self.assertFalse(is_point_in_time("2025-06-05T00:00:00Z", "2025-06-05T00:00:00Z", as_of))

        # 4. Document published before as_of but embargoed/available after as_of: must reject
        self.assertFalse(is_point_in_time("2025-05-30T10:00:00Z", "2025-06-02T10:00:00Z", as_of))

    def test_inference_helpers(self):
        """Verify automated semantic mapping of source types, event types, and horizons."""
        self.assertEqual(infer_source_type("SEC EDGAR Direct", "Form-10-K"), SourceType.SEC_FILING)
        self.assertEqual(infer_source_type("Federal Reserve FRED", "MacroReport"), SourceType.MACRO_RELEASE)
        self.assertEqual(infer_source_type("Company IR Portal", "IRRelease"), SourceType.COMPANY_IR)
        self.assertEqual(infer_source_type("Quarterly Call", "EarningsTranscript"), SourceType.EARNINGS_CALL_TRANSCRIPT)
        self.assertEqual(infer_source_type("Reuters Business", "NewsArticle"), SourceType.REPUTABLE_NEWS)

        self.assertEqual(infer_event_type("Quarterly EPS surged 45% YoY", "Earnings"), EventType.EARNINGS_RELEASE)
        self.assertEqual(infer_event_type("Management raised full-year operating guidance", "Guidance"), EventType.GUIDANCE_UPDATE)
        self.assertEqual(infer_event_type("FOMC leaves benchmark fed funds rate unchanged", "Macro"), EventType.MACRO_RATE_DECISION)
        self.assertEqual(infer_event_type("CEO purchased 50,000 shares in open market", "Form-4"), EventType.INSIDER_TRANSACTION)

    def test_db_persistence_and_foreign_key(self):
        """Verify that EvidenceRecord instances are saved to SQLite and foreign-keyed to signals."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            init_db(db_path)

            sample_signal = {
                "symbol": "NVDA",
                "decision": "BUY",
                "confidence": 0.85,
                "quant_score": 78.5,
                "evidence_records": [
                    {
                        "document_id": "doc_nvda_1",
                        "ticker": "NVDA",
                        "source_type": SourceType.SEC_FILING.value,
                        "source_url": "https://sec.gov/1",
                        "publication_timestamp": "2025-01-10T10:00:00Z",
                        "available_timestamp": "2025-01-10T10:00:00Z",
                        "retrieval_timestamp": "2025-01-10T12:00:00Z",
                        "passage_or_excerpt": "Quarterly net income reached $12.3B.",
                        "event_type": EventType.EARNINGS_RELEASE.value,
                        "event_horizon": EventHorizon.MEDIUM_TERM_20D.value,
                        "reliability": 1.00,
                        "importance": 1.00,
                        "hybrid_score": 0.88,
                        "sub_query": "Updated Guidance & Earnings Takeaways",
                    },
                    {
                        "document_id": "doc_nvda_2",
                        "ticker": "NVDA",
                        "source_type": SourceType.COMPANY_IR.value,
                        "source_url": "https://ir.nvda.com/2",
                        "publication_timestamp": "2025-01-12T14:00:00Z",
                        "available_timestamp": "2025-01-12T14:00:00Z",
                        "retrieval_timestamp": "2025-01-12T15:00:00Z",
                        "passage_or_excerpt": "Blackwell Ultra server architecture shipping at scale.",
                        "event_type": EventType.PRODUCT_LAUNCH.value,
                        "event_horizon": EventHorizon.SHORT_TERM_7D.value,
                        "reliability": 0.95,
                        "importance": 0.90,
                        "hybrid_score": 0.81,
                        "sub_query": "Recent Material Events (Last 24h / 7d)",
                    },
                ],
            }

            save_results([sample_signal], model_used="test_runner", db_path=db_path)

            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("SELECT id FROM signals WHERE symbol = 'NVDA'")
            row = cur.fetchone()
            self.assertIsNotNone(row)
            signal_id = row[0]

            records = get_evidence_records_for_signal(signal_id, db_path=db_path)
            self.assertEqual(len(records), 2)

            doc_ids = {r["document_id"] for r in records}
            self.assertIn("doc_nvda_1", doc_ids)
            self.assertIn("doc_nvda_2", doc_ids)

            self.assertEqual(records[0]["signal_id"], signal_id)
            self.assertEqual(records[1]["signal_id"], signal_id)

            r1 = next(r for r in records if r["document_id"] == "doc_nvda_1")
            self.assertEqual(r1["source_type"], "SEC_FILING")
            self.assertEqual(r1["event_type"], "EARNINGS_RELEASE")
            self.assertEqual(r1["publication_timestamp"], "2025-01-10T10:00:00Z")
            self.assertEqual(r1["reliability"], 1.00)
            self.assertIn("net income", r1["passage_or_excerpt"])

        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_query_evidence_as_of_db_zero_lookahead(self):
        """Verify that query_evidence_as_of strictly excludes post-decision documents."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            init_db(db_path)

            recs = [
                EvidenceRecord(
                    document_id="doc_aapl_past",
                    ticker="AAPL",
                    source_type=SourceType.SEC_FILING.value,
                    source_url="",
                    publication_timestamp="2025-01-15T12:00:00Z",
                    available_timestamp="2025-01-15T12:00:00Z",
                    retrieval_timestamp="2025-01-15T12:00:00Z",
                    passage_or_excerpt="Past earnings data.",
                    event_type=EventType.EARNINGS_RELEASE.value,
                    event_horizon=EventHorizon.MEDIUM_TERM_20D.value,
                    hybrid_score=0.95,
                ),
                EvidenceRecord(
                    document_id="doc_aapl_future",
                    ticker="AAPL",
                    source_type=SourceType.COMPANY_IR.value,
                    source_url="",
                    publication_timestamp="2025-02-10T12:00:00Z",
                    available_timestamp="2025-02-10T12:00:00Z",
                    retrieval_timestamp="2025-02-10T12:00:00Z",
                    passage_or_excerpt="Future unreleased product roadmap.",
                    event_type=EventType.PRODUCT_LAUNCH.value,
                    event_horizon=EventHorizon.SHORT_TERM_7D.value,
                    hybrid_score=0.99,
                ),
            ]

            save_evidence_records(recs, signal_id=1, db_path=db_path)

            # Query as-of Jan 31, 2025: should only return the past record!
            results_jan = query_evidence_as_of("AAPL", as_of_timestamp="2025-01-31T23:59:59Z", db_path=db_path)
            self.assertEqual(len(results_jan), 1)
            self.assertEqual(results_jan[0]["document_id"], "doc_aapl_past")

            # Query as-of Feb 15, 2025: should return both records
            results_feb = query_evidence_as_of("AAPL", as_of_timestamp="2025-02-15T23:59:59Z", db_path=db_path)
            self.assertEqual(len(results_feb), 2)
            self.assertEqual({r["document_id"] for r in results_feb}, {"doc_aapl_past", "doc_aapl_future"})

        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_rag_service_evidence_records_in_payload(self):
        """Verify that RAGService extracts and embeds 10-field EvidenceRecords in the payload."""
        rag = RAGService(top_k=4)

        mock_gloomberb = {
            "symbol": "MSFT",
            "profile": {"sector": "Technology", "description": "Global enterprise software and cloud provider."},
            "news": [
                {
                    "source": "Reuters",
                    "title": "Azure Cloud Revenue Accelerates 33% YoY",
                    "summary": "Enterprise AI workload adoption drives acceleration across cloud services.",
                    "published_at": "2026-10-14T09:00:00Z",
                    "url": "https://reuters.com/news/azure",
                }
            ],
            "filings": [
                {
                    "form": "10-Q",
                    "date": "2026-09-30",
                    "summary": "Operating cash flow rose 18% to $34.2 billion for the quarter.",
                    "url": "https://sec.gov/edgar/msft-10q",
                }
            ],
            "peer_valuation": {"direct_peer_benchmarks": []},
            "financials": {"key_ratios": {}},
            "macro_econ": {},
        }

        mock_technical = {
            "current_price": 420.0,
            "atr": 6.5,
            "high_20d": 435.0,
            "low_20d": 405.0,
            "suggested_stop_loss": 408.0,
            "suggested_target_price": 445.0,
            "analyst_target_price": 475.0,
            "rsi14": 56.5,
            "ema20": 418.0,
            "ema50": 410.0,
        }

        payload = rag.get_nemotron_payload(mock_gloomberb, mock_technical, prompt_profile="full")
        self.assertIn("evidence_records", payload)
        records = payload["evidence_records"]
        self.assertGreater(len(records), 0)

        first_rec = records[0]
        required_keys = [
            "document_id",
            "ticker",
            "source_type",
            "source_url",
            "publication_timestamp",
            "available_timestamp",
            "retrieval_timestamp",
            "passage_or_excerpt",
            "event_type",
            "event_horizon",
        ]
        for k in required_keys:
            self.assertIn(k, first_rec, f"Missing {k} in evidence record")
            self.assertIsNotNone(first_rec[k])

        self.assertEqual(first_rec["ticker"], "MSFT")
        self.assertIn(first_rec["source_type"], [s.value for s in SourceType])
        self.assertIn(first_rec["event_type"], [e.value for e in EventType])

    def test_rag_service_as_of_zero_lookahead_filtering(self):
        """Verify that passing as_of_timestamp to RAGService strictly purges future documents."""
        rag = RAGService(top_k=4)

        mock_gloomberb = {
            "symbol": "TSLA",
            "profile": {"sector": "Consumer Cyclical", "description": "Electric vehicle and clean energy leader."},
            "news": [
                {
                    "source": "Gloomberb News",
                    "title": "Historical Delivery Numbers",
                    "summary": "Deliveries met consensus expectations.",
                    "published_at": "2025-01-02T10:00:00Z",
                },
                {
                    "source": "Gloomberb News",
                    "title": "Future Cybercab Robotaxi Event",
                    "summary": "Full autonomous robotaxi commercial deployment roadmap.",
                    "published_at": "2025-06-15T10:00:00Z",
                },
            ],
            "filings": [],
            "peer_valuation": {},
            "financials": {},
            "macro_econ": {},
        }

        # Query with as_of cutoff at 2025-03-01: Future June 2025 news MUST be excluded!
        docs = rag.convert_gloomberb_to_documents(mock_gloomberb, as_of_timestamp="2025-03-01T00:00:00Z")

        contents = [d.page_content for d in docs]
        self.assertTrue(any("Historical Delivery Numbers" in c for c in contents))
        self.assertFalse(any("Future Cybercab Robotaxi Event" in c for c in contents))


if __name__ == "__main__":
    unittest.main()
