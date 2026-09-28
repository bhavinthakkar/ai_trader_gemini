import unittest
from unittest.mock import patch, MagicMock

from main import (
    _clean_telegram_markdown,
    _truncate_telegram_text,
    split_telegram_text,
    format_telegram_digest,
    send_telegram_digest,
)


class TestTelegramDigest(unittest.TestCase):
    def test_clean_telegram_markdown(self):
        self.assertEqual(_clean_telegram_markdown("test_case_value"), "test case value")
        self.assertEqual(_clean_telegram_markdown("*bold*"), "bold")
        self.assertEqual(_clean_telegram_markdown("`code`"), "'code'")
        self.assertEqual(_clean_telegram_markdown("- bullet item"), "bullet item")
        self.assertEqual(_clean_telegram_markdown("* bullet item"), "bullet item")
        self.assertEqual(_clean_telegram_markdown("• bullet item"), "bullet item")
        self.assertEqual(_clean_telegram_markdown(""), "")
        self.assertEqual(_clean_telegram_markdown(None), "")

    def test_truncate_telegram_text(self):
        short = "Simple short rationale."
        self.assertEqual(_truncate_telegram_text(short, max_chars=50), short)

        long_text = "This is a very detailed and verbose financial explanation discussing technical momentum and valuation metrics across the portfolio."
        truncated = _truncate_telegram_text(long_text, max_chars=40)
        self.assertTrue(len(truncated) <= 40)
        self.assertTrue(truncated.endswith("..."))
        # Check that it truncated at word boundary and cleaned special chars
        self.assertNotIn("_", truncated)

    def test_split_telegram_text_empty_and_short(self):
        self.assertEqual(split_telegram_text(""), [])
        self.assertEqual(split_telegram_text("hello world"), ["hello world"])

    def test_split_telegram_text_under_limit(self):
        text = "Block 1\n\nBlock 2\n\nBlock 3"
        chunks = split_telegram_text(text, max_chars=100)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)

    def test_split_telegram_text_by_paragraphs(self):
        block1 = "A" * 2000
        block2 = "B" * 2000
        text = f"{block1}\n\n{block2}"
        chunks = split_telegram_text(text, max_chars=3800)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0], block1)
        self.assertEqual(chunks[1], block2)
        for c in chunks:
            self.assertLessEqual(len(c), 3800)

    def test_split_telegram_text_large_single_paragraph(self):
        # A single paragraph with lines that exceeds max_chars
        lines = [f"Line {i}: " + ("x" * 100) for i in range(50)]
        text = "\n".join(lines)
        chunks = split_telegram_text(text, max_chars=1000)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c), 1000)

    def test_split_telegram_text_huge_unbroken_line(self):
        # A single line with no spaces or newlines exceeding limit
        huge_line = "Z" * 5000
        chunks = split_telegram_text(huge_line, max_chars=1000)
        self.assertEqual(len(chunks), 5)
        for c in chunks:
            self.assertEqual(len(c), 1000)

    def test_format_telegram_digest_truncation(self):
        sample_results = [
            {
                "stock": "NVDA",
                "decision": "BUY",
                "confidence": 0.85,
                "buy_score": 0.85,
                "hold_score": 0.10,
                "sell_score": 0.05,
                "horizon_days": 10,
                "quant_score": 78.5,
                "pillar_scores": {
                    "trend": 85.0,
                    "sector": 70.0,
                    "alpha": 80.0,
                    "valuation_history": 65.0,
                    "peer_valuation": 72.0,
                },
                "data_completeness": 0.95,
                "reward_risk_ratio": 2.2,
                "breakeven_win_rate": 0.31,
                "primary_driver": "QUANT_STRUCTURE",
                "bull_case": [
                    "Strong momentum breakout above 50 EMA on above-average volume.",
                    "Forward P/E represents a 15% discount to historical 3-year median and competitor peers.",
                    "Extra third bull point that should be truncated away.",
                ],
                "bear_case": [
                    "Broader tech sector multiple contraction on persistent interest rate volatility and macro headwinds.",
                    "Extra bear point that should be truncated away.",
                ],
                "key_risks": [
                    "ATR trailing stop violation at $118.50 with sudden semiconductor export restrictions.",
                    "Secondary risk that should be omitted in concise summary.",
                ],
                "missing_information": [
                    "Latest quarter foundry capex allocation schedule.",
                    "Second missing info item.",
                ],
            }
        ]

        digest = format_telegram_digest(sample_results, model_label="Nemotron_3")
        # Check header
        self.assertIn("Gloomberb Multi-Pillar RAG Digest (Nemotron 3)", digest)
        # Check stock title
        self.assertIn("🟢 *NVDA* | *BUY*", digest)
        # Check that BUY includes at most 2 bull cases
        self.assertIn("Strong momentum breakout", digest)
        self.assertIn("Forward P/E represents", digest)
        self.assertNotIn("Extra third bull point", digest)
        # Check that bear case is limited to 1 for BUY
        self.assertIn("Broader tech sector multiple contraction", digest)
        self.assertNotIn("Extra bear point", digest)
        # Check key risk limited to 1
        self.assertIn("ATR trailing stop violation", digest)
        self.assertNotIn("Secondary risk", digest)
        # Check missing info limited to 1
        self.assertIn("Latest quarter foundry capex", digest)
        self.assertNotIn("Second missing info item", digest)

        # Entire digest for one stock should be well under 1500 chars (around 600-800 chars)
        self.assertLess(len(digest), 1500)

    @patch("requests.post")
    def test_send_telegram_digest_single_message(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        send_telegram_digest("fake_token", "fake_chat", "Short test message")
        self.assertEqual(mock_post.call_count, 1)
        args, kwargs = mock_post.call_args
        self.assertEqual(kwargs["json"]["text"], "Short test message")
        self.assertEqual(kwargs["json"]["parse_mode"], "Markdown")

    @patch("requests.post")
    def test_send_telegram_digest_multipart_guaranteed_size(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Create text of 8000 chars
        large_text = "\n\n".join([f"Stock {i}: " + ("x" * 900) for i in range(8)])
        with patch("time.sleep", return_value=None):
            send_telegram_digest("fake_token", "fake_chat", large_text)

        self.assertGreater(mock_post.call_count, 1)
        for call in mock_post.call_args_list:
            chunk = call[1]["json"]["text"]
            self.assertLessEqual(len(chunk), 3800)

    @patch("requests.post")
    def test_send_telegram_digest_markdown_parse_error_fallback(self, mock_post):
        # First call fails with entity parsing error, second succeeds
        failed_response = MagicMock()
        failed_response.status_code = 400
        failed_response.text = '{"ok":false,"description":"Bad Request: can\'t parse entities"}'

        ok_response = MagicMock()
        ok_response.status_code = 200

        mock_post.side_effect = [failed_response, ok_response]

        send_telegram_digest("fake_token", "fake_chat", "Broken _markdown text")
        self.assertEqual(mock_post.call_count, 2)
        # Second call should not have parse_mode
        first_call_json = mock_post.call_args_list[0][1]["json"]
        second_call_json = mock_post.call_args_list[1][1]["json"]
        self.assertEqual(first_call_json.get("parse_mode"), "Markdown")
        self.assertNotIn("parse_mode", second_call_json)


if __name__ == "__main__":
    unittest.main()
