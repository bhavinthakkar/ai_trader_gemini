import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import sanitize_plain_english_reasoning, get_plain_english_takeaways


class TestPlainEnglishReasoning(unittest.TestCase):
    def test_sanitize_removes_fed_policy_repetition(self):
        dense_reasoning = (
            "The final directional decision is BUY. Price is above EMA20 at $21.70 with positive momentum. "
            "Fed policy context is neutral to mildly accommodative: fed funds 3.88%, real 10Y 2.78%, 10Y 4.98%, "
            "10Y 5-day change +8.0 bps, 2Y10Y spread +0.25% normal/steepening; expected cut/hike path and FOMC timing are missing. "
            "Strong revenue growth of 28.8% supports valuation."
        )
        cleaned = sanitize_plain_english_reasoning(dense_reasoning)
        
        # Check that Fed policy clauses were removed
        self.assertNotIn("fed funds 3.88%", cleaned.lower())
        self.assertNotIn("real 10y", cleaned.lower())
        self.assertNotIn("2y10y spread", cleaned.lower())
        self.assertNotIn("fomc timing", cleaned.lower())
        
        # Check that business context was preserved
        self.assertIn("Price is above EMA20", cleaned)
        self.assertIn("Strong revenue growth of 28.8%", cleaned)

    def test_sanitize_semicolon_clauses(self):
        mixed = (
            "Bullish continuation setup; Fed funds at 3.88% remains restrictive; RSI14 is healthy at 54.2."
        )
        cleaned = sanitize_plain_english_reasoning(mixed)
        self.assertNotIn("fed funds", cleaned.lower())
        self.assertIn("Bullish continuation setup", cleaned)
        self.assertIn("RSI14 is healthy at 54.2", cleaned)

    def test_get_plain_english_takeaways(self):
        sig = {
            "decision": "BUY",
            "symbol": "MUFG",
            "company_name": "Mitsubishi UFJ Financial",
            "bull_case": "Deep valuation discount with forward P/E 10.4x vs peer avg 22.9x; Wall Street mean target implies 12% upside",
            "key_risks": "Upcoming earnings announcement in 2 days; Fed funds at 3.88% and real 10Y yield remain restrictive; stop-loss placed at $21.20",
        }
        takeaways = get_plain_english_takeaways(sig)
        
        bull_bullets = takeaways["bull_bullets"]
        risk_bullets = takeaways["risk_bullets"]
        
        self.assertTrue(len(bull_bullets) >= 1)
        self.assertTrue(any("valuation discount" in b.lower() for b in bull_bullets))
        
        # Verify Fed policy was stripped from key risks
        for r in risk_bullets:
            self.assertNotIn("fed funds", r.lower())
            self.assertNotIn("real 10y", r.lower())


if __name__ == "__main__":
    unittest.main()
