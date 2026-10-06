import unittest
from unittest import mock
from fastapi.testclient import TestClient
from api import app, RUNNING_JOBS


class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_check(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["app"], "AI Trader API")

    def test_fetch_latest_signals(self):
        resp = self.client.get("/api/signals/latest?days=7")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("signals", data)
        self.assertIn("count", data)
        self.assertIsInstance(data["signals"], list)

    def test_fetch_latest_signals_filters(self):
        resp = self.client.get("/api/signals/latest?days=7&decision=HOLD")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        for s in data["signals"]:
            self.assertEqual(s["decision"], "HOLD")

    def test_fetch_summary_metrics(self):
        resp = self.client.get("/api/signals/summary?days=7")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("summary", data)
        self.assertIn("performance", data)
        self.assertIn("total_tracked", data["summary"])

    def test_fetch_signal_history(self):
        resp = self.client.get("/api/signals/history?limit=10")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("records", data)
        self.assertLessEqual(len(data["records"]), 10)

    def test_resolve_ticker_us(self):
        resp = self.client.get("/api/resolve?query=AAPL&prefer_exchange=US")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["resolved_symbol"], "AAPL")
        self.assertEqual(data["currency_code"], "USD")

    def test_resolve_ticker_isin(self):
        resp = self.client.get("/api/resolve?query=US67066G1040")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["resolved_symbol"], "NVD.DE")
        self.assertEqual(data["currency_code"], "EUR")

    def test_portfolio_reviews(self):
        resp = self.client.get("/api/portfolio/reviews?limit=5")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("reviews", data)

    @unittest.mock.patch("api._run_single_analysis_job")
    def test_analyze_trigger(self, mock_job):
        # Test queuing endpoint (mocking background task)
        resp = self.client.post("/api/analyze", json={"symbol": "AAPL", "model": "free", "is_eu": False})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "queued")
        self.assertIn("job_id", data)

        # Check job status endpoint
        job_resp = self.client.get(f"/api/jobs/{data['job_id']}")
        self.assertEqual(job_resp.status_code, 200)
        job_data = job_resp.json()
        self.assertEqual(job_data["target_symbol"], "AAPL")
        self.assertEqual(job_data["model"], "free")
        mock_job.assert_called_once()

    @unittest.mock.patch("api._run_single_analysis_job")
    def test_analyze_trigger_defaults_to_ling(self, mock_job):
        resp = self.client.post("/api/analyze", json={"symbol": "NVDA"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["model"], "ling")

        job_resp = self.client.get(f"/api/jobs/{data['job_id']}")
        self.assertEqual(job_resp.status_code, 200)
        self.assertEqual(job_resp.json()["model"], "ling")

    @unittest.mock.patch("api._run_single_analysis_job")
    def test_analyze_trigger_redirects_bunny_to_ling(self, mock_job):
        resp = self.client.post("/api/analyze", json={"symbol": "NVDA", "model": "bunny"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["model"], "ling")

        job_resp = self.client.get(f"/api/jobs/{data['job_id']}")
        self.assertEqual(job_resp.status_code, 200)
        self.assertEqual(job_resp.json()["model"], "ling")


if __name__ == "__main__":
    unittest.main()

