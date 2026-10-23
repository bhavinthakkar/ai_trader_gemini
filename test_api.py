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
    def test_analyze_trigger_defaults_to_free(self, mock_job):
        resp = self.client.post("/api/analyze", json={"symbol": "NVDA"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["model"], "free")

        job_resp = self.client.get(f"/api/jobs/{data['job_id']}")
        self.assertEqual(job_resp.status_code, 200)
        self.assertEqual(job_resp.json()["model"], "free")

    @unittest.mock.patch("api._run_single_analysis_job")
    def test_analyze_trigger_redirects_bunny_to_free(self, mock_job):
        resp = self.client.post("/api/analyze", json={"symbol": "NVDA", "model": "bunny"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["model"], "free")

        job_resp = self.client.get(f"/api/jobs/{data['job_id']}")
        self.assertEqual(job_resp.status_code, 200)
        self.assertEqual(job_resp.json()["model"], "free")

    def test_ml_context_available(self):
        resp = self.client.get("/api/ml/context/MRVL")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["available"])
        self.assertEqual(data["symbol"], "MRVL")
        self.assertIsNotNone(data["data"])
        self.assertGreater(data["data"]["score_0_100"], 80.0)
        self.assertEqual(data["data"]["signal"], "STRONG OUTPERFORM")

    def test_ml_context_unavailable(self):
        resp = self.client.get("/api/ml/context/CRUS")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertFalse(data["available"])
        self.assertEqual(data["symbol"], "CRUS")
        self.assertIsNone(data["data"])

    def test_ml_leaderboard(self):
        resp = self.client.get("/api/ml/leaderboard?limit=5")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertGreater(data["count"], 0)
        self.assertGreater(data["universe_size"], 0)
        self.assertLessEqual(len(data["top_picks"]), 5)
        self.assertLessEqual(len(data["bottom_picks"]), 5)

    def test_fetch_shadow_decisions(self):
        resp = self.client.get("/api/shadow/decisions?limit=10")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("count", data)
        self.assertIn("decisions", data)
        self.assertIsInstance(data["decisions"], list)

    def test_fetch_shadow_metrics(self):
        resp = self.client.get("/api/shadow/metrics?limit=10")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("status", data)
        self.assertIn(data["status"], ["SUCCESS", "NO_DATA"])

    def test_fetch_execution_orders(self):
        resp = self.client.get("/api/execution/orders?limit=10")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("count", data)
        self.assertIn("orders", data)
        self.assertIsInstance(data["orders"], list)

    def test_kill_switch_trip_and_reset(self):
        # 1. Check initial state
        resp = self.client.get("/api/execution/kill-switch")
        self.assertEqual(resp.status_code, 200)

        # 2. Trip kill switch
        trip_resp = self.client.post("/api/execution/kill-switch/trip", json={"reason": "MANUAL_OPERATOR_TRIP", "operator": "PytestClient"})
        self.assertEqual(trip_resp.status_code, 200)
        self.assertTrue(trip_resp.json()["is_tripped"])

        # 3. Verify tripped state
        status_resp = self.client.get("/api/execution/kill-switch")
        self.assertTrue(status_resp.json()["is_tripped"])

        # 4. Reset kill switch
        reset_resp = self.client.post("/api/execution/kill-switch/reset?operator=PytestClient")
        self.assertEqual(reset_resp.status_code, 200)
        self.assertFalse(reset_resp.json()["is_tripped"])


if __name__ == "__main__":
    unittest.main()


