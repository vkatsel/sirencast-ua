"""
Integration tests for the SirenCast UA web server and API endpoints.
Verifies HTTP endpoints (/api/health, /api/forecast, /) and query parameters.
"""
import json
import threading
import unittest
import urllib.request
from socketserver import TCPServer

from src.web import SirenCastHTTPRequestHandler


class TestWebServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Bind to port 0 for automatic available ephemeral port allocation
        TCPServer.allow_reuse_address = True
        cls.server = TCPServer(("127.0.0.1", 0), SirenCastHTTPRequestHandler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_health_endpoint(self):
        """Verify /api/health returns HTTP 200 and valid JSON status."""
        url = f"http://127.0.0.1:{self.port}/api/health"
        with urllib.request.urlopen(url, timeout=15) as resp:
            self.assertEqual(resp.status, 200)
            self.assertEqual(resp.headers.get("Content-Type"), "application/json; charset=utf-8")
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data.get("status"), "ok")
            self.assertEqual(data.get("service"), "SirenCast-UA")

    def test_forecast_endpoint_default(self):
        """Verify /api/forecast returns complete multi-metric contract."""
        url = f"http://127.0.0.1:{self.port}/api/forecast"
        with urllib.request.urlopen(url, timeout=15) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("threat_index_pct", data)
            self.assertIn("raw_probability_pct", data)
            self.assertIn("risk_level", data)
            self.assertIn("metadata", data)
            self.assertIn("top_drivers", data)
            self.assertGreaterEqual(data["threat_index_pct"], 0.0)
            self.assertLessEqual(data["threat_index_pct"], 100.0)

    def test_forecast_odds_weight_parameter(self):
        """Verify ?w= query parameter dynamically modifies calibration odds."""
        url = f"http://127.0.0.1:{self.port}/api/forecast?w=10.0"
        with urllib.request.urlopen(url, timeout=15) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data.get("odds_weight"), 10.0)

    def test_static_html_served(self):
        """Verify root / serves HTML dashboard."""
        url = f"http://127.0.0.1:{self.port}/"
        with urllib.request.urlopen(url, timeout=15) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("text/html", resp.headers.get("Content-Type"))
            html_content = resp.read().decode("utf-8")
            self.assertIn("SirenCast UA", html_content)


if __name__ == "__main__":
    unittest.main()
