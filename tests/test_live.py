import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone
import pandas as pd

from src.live_data import (
    CACHE_TTL_SECONDS,
    OBLAST_MAPPING,
    RAION_MAPPING,
    fetch_live_alerts_raw,
    parse_live_siren_states,
)
from src.live_forecast import (
    get_trained_live_model,
    build_live_feature_vector,
    predict_live_risk,
)


class TestLiveModule(unittest.TestCase):
    def test_cache_ttl_protection(self):
        """Verify that cache TTL strictly adheres to the Ubilling 2 RPS rate limit (>= 5s)."""
        self.assertGreaterEqual(CACHE_TTL_SECONDS, 5.0)

    def test_parse_live_siren_states_with_districts(self):
        """Verify parsing raw states into regions and Kyivska oblast raions."""
        mock_raw = {
            "source": "ubilling",
            "raw": {
                "м. Київ": {"enabled": False, "changed": "1970-01-01 03:00:00"},
                "Київська область": {
                    "enabled": True,
                    "changed": "2026-10-11 01:00:00",
                    "districts": {
                        "Броварський": {"enabled": True},
                        "Бориспільський": {"enabled": False},
                    },
                },
                "Чернігівська область": {"enabled": True, "changed": "2026-10-11 00:50:00"},
            },
        }

        with patch("src.live_data.fetch_live_alerts_raw", return_value=mock_raw):
            data = parse_live_siren_states()
            self.assertFalse(data["regions"]["Kyiv City"])
            self.assertTrue(data["regions"]["Kyivska oblast"])
            self.assertTrue(data["regions"]["Chernihivska oblast"])
            self.assertTrue(data["kyiv_raions"]["Brovarskyi raion"])
            self.assertFalse(data["kyiv_raions"]["Boryspilskyi raion"])
            self.assertGreaterEqual(data["national_active_count"], 2)

    def test_build_live_feature_vector(self):
        """Verify that build_live_feature_vector produces a valid 1-row DataFrame."""
        mock_alerts = {
            "regions": {"Kyiv City": False, "Chernihivska oblast": True},
            "kyiv_raions": {"Brovarskyi raion": True},
            "timestamp_utc": datetime.now(timezone.utc),
            "national_active_count": 5,
        }
        mock_texts = [
            "Увага! БпЛА Shahed, курс на Київ!",
            "Працює ППО по шахедах на півночі Київщини",
        ]

        df_vec, meta = build_live_feature_vector(
            simulated_alerts=mock_alerts, simulated_telegram_texts=mock_texts
        )

        self.assertEqual(len(df_vec), 1)
        self.assertEqual(meta["is_kyiv_active"], 0)
        self.assertEqual(meta["telegram_threats"]["shahed"], 2)
        self.assertEqual(meta["telegram_threats"]["kyiv_targeted"], 1)

    def test_predict_live_risk_contract(self):
        """Verify predict_live_risk returns valid percentage and categorical risk levels."""
        mock_alerts = {
            "regions": {"Kyiv City": False, "Chernihivska oblast": False},
            "kyiv_raions": {},
            "timestamp_utc": datetime.now(timezone.utc),
            "national_active_count": 0,
        }
        mock_texts = ["Відбій тривоги по областях. Чисто."]

        res = predict_live_risk(
            simulated_alerts=mock_alerts, simulated_telegram_texts=mock_texts
        )

        self.assertIn("probability_pct", res)
        self.assertIn("risk_level", res)
        self.assertIn("risk_color", res)
        self.assertIn("top_drivers", res)

        pct = res["probability_pct"]
        self.assertGreaterEqual(pct, 0.0)
        self.assertLessEqual(pct, 100.0)
        self.assertIn(res["risk_level"], ["LOW RISK", "ELEVATED RISK", "HIGH RISK", "CRITICAL IMMINENT RISK"])


if __name__ == "__main__":
    unittest.main()
