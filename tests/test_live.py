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
        self.assertEqual(df_vec.shape[1], 85)
        self.assertEqual(df_vec.isna().sum().sum(), 0)
        self.assertEqual(meta["is_kyiv_active"], 0)
        self.assertEqual(meta["telegram_threats"]["shahed"], 2)
        self.assertEqual(meta["telegram_threats"]["kyiv_targeted"], 1)

        # Assert correct spatial features mapped from inputs
        self.assertEqual(df_vec["adj_chernihivska_active"].values[0], 1)
        self.assertEqual(df_vec["raion_brovarskyi_active"].values[0], 1)
        self.assertEqual(df_vec["raion_boryspilskyi_active"].values[0], 0)
        self.assertEqual(df_vec["east_flank_active_count"].values[0], 0)
        self.assertEqual(df_vec["tg_kyiv_targeted_30m"].values[0], 1)

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
        self.assertIn("top_drivers_ua", res)
        self.assertEqual(len(res["top_drivers"]), len(res["top_drivers_ua"]))

        # Spatial telemetry dictionary integrity
        self.assertIn("raions", res["metadata"])
        self.assertIn("adjacent_regions", res["metadata"])
        self.assertEqual(len(res["metadata"]["raions"]), 7)
        self.assertEqual(len(res["metadata"]["adjacent_regions"]), 5)

        pct = res["probability_pct"]
        self.assertGreaterEqual(pct, 0.0)
        self.assertLessEqual(pct, 100.0)
        self.assertIn(res["risk_level"], ["LOW RISK", "ELEVATED RISK", "HIGH RISK", "CRITICAL IMMINENT RISK"])

        # Multi-dimensional metric contract checks
        self.assertIn("threat_index_pct", res)
        self.assertIn("raw_probability_pct", res)
        self.assertIn("ecdf_percentile", res)
        self.assertIn("hazard_multiplier", res)
        self.assertGreaterEqual(res["ecdf_percentile"], 0.0)
        self.assertLessEqual(res["ecdf_percentile"], 100.0)
        self.assertGreaterEqual(res["hazard_multiplier"], 1.0)

    def test_calibrate_odds(self):
        """Verify Bayesian odds calibration behavior and monotonicity."""
        from src.live_forecast import calibrate_odds
        # Boundary cases
        self.assertAlmostEqual(calibrate_odds(0.0), 0.0, delta=1e-3)
        self.assertAlmostEqual(calibrate_odds(1.0), 1.0, delta=1e-3)

        # Monotonicity
        p1 = calibrate_odds(0.05, weight=5.0)
        p2 = calibrate_odds(0.20, weight=5.0)
        p3 = calibrate_odds(0.50, weight=5.0)
        self.assertLess(p1, p2)
        self.assertLess(p2, p3)

        # Higher weight elevates posterior
        p_w1 = calibrate_odds(0.10, weight=1.0)
        p_w5 = calibrate_odds(0.10, weight=5.0)
        self.assertAlmostEqual(p_w1, 0.10, places=4)
        self.assertGreater(p_w5, p_w1)

    def test_live_features_exact_match_model(self):
        """Verify that live feature vector contains all features expected by the trained model with zero drift."""
        model, feat_cols, _ = get_trained_live_model()
        df_vec, _ = build_live_feature_vector()
        self.assertEqual(set(df_vec.columns), set(feat_cols), "Feature drift detected between live vector and trained model!")
        self.assertEqual(len(df_vec.columns), len(feat_cols))


    def test_suspilne_kyiv_regex_patterns(self):
        """Verify that Suspilne alert regexes capture alerts and reject news stories."""
        from src.live_data import RE_SUSPILNE_KYIV_START, RE_SUSPILNE_KYIV_CLEAR
        start_yellow = "🟡 м. Київ — повітряна тривога, жовтий рівень: Дронова загроза (жовтий рівень)"
        start_red = "🔴 м. Київ — повітряна тривога, червоний рівень: Ракетна загроза (червоний рівень)"
        clear_msg = "🟢 м. Київ — відбій повітряної тривоги"
        news_lake = "Межі заказника Червонохуторські озера, що у Дарницькому районі Києва..."
        news_pig = "Свиня народила поросят у приватному секторі..."

        self.assertTrue(bool(RE_SUSPILNE_KYIV_START.search(start_yellow)))
        self.assertTrue(bool(RE_SUSPILNE_KYIV_START.search(start_red)))
        self.assertFalse(bool(RE_SUSPILNE_KYIV_CLEAR.search(start_yellow)))

        self.assertTrue(bool(RE_SUSPILNE_KYIV_CLEAR.search(clear_msg)))
        self.assertFalse(bool(RE_SUSPILNE_KYIV_START.search(clear_msg)))

        self.assertFalse(bool(RE_SUSPILNE_KYIV_START.search(news_lake)))
        self.assertFalse(bool(RE_SUSPILNE_KYIV_CLEAR.search(news_lake)))
        self.assertFalse(bool(RE_SUSPILNE_KYIV_START.search(news_pig)))
        self.assertFalse(bool(RE_SUSPILNE_KYIV_CLEAR.search(news_pig)))

    def test_kyiv_alert_timeline_yellow_red_transition(self):
        """
        Verify that yellow -> red -> yellow transitions do not reset alert start time,
        and continuous duration is measured from the initial alert onset.
        """
        from src.live_data import resolve_kyiv_alert_timeline
        events = [
            {"type": "START", "datetime_iso": "2026-10-12T14:00:00+00:00", "text": "🟡 м. Київ — повітряна тривога, жовтий рівень"},
            {"type": "START", "datetime_iso": "2026-10-12T14:15:00+00:00", "text": "🔴 м. Київ — повітряна тривога, червоний рівень"},
            {"type": "START", "datetime_iso": "2026-10-12T14:30:00+00:00", "text": "🟡 м. Київ — повітряна тривога, жовтий рівень"},
        ]
        now_eval = datetime.fromisoformat("2026-10-12T14:40:00+00:00")
        state = resolve_kyiv_alert_timeline(events, now_utc=now_eval)

        self.assertTrue(state["is_active"])
        self.assertEqual(state["time_since_last_alert_min"], 0.0)
        # Should be 40 minutes active from 14:00, NOT reset to 10m or 25m!
        self.assertEqual(state["minutes_active"], 40.0)
        self.assertIn("40 хв", state["detail_ua"])
        self.assertIn("40m", state["detail_en"])

    def test_kyiv_alert_timeline_all_clear_duration(self):
        """Verify that after an all-clear, time_since_last_alert_min accurately measures quiet duration."""
        from src.live_data import resolve_kyiv_alert_timeline
        events = [
            {"type": "START", "datetime_iso": "2026-10-12T14:00:00+00:00", "text": "🟡 м. Київ — повітряна тривога"},
            {"type": "CLEAR", "datetime_iso": "2026-10-12T14:45:00+00:00", "text": "🟢 м. Київ — відбій повітряної тривоги"},
        ]
        now_eval = datetime.fromisoformat("2026-10-12T16:00:00+00:00")
        state = resolve_kyiv_alert_timeline(events, now_utc=now_eval)

        self.assertFalse(state["is_active"])
        self.assertEqual(state["minutes_active"], 0.0)
        # 75 minutes from 14:45 to 16:00
        self.assertEqual(state["time_since_last_alert_min"], 75.0)
        self.assertIn("1 год 15 хв", state["detail_ua"])
        self.assertIn("1h 15m", state["detail_en"])

    def test_suspilne_news_filtered_from_telegram_stream(self):
        """Verify that general news items from Suspilne are discarded from live stream."""
        from src.live_data import fetch_live_telegram_stream
        mock_suspilne = [
            {"channel": "suspilne_kyiv", "text": "🟢 м. Київ — відбій повітряної тривоги", "datetime_iso": "2026-10-12T14:45:00+00:00", "is_active": True},
            {"channel": "suspilne_kyiv", "text": "У приватному секторі свиня народила поросят, фермер вражений", "datetime_iso": "2026-10-12T14:50:00+00:00", "is_active": True},
            {"channel": "suspilne_kyiv", "text": "Чому у столиці перебої з водою: роз'яснення", "datetime_iso": "2026-10-12T14:55:00+00:00", "is_active": True},
        ]

        with patch("src.live_data.fetch_live_telegram_channel", side_effect=lambda ch, **kwargs: mock_suspilne if ch == "suspilne_kyiv" else []):
            with patch.dict("src.live_data._telegram_cache", {"timestamp": 0.0, "data": None}):
                stream = fetch_live_telegram_stream()
                suspilne_texts = [m["text"] for m in stream if m["channel"] == "suspilne_kyiv"]
                self.assertEqual(len(suspilne_texts), 1)
                self.assertIn("відбій", suspilne_texts[0])
                self.assertNotIn("свиня", " ".join(suspilne_texts))


if __name__ == "__main__":
    unittest.main()
