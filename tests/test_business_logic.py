"""
Domain and business logic validation tests for SirenCast UA.
Verifies defense operational rules, threat escalation monotonicity,
directional targeting sensitivity, and resolution suppression.
"""
import unittest
from datetime import datetime, timezone

from src.live_forecast import predict_live_risk, calibrate_odds


class TestBusinessLogic(unittest.TestCase):
    def setUp(self):
        self.now_utc = datetime.now(timezone.utc)

    def test_threat_escalation_monotonicity(self):
        """
        Verify that threat index increases monotonically across standard military escalation levels:
        Calm Sky < Border Shahed approach < Direct Kyiv vector ballistic < Active siren in Kyiv.
        """
        # Level 1: Calm Sky
        alerts_calm = {
            "regions": {"Kyiv City": False, "Chernihivska oblast": False, "Sumska oblast": False},
            "kyiv_raions": {},
            "timestamp_utc": self.now_utc,
            "national_active_count": 0,
        }
        res_calm = predict_live_risk(simulated_alerts=alerts_calm, simulated_telegram_texts=["Відбій. Небо чисте."])

        # Level 2: Approaching Shaheds in transit corridors (Chernihiv/Sumy)
        alerts_approach = {
            "regions": {"Kyiv City": False, "Chernihivska oblast": True, "Sumska oblast": True},
            "kyiv_raions": {},
            "timestamp_utc": self.now_utc,
            "national_active_count": 5,
        }
        res_approach = predict_live_risk(
            simulated_alerts=alerts_approach,
            simulated_telegram_texts=["Група шахедів курсом на Чернігівщину!"],
        )

        # Level 3: Direct Kyiv corridor vector + Inner ring active raions + Ballistic
        alerts_vector = {
            "regions": {"Kyiv City": False, "Chernihivska oblast": True, "Kyivska oblast": True},
            "kyiv_raions": {"Brovarskyi raion": True, "Boryspilskyi raion": True},
            "timestamp_utc": self.now_utc,
            "national_active_count": 10,
        }
        res_vector = predict_live_risk(
            simulated_alerts=alerts_vector,
            simulated_telegram_texts=["Швидкісна ціль повз Бровари на Київ! Балістика!"],
        )

        # Level 4: Siren already howling in Kyiv City
        alerts_active = {
            "regions": {"Kyiv City": True, "Kyivska oblast": True},
            "kyiv_raions": {"Brovarskyi raion": True},
            "timestamp_utc": self.now_utc,
            "national_active_count": 14,
        }
        res_active = predict_live_risk(
            simulated_alerts=alerts_active,
            simulated_telegram_texts=["Повітряна тривога в Києві триває"],
        )

        # Strict Monotonicity: Level 1 < Level 2 < Level 3 < Level 4
        self.assertLess(res_calm["threat_index_pct"], res_approach["threat_index_pct"])
        self.assertLess(res_approach["threat_index_pct"], res_vector["threat_index_pct"])
        self.assertLess(res_vector["threat_index_pct"], res_active["threat_index_pct"])

        # Risk categories match operational expectation
        self.assertEqual(res_calm["risk_level"], "LOW RISK")
        self.assertIn(res_vector["risk_level"], ["ELEVATED RISK", "HIGH RISK", "CRITICAL IMMINENT RISK"])
        self.assertEqual(res_active["risk_level"], "CRITICAL IMMINENT RISK")

    def test_directional_vector_sensitivity(self):
        """
        Verify that explicit vector towards Kyiv/Brovary triggers tactical threat detection
        in metadata, net threat score, and AI decision drivers compared to neutral/non-targeted messages.
        """
        alerts = {
            "regions": {"Kyiv City": False, "Chernihivska oblast": True},
            "kyiv_raions": {},
            "timestamp_utc": self.now_utc,
            "national_active_count": 4,
        }

        res_targeted = predict_live_risk(
            simulated_alerts=alerts,
            simulated_telegram_texts=["Увага! БпЛА Shahed курсом на Київ через Бровари!"],
        )
        res_neutral = predict_live_risk(
            simulated_alerts=alerts,
            simulated_telegram_texts=["БпЛА на півдні Одещини у бік моря."],
        )

        # Tactical NLP threat signals correctly attributed
        self.assertEqual(res_targeted["metadata"]["telegram_threats"]["kyiv_targeted"], 1)
        self.assertEqual(res_neutral["metadata"]["telegram_threats"]["kyiv_targeted"], 0)
        self.assertGreater(
            res_targeted["metadata"]["telegram_threats"]["net_score"],
            res_neutral["metadata"]["telegram_threats"]["net_score"],
        )

        # AI decision drivers correctly detect the vector to the capital
        self.assertTrue(
            any("towards Kyiv" in d for d in res_targeted["top_drivers"]),
            "Kyiv vector driver was not triggered!",
        )
        self.assertFalse(
            any("towards Kyiv" in d for d in res_neutral["top_drivers"]),
            "Kyiv vector driver should not be present in neutral airspace!",
        )

    def test_resolution_suppression(self):
        """
        Verify that resolution and clearing messages ('чисто', 'відбій') suppress threat score
        and do not falsely trigger high risk alerts.
        """
        alerts = {
            "regions": {"Kyiv City": False},
            "kyiv_raions": {},
            "timestamp_utc": self.now_utc,
            "national_active_count": 1,
        }

        res_clearing = predict_live_risk(
            simulated_alerts=alerts,
            simulated_telegram_texts=["Київщина — чисто. По шахедах відбій, усі цілі знищено."],
        )

        # Net threat score must not elevate
        self.assertEqual(res_clearing["metadata"]["telegram_threats"]["net_score"], 0)
        self.assertEqual(res_clearing["risk_level"], "LOW RISK")

    def test_odds_multiplier_scaling(self):
        """
        Verify that changing Bayesian odds weight dynamically scales operational score
        without altering the underlying raw probability.
        """
        alerts = {
            "regions": {"Kyiv City": False, "Chernihivska oblast": True},
            "kyiv_raions": {"Brovarskyi raion": True},
            "timestamp_utc": self.now_utc,
            "national_active_count": 5,
        }

        res_w1 = predict_live_risk(simulated_alerts=alerts, odds_weight=1.0)
        res_w5 = predict_live_risk(simulated_alerts=alerts, odds_weight=5.0)
        res_w10 = predict_live_risk(simulated_alerts=alerts, odds_weight=10.0)

        # Raw statistical probabilities are identical
        self.assertAlmostEqual(res_w1["raw_probability_pct"], res_w5["raw_probability_pct"], places=2)
        self.assertAlmostEqual(res_w5["raw_probability_pct"], res_w10["raw_probability_pct"], places=2)

        # Calibrated threat scales monotonically with weight
        self.assertLess(res_w1["threat_index_pct"], res_w5["threat_index_pct"])
        self.assertLess(res_w5["threat_index_pct"], res_w10["threat_index_pct"])


if __name__ == "__main__":
    unittest.main()
