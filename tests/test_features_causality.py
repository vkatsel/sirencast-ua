"""
Automated unit tests for 30m feature store and causality verification.
Ensures zero data leakage, correct lag alignment, and valid spatial bounds.
"""
import unittest
import numpy as np
import pandas as pd
from src.features_ts import get_feature_dataset
from src.config import INNER_RING_RAIONS, ADJACENT_OBLASTS


class TestFeaturesCausality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = get_feature_dataset()

    def test_dataset_dimensions_and_nans(self):
        """Assert dataset has expected shape and no unexpected NaNs."""
        self.assertGreater(len(self.df), 60000, f"Expected >60,000 rows, got {len(self.df)}")
        self.assertGreaterEqual(self.df.shape[1], 70, f"Expected >=70 columns, got {self.df.shape[1]}")
        null_counts = self.df.isna().sum().sum()
        self.assertEqual(null_counts, 0, f"Found {null_counts} NaNs in feature dataset!")

    def test_target_distribution(self):
        """Assert realistic class imbalance for 30m Kyiv onset target (between 1.5% and 5%)."""
        pos_rate = self.df["target_kyiv_start_30m"].mean()
        self.assertGreater(pos_rate, 0.015, f"Positive rate unexpectedly low: {pos_rate:.2%}")
        self.assertLess(pos_rate, 0.05, f"Positive rate unexpectedly high: {pos_rate:.2%}")

    def test_strict_lag_alignment(self):
        """
        Verify that lag features (e.g., lag30m, lag60m) are mathematically identical
        to the unlagged series shifted by 1 and 2 steps.
        Guarantees no future lookahead.
        """
        # Inner ring count lag check
        unlagged = self.df["inner_ring_active_count"].values
        lag30 = self.df["inner_ring_active_count_lag30m"].values
        lag60 = self.df["inner_ring_active_count_lag60m"].values

        np.testing.assert_array_equal(
            lag30[1:], unlagged[:-1],
            err_msg="lag30m does not match unlagged shifted by 1!"
        )
        np.testing.assert_array_equal(
            lag60[2:], unlagged[:-2],
            err_msg="lag60m does not match unlagged shifted by 2!"
        )

        # Adjacent ring count lag check
        adj_unlagged = self.df["adjacent_active_count"].values
        adj_lag30 = self.df["adjacent_active_count_lag30m"].values
        np.testing.assert_array_equal(
            adj_lag30[1:], adj_unlagged[:-1],
            err_msg="adjacent lag30m does not match unlagged shifted by 1!"
        )

    def test_spatial_radar_bounds(self):
        """Assert all multi-tier spatial features stay within exact physical bounds."""
        # Inner ring: 0 to 7 raions
        self.assertTrue((self.df["inner_ring_active_count"] >= 0).all())
        self.assertTrue((self.df["inner_ring_active_count"] <= 7).all())

        # Adjacent ring: 0 to 5 oblasts
        self.assertTrue((self.df["adjacent_active_count"] >= 0).all())
        self.assertTrue((self.df["adjacent_active_count"] <= 5).all())

        # Eastern flank: 0 to 4 oblasts
        self.assertTrue((self.df["east_flank_active_count"] >= 0).all())
        self.assertTrue((self.df["east_flank_active_count"] <= 4).all())

        # Southern flank: 0 to 3 oblasts
        self.assertTrue((self.df["south_flank_active_count"] >= 0).all())
        self.assertTrue((self.df["south_flank_active_count"] <= 3).all())

        # National macro: 0 to 25 oblasts
        self.assertTrue((self.df["national_active_count"] >= 0).all())
        self.assertTrue((self.df["national_active_count"] <= 25).all())

    def test_cyclical_calendar_encoding(self):
        """Assert sin and cos coordinates stay on unit circle."""
        hour_radius = np.sqrt(self.df["hour_sin"] ** 2 + self.df["hour_cos"] ** 2)
        np.testing.assert_allclose(hour_radius, 1.0, atol=1e-5, err_msg="hour sin/cos not on unit circle!")

        dow_radius = np.sqrt(self.df["dow_sin"] ** 2 + self.df["dow_cos"] ** 2)
        np.testing.assert_allclose(dow_radius, 1.0, atol=1e-5, err_msg="dow sin/cos not on unit circle!")


if __name__ == "__main__":
    unittest.main()
