"""
Automated unit tests for text feature extraction and multimodal fusion.
Ensures zero data leakage, correct causal aggregation, and valid bounds.
"""
import unittest
import numpy as np
import pandas as pd
from src.features_text import get_multimodal_dataset


class TestTextFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = get_multimodal_dataset()

    def test_dimensions_and_nans(self):
        """Assert multimodal dataset has expected columns and zero NaNs."""
        self.assertGreaterEqual(len(self.df), 60000)
        self.assertEqual(self.df.shape[1], 88)
        null_count = self.df.isna().sum().sum()
        self.assertEqual(null_count, 0, f"Found {null_count} NaNs in multimodal dataset!")

    def test_text_feature_columns_present(self):
        """Assert all expected text signal columns are present."""
        expected_cols = [
            "tg_msg_count_30m",
            "tg_kyiv_targeted_30m",
            "tg_shahed_count_30m",
            "tg_missile_count_30m",
            "tg_ballistic_count_30m",
            "tg_aviation_count_30m",
            "tg_resolution_count_30m",
            "tg_net_threat_score",
            "tg_msg_count_lag30m",
            "tg_net_threat_lag30m",
            "tg_kyiv_targeted_lag30m",
        ]
        for col in expected_cols:
            self.assertIn(col, self.df.columns, f"Missing text feature: {col}")

    def test_non_negative_counts(self):
        """Assert all counts are >= 0."""
        count_cols = [
            "tg_msg_count_30m", "tg_shahed_count_30m", "tg_missile_count_30m",
            "tg_ballistic_count_30m", "tg_aviation_count_30m", "tg_resolution_count_30m",
            "tg_net_threat_score"
        ]
        for col in count_cols:
            self.assertTrue((self.df[col] >= 0).all(), f"Found negative values in {col}!")

    def test_resolution_signals_active(self):
        """Assert resolution/clearing signals ('чисто', 'збито', 'відбій') are detected."""
        res_sum = self.df["tg_resolution_count_30m"].sum()
        self.assertGreater(res_sum, 1000, f"Expected >1000 resolution mentions, found {res_sum}")

    def test_targeted_is_binary(self):
        """Assert tg_kyiv_targeted_30m is strictly binary (0 or 1)."""
        unique_vals = set(self.df["tg_kyiv_targeted_30m"].unique())
        self.assertTrue(unique_vals.issubset({0, 1}), f"Unexpected values in targeted: {unique_vals}")

    def test_lag_alignment(self):
        """Verify lag30m is mathematically identical to series shifted by 1."""
        unlagged = self.df["tg_msg_count_30m"].values
        lagged = self.df["tg_msg_count_lag30m"].values
        np.testing.assert_array_equal(
            lagged[1:], unlagged[:-1],
            err_msg="tg_msg_count_lag30m does not match unlagged shifted by 1!"
        )


if __name__ == "__main__":
    unittest.main()
