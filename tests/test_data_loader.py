"""
Verification tests for the data loading and cleaning pipeline.
Ensures zero duplicates, valid timestamps, non-negative durations,
and complete spatial coverage.
"""
import unittest
import pandas as pd
from src.config import (
    PROCESSED_ALERTS_PARQUET,
    TARGET_REGION,
    INNER_RING_RAIONS,
    ALL_OBLASTS,
)
from src.data_loader import get_clean_alerts


class TestDataLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = get_clean_alerts()

    def test_zero_duplicates(self):
        """Assert that no duplicate alert records remain in cleaned data."""
        initial_count = len(self.df)
        dedup_count = len(self.df.drop_duplicates(subset=["oblast", "raion", "hromada", "level", "started_at", "finished_at"]))
        self.assertEqual(initial_count, dedup_count, f"Found {initial_count - dedup_count} remaining duplicates!")

    def test_valid_utc_timestamps(self):
        """Assert that timestamps are parsed and in UTC."""
        self.assertIsNotNone(self.df["started_at"].dt.tz, "started_at must be timezone-aware (UTC)")
        self.assertIsNotNone(self.df["finished_at"].dt.tz, "finished_at must be timezone-aware (UTC)")
        self.assertEqual(self.df["started_at"].isna().sum(), 0, "Null values found in started_at")
        self.assertEqual(self.df["finished_at"].isna().sum(), 0, "Null values found in finished_at")

    def test_non_negative_durations(self):
        """Assert finished_at >= started_at for every single alert."""
        invalid_durations = (self.df["finished_at"] < self.df["started_at"]).sum()
        self.assertEqual(invalid_durations, 0, f"Found {invalid_durations} records with finished_at < started_at")

    def test_target_coverage(self):
        """Assert target region (Kyiv City) exists and has alert records."""
        kyiv_records = self.df[self.df["oblast"] == TARGET_REGION]
        self.assertGreater(len(kyiv_records), 1000, f"Expected >1000 Kyiv records, found {len(kyiv_records)}")

    def test_inner_ring_coverage(self):
        """Assert all 7 raions of Kyivska oblast are represented."""
        kyiv_obl = self.df[self.df["oblast"] == "Kyivska oblast"]
        available_raions = set(kyiv_obl["raion"].unique())
        for raion in INNER_RING_RAIONS:
            self.assertIn(raion, available_raions, f"Missing inner ring raion: {raion}")

    def test_all_oblasts_coverage(self):
        """Assert all 25 Ukrainian oblasts/regions are represented."""
        unique_oblasts = set(self.df["oblast"].unique())
        for oblast in ALL_OBLASTS:
            self.assertIn(oblast, unique_oblasts, f"Missing oblast: {oblast}")


if __name__ == "__main__":
    unittest.main()
