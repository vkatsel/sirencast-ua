"""
Data loading and preprocessing module for SirenCast UA.
Handles loading raw alert archives, deduplication, datetime parsing,
and validation checks.
"""
import logging
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

from src.config import (
    RAW_OFFICIAL_DATA,
    PROCESSED_DATA_DIR,
    PROCESSED_ALERTS_PARQUET,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_and_clean_official_data(
    raw_path: Path = RAW_OFFICIAL_DATA,
    save_parquet: bool = True,
    output_path: Path = PROCESSED_ALERTS_PARQUET,
) -> Tuple[pd.DataFrame, dict]:
    """
    Loads official air alerts CSV, removes exact duplicates,
    validates datetimes, and optionally saves to Parquet.

    Returns:
        Tuple[pd.DataFrame, dict]: Cleaned dataframe and summary stats dictionary.
    """
    logger.info(f"Loading raw official data from: {raw_path}")
    raw_df = pd.read_csv(raw_path, dtype={"raion": str, "hromada": str})
    initial_rows = len(raw_df)
    logger.info(f"Initial raw rows: {initial_rows:,}")

    # Step 1: Drop exact duplicates
    clean_df = raw_df.drop_duplicates().copy()
    duplicates_removed = initial_rows - len(clean_df)
    logger.info(f"Removed duplicates: {duplicates_removed:,} rows ({duplicates_removed / initial_rows:.2%})")

    # Step 2: Parse datetime columns to UTC
    logger.info("Parsing UTC timestamps...")
    clean_df["started_at"] = pd.to_datetime(clean_df["started_at"], utc=True)
    clean_df["finished_at"] = pd.to_datetime(clean_df["finished_at"], utc=True)

    # Step 3: Anomaly checks
    null_starts = clean_df["started_at"].isna().sum()
    null_finishes = clean_df["finished_at"].isna().sum()
    if null_starts > 0 or null_finishes > 0:
        logger.warning(f"Dropping missing timestamps: {null_starts} starts, {null_finishes} finishes")
        clean_df = clean_df.dropna(subset=["started_at", "finished_at"])

    # Duration calculation
    clean_df["duration_sec"] = (clean_df["finished_at"] - clean_df["started_at"]).dt.total_seconds()
    negative_durations = (clean_df["duration_sec"] < 0).sum()
    if negative_durations > 0:
        logger.warning(f"Found {negative_durations} alerts with finished_at < started_at. Dropping them.")
        clean_df = clean_df[clean_df["duration_sec"] >= 0]

    # Fill NaNs in raion and hromada with empty string
    clean_df["raion"] = clean_df["raion"].fillna("")
    clean_df["hromada"] = clean_df["hromada"].fillna("")

    # Sort chronologically by started_at
    clean_df = clean_df.sort_values(by="started_at").reset_index(drop=True)

    # Summary metrics
    stats = {
        "initial_rows": initial_rows,
        "clean_rows": len(clean_df),
        "duplicates_removed": duplicates_removed,
        "negative_durations": int(negative_durations),
        "min_date": str(clean_df["started_at"].min()),
        "max_date": str(clean_df["started_at"].max()),
        "unique_oblasts": int(clean_df["oblast"].nunique()),
    }
    logger.info(f"Cleaned dataset ready: {len(clean_df):,} rows from {stats['min_date']} to {stats['max_date']}")

    # Step 4: Save to Parquet
    if save_parquet:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving cleaned dataset to Parquet: {output_path}")
        clean_df.to_parquet(output_path, index=False, engine="pyarrow")

    return clean_df, stats


def get_clean_alerts(force_reload: bool = False) -> pd.DataFrame:
    """
    Returns the cleaned DataFrame. If parquet file exists and force_reload=False,
    loads from Parquet (very fast). Otherwise, runs cleaning pipeline.
    """
    if PROCESSED_ALERTS_PARQUET.exists() and not force_reload:
        logger.info(f"Loading cached clean alerts from: {PROCESSED_ALERTS_PARQUET}")
        return pd.read_parquet(PROCESSED_ALERTS_PARQUET, engine="pyarrow")
    df, _ = load_and_clean_official_data(save_parquet=True)
    return df


if __name__ == "__main__":
    df, stats = load_and_clean_official_data()
    print("\n--- Summary Statistics ---")
    for k, v in stats.items():
        print(f"{k}: {v}")
