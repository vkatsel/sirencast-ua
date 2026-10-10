"""
Feature engineering module for 30-minute time series grid in SirenCast UA.
Constructs causal spatial features (Inner Ring raions, Adjacent Ring oblasts,
Deep Flanks, National Macro-Wave) and the Kyiv City prediction target.
Strictly causal: features at time t use only data prior to t.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import (
    TARGET_REGION,
    INNER_RING_RAIONS,
    ADJACENT_OBLASTS,
    EASTERN_FLANK_OBLASTS,
    SOUTHERN_FLANK_OBLASTS,
    ALL_OBLASTS,
    TIME_FREQ,
    PROCESSED_FEATURES_PARQUET,
)
from src.data_loader import get_clean_alerts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def merge_intervals(starts: np.ndarray, ends: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sorts and merges overlapping/contiguous time intervals [start, end].
    Assumes starts and ends are integer timestamps.
    """
    if len(starts) == 0:
        return np.array([], dtype="int64"), np.array([], dtype="int64")

    order = np.argsort(starts)
    s_sorted = starts[order]
    e_sorted = ends[order]

    merged_s = [s_sorted[0]]
    merged_e = [e_sorted[0]]

    for s, e in zip(s_sorted[1:], e_sorted[1:]):
        if s <= merged_e[-1]:
            if e > merged_e[-1]:
                merged_e[-1] = e
        else:
            merged_s.append(s)
            merged_e.append(e)

    return np.array(merged_s, dtype="int64"), np.array(merged_e, dtype="int64")


def compute_active_mask(grid_ts: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """
    Returns boolean array indicating if an alert is active at each grid timestamp t
    (i.e., starts[i] <= t < ends[i]).
    """
    if len(starts) == 0:
        return np.zeros(len(grid_ts), dtype=bool)

    idx = np.searchsorted(starts, grid_ts, side="right") - 1
    valid = idx >= 0
    active = np.zeros(len(grid_ts), dtype=bool)
    active[valid] = grid_ts[valid] < ends[idx[valid]]
    return active


def compute_minutes_active(
    grid_ts: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    units_per_min: float,
) -> np.ndarray:
    """
    Returns float array of minutes the active alert has been running at time t.
    If not active, returns 0.0.
    """
    if len(starts) == 0:
        return np.zeros(len(grid_ts), dtype=np.float32)

    idx = np.searchsorted(starts, grid_ts, side="right") - 1
    valid = idx >= 0
    mins = np.zeros(len(grid_ts), dtype=np.float32)
    is_active = np.zeros(len(grid_ts), dtype=bool)
    is_active[valid] = grid_ts[valid] < ends[idx[valid]]

    active_indices = np.where(is_active)[0]
    matched_starts = starts[idx[active_indices]]
    elapsed_units = grid_ts[active_indices] - matched_starts
    mins[active_indices] = (elapsed_units / units_per_min).astype(np.float32)
    return mins


def compute_time_since_last_ended_min(
    grid_ts: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    units_per_min: float,
    max_cap_min: float = 10080.0,
) -> np.ndarray:
    """
    Returns minutes since the most recent previous alert finished.
    If an alert is active at time t, returns 0.0.
    If no prior alert has occurred yet, returns max_cap_min (default 7 days).
    """
    if len(ends) == 0:
        return np.full(len(grid_ts), max_cap_min, dtype=np.float32)

    active = compute_active_mask(grid_ts, starts, ends)
    idx_end = np.searchsorted(ends, grid_ts, side="right") - 1

    time_since = np.full(len(grid_ts), max_cap_min, dtype=np.float32)
    valid = (idx_end >= 0) & (~active)

    matched_ends = ends[idx_end[valid]]
    diff_min = (grid_ts[valid] - matched_ends) / units_per_min
    time_since[valid] = np.clip(diff_min, 0.0, max_cap_min).astype(np.float32)
    time_since[active] = 0.0
    return time_since


def compute_rolling_metrics(
    grid_ts: np.ndarray,
    raw_starts: np.ndarray,
    window_hours: float,
    units_per_min: float,
) -> np.ndarray:
    """
    Returns the count of alerts that started strictly within [t - window_hours, t].
    """
    if len(raw_starts) == 0:
        return np.zeros(len(grid_ts), dtype=np.int32)

    window_units = int(window_hours * 60.0 * units_per_min)
    left_ts = grid_ts - window_units
    idx_right = np.searchsorted(raw_starts, grid_ts, side="right")
    idx_left = np.searchsorted(raw_starts, left_ts, side="right")
    return (idx_right - idx_left).astype(np.int32)


def build_30m_feature_store(
    df: Optional[pd.DataFrame] = None,
    start_date: str = "2023-01-01 00:00:00+00:00",
    end_date: str = "2026-09-07 00:00:00+00:00",
    freq: str = TIME_FREQ,
    save_parquet: bool = True,
    output_path: Path = PROCESSED_FEATURES_PARQUET,
) -> pd.DataFrame:
    """
    Builds the complete tabular time series dataset on a regular 30m grid.

    Columns generated:
    - timestamp: DatetimeIndex (UTC)
    - target_kyiv_start_30m: 1 if alert starts in [t, t+30m), 0 otherwise
    - target_kyiv_active_30m: 1 if alert active in [t, t+30m], 0 otherwise
    - kyiv_is_active: status at t
    - kyiv_time_since_last_alert_min: silence duration
    - kyiv_alerts_last_6h, 24h: recent alert counts
    - inner_ring: statuses for each of 7 raions + active_count + lag30m + lag60m
    - adjacent_ring: statuses for 5 adjacent oblasts + active_count + lag30m + lag60m
    - flanks: eastern & southern sector active counts + lags
    - national: total active oblasts across Ukraine + wave speed
    - cyclical time: hour_sin, hour_cos, dow_sin, dow_cos, is_weekend
    """
    if df is None:
        df = get_clean_alerts()

    logger.info(f"Building 30m grid from {start_date} to {end_date} (freq={freq})...")
    grid = pd.date_range(start=start_date, end=end_date, freq=freq, tz="UTC")
    grid_ts = grid.astype("int64").values
    n_rows = len(grid)
    logger.info(f"Generated {n_rows:,} time steps.")

    feat_df = pd.DataFrame(index=grid)
    feat_df.index.name = "timestamp"

    # Time conversion factors (invariant to microsecond vs nanosecond internal unit)
    step_val = int(grid_ts[1] - grid_ts[0])
    step_minutes = pd.Timedelta(freq).total_seconds() / 60.0
    units_per_min = float(step_val) / step_minutes

    # -------------------------------------------------------------
    # 1. TARGET & LOCAL KYIV FEATURES
    # -------------------------------------------------------------
    logger.info("Computing Kyiv City target and local history features...")
    kyiv_df = df[df["oblast"] == TARGET_REGION]
    kyiv_raw_starts = np.sort(kyiv_df["started_at"].astype("int64").values)
    kyiv_raw_ends = kyiv_df["finished_at"].astype("int64").values[np.argsort(kyiv_df["started_at"].astype("int64").values)]
    kyiv_m_starts, kyiv_m_ends = merge_intervals(kyiv_raw_starts, kyiv_raw_ends)

    # Target: Alert starts in [t, t + step_val)
    idx_next = np.searchsorted(kyiv_raw_starts, grid_ts + step_val, side="left")
    idx_curr = np.searchsorted(kyiv_raw_starts, grid_ts, side="left")
    feat_df["target_kyiv_start_30m"] = ((idx_next - idx_curr) > 0).astype(np.int32)

    # Kyiv active at time t
    kyiv_active_now = compute_active_mask(grid_ts, kyiv_m_starts, kyiv_m_ends)
    feat_df["kyiv_is_active"] = kyiv_active_now.astype(np.int32)

    # Target: Active at any point in [t, t + step_val]
    kyiv_active_next = compute_active_mask(grid_ts + step_val, kyiv_m_starts, kyiv_m_ends)
    feat_df["target_kyiv_active_30m"] = (kyiv_active_now | kyiv_active_next | (feat_df["target_kyiv_start_30m"] == 1)).astype(np.int32)

    # Kyiv local history
    feat_df["kyiv_time_since_last_alert_min"] = compute_time_since_last_ended_min(
        grid_ts, kyiv_m_starts, kyiv_m_ends, units_per_min=units_per_min
    )
    feat_df["kyiv_minutes_active"] = compute_minutes_active(
        grid_ts, kyiv_m_starts, kyiv_m_ends, units_per_min=units_per_min
    )
    feat_df["kyiv_alerts_last_6h"] = compute_rolling_metrics(
        grid_ts, kyiv_raw_starts, window_hours=6.0, units_per_min=units_per_min
    )
    feat_df["kyiv_alerts_last_24h"] = compute_rolling_metrics(
        grid_ts, kyiv_raw_starts, window_hours=24.0, units_per_min=units_per_min
    )

    # -------------------------------------------------------------
    # 2. TIER 1: INNER RING (7 RAIONS OF KYIV REGION)
    # -------------------------------------------------------------
    logger.info("Computing Tier 1 (7 Inner Ring Raions of Kyivska oblast)...")
    kyiv_obl_df = df[df["oblast"] == "Kyivska oblast"]
    obl_level_starts = kyiv_obl_df[kyiv_obl_df["level"] == "oblast"]["started_at"].astype("int64").values
    obl_level_ends = kyiv_obl_df[kyiv_obl_df["level"] == "oblast"]["finished_at"].astype("int64").values

    inner_active_matrix = []
    inner_active_names = []

    for raion_name in INNER_RING_RAIONS:
        clean_name = raion_name.replace(" raion", "").lower()
        raion_specific = kyiv_obl_df[kyiv_obl_df["raion"] == raion_name]
        r_starts = np.concatenate([obl_level_starts, raion_specific["started_at"].astype("int64").values])
        r_ends = np.concatenate([obl_level_ends, raion_specific["finished_at"].astype("int64").values])

        m_starts, m_ends = merge_intervals(r_starts, r_ends)
        r_active = compute_active_mask(grid_ts, m_starts, m_ends)
        r_mins = compute_minutes_active(grid_ts, m_starts, m_ends, units_per_min=units_per_min)

        col_active = f"raion_{clean_name}_active"
        col_mins = f"raion_{clean_name}_mins_active"
        feat_df[col_active] = r_active.astype(np.int32)
        feat_df[col_mins] = r_mins

        # Lags
        feat_df[f"{col_active}_lag30m"] = feat_df[col_active].shift(1).fillna(0).astype(np.int32)
        feat_df[f"{col_active}_lag60m"] = feat_df[col_active].shift(2).fillna(0).astype(np.int32)

        inner_active_matrix.append(r_active.astype(np.int32))
        inner_active_names.append(col_active)

    # Inner ring aggregates
    inner_mat = np.array(inner_active_matrix)
    feat_df["inner_ring_active_count"] = inner_mat.sum(axis=0).astype(np.int32)
    feat_df["inner_ring_active_count_lag30m"] = feat_df["inner_ring_active_count"].shift(1).fillna(0).astype(np.int32)
    feat_df["inner_ring_active_count_lag60m"] = feat_df["inner_ring_active_count"].shift(2).fillna(0).astype(np.int32)
    feat_df["inner_ring_diff_30m"] = feat_df["inner_ring_active_count"] - feat_df["inner_ring_active_count_lag30m"]

    # -------------------------------------------------------------
    # 3. TIER 2: ADJACENT OBLASTS (5 SURROUNDING REGIONS)
    # -------------------------------------------------------------
    logger.info("Computing Tier 2 (5 Adjacent oblasts)...")
    adj_active_matrix = []

    for obl in ADJACENT_OBLASTS:
        clean_name = obl.replace(" oblast", "").lower()
        sub_df = df[df["oblast"] == obl]
        m_starts, m_ends = merge_intervals(
            sub_df["started_at"].astype("int64").values,
            sub_df["finished_at"].astype("int64").values,
        )
        o_active = compute_active_mask(grid_ts, m_starts, m_ends)
        col_active = f"adj_{clean_name}_active"
        feat_df[col_active] = o_active.astype(np.int32)
        feat_df[f"{col_active}_lag30m"] = feat_df[col_active].shift(1).fillna(0).astype(np.int32)
        feat_df[f"{col_active}_lag60m"] = feat_df[col_active].shift(2).fillna(0).astype(np.int32)
        adj_active_matrix.append(o_active.astype(np.int32))

    adj_mat = np.array(adj_active_matrix)
    feat_df["adjacent_active_count"] = adj_mat.sum(axis=0).astype(np.int32)
    feat_df["adjacent_active_count_lag30m"] = feat_df["adjacent_active_count"].shift(1).fillna(0).astype(np.int32)
    feat_df["adjacent_active_count_lag60m"] = feat_df["adjacent_active_count"].shift(2).fillna(0).astype(np.int32)
    feat_df["adjacent_diff_30m"] = feat_df["adjacent_active_count"] - feat_df["adjacent_active_count_lag30m"]

    # -------------------------------------------------------------
    # 4. TIER 3: DEEP FLANKS (EASTERN & SOUTHERN CORRIDORS)
    # -------------------------------------------------------------
    logger.info("Computing Tier 3 (Eastern and Southern Corridors)...")
    east_active_list = []
    for obl in EASTERN_FLANK_OBLASTS:
        sub_df = df[df["oblast"] == obl]
        m_starts, m_ends = merge_intervals(
            sub_df["started_at"].astype("int64").values,
            sub_df["finished_at"].astype("int64").values,
        )
        east_active_list.append(compute_active_mask(grid_ts, m_starts, m_ends).astype(np.int32))
    feat_df["east_flank_active_count"] = np.array(east_active_list).sum(axis=0).astype(np.int32)
    feat_df["east_flank_active_count_lag30m"] = feat_df["east_flank_active_count"].shift(1).fillna(0).astype(np.int32)
    feat_df["east_flank_active_count_lag60m"] = feat_df["east_flank_active_count"].shift(2).fillna(0).astype(np.int32)
    feat_df["east_flank_active_count_lag90m"] = feat_df["east_flank_active_count"].shift(3).fillna(0).astype(np.int32)

    south_active_list = []
    for obl in SOUTHERN_FLANK_OBLASTS:
        sub_df = df[df["oblast"] == obl]
        m_starts, m_ends = merge_intervals(
            sub_df["started_at"].astype("int64").values,
            sub_df["finished_at"].astype("int64").values,
        )
        south_active_list.append(compute_active_mask(grid_ts, m_starts, m_ends).astype(np.int32))
    feat_df["south_flank_active_count"] = np.array(south_active_list).sum(axis=0).astype(np.int32)
    feat_df["south_flank_active_count_lag30m"] = feat_df["south_flank_active_count"].shift(1).fillna(0).astype(np.int32)
    feat_df["south_flank_active_count_lag60m"] = feat_df["south_flank_active_count"].shift(2).fillna(0).astype(np.int32)
    feat_df["south_flank_active_count_lag90m"] = feat_df["south_flank_active_count"].shift(3).fillna(0).astype(np.int32)

    # -------------------------------------------------------------
    # 5. TIER 4: NATIONAL MACRO-WAVE (ALL 25 OBLASTS)
    # -------------------------------------------------------------
    logger.info("Computing Tier 4 (National Macro-Wave across all 25 regions)...")
    national_active_list = []
    for obl in ALL_OBLASTS:
        sub_df = df[df["oblast"] == obl]
        m_starts, m_ends = merge_intervals(
            sub_df["started_at"].astype("int64").values,
            sub_df["finished_at"].astype("int64").values,
        )
        national_active_list.append(compute_active_mask(grid_ts, m_starts, m_ends).astype(np.int32))

    feat_df["national_active_count"] = np.array(national_active_list).sum(axis=0).astype(np.int32)
    feat_df["national_active_count_lag30m"] = feat_df["national_active_count"].shift(1).fillna(0).astype(np.int32)
    feat_df["national_active_count_lag60m"] = feat_df["national_active_count"].shift(2).fillna(0).astype(np.int32)
    feat_df["national_wave_speed_30m"] = feat_df["national_active_count"] - feat_df["national_active_count_lag30m"]
    feat_df["national_wave_speed_60m"] = feat_df["national_active_count"] - feat_df["national_active_count_lag60m"]

    # -------------------------------------------------------------
    # 6. CALENDAR & CYCLICAL ENCODING
    # -------------------------------------------------------------
    logger.info("Computing Cyclical Calendar features...")
    hours = grid.hour + grid.minute / 60.0
    feat_df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0).astype(np.float32)
    feat_df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0).astype(np.float32)

    dows = grid.dayofweek
    feat_df["dow_sin"] = np.sin(2 * np.pi * dows / 7.0).astype(np.float32)
    feat_df["dow_cos"] = np.cos(2 * np.pi * dows / 7.0).astype(np.float32)
    feat_df["is_weekend"] = (dows >= 5).astype(np.int32)

    # Reset index to column for clean parquet serialization
    result_df = feat_df.reset_index()

    logger.info(f"Feature dataset complete: {result_df.shape[0]:,} rows x {result_df.shape[1]} columns.")
    target_pos = result_df["target_kyiv_start_30m"].sum()
    logger.info(f"Positive targets (Kyiv starts): {target_pos:,} ({target_pos / len(result_df):.2%})")

    if save_parquet:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving features to: {output_path}")
        result_df.to_parquet(output_path, index=False, engine="pyarrow")

    return result_df


def get_feature_dataset(force_reload: bool = False) -> pd.DataFrame:
    """
    Returns feature dataset. Loads from parquet if present, else computes.
    """
    if PROCESSED_FEATURES_PARQUET.exists() and not force_reload:
        logger.info(f"Loading cached feature dataset from: {PROCESSED_FEATURES_PARQUET}")
        return pd.read_parquet(PROCESSED_FEATURES_PARQUET, engine="pyarrow")
    return build_30m_feature_store(save_parquet=True)


if __name__ == "__main__":
    df_feat = build_30m_feature_store()
    print("\nSample shape:", df_feat.shape)
    print("Columns sample:", list(df_feat.columns[:15]))
