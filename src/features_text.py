"""
Text feature extraction module for SirenCast UA.
Parses Telegram Desktop exports (Monitor and PSZSU channels),
extracts causal NLP threat signals with contextual direction and
mitigation filtering ('чисто', 'збито', 'відбій'), and fuses with the 30m grid.
Strictly causal: features at time t use only messages with timestamp < t.
"""
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import (
    BASE_DIR,
    MONITOR_JSON,
    PSZSU_JSON,
    CLEAN_TELEGRAM_PARQUET,
    PROCESSED_MULTIMODAL_PARQUET,
    PROCESSED_DATA_DIR,
)
from src.features_ts import get_feature_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Precompiled regex patterns for threat and resolution signals (Cyrillic-safe without ASCII \b)
RE_KYIV_TARGETED = re.compile(
    r"(?:курс|напрямок|напрямку|вектор|летить|рух|захід|заходять|повз|ціль|цілі|швидкісна|ракета|ракети)\s+(?:на\s+|у\s+|через\s+)?(?:київ|київщин|бровар|васильків|бориспіль|вишгород|буч|ірпін|вишеньки|позняк)"
    r"|(?:київ|київщина|бровари|васильків|бориспіль|вишгород|буча)\s*[-:—–]?\s*(?:увага|загроза|ракета|бпла|шахед|тривога|укриття|в\s+укриття)",
    re.IGNORECASE,
)

RE_RESOLUTION = re.compile(
    r"(?:відбій|чисто|збит|знищен|локаційно\s+втрачен|подавлен|дорозвідк)",
    re.IGNORECASE,
)

RE_SHAHED = re.compile(r"(?:шахед|бпла|дрон|мопед|геран)", re.IGNORECASE)
RE_MISSILE = re.compile(r"(?:ракета|крилат|х-101|х-555|х-22|х-59|калібр|пуск|залп)", re.IGNORECASE)
RE_BALLISTIC = re.compile(r"(?:балісти[кч]|іскандер|с-300|с-400|бєлгород|курськ)", re.IGNORECASE)
RE_AVIATION = re.compile(r"(?:ту-95|ту-22|ту-160|міг-31|кинджал|оленья|енгельс|саваслейк)", re.IGNORECASE)


def extract_plain_text(raw_text) -> str:
    """Extracts string from Telegram text (which can be str or list of entity dicts)."""
    if isinstance(raw_text, str):
        return raw_text
    elif isinstance(raw_text, list):
        return "".join([x if isinstance(x, str) else x.get("text", "") for x in raw_text])
    return ""


def load_raw_telegram_stream(
    parquet_path: Path = CLEAN_TELEGRAM_PARQUET,
    monitor_path: Path = MONITOR_JSON,
    pszsu_path: Path = PSZSU_JSON,
    min_year: int = 2023,
) -> pd.DataFrame:
    """
    Loads Monitor and PSZSU messages. Prioritizes the lightweight clean Parquet (8MB).
    Falls back to raw JSON exports if Parquet is not found.
    """
    if parquet_path.exists():
        logger.info(f"Loading pruned Telegram messages directly from {parquet_path}...")
        df_msgs = pd.read_parquet(parquet_path, engine="pyarrow")
        if "timestamp_utc" not in df_msgs.columns:
            df_msgs["timestamp_utc"] = pd.to_datetime(df_msgs["unixtime"], unit="s", utc=True)
        return df_msgs[df_msgs["timestamp_utc"].dt.year >= min_year].reset_index(drop=True)

    logger.info(f"Clean parquet not found. Parsing raw Telegram exports (min_year={min_year})...")
    messages = []

    for path, channel_name in [(monitor_path, "monitor"), (pszsu_path, "pszsu")]:
        if not path.exists():
            logger.warning(f"File not found: {path}")
            continue
        logger.info(f"Parsing {channel_name} from {path}...")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for m in data.get("messages", []):
            if m.get("type") != "message" or not m.get("date_unixtime"):
                continue
            ts_sec = int(m["date_unixtime"])
            text = extract_plain_text(m.get("text", "")).strip()
            if not text:
                continue
            messages.append((ts_sec, channel_name, text))

    df_msgs = pd.DataFrame(messages, columns=["unixtime", "channel", "text"])
    df_msgs["timestamp_utc"] = pd.to_datetime(df_msgs["unixtime"], unit="s", utc=True)
    df_msgs = df_msgs[df_msgs["timestamp_utc"].dt.year >= min_year]
    df_msgs = df_msgs.sort_values(by="unixtime").reset_index(drop=True)

    logger.info(f"Total valid messages from {min_year}+: {len(df_msgs):,}")
    return df_msgs


def extract_message_signals(df_msgs: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized extraction of NLP signals per message.
    """
    logger.info("Extracting threat, direction, and resolution NLP signals...")
    texts = df_msgs["text"].astype(str)

    # Boolean flags
    is_resolution = texts.str.contains(RE_RESOLUTION, regex=True)
    is_kyiv_raw = texts.str.contains(RE_KYIV_TARGETED, regex=True)

    # Target Kyiv signal is valid ONLY if not pure resolution ("Київ чисто" / "Васильків збиття")
    df_msgs["sig_kyiv_targeted"] = (is_kyiv_raw & (~is_resolution)).astype(np.int32)
    df_msgs["sig_resolution"] = is_resolution.astype(np.int32)
    df_msgs["sig_shahed"] = texts.str.contains(RE_SHAHED, regex=True).astype(np.int32)
    df_msgs["sig_missile"] = texts.str.contains(RE_MISSILE, regex=True).astype(np.int32)
    df_msgs["sig_ballistic"] = texts.str.contains(RE_BALLISTIC, regex=True).astype(np.int32)
    df_msgs["sig_aviation"] = texts.str.contains(RE_AVIATION, regex=True).astype(np.int32)

    return df_msgs


def aggregate_text_features_to_grid(
    df_grid: pd.DataFrame,
    df_msgs: pd.DataFrame,
    window_minutes: int = 30,
) -> pd.DataFrame:
    """
    Aggregates text messages into each 30m grid slot [t - window_minutes, t).
    Guarantees strict causality: message timestamp < t.
    Uses exact Unix epoch seconds for timestamp comparisons.
    """
    logger.info(f"Aggregating text signals onto 30m grid (causal window = {window_minutes}m)...")
    # Exact Unix epoch seconds
    grid_sec = df_grid["timestamp"].astype("datetime64[s, UTC]").astype("int64").values
    msg_sec = df_msgs["unixtime"].astype("int64").values
    window_sec = window_minutes * 60

    # Pre-extract numpy signal arrays
    kyiv_sig = df_msgs["sig_kyiv_targeted"].values
    resol_sig = df_msgs["sig_resolution"].values
    shahed_sig = df_msgs["sig_shahed"].values
    missile_sig = df_msgs["sig_missile"].values
    ballistic_sig = df_msgs["sig_ballistic"].values
    aviation_sig = df_msgs["sig_aviation"].values

    # Prefix sums for fast O(1) interval aggregation
    cumsum_kyiv = np.concatenate([[0], np.cumsum(kyiv_sig)])
    cumsum_resol = np.concatenate([[0], np.cumsum(resol_sig)])
    cumsum_shahed = np.concatenate([[0], np.cumsum(shahed_sig)])
    cumsum_missile = np.concatenate([[0], np.cumsum(missile_sig)])
    cumsum_ballistic = np.concatenate([[0], np.cumsum(ballistic_sig)])
    cumsum_aviation = np.concatenate([[0], np.cumsum(aviation_sig)])

    # Left and right bounds for [t - window_sec, t)
    # Right index: messages strictly < t
    idx_right = np.searchsorted(msg_sec, grid_sec, side="left")
    idx_left = np.searchsorted(msg_sec, grid_sec - window_sec, side="left")

    msg_count = (idx_right - idx_left).astype(np.int32)
    kyiv_count = cumsum_kyiv[idx_right] - cumsum_kyiv[idx_left]
    resol_count = cumsum_resol[idx_right] - cumsum_resol[idx_left]
    shahed_count = cumsum_shahed[idx_right] - cumsum_shahed[idx_left]
    missile_count = cumsum_missile[idx_right] - cumsum_missile[idx_left]
    ballistic_count = cumsum_ballistic[idx_right] - cumsum_ballistic[idx_left]
    aviation_count = cumsum_aviation[idx_right] - cumsum_aviation[idx_left]

    # Net threat score: weighted sum of threats minus clearing/resolution signals
    threat_sum = kyiv_count * 3 + missile_count * 2 + ballistic_count * 2 + aviation_count * 2 + shahed_count
    net_threat = np.maximum(0, threat_sum - resol_count * 2).astype(np.float32)

    text_feats = pd.DataFrame(index=df_grid.index)
    text_feats["tg_msg_count_30m"] = msg_count
    text_feats["tg_kyiv_targeted_30m"] = (kyiv_count > 0).astype(np.int32)
    text_feats["tg_shahed_count_30m"] = shahed_count
    text_feats["tg_missile_count_30m"] = missile_count
    text_feats["tg_ballistic_count_30m"] = ballistic_count
    text_feats["tg_aviation_count_30m"] = aviation_count
    text_feats["tg_resolution_count_30m"] = resol_count
    text_feats["tg_net_threat_score"] = net_threat

    # Lags
    text_feats["tg_msg_count_lag30m"] = text_feats["tg_msg_count_30m"].shift(1).fillna(0).astype(np.int32)
    text_feats["tg_net_threat_lag30m"] = text_feats["tg_net_threat_score"].shift(1).fillna(0).astype(np.float32)
    text_feats["tg_kyiv_targeted_lag30m"] = text_feats["tg_kyiv_targeted_30m"].shift(1).fillna(0).astype(np.int32)

    return text_feats


def build_multimodal_dataset(
    save_parquet: bool = True,
    output_path: Path = PROCESSED_MULTIMODAL_PARQUET,
) -> pd.DataFrame:
    """
    Fuses 30m Time Series feature dataset with Telegram NLP features.
    """
    df_ts = get_feature_dataset()
    logger.info(f"Loaded TS dataset: {df_ts.shape}")

    df_msgs = load_raw_telegram_stream()
    df_msgs = extract_message_signals(df_msgs)

    text_feats = aggregate_text_features_to_grid(df_ts, df_msgs)

    logger.info("Fusing TS and Text feature matrices...")
    multimodal_df = pd.concat([df_ts, text_feats], axis=1)

    logger.info(f"Multimodal dataset complete: {multimodal_df.shape[0]:,} rows x {multimodal_df.shape[1]} columns.")

    if save_parquet:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving multimodal dataset to {output_path}...")
        multimodal_df.to_parquet(output_path, index=False, engine="pyarrow")

    return multimodal_df


def get_multimodal_dataset(force_reload: bool = False) -> pd.DataFrame:
    """
    Returns multimodal feature dataset. Loads from cache if present.
    """
    if PROCESSED_MULTIMODAL_PARQUET.exists() and not force_reload:
        logger.info(f"Loading cached multimodal dataset from {PROCESSED_MULTIMODAL_PARQUET}...")
        return pd.read_parquet(PROCESSED_MULTIMODAL_PARQUET, engine="pyarrow")
    return build_multimodal_dataset(save_parquet=True)


if __name__ == "__main__":
    df_multi = build_multimodal_dataset()
    print("\nMultimodal dataset shape:", df_multi.shape)
    print("New text columns:", [c for c in df_multi.columns if c.startswith("tg_")])
