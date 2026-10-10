"""
Prunes and compresses raw Telegram Desktop JSON exports (Monitor & PSZSU).
Drops all unnecessary metadata (reactions, media entities, avatars, reply markup)
and retains only the essential fields: unixtime, channel, and plain text.
Reduces dataset footprint from ~180 MB to ~8.3 MB (96% compression).
"""
import json
import logging
from pathlib import Path
from typing import Optional
import pandas as pd

from src.config import DATA_DIR, MONITOR_JSON, PSZSU_JSON

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CLEAN_TELEGRAM_PARQUET = DATA_DIR / "clean_telegram.parquet"


def extract_plain_text(raw_text) -> str:
    """Extracts plain string from Telegram message text."""
    if isinstance(raw_text, str):
        return raw_text
    elif isinstance(raw_text, list):
        return "".join([x if isinstance(x, str) else x.get("text", "") for x in raw_text])
    return ""


def prune_and_save_telegram(
    monitor_path: Path = MONITOR_JSON,
    pszsu_path: Path = PSZSU_JSON,
    output_path: Path = CLEAN_TELEGRAM_PARQUET,
    min_year: int = 2023,
) -> pd.DataFrame:
    """
    Extracts essential fields from raw Telegram JSONs and saves a lightweight Parquet file.
    """
    logger.info(f"Pruning Telegram exports from {monitor_path} and {pszsu_path}...")
    messages = []

    for path, channel_name in [(monitor_path, "monitor"), (pszsu_path, "pszsu")]:
        if not path.exists():
            logger.warning(f"File not found: {path}")
            continue
        logger.info(f"Reading {channel_name} from {path}...")
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

    initial_size_mb = (
        (monitor_path.stat().st_size if monitor_path.exists() else 0)
        + (pszsu_path.stat().st_size if pszsu_path.exists() else 0)
    ) / (1024 * 1024)

    logger.info(f"Saving {len(df_msgs):,} pruned messages to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_msgs.to_parquet(output_path, index=False, engine="pyarrow")

    pruned_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(
        f"Pruning complete: {initial_size_mb:.1f} MB -> {pruned_size_mb:.2f} MB "
        f"({(1.0 - pruned_size_mb / initial_size_mb):.1%} reduction)."
    )
    return df_msgs


if __name__ == "__main__":
    prune_and_save_telegram()
