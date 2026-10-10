"""
Live risk forecasting module for SirenCast UA.
Synthesizes real-time physical radar states and Telegram threat warnings into an 88-feature vector,
runs calibrated inference through LightGBM, and produces actionable percentage risk probabilities.
"""
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

from src.config import BASE_DIR
from src.features_text import (
    RE_AVIATION,
    RE_BALLISTIC,
    RE_KYIV_TARGETED,
    RE_MISSILE,
    RE_RESOLUTION,
    RE_SHAHED,
    get_multimodal_dataset,
)
from src.live_data import parse_live_siren_states, fetch_live_telegram_stream
from src.models import create_lightgbm_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_CACHE_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_CACHE_DIR / "lightgbm_multimodal_live.joblib"


def get_trained_live_model():
    """
    Loads pre-trained multimodal LightGBM model from cache,
    or trains and saves it if not already cached.
    """
    if MODEL_PATH.exists():
        try:
            bundle = joblib.load(MODEL_PATH)
            return bundle["model"], bundle["feature_cols"]
        except Exception:
            pass

    logger.info("Training full multimodal LightGBM model for live inference...")
    df = get_multimodal_dataset()
    ignore_cols = ["timestamp", "target_kyiv_start_30m", "target_kyiv_active_30m"]
    feat_cols = [c for c in df.columns if c not in ignore_cols]

    X = df[feat_cols]
    y = df["target_kyiv_active_30m"].values

    model = create_lightgbm_model()
    model.fit(X, y)

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_cols": feat_cols}, MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")

    return model, feat_cols


def build_live_feature_vector(
    simulated_alerts: Optional[dict] = None,
    simulated_telegram_texts: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Constructs the real-time feature vector for moment t using live or simulated signals.
    """
    df_hist = get_multimodal_dataset()
    last_row = df_hist.iloc[-1].copy()

    # 1. Physical Siren Telemetry
    if simulated_alerts is not None:
        alert_info = simulated_alerts
    else:
        alert_info = parse_live_siren_states()

    now_utc = alert_info.get("timestamp_utc", datetime.now(timezone.utc))
    regions = alert_info.get("regions", {})
    raions = alert_info.get("kyiv_raions", {})
    national_count = alert_info.get("national_active_count", 0)

    # 2. Telegram Warnings
    if simulated_telegram_texts is not None:
        tg_texts = simulated_telegram_texts
    else:
        tg_msgs = fetch_live_telegram_stream()
        tg_texts = [m["text"] for m in tg_msgs]

    # NLP Signal Extraction
    combined_texts = " ".join(tg_texts)
    has_kyiv_target = bool(re.search(RE_KYIV_TARGETED, combined_texts))
    is_resolution = bool(re.search(RE_RESOLUTION, combined_texts))
    effective_kyiv_target = 1 if (has_kyiv_target and not is_resolution) else 0

    shahed_count = len(re.findall(RE_SHAHED, combined_texts))
    missile_count = len(re.findall(RE_MISSILE, combined_texts))
    ballistic_count = len(re.findall(RE_BALLISTIC, combined_texts))
    aviation_count = len(re.findall(RE_AVIATION, combined_texts))
    resolution_count = len(re.findall(RE_RESOLUTION, combined_texts))

    threat_sum = effective_kyiv_target * 3 + missile_count * 2 + ballistic_count * 2 + aviation_count * 2 + shahed_count
    net_threat = max(0, threat_sum - resolution_count * 2)

    # 3. Assemble Feature Dictionary
    f = dict(last_row)

    # Calendar Cyclical
    hour = now_utc.hour + now_utc.minute / 60.0
    dow = now_utc.weekday()
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    f["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    f["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    f["is_weekend"] = 1 if dow >= 5 else 0

    # Tier 0 (Kyiv City)
    is_kyiv_active = 1 if regions.get("Kyiv City", False) else 0
    f["kyiv_is_active"] = is_kyiv_active
    f["kyiv_active_duration_min"] = 30.0 if is_kyiv_active else 0.0

    # Tier 1 (7 Kyivska Raions)
    raion_active_count = sum(1 for v in raions.values() if v)
    f["inner_ring_active_count"] = raion_active_count
    f["inner_ring_any_active"] = 1 if raion_active_count > 0 else 0
    f["inner_ring_all_active"] = 1 if raion_active_count == 7 else 0

    for eng_raion, is_active in raions.items():
        base_name = eng_raion.split()[0].lower()
        col = f"raion_{base_name}_active"
        if col in f:
            f[col] = 1 if is_active else 0

    # Tier 2 (Adjacent Oblasts)
    adj_names = ["Chernihivska oblast", "Sumska oblast", "Poltavska oblast", "Cherkaska oblast", "Zhytomyrska oblast"]
    adj_active = [regions.get(r, False) for r in adj_names]
    f["adjacent_active_count"] = sum(1 for v in adj_active if v)
    f["adjacent_any_active"] = 1 if any(adj_active) else 0
    f["adjacent_all_active"] = 1 if all(adj_active) else 0

    # Tier 3 (Flanks)
    east_names = ["Kharkivska oblast", "Dnipropetrovska oblast", "Zaporizka oblast"]
    south_names = ["Mykolaivska oblast", "Odeska oblast", "Khersonska oblast"]
    f["eastern_flank_active_count"] = sum(1 for r in east_names if regions.get(r, False))
    f["southern_flank_active_count"] = sum(1 for r in south_names if regions.get(r, False))

    # Tier 4 (Macro Surge Speed)
    last_national = last_row.get("national_active_count", national_count)
    f["national_active_count"] = national_count
    f["national_delta_30m"] = national_count - last_national
    f["national_surge_speed"] = (national_count - last_national) / 0.5  # per hour

    # Telegram NLP Features
    f["tg_msg_count_30m"] = len(tg_texts)
    f["tg_kyiv_targeted_30m"] = effective_kyiv_target
    f["tg_shahed_count_30m"] = shahed_count
    f["tg_missile_count_30m"] = missile_count
    f["tg_ballistic_count_30m"] = ballistic_count
    f["tg_aviation_count_30m"] = aviation_count
    f["tg_resolution_count_30m"] = resolution_count
    f["tg_net_threat_score"] = float(net_threat)

    metadata = {
        "timestamp_utc": now_utc,
        "is_kyiv_active": is_kyiv_active,
        "raion_active_count": raion_active_count,
        "adjacent_active_count": sum(1 for v in adj_active if v),
        "national_active_count": national_count,
        "national_delta": national_count - last_national,
        "telegram_threats": {
            "kyiv_targeted": effective_kyiv_target,
            "shahed": shahed_count,
            "missile": missile_count,
            "ballistic": ballistic_count,
            "aviation": aviation_count,
            "resolution": resolution_count,
            "net_score": net_threat,
        },
        "sample_messages": tg_texts[:3],
    }

    df_live = pd.DataFrame([f])
    return df_live, metadata


def predict_live_risk(
    simulated_alerts: Optional[dict] = None,
    simulated_telegram_texts: Optional[List[str]] = None,
) -> Dict:
    """
    Computes calibrated probability forecast for Kyiv City on the [t, t+30min] horizon.
    Returns:
        dict: {
            "probability_pct": float (e.g. 24.5),
            "risk_level": "LOW" | "ELEVATED" | "HIGH" | "CRITICAL",
            "risk_color": "green" | "yellow" | "orange" | "red",
            "metadata": dict,
            "top_drivers": list[str]
        }
    """
    model, feat_cols = get_trained_live_model()
    df_live, meta = build_live_feature_vector(simulated_alerts, simulated_telegram_texts)

    X_live = df_live[feat_cols]
    prob = float(model.predict_proba(X_live)[0, 1])
    pct = round(prob * 100.0, 1)

    if pct < 15.0:
        level = "LOW RISK"
        color = "green"
    elif pct < 35.0:
        level = "ELEVATED RISK"
        color = "yellow"
    elif pct < 65.0:
        level = "HIGH RISK"
        color = "orange"
    else:
        level = "CRITICAL IMMINENT RISK"
        color = "red"

    # Identify primary explanatory drivers
    drivers = []
    if meta["is_kyiv_active"]:
        drivers.append("Siren currently active in Kyiv City (continuation probability)")
    if meta["telegram_threats"]["kyiv_targeted"]:
        drivers.append("Telegram warning: aerial target heading directly towards Kyiv / Brovary")
    if meta["raion_active_count"] > 0:
        drivers.append(f"{meta['raion_active_count']} / 7 Inner Ring raions of Kyivska oblast under alert")
    if meta["adjacent_active_count"] > 0:
        drivers.append(f"{meta['adjacent_active_count']} / 5 Bordering oblasts active (Chernihiv/Sumy/Poltava)")
    if meta["telegram_threats"]["ballistic"]:
        drivers.append("Ballistic threat active in radar sector")
    if meta["telegram_threats"]["aviation"]:
        drivers.append("Tactical aviation / missile carrier takeoff reported")
    if meta["national_delta"] > 2:
        drivers.append(f"Rapid national alert surge (+{meta['national_delta']} oblasts in 30m)")

    if not drivers:
        drivers.append("Calm airspace: no approaching threats detected across all 4 spatial tiers")

    return {
        "probability_pct": pct,
        "risk_pct": pct,
        "probability_raw": prob,
        "risk_level": level,
        "risk_band": level,
        "risk_color": color,
        "metadata": meta,
        "top_drivers": drivers,
    }


def format_cli_dashboard(result: Optional[Dict] = None) -> str:
    """Formats live forecast results into a clean terminal dashboard."""
    if result is None:
        result = predict_live_risk()
    pct = result["probability_pct"]
    level = result["risk_level"]
    meta = result["metadata"]
    drivers = result["top_drivers"]
    t_str = meta["timestamp_utc"].strftime("%Y-%m-%d %H:%M:%S UTC")

    # Visual percentage progress gauge
    filled = int(round(pct / 5.0))
    bar = "=" * filled + "-" * (20 - filled)

    kyiv_status = "[ALERT ACTIVE]" if meta['is_kyiv_active'] else "[ALL CLEAR]"
    vector_status = "[INCOMING VECTOR]" if meta['telegram_threats']['kyiv_targeted'] else "None"

    lines = [
        "",
        "+==============================================================================+",
        "|                       SIRENCAST UA - LIVE FORECAST                           |",
        "|            30-Minute Probabilistic Early Warning for Kyiv City               |",
        "+==============================================================================+",
        f"  Timestamp: {t_str}",
        "",
        f"  FORECASTED THREAT PROBABILITY:  [{bar}]  {pct:.1f}%",
        f"  STATUS LEVEL:                   {level}",
        "",
        "  +-- SPATIAL RADAR TIERS -----------------------------------------------------+",
        f"  | Tier 0 (Kyiv City):       {kyiv_status:<20}                             |",
        f"  | Tier 1 (Kyivska Raions):  {meta['raion_active_count']} / 7 active surrounding raions           |",
        f"  | Tier 2 (Bordering Obl.):  {meta['adjacent_active_count']} / 5 adjacent oblasts active             |",
        f"  | Tier 4 (National Count):  {meta['national_active_count']} / 25 regions active (delta={meta['national_delta']:+d})          |",
        "  +----------------------------------------------------------------------------+",
        "",
        "  +-- TELEGRAM REAL-TIME SIGNALS ----------------------------------------------+",
        f"  | Kyiv Vector Targeted:     {vector_status:<20}                             |",
        f"  | Detected Threat Units:    Shaheds: {meta['telegram_threats']['shahed']}, Missiles: {meta['telegram_threats']['missile']}, Ballistics: {meta['telegram_threats']['ballistic']} |",
        f"  | Net Threat Score:         {meta['telegram_threats']['net_score']:.1f}                                             |",
        "  +----------------------------------------------------------------------------+",
        "",
        "  PRIMARY EXPLANATORY DRIVERS:",
    ]
    for d in drivers:
        lines.append(f"   * {d}")

    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    res = predict_live_risk()
    print(format_cli_dashboard(res))
