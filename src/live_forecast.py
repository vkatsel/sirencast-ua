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
from src.models import create_lightgbm_model, compute_time_decay_weights

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_CACHE_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_CACHE_DIR / "lightgbm_multimodal_live.joblib"


def calibrate_odds(prob: float, weight: float = 5.0) -> float:
    """
    Bayesian cost-sensitive odds calibration for civil defense decision-making:
        P_cal = (w * p) / ((1 - p) + w * p)
    Maps low base rates into an intuitive operational threat scale (0-100%)
    while preserving 100% of rank ordering (monotonic).
    """
    p = np.clip(prob, 1e-6, 1.0 - 1e-6)
    cal_odds = (p / (1.0 - p)) * weight
    return float(cal_odds / (1.0 + cal_odds))


def get_trained_live_model():
    """
    Loads pre-trained multimodal LightGBM model and cached empirical ECDF distribution,
    or trains with exponential time-decay (half-life = 270 days) and saves if not cached.
    """
    if MODEL_PATH.exists():
        try:
            bundle = joblib.load(MODEL_PATH)
            if "sorted_hist_preds" in bundle:
                return bundle["model"], bundle["feature_cols"], bundle["sorted_hist_preds"]
        except Exception:
            pass

    logger.info("Training multimodal LightGBM model with time-decay (T_half=270d) for live inference...")
    df = get_multimodal_dataset()
    ignore_cols = ["timestamp", "target_kyiv_start_30m", "target_kyiv_active_30m"]
    feat_cols = [c for c in df.columns if c not in ignore_cols]

    X = df[feat_cols]
    y = df["target_kyiv_active_30m"].values

    sample_weights = compute_time_decay_weights(df["timestamp"], half_life_days=270.0)
    model = create_lightgbm_model()
    model.fit(X, y, sample_weight=sample_weights)

    # Precalculate and sort historical predictions for fast O(log N) ECDF percentile lookup
    hist_preds = model.predict_proba(X)[:, 1]
    sorted_hist_preds = np.sort(hist_preds)

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "feature_cols": feat_cols,
        "sorted_hist_preds": sorted_hist_preds,
    }, MODEL_PATH)
    logger.info(f"Model and ECDF distribution saved to {MODEL_PATH}")

    return model, feat_cols, sorted_hist_preds


_cached_last_row = None


def get_cached_baseline_row():
    global _cached_last_row
    if _cached_last_row is None:
        df_hist = get_multimodal_dataset()
        _cached_last_row = df_hist.iloc[-1].copy()
    return _cached_last_row


def build_live_feature_vector(
    simulated_alerts: Optional[dict] = None,
    simulated_telegram_texts: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Constructs the real-time feature vector for moment t using live or simulated signals.
    """
    last_row = get_cached_baseline_row().copy()

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
        active_tg_texts = simulated_telegram_texts
        formatted_samples = [{"channel": "simulation", "text": t, "age_minutes": 0.0, "is_active": True} for t in simulated_telegram_texts[:5]]
    else:
        tg_msgs = fetch_live_telegram_stream()
        # Strictly evaluate active messages published within the recent 45-minute window!
        active_msgs = [m for m in tg_msgs if m.get("is_active", False)]
        active_tg_texts = [m["text"] for m in active_msgs]
        
        # Prepare sample messages for the UI, preferring active ones or latest
        display_msgs = active_msgs if active_msgs else tg_msgs[-5:]
        formatted_samples = [
            {
                "channel": m.get("channel", "telegram"),
                "text": m.get("text", ""),
                "age_minutes": m.get("age_minutes", 0.0),
                "is_active": m.get("is_active", False),
                "datetime_iso": m.get("datetime_iso", ""),
            }
            for m in display_msgs[:5]
        ]

    # NLP Signal Extraction strictly from messages published within the current threat window
    combined_texts = " ".join(active_tg_texts)
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
        col_active = f"raion_{base_name}_active"
        col_mins = f"raion_{base_name}_mins_active"
        if col_active in f:
            f[col_active] = 1 if is_active else 0
        if col_mins in f:
            f[col_mins] = 30.0 if is_active else 0.0

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
    surge_speed = (national_count - last_national) / 0.5  # per hour
    f["national_wave_speed_30m"] = surge_speed
    f["national_wave_speed_60m"] = surge_speed * 0.8

    # Telegram NLP Features
    f["tg_msg_count_30m"] = len(active_tg_texts)
    f["tg_kyiv_targeted_30m"] = effective_kyiv_target
    f["tg_shahed_count_30m"] = shahed_count
    f["tg_missile_count_30m"] = missile_count
    f["tg_ballistic_count_30m"] = ballistic_count
    f["tg_aviation_count_30m"] = aviation_count
    f["tg_resolution_count_30m"] = resolution_count
    f["tg_net_threat_score"] = float(net_threat)
    f["tg_net_threat_lag30m"] = float(net_threat)
    f["tg_kyiv_targeted_lag30m"] = effective_kyiv_target

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
        "sample_messages": formatted_samples,
    }

    df_live = pd.DataFrame([f])
    return df_live, metadata


def predict_live_risk(
    simulated_alerts: Optional[dict] = None,
    simulated_telegram_texts: Optional[List[str]] = None,
    odds_weight: float = 5.0,
) -> Dict:
    """
    Computes multi-dimensional risk forecast for Kyiv City on the [t, t+30min] horizon:
    1. Operational Threat Index (%): Bayesian cost-calibrated decision posterior (0-100%).
    2. Raw Statistical Probability (%): Unweighted frequentist chance of siren onset.
    3. Empirical Threat Percentile (%): ECDF quantile vs entire 2023-2026 dataset history.
    4. Hazard Surge Multiplier: Factor increase over quiet baseline airspace (~1.8%).

    Returns:
        dict: contract containing all four dimensions, categorical level, metadata, and drivers.
    """
    model, feat_cols, sorted_hist_preds = get_trained_live_model()
    df_live, meta = build_live_feature_vector(simulated_alerts, simulated_telegram_texts)

    X_live = df_live[feat_cols]
    prob_raw = float(model.predict_proba(X_live)[0, 1])
    raw_pct = round(prob_raw * 100.0, 1)

    # 1. Operational Threat Index via Bayesian odds calibration (preserves monotonicity)
    threat_prob = calibrate_odds(prob_raw, weight=odds_weight)
    threat_pct = round(threat_prob * 100.0, 1)

    # 2. Empirical ECDF Percentile (rank among all historical 30-min intervals)
    idx = int(np.searchsorted(sorted_hist_preds, prob_raw, side="right"))
    ecdf_pct = float(round((idx / len(sorted_hist_preds)) * 100.0, 1))

    # 3. Hazard Surge Multiplier vs quiet calm baseline (~1.8% historical prior)
    p_baseline = 0.018
    hazard_multiplier = round(max(1.0, prob_raw / p_baseline), 1)

    # Risk level categorized by Operational Threat Index:
    if threat_pct < 10.0:
        level = "LOW RISK"
        color = "green"
    elif threat_pct < 35.0:
        level = "ELEVATED RISK"
        color = "yellow"
    elif threat_pct < 70.0:
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
        drivers.append(f"{meta['adjacent_active_count']} / 5 Adjacent & approach corridors active (Chernihiv/Sumy/Poltava)")
    if meta["telegram_threats"]["ballistic"]:
        drivers.append("Ballistic threat active in radar sector")
    if meta["telegram_threats"]["aviation"]:
        drivers.append("Tactical aviation / missile carrier takeoff reported")
    if meta["national_delta"] > 2:
        drivers.append(f"Rapid national alert surge (+{meta['national_delta']} oblasts in 30m)")

    if not drivers:
        drivers.append("Calm airspace: no approaching threats detected across all 4 spatial tiers")

    return {
        "threat_index_pct": threat_pct,
        "raw_probability_pct": raw_pct,
        "probability_pct": threat_pct,  # backward compatibility
        "risk_pct": threat_pct,
        "probability_raw": prob_raw,
        "ecdf_percentile": ecdf_pct,
        "hazard_multiplier": hazard_multiplier,
        "risk_level": level,
        "risk_band": level,
        "risk_color": color,
        "odds_weight": odds_weight,
        "metadata": meta,
        "top_drivers": drivers,
    }


def format_cli_dashboard(result: Optional[Dict] = None) -> str:
    """Formats live forecast results into a clean terminal dashboard with dual metrics."""
    if result is None:
        result = predict_live_risk()
    threat_pct = result["threat_index_pct"]
    raw_pct = result["raw_probability_pct"]
    ecdf_pct = result["ecdf_percentile"]
    hazard_mult = result["hazard_multiplier"]
    level = result["risk_level"]
    meta = result["metadata"]
    drivers = result["top_drivers"]
    t_str = meta["timestamp_utc"].strftime("%Y-%m-%d %H:%M:%S UTC")

    # Visual percentage progress gauge for operational threat
    filled = int(round(threat_pct / 5.0))
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
        f"  OPERATIONAL THREAT SCORE:       [{bar}]  {threat_pct:.1f}%  ({level})",
        f"  RAW STATISTICAL PROBABILITY:    {raw_pct:.1f}% in next 30 min",
        f"  HAZARD SURGE MULTIPLIER:        {hazard_mult:.1f}x vs quiet baseline",
        "",
        "  +-- SPATIAL RADAR TIERS -----------------------------------------------------+",
        f"  | Tier 0 (Kyiv City):       {kyiv_status:<20}                             |",
        f"  | Tier 1 (Kyivska Raions):  {meta['raion_active_count']} / 7 active surrounding raions           |",
        f"  | Tier 2 (Adjacent/Transit):{meta['adjacent_active_count']} / 5 approach corridors active           |",
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

    sample_msgs = meta.get("sample_messages", [])
    if sample_msgs:
        lines.append("")
        lines.append("  RECENT TELEGRAM INTEL FEED:")
        for sm in sample_msgs[:4]:
            ch = sm.get("channel", "tg")
            age = sm.get("age_minutes", 0.0)
            age_str = f"{int(age)}m ago" if age < 60 else f"{age / 60.0:.1f}h ago"
            status_tag = "[ACTIVE]" if sm.get("is_active", False) else "[ARCHIVE]"
            txt = sm.get("text", "")[:75].encode("ascii", errors="backslashreplace").decode("ascii")
            lines.append(f"   * {status_tag} ({age_str}) @{ch}: {txt}...")

    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    res = predict_live_risk()
    print(format_cli_dashboard(res))
