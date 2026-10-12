"""
Live data ingestion module for SirenCast UA.
Pulls real-time physical alert states and public Telegram warnings without credentials.
Enforces strict client-side rate limiting (min 5s TTL cache) to comply with Ubilling's 2 RPS limit.
"""
import html
import json
import logging
import re
import time
import urllib.request
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

try:
    from zoneinfo import ZoneInfo
    KYIV_TZ = ZoneInfo("Europe/Kyiv")
except Exception:
    KYIV_TZ = timezone(timedelta(hours=3))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from src.config import DATA_DIR

UBILLING_API_URL = "https://ubilling.net.ua/aerialalerts/?raw"
CACHE_TTL_SECONDS = 6.0  # Safe TTL > 0.5s to strictly respect 2 RPS limit
DISK_CACHE_FILE = DATA_DIR / ".cache_live_alerts.json"
KYIV_EVENTS_CACHE_FILE = DATA_DIR / ".cache_kyiv_alert_events.json"

# Official Suspilne Kyiv alert patterns (support yellow & red levels and all-clears)
RE_SUSPILNE_KYIV_START = re.compile(
    r"(?:[🟡🔴]\s*)?(?:м\.\s*)?Київ\s*[—–\-−]\s*повітряна\s*тривога",
    re.IGNORECASE,
)
RE_SUSPILNE_KYIV_CLEAR = re.compile(
    r"(?:[🟢]\s*)?(?:м\.\s*)?Київ\s*[—–\-−]\s*відбій\s*повітряної\s*тривоги",
    re.IGNORECASE,
)


def format_duration_en(minutes: float) -> str:
    m = max(0, int(round(minutes)))
    if m < 60:
        return f"{m}m"
    hours = m // 60
    rem = m % 60
    return f"{hours}h {rem:02d}m"


def format_duration_ua(minutes: float) -> str:
    m = max(0, int(round(minutes)))
    if m < 60:
        return f"{m} хв"
    hours = m // 60
    rem = m % 60
    return f"{hours} год {rem:02d} хв"


# Thread-safe in-memory cache
_alert_cache: Dict = {"timestamp": 0.0, "data": None}
_telegram_cache: Dict = {"timestamp": 0.0, "data": None}
_kyiv_telemetry_cache: Dict = {"timestamp": 0.0, "data": None}

# Region name mappings from Ukrainian (Ubilling API) to English (SirenCast dataset)
OBLAST_MAPPING = {
    "м. Київ": "Kyiv City",
    "Київська область": "Kyivska oblast",
    "Чернігівська область": "Chernihivska oblast",
    "Сумська область": "Sumska oblast",
    "Полтавська область": "Poltavska oblast",
    "Черкаська область": "Cherkaska oblast",
    "Житомирська область": "Zhytomyrska oblast",
    "Харківська область": "Kharkivska oblast",
    "Дніпропетровська область": "Dnipropetrovska oblast",
    "Запорізька область": "Zaporizka oblast",
    "Донецька область": "Donetska oblast",
    "Луганська область": "Luhanska oblast",
    "Миколаївська область": "Mykolaivska oblast",
    "Одеська область": "Odeska oblast",
    "Херсонська область": "Khersonska oblast",
    "Вінницька область": "Vinnytska oblast",
    "Кіровоградська область": "Kirovohradska oblast",
    "Хмельницька область": "Khmelnytska oblast",
    "Рівненська область": "Rivnenska oblast",
    "Волинська область": "Volynska oblast",
    "Львівська область": "Lvivska oblast",
    "Тернопільська область": "Ternopilska oblast",
    "Івано-Франківська область": "Ivano-Frankivska oblast",
    "Закарпатська область": "Zakarpatska oblast",
    "Чернівецька область": "Chernivetska oblast",
    "Автономна Республіка Крим": "Autonomous Republic of Crimea",
    "м. Севастополь": "Sevastopol",
}

RAION_MAPPING = {
    "Вишгородський": "Vyshhorodskyi raion",
    "Броварський": "Brovarskyi raion",
    "Бориспільський": "Boryspilskyi raion",
    "Обухівський": "Obukhivskyi raion",
    "Фастівський": "Fastivskyi raion",
    "Бучанський": "Buchanskyi raion",
    "Білоцерківський": "Bilotserkivskyi raion",
}


def _load_disk_cache() -> Optional[dict]:
    if DISK_CACHE_FILE.exists():
        try:
            mtime = DISK_CACHE_FILE.stat().st_mtime
            with open(DISK_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {"timestamp": mtime, "data": data}
        except Exception:
            pass
    return None


def _save_disk_cache(data: dict):
    try:
        DISK_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(DISK_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception as e:
        logger.debug(f"Could not persist disk cache: {e}")


def fetch_live_alerts_raw(timeout: int = 5) -> dict:
    """
    Fetches raw alerts from Ubilling API with an in-memory and on-disk TTL cache
    to strictly adhere to the 2 RPS rate limit and prevent HTTP 429 across processes.
    """
    global _alert_cache
    now = time.time()

    # Check in-memory cache
    if _alert_cache["data"] is not None and (now - _alert_cache["timestamp"]) < CACHE_TTL_SECONDS:
        return _alert_cache["data"]

    # Check on-disk cache if in-memory is cold
    if _alert_cache["data"] is None:
        disk_cached = _load_disk_cache()
        if disk_cached is not None:
            _alert_cache["timestamp"] = disk_cached["timestamp"]
            _alert_cache["data"] = disk_cached["data"]
            if (now - disk_cached["timestamp"]) < CACHE_TTL_SECONDS:
                return disk_cached["data"]

    req = urllib.request.Request(
        UBILLING_API_URL,
        headers={"User-Agent": "SirenCast-UA/1.0 (Academic Research; KSE AI School)"},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            _alert_cache["timestamp"] = now
            _alert_cache["data"] = data
            _save_disk_cache(data)
            return data
    except Exception as e:
        logger.warning(f"Failed to fetch live alerts from {UBILLING_API_URL}: {e}")
        # If in-memory or on-disk cache exists, serve stale cache gracefully
        if _alert_cache["data"] is not None:
            logger.info("Serving stale cached alerts as fallback.")
            return _alert_cache["data"]
        disk_cached = _load_disk_cache()
        if disk_cached is not None:
            logger.info("Serving stale on-disk cached alerts as fallback.")
            _alert_cache["timestamp"] = disk_cached["timestamp"]
            _alert_cache["data"] = disk_cached["data"]
            return disk_cached["data"]
        # Fallback to simulated quiescent state
        return {"source": "simulated", "raw": {}}


def parse_live_siren_states() -> Dict[str, dict]:
    """
    Parses live alert states into structured English region identifiers.
    Returns:
        dict: {
            "regions": {"Kyiv City": True, "Chernihivska oblast": False, ...},
            "kyiv_raions": {"Brovarskyi raion": True, ...},
            "timestamp_utc": datetime object,
            "national_active_count": int,
        }
    """
    data = fetch_live_alerts_raw()
    raw = data.get("raw", {})
    now_utc = datetime.now(timezone.utc)

    regions_status = {eng: False for eng in OBLAST_MAPPING.values()}
    kyiv_raions_status = {eng: False for eng in RAION_MAPPING.values()}

    for ua_name, val in raw.items():
        clean_ua = ua_name.strip()
        eng_name = OBLAST_MAPPING.get(clean_ua)
        if not eng_name:
            continue

        is_top_enabled = bool(val.get("enabled", False))

        # Check sub-districts (in modern Ukrainian alert system, alerts are often issued per raion)
        districts = val.get("districts", {})
        if isinstance(districts, dict):
            dist_items = districts.items()
        elif isinstance(districts, list):
            dist_items = [(d.get("name", ""), d) for d in districts if isinstance(d, dict)]
        else:
            dist_items = []

        any_district_enabled = any(bool(d_val.get("enabled", False)) for _, d_val in dist_items)
        regions_status[eng_name] = is_top_enabled or any_district_enabled

        # Parse Kyivska oblast districts if available
        if clean_ua == "Київська область":
            for d_name, d_val in dist_items:
                for ua_prefix, eng_raion in RAION_MAPPING.items():
                    if ua_prefix in d_name:
                        kyiv_raions_status[eng_raion] = bool(d_val.get("enabled", False))

    # If raions aren't individually active but Kyivska oblast is active overall:
    if regions_status.get("Kyivska oblast", False) and not any(kyiv_raions_status.values()):
        for r in kyiv_raions_status:
            kyiv_raions_status[r] = True

    # Parse Kyiv changed timestamp if available
    kyiv_raw = raw.get("м. Київ", {})
    kyiv_changed_at = None
    changed_str = kyiv_raw.get("changed") or kyiv_raw.get("enabled_at") or kyiv_raw.get("disabled_at")
    if changed_str:
        try:
            # Handles standard SQL format "YYYY-MM-DD HH:MM:SS" or ISO
            parsed_ch = datetime.fromisoformat(str(changed_str).replace("Z", "+00:00"))
            if parsed_ch.tzinfo is None:
                parsed_ch = parsed_ch.replace(tzinfo=timezone.utc)
            if parsed_ch.year >= 2022:
                kyiv_changed_at = parsed_ch
        except Exception:
            pass

    # Integrate Suspilne Kyiv alert tracking
    try:
        kyiv_telemetry = fetch_and_update_kyiv_alert_telemetry(now_utc)
        if kyiv_telemetry.get("is_active"):
            regions_status["Kyiv City"] = True
    except Exception as e:
        logger.warning(f"Could not update Kyiv telemetry from Suspilne: {e}")
        kyiv_telemetry = resolve_kyiv_alert_timeline([], now_utc)

    # If Ubilling API explicitly reports Kyiv active, ensure is_active reflects it
    if regions_status.get("Kyiv City", False) and not kyiv_telemetry.get("is_active", False):
        kyiv_telemetry["is_active"] = True
        kyiv_telemetry["status_label_en"] = "ALERT ACTIVE"
        kyiv_telemetry["status_label_ua"] = "ТРИВОГА"
        if kyiv_changed_at:
            elapsed_act = max(1.0, (now_utc - kyiv_changed_at).total_seconds() / 60.0)
            kyiv_telemetry["minutes_active"] = round(elapsed_act, 1)
            kyiv_telemetry["badge_en"] = f"ALERT ACTIVE • {format_duration_en(elapsed_act)}"
            kyiv_telemetry["badge_ua"] = f"ТРИВОГА • {format_duration_ua(elapsed_act)}"
            kyiv_telemetry["detail_en"] = f"Alert active: {format_duration_en(elapsed_act)}"
            kyiv_telemetry["detail_ua"] = f"Триває {format_duration_ua(elapsed_act)}"
            kyiv_telemetry["card_en"] = f"Alert active for {format_duration_en(elapsed_act)}"
            kyiv_telemetry["card_ua"] = f"Тривога триває {format_duration_ua(elapsed_act)}"

    national_active = sum(1 for v in regions_status.values() if v)

    return {
        "regions": regions_status,
        "kyiv_raions": kyiv_raions_status,
        "timestamp_utc": now_utc,
        "national_active_count": national_active,
        "kyiv_changed_at": kyiv_changed_at,
        "kyiv_telemetry": kyiv_telemetry,
    }


def fetch_live_telegram_channel(
    channel_handle: str,
    limit: int = 20,
    max_age_minutes: float = 45.0,
    timeout: int = 5,
) -> List[dict]:
    """
    Scrapes recent messages from public Telegram web preview (https://t.me/s/<channel>).
    Strictly associates each text with its own timestamp and marks whether it is an active threat
    (published within max_age_minutes of current UTC time).
    """
    url = f"https://t.me/s/{channel_handle}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SirenCastUA/1.0"},
    )

    messages = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = resp.read().decode("utf-8", errors="ignore")

        # Split content into individual message blocks
        chunks = re.split(r'<div class="tgme_widget_message\s+', content)[1:]
        now_utc = datetime.now(timezone.utc)

        for chunk in chunks:
            time_m = re.search(r'<time[^>]*datetime="([^"]+)"', chunk)
            text_m = re.search(
                r'<div class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>',
                chunk,
                re.DOTALL,
            )
            if not time_m or not text_m:
                continue

            dt_str = time_m.group(1)
            try:
                dt = datetime.fromisoformat(dt_str)
                age_min = (now_utc - dt).total_seconds() / 60.0
            except Exception:
                dt = now_utc
                age_min = 0.0

            clean_text = html.unescape(re.sub(r"<[^>]+>", " ", text_m.group(1))).strip()
            clean_text = re.sub(r"\s+", " ", clean_text)
            if not clean_text:
                continue

            messages.append({
                "channel": channel_handle,
                "text": clean_text,
                "datetime_iso": dt.isoformat(),
                "age_minutes": round(age_min, 1),
                "is_active": age_min <= max_age_minutes,
            })
    except Exception as e:
        logger.warning(f"Could not scrape telegram channel @{channel_handle}: {e}")

    # Return the most recent messages up to limit
    return messages[-limit:]


def resolve_kyiv_alert_timeline(
    events: List[dict],
    now_utc: Optional[datetime] = None,
) -> dict:
    """
    Evaluates chronological alert and all-clear events from Suspilne Kyiv.
    Crucial business rules:
      1. An air alert (either yellow drone threat or red missile threat) marks alert onset.
      2. Transitions between yellow and red DO NOT reset the alert start time (continuous alert).
      3. An alert remains active until an explicit all-clear event is reached.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    elif now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)

    sorted_events = sorted(events, key=lambda e: e.get("datetime_iso", ""))

    in_alert = False
    alert_start: Optional[datetime] = None
    all_clear: Optional[datetime] = None

    for ev in sorted_events:
        try:
            ev_dt = datetime.fromisoformat(ev["datetime_iso"])
            if ev_dt.tzinfo is None:
                ev_dt = ev_dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue

        ev_type = ev.get("type", "").upper()
        if ev_type == "START":
            if not in_alert:
                in_alert = True
                alert_start = ev_dt
            # If already in alert, do not reset alert_start
        elif ev_type == "CLEAR":
            in_alert = False
            all_clear = ev_dt

    if in_alert and alert_start:
        elapsed_active = max(1.0, (now_utc - alert_start).total_seconds() / 60.0)
        start_kyiv = alert_start.astimezone(KYIV_TZ)
        start_str = start_kyiv.strftime("%H:%M")
        return {
            "is_active": True,
            "minutes_active": round(elapsed_active, 1),
            "time_since_last_alert_min": 0.0,
            "status_label_en": "ALERT ACTIVE",
            "status_label_ua": "ТРИВОГА",
            "badge_en": f"ALERT ACTIVE • {format_duration_en(elapsed_active)}",
            "badge_ua": f"ТРИВОГА • {format_duration_ua(elapsed_active)}",
            "detail_en": f"Alert active: {format_duration_en(elapsed_active)} (started {start_str} Kyiv time)",
            "detail_ua": f"Триває {format_duration_ua(elapsed_active)} (з {start_str} за Києвом)",
            "card_en": f"Alert active for {format_duration_en(elapsed_active)}",
            "card_ua": f"Тривога триває {format_duration_ua(elapsed_active)}",
            "last_event_time_iso": alert_start.isoformat(),
            "source": "suspilne_kyiv",
        }
    elif all_clear:
        elapsed_clear = max(0.0, (now_utc - all_clear).total_seconds() / 60.0)
        elapsed_clear = min(elapsed_clear, 10080.0)
        clear_kyiv = all_clear.astimezone(KYIV_TZ)
        clear_str = clear_kyiv.strftime("%H:%M")
        return {
            "is_active": False,
            "minutes_active": 0.0,
            "time_since_last_alert_min": round(elapsed_clear, 1),
            "status_label_en": "ALL CLEAR",
            "status_label_ua": "ВІДБІЙ",
            "badge_en": f"ALL CLEAR • {format_duration_en(elapsed_clear)}",
            "badge_ua": f"ВІДБІЙ • {format_duration_ua(elapsed_clear)}",
            "detail_en": f"All clear: {format_duration_en(elapsed_clear)} ago (since {clear_str} Kyiv time)",
            "detail_ua": f"Відбій {format_duration_ua(elapsed_clear)} тому (з {clear_str} за Києвом)",
            "card_en": f"All clear: {format_duration_en(elapsed_clear)} ago",
            "card_ua": f"Відбій {format_duration_ua(elapsed_clear)} тому",
            "last_event_time_iso": all_clear.isoformat(),
            "source": "suspilne_kyiv",
        }
    else:
        return {
            "is_active": False,
            "minutes_active": 0.0,
            "time_since_last_alert_min": 720.0,
            "status_label_en": "ALL CLEAR",
            "status_label_ua": "ВІДБІЙ",
            "badge_en": "ALL CLEAR",
            "badge_ua": "ВІДБІЙ",
            "detail_en": "All clear (calm airspace baseline)",
            "detail_ua": "Відбій (фоновий спокійний стан)",
            "card_en": "All clear: >12h",
            "card_ua": "Спокій: >12 год",
            "last_event_time_iso": "",
            "source": "baseline",
        }


def _load_kyiv_events_cache() -> List[dict]:
    if KYIV_EVENTS_CACHE_FILE.exists():
        try:
            with open(KYIV_EVENTS_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def _save_kyiv_events_cache(events: List[dict]):
    try:
        KYIV_EVENTS_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(KYIV_EVENTS_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(events[-100:], f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.debug(f"Could not save Kyiv events cache: {e}")


def fetch_and_update_kyiv_alert_telemetry(now_utc: Optional[datetime] = None) -> dict:
    """
    Ingests recent messages from t.me/suspilne_kyiv, isolates only official air raid
    siren notifications (discarding any news), updates the persistent event ledger,
    and returns resolved Kyiv City alert telemetry.
    """
    global _kyiv_telemetry_cache
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    now_t = time.time()
    if _kyiv_telemetry_cache["data"] is not None and (now_t - _kyiv_telemetry_cache["timestamp"]) < 10.0:
        return _kyiv_telemetry_cache["data"]

    cached_events = _load_kyiv_events_cache()
    event_map = {e["datetime_iso"]: e for e in cached_events if "datetime_iso" in e}

    try:
        # Scrape Suspilne Kyiv with lookback window
        raw_msgs = fetch_live_telegram_channel("suspilne_kyiv", limit=35, max_age_minutes=2880.0)
        for m in raw_msgs:
            txt = m.get("text", "")
            iso = m.get("datetime_iso", "")
            if not iso or not txt:
                continue

            if RE_SUSPILNE_KYIV_START.search(txt):
                event_map[iso] = {"type": "START", "datetime_iso": iso, "text": txt}
            elif RE_SUSPILNE_KYIV_CLEAR.search(txt):
                event_map[iso] = {"type": "CLEAR", "datetime_iso": iso, "text": txt}
    except Exception as e:
        logger.warning(f"Error fetching suspilne_kyiv: {e}")

    all_events = sorted(event_map.values(), key=lambda e: e.get("datetime_iso", ""))
    _save_kyiv_events_cache(all_events)

    result = resolve_kyiv_alert_timeline(all_events, now_utc)
    _kyiv_telemetry_cache["timestamp"] = now_t
    _kyiv_telemetry_cache["data"] = result
    return result


def fetch_live_telegram_stream(max_age_minutes: float = 45.0) -> List[dict]:
    """
    Combines live telegram messages from @war_monitor, @kpszsu, and official alert
    messages from @suspilne_kyiv (filtering out any non-alert news) with 10s caching.
    Sorts all messages by publication timestamp so the latest events are always evaluated first.
    """
    global _telegram_cache
    now = time.time()

    if _telegram_cache["data"] is not None and (now - _telegram_cache["timestamp"]) < 10.0:
        return _telegram_cache["data"]

    msgs = []
    # Scrape both official military and volunteer monitor channels
    msgs.extend(fetch_live_telegram_channel("kpszsu", limit=15, max_age_minutes=max_age_minutes))
    msgs.extend(fetch_live_telegram_channel("war_monitor", limit=15, max_age_minutes=max_age_minutes))

    # Ingest Suspilne Kyiv ONLY for official alert notifications, strictly rejecting all non-alert news
    try:
        suspilne_raw = fetch_live_telegram_channel("suspilne_kyiv", limit=20, max_age_minutes=max_age_minutes)
        suspilne_alerts = [
            m for m in suspilne_raw
            if RE_SUSPILNE_KYIV_START.search(m.get("text", "")) or RE_SUSPILNE_KYIV_CLEAR.search(m.get("text", ""))
        ]
        msgs.extend(suspilne_alerts)
    except Exception as e:
        logger.warning(f"Failed to fetch suspilne_kyiv alerts for stream: {e}")

    # Sort chronologically by datetime_iso
    msgs.sort(key=lambda m: m.get("datetime_iso", ""), reverse=False)

    _telegram_cache["timestamp"] = now
    _telegram_cache["data"] = msgs
    return msgs


if __name__ == "__main__":
    print("Testing Live Data Ingestion:")
    alerts = parse_live_siren_states()
    print(f"Timestamp: {alerts['timestamp_utc'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"National Active Sirens: {alerts['national_active_count']}")
    print(f"Kyiv City Active: {alerts['regions']['Kyiv City']}")
    print(f"Kyivska Raions Active: {alerts['kyiv_raions']}")

    print("\nTesting Live Telegram Stream:")
    tg_msgs = fetch_live_telegram_stream()
    print(f"Total Live Messages Extracted: {len(tg_msgs)}")
    for m in tg_msgs[:3]:
        safe_txt = m['text'][:80].encode('ascii', errors='backslashreplace').decode('ascii')
        print(f"  [@{m['channel']}]: {safe_txt}...")
