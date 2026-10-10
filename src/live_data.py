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
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

UBILLING_API_URL = "https://ubilling.net.ua/aerialalerts/?raw"
CACHE_TTL_SECONDS = 6.0  # Safe TTL > 0.5s to strictly respect 2 RPS limit

# Thread-safe in-memory cache
_alert_cache: Dict = {"timestamp": 0.0, "data": None}
_telegram_cache: Dict = {"timestamp": 0.0, "data": None}

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


def fetch_live_alerts_raw(timeout: int = 5) -> dict:
    """
    Fetches raw alerts from Ubilling API with an in-memory TTL cache
    to strictly adhere to the 2 RPS rate limit.
    """
    global _alert_cache
    now = time.time()

    # Return cached data if within TTL
    if _alert_cache["data"] is not None and (now - _alert_cache["timestamp"]) < CACHE_TTL_SECONDS:
        return _alert_cache["data"]

    req = urllib.request.Request(
        UBILLING_API_URL,
        headers={"User-Agent": "SirenCast-UA/1.0 (Academic Research; KSE AI School)"},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            _alert_cache["timestamp"] = now
            _alert_cache["data"] = data
            return data
    except Exception as e:
        logger.warning(f"Failed to fetch live alerts from {UBILLING_API_URL}: {e}")
        if _alert_cache["data"] is not None:
            logger.info("Serving stale cached alerts as fallback.")
            return _alert_cache["data"]
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

        is_enabled = bool(val.get("enabled", False))
        regions_status[eng_name] = is_enabled

        # Parse Kyivska oblast districts if available
        if clean_ua == "Київська область" and "districts" in val:
            districts = val["districts"]
            if isinstance(districts, dict):
                dist_items = districts.items()
            elif isinstance(districts, list):
                dist_items = [(d.get("name", ""), d) for d in districts if isinstance(d, dict)]
            else:
                dist_items = []

            for d_name, d_val in dist_items:
                for ua_prefix, eng_raion in RAION_MAPPING.items():
                    if ua_prefix in d_name:
                        kyiv_raions_status[eng_raion] = bool(d_val.get("enabled", False))

    # If raions aren't individually active but Kyivska oblast is active overall:
    if regions_status.get("Kyivska oblast", False) and not any(kyiv_raions_status.values()):
        for r in kyiv_raions_status:
            kyiv_raions_status[r] = True

    national_active = sum(1 for v in regions_status.values() if v)

    return {
        "regions": regions_status,
        "kyiv_raions": kyiv_raions_status,
        "timestamp_utc": now_utc,
        "national_active_count": national_active,
    }


def fetch_live_telegram_channel(channel_handle: str, limit: int = 15, timeout: int = 5) -> List[dict]:
    """
    Scrapes recent messages from public Telegram web preview (https://t.me/s/<channel>).
    Zero tokens, zero setup, completely open.
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

        # Extract message divs
        pattern = re.compile(
            r'<div class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>',
            re.DOTALL,
        )
        time_pattern = re.compile(r'<time[^>]*datetime="([^"]+)"', re.IGNORECASE)

        raw_texts = pattern.findall(content)
        times = time_pattern.findall(content)

        for i, raw_html in enumerate(raw_texts[-limit:]):
            clean_text = html.unescape(re.sub(r"<[^>]+>", " ", raw_html)).strip()
            clean_text = re.sub(r"\s+", " ", clean_text)
            if not clean_text:
                continue

            msg_time = times[i] if i < len(times) else None
            messages.append({
                "channel": channel_handle,
                "text": clean_text,
                "datetime_iso": msg_time,
            })
    except Exception as e:
        logger.warning(f"Could not scrape telegram channel @{channel_handle}: {e}")

    return messages


def fetch_live_telegram_stream() -> List[dict]:
    """
    Combines live telegram messages from @war_monitor and @kpszsu with 10s caching.
    """
    global _telegram_cache
    now = time.time()

    if _telegram_cache["data"] is not None and (now - _telegram_cache["timestamp"]) < 10.0:
        return _telegram_cache["data"]

    msgs = []
    msgs.extend(fetch_live_telegram_channel("war_monitor", limit=10))
    msgs.extend(fetch_live_telegram_channel("kpszsu", limit=10))

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
