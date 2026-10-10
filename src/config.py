"""
Configuration settings for SirenCast UA.
"""
import sys
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_OFFICIAL_DATA = DATA_DIR / "official_data_en.csv"
RAW_VOLUNTEER_DATA = DATA_DIR / "volunteer_data_en.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_ALERTS_PARQUET = PROCESSED_DATA_DIR / "clean_alerts.parquet"
PROCESSED_FEATURES_PARQUET = PROCESSED_DATA_DIR / "features_30m.parquet"
PROCESSED_MULTIMODAL_PARQUET = PROCESSED_DATA_DIR / "features_multimodal_30m.parquet"
TELEGRAM_DATA_DIR = DATA_DIR / "telegram"
MONITOR_JSON = TELEGRAM_DATA_DIR / "monitor.json"
PSZSU_JSON = TELEGRAM_DATA_DIR / "pszsu.json"
CLEAN_TELEGRAM_PARQUET = DATA_DIR / "clean_telegram.parquet"
ARTIFACTS_DIR = BASE_DIR / ".agents" / "artifacts"

# Time Series Settings
TIME_STEP_MINUTES = 30
TIME_FREQ = f"{TIME_STEP_MINUTES}min"

# Target Definition
TARGET_REGION = "Kyiv City"

# Spatial Radar Hierarchy
INNER_RING_RAIONS = [
    "Vyshhorodskyi raion",
    "Brovarskyi raion",
    "Boryspilskyi raion",
    "Obukhivskyi raion",
    "Bilotserkivskyi raion",
    "Fastivskyi raion",
    "Buchanskyi raion",
]

ADJACENT_OBLASTS = [
    "Chernihivska oblast",
    "Sumska oblast",
    "Poltavska oblast",
    "Cherkaska oblast",
    "Zhytomyrska oblast",
]

EASTERN_FLANK_OBLASTS = [
    "Kharkivska oblast",
    "Dnipropetrovska oblast",
    "Zaporizka oblast",
    "Donetska oblast",
]

SOUTHERN_FLANK_OBLASTS = [
    "Odeska oblast",
    "Mykolaivska oblast",
    "Khersonska oblast",
]

ALL_OBLASTS = [
    "Cherkaska oblast",
    "Chernihivska oblast",
    "Chernivetska oblast",
    "Dnipropetrovska oblast",
    "Donetska oblast",
    "Ivano-Frankivska oblast",
    "Kharkivska oblast",
    "Khersonska oblast",
    "Khmelnytska oblast",
    "Kirovohradska oblast",
    "Kyiv City",
    "Kyivska oblast",
    "Luhanska oblast",
    "Lvivska oblast",
    "Mykolaivska oblast",
    "Odeska oblast",
    "Poltavska oblast",
    "Rivnenska oblast",
    "Sumska oblast",
    "Ternopilska oblast",
    "Vinnytska oblast",
    "Volynska oblast",
    "Zakarpatska oblast",
    "Zaporizka oblast",
    "Zhytomyrska oblast",
]
