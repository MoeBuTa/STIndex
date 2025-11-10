"""
Constants for STIndex project.

Defines project directories, API endpoints, and other constants.
"""

from pathlib import Path

# Project directories
PROJECT_DIR = Path(__file__).resolve().parents[1].resolve().parents[0]

DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = str(DATA_DIR)

LOG_DIR = str(PROJECT_DIR / "data" / "logs")
OUTPUT_DIR = PROJECT_DIR / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = str(OUTPUT_DIR)

CFG_DIR = PROJECT_DIR / "cfg"
CFG_EXTRACTION_INFERENCE_DIR = CFG_DIR / "extraction" / "inference"
CFG_EXTRACTION_TRAINING_DIR = CFG_DIR / "extraction" / "training"
CFG_EXTRACTION_POSTPROCESS_DIR = CFG_DIR / "extraction" / "postprocess"
CFG_EXTRACTION_EVALUATION_DIR = CFG_DIR / "extraction" / "evaluation"
CFG_PREPROCESS_DIR = CFG_DIR / "preprocess"

# Cache directories
CACHE_DIR = Path.home() / ".stindex"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

GEOCODE_CACHE_DIR = CACHE_DIR / "geocode_cache"
GEOCODE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# API endpoints and services
NOMINATIM_ENDPOINT = "https://nominatim.openstreetmap.org/search"
DEFAULT_USER_AGENT = "stindex-spatiotemporal-extraction/1.0"

# Rate limiting (seconds)
NOMINATIM_RATE_LIMIT = 1.0
GEOCODER_REQUEST_TIMEOUT = 10.0

# Default model settings
DEFAULT_LLM_PROVIDER = "openai"
DEFAULT_MODEL_NAME = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 2048

# Extraction defaults
DEFAULT_MIN_CONFIDENCE = 0.5
DEFAULT_ENABLE_CACHE = True

# Spatial extraction
SPACY_MODEL = "en_core_web_sm"
DEFAULT_GEOCODER = "nominatim"

# Temporal extraction
ISO_8601_DATE_FORMAT = "%Y-%m-%d"
ISO_8601_DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
