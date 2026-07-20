"""Application configuration, sourced from environment variables."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = Path(os.getenv("ADR_MODEL_DIR", BASE_DIR.parent / "models"))

MAX_UPLOAD_MB = float(os.getenv("ADR_MAX_UPLOAD_MB", "15"))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)

# Comma-separated list of allowed CORS origins; "*" allows any origin.
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ADR_ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]

MIN_EMBEDDED_TEXT_CHARS = int(os.getenv("ADR_MIN_EMBEDDED_TEXT_CHARS", "20"))
OCR_DPI = int(os.getenv("ADR_OCR_DPI", "200"))
