"""Rule-based extraction of drugs, dosages, patient age, and conditions from
prescription text."""

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from app.config import DATA_DIR

with open(DATA_DIR / "drug_reference.json", encoding="utf-8") as f:
    _DRUG_REF = json.load(f)

with open(DATA_DIR / "condition_reference.json", encoding="utf-8") as f:
    _CONDITION_REF = json.load(f)["conditions"]

# Map every alias -> canonical drug key, longest alias first so multi-word
# aliases ("insulin glargine") are matched before shorter ones ("insulin").
_ALIAS_TO_KEY: list[tuple[str, str]] = []
for key, info in _DRUG_REF["drugs"].items():
    for alias in info["aliases"]:
        _ALIAS_TO_KEY.append((alias.lower(), key))
_ALIAS_TO_KEY.sort(key=lambda pair: len(pair[0]), reverse=True)

_DOSAGE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(mcg|mg|milligrams?|g|grams?|ml|milliliters?|iu|units?)\b",
    re.IGNORECASE,
)
_UNIT_TO_MG = {"mcg": 0.001, "mg": 1, "milligram": 1, "milligrams": 1,
               "g": 1000, "gram": 1000, "grams": 1000,
               "ml": 1, "milliliter": 1, "milliliters": 1,
               "iu": 1, "unit": 1, "units": 1}

_FREQUENCY_PATTERNS = {
    "once daily": r"\bonce\s+(a\s+)?day\b|\bod\b|\bqd\b|\bonce\s+daily\b",
    "twice daily": r"\btwice\s+(a\s+)?day\b|\bbid\b|\btwice\s+daily\b",
    "three times daily": r"\bthree\s+times\s+(a\s+)?day\b|\btid\b",
    "four times daily": r"\bfour\s+times\s+(a\s+)?day\b|\bqid\b",
    "as needed": r"\bprn\b|\bas\s+needed\b",
}

_AGE_PATTERNS = [
    re.compile(r"\bage\s*[:\-=]?\s*(\d{1,3})\b", re.IGNORECASE),
    re.compile(r"\bage\s*\(\s*(\d{1,3})\s*\)", re.IGNORECASE),
    re.compile(r"\b(\d{1,3})\s*(?:years?[\s-]?old|yrs?[\s-]?old|y/o|yo)\b", re.IGNORECASE),
]


@dataclass
class DrugMention:
    key: str
    name: str
    dosage_mg: Optional[float] = None
    frequency: Optional[str] = None
    raw_line: str = ""


@dataclass
class ParsedPrescription:
    drugs: list[DrugMention] = field(default_factory=list)
    age: Optional[int] = None
    age_estimated: bool = True
    conditions: list[str] = field(default_factory=list)
    raw_text: str = ""


def _extract_age(text: str) -> tuple[Optional[int], bool]:
    for pattern in _AGE_PATTERNS:
        match = pattern.search(text)
        if match:
            age = int(match.group(1))
            if 0 < age < 120:
                return age, False
    return None, True


def _extract_conditions(text: str) -> list[str]:
    lower = text.lower()
    found = []
    for condition, keywords in _CONDITION_REF.items():
        if any(kw in lower for kw in keywords):
            found.append(condition)
    return found


def _extract_frequency(line: str) -> Optional[str]:
    for label, pattern in _FREQUENCY_PATTERNS.items():
        if re.search(pattern, line, re.IGNORECASE):
            return label
    return None


def _extract_dosage_mg(line: str) -> Optional[float]:
    match = _DOSAGE_RE.search(line)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).lower()
    multiplier = _UNIT_TO_MG.get(unit, 1)
    return round(value * multiplier, 4)


def _find_drugs_in_line(line: str) -> list[str]:
    lower = line.lower()
    matched_keys = []
    for alias, key in _ALIAS_TO_KEY:
        if key in matched_keys:
            continue
        if re.search(rf"\b{re.escape(alias)}\b", lower):
            matched_keys.append(key)
    return matched_keys


def parse(text: str) -> ParsedPrescription:
    """Parse raw prescription text into structured drug/patient data."""
    age, age_estimated = _extract_age(text)
    conditions = _extract_conditions(text)

    seen_keys: set[str] = set()
    drugs: list[DrugMention] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        for key in _find_drugs_in_line(stripped):
            if key in seen_keys:
                continue
            seen_keys.add(key)
            drugs.append(
                DrugMention(
                    key=key,
                    name=key.replace("_", " ").title(),
                    dosage_mg=_extract_dosage_mg(stripped),
                    frequency=_extract_frequency(stripped),
                    raw_line=stripped,
                )
            )

    # Fallback: scan the whole document (not line-by-line) for any drugs
    # missed because the layout doesn't put dosage/name on one line.
    if not drugs:
        for key in _find_drugs_in_line(text):
            if key in seen_keys:
                continue
            seen_keys.add(key)
            drugs.append(DrugMention(key=key, name=key.replace("_", " ").title()))

    return ParsedPrescription(
        drugs=drugs,
        age=age if age is not None else 55,
        age_estimated=age_estimated,
        conditions=conditions,
        raw_text=text,
    )
