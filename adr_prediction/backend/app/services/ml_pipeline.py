"""Feature engineering, model inference, and SHAP explainability for ADR risk."""

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from app.config import DATA_DIR, MODEL_DIR
from app.services.prescription_nlp import DrugMention, ParsedPrescription

logger = logging.getLogger(__name__)

with open(DATA_DIR / "drug_reference.json", encoding="utf-8") as f:
    _DRUG_REF = json.load(f)

_DRUGS = _DRUG_REF["drugs"]
_INTERACTIONS = _DRUG_REF["interactions"]

FEATURE_NAMES = [
    "age",
    "drug_count",
    "condition_count",
    "high_risk_drug_count",
    "max_dosage_ratio",
    "interaction_score",
    "polypharmacy",
]

_model = None
_shap_explainer = None
_metadata: dict = {}


def _model_path() -> Path:
    return MODEL_DIR / "adr_model.joblib"


def _metadata_path() -> Path:
    return MODEL_DIR / "metadata.json"


def is_model_loaded() -> bool:
    return _model is not None


def model_version() -> Optional[str]:
    return _metadata.get("trained_at") if _metadata else None


def is_high_risk_drug(key: str) -> bool:
    return bool(_DRUGS.get(key, {}).get("high_risk"))


def load_model() -> None:
    """Load the trained model artifact into module-level singletons."""
    global _model, _shap_explainer, _metadata

    path = _model_path()
    if not path.exists():
        logger.warning("No model artifact found at %s. Training a fresh one now...", path)
        from app.services.model_training import train_default

        train_default()

    _model = joblib.load(path)

    meta_path = _metadata_path()
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            _metadata = json.load(f)

    try:
        import shap

        _shap_explainer = shap.TreeExplainer(_model)
    except Exception as exc:  # pragma: no cover - SHAP is optional at runtime
        logger.warning("SHAP explainer unavailable: %s", exc)
        _shap_explainer = None


def _interaction_pairs(drug_keys: set[str]) -> list[dict]:
    detected = []
    for entry in _INTERACTIONS:
        a, b = entry["drugs"]
        if a in drug_keys and b in drug_keys:
            detected.append(entry)
    return detected


def _max_dosage_ratio(drugs: list[DrugMention]) -> float:
    ratios = []
    for drug in drugs:
        if drug.dosage_mg is None:
            continue
        ref = _DRUGS.get(drug.key, {})
        max_mg = ref.get("typical_max_mg")
        if max_mg:
            ratios.append(drug.dosage_mg / max_mg)
    return round(min(max(ratios), 3.0), 4) if ratios else 0.0


def extract_features(parsed: ParsedPrescription) -> dict:
    """Turn a parsed prescription into the model's numeric feature vector."""
    drug_keys = {d.key for d in parsed.drugs}
    high_risk_count = sum(1 for k in drug_keys if _DRUGS.get(k, {}).get("high_risk"))
    interactions = _interaction_pairs(drug_keys)

    interaction_score = 0.0
    if interactions:
        interaction_score = min(
            max(entry["severity"] for entry in interactions) + 0.1 * (len(interactions) - 1),
            1.0,
        )

    features = {
        "age": float(parsed.age),
        "drug_count": float(len(drug_keys)),
        "condition_count": float(len(parsed.conditions)),
        "high_risk_drug_count": float(high_risk_count),
        "max_dosage_ratio": _max_dosage_ratio(parsed.drugs),
        "interaction_score": float(interaction_score),
        "polypharmacy": 1.0 if len(drug_keys) >= 5 else 0.0,
    }
    return features


def _feature_vector(features: dict) -> np.ndarray:
    return np.array([[features[name] for name in FEATURE_NAMES]], dtype=float)


def predict(features: dict) -> dict:
    """Return the ADR risk prediction for a single feature vector."""
    if _model is None:
        raise RuntimeError("Model is not loaded. Run scripts/train_model.py first.")

    X = _feature_vector(features)
    proba = _model.predict_proba(X)[0]
    risk_score = float(proba[1])
    confidence = float(max(proba))

    if risk_score < 0.33:
        severity = "Low"
    elif risk_score < 0.66:
        severity = "Moderate"
    else:
        severity = "High"

    return {
        "adr_risk_score": round(risk_score, 4),
        "severity": severity,
        "confidence": round(confidence, 4),
    }


def explain(features: dict) -> dict:
    """Return per-feature SHAP contributions for the positive (ADR) class."""
    if _shap_explainer is None:
        return {}

    X = _feature_vector(features)
    try:
        raw = _shap_explainer.shap_values(X)
    except Exception as exc:  # pragma: no cover
        logger.warning("SHAP explanation failed: %s", exc)
        return {}

    # Normalize across SHAP/sklearn version differences in output shape.
    values = np.asarray(raw)
    if values.ndim == 3:
        # shape (n_samples, n_features, n_classes) -> take positive class
        row = values[0, :, 1]
    elif isinstance(raw, list):
        # older SHAP: list of per-class arrays, each (n_samples, n_features)
        row = np.asarray(raw[1])[0]
    else:
        row = values[0]

    return {name: round(float(val), 5) for name, val in zip(FEATURE_NAMES, row)}


def prr_proxy(parsed: ParsedPrescription) -> dict:
    """Approximate a Proportional Reporting Ratio signal for the drug list.

    This mirrors app/services/prr_calculator.py's PRR concept but, since a
    single prescription has no aggregate report counts, uses the curated
    per-drug `prr_signal` reference values as a stand-in signal strength.
    Swap in prr_calculator.calculate_prr() against real FAERS-style report
    data for production-grade pharmacovigilance signal detection.
    """
    signals = [
        _DRUGS[d.key]["prr_signal"] for d in parsed.drugs if d.key in _DRUGS
    ]
    if not signals:
        return {"prr": 1.0, "prr_signal": False, "basis": "no reference drugs identified"}

    value = round(max(signals), 2)
    return {
        "prr": value,
        "prr_signal": value >= 2.0,
        "basis": "max per-drug reference PRR proxy (FDA threshold: PRR >= 2.0)",
    }


def build_recommendation(severity: str, interactions: list[dict]) -> str:
    if interactions:
        top = max(interactions, key=lambda e: e["severity"])
        a, b = top["drugs"]
        a_name = a.replace("_", " ").title()
        b_name = b.replace("_", " ").title()
        return f"Avoid combination: {a_name} + {b_name} ({top['reason']}). Consider alternative therapy."
    if severity == "High":
        return "High ADR risk. Consult prescribing physician before dispensing; consider dose adjustment or alternative agents."
    if severity == "Moderate":
        return "Moderate ADR risk. Monitor patient closely for adverse reactions after starting therapy."
    return "Low ADR risk. Routine monitoring is sufficient."


def analyze(parsed: ParsedPrescription) -> dict:
    """Full pipeline: features -> prediction -> explanation -> report."""
    features = extract_features(parsed)
    prediction = predict(features)
    shap_values = explain(features)
    prr = prr_proxy(parsed)
    drug_keys = {d.key for d in parsed.drugs}
    interactions = _interaction_pairs(drug_keys)
    recommendation = build_recommendation(prediction["severity"], interactions)

    return {
        "prediction": prediction,
        "features": features,
        "shap_values": shap_values,
        "prr": prr,
        "interactions": [
            {
                "drugs": [d.replace("_", " ").title() for d in entry["drugs"]],
                "severity": entry["severity"],
                "reason": entry["reason"],
            }
            for entry in interactions
        ],
        "recommendation": recommendation,
    }
