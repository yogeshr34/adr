"""Synthetic training data generation and model training for the ADR classifier.

There is no licensed clinical/FAERS dataset bundled with this repo, so we
generate a large synthetic cohort whose labels follow a clinically-informed
risk function (age, polypharmacy, high-risk drugs, dosage, drug interactions
all increase ADR probability, matching real pharmacovigilance literature)
plus noise. The feature *extraction* (app/services/ml_pipeline.py) is fully
rule-based and real; only the training labels are synthetic.

To use a real dataset instead: build a DataFrame with columns matching
FEATURE_NAMES plus a binary `adr_label` column and pass it to train_and_save().
"""

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from app.config import MODEL_DIR
from app.services.ml_pipeline import FEATURE_NAMES

RNG = np.random.default_rng(42)


def make_synthetic_dataset(n_samples: int = 4000) -> pd.DataFrame:
    age = RNG.uniform(18, 90, n_samples)
    drug_count = np.clip(RNG.poisson(2.2, n_samples), 0, 10).astype(float)
    condition_count = np.clip(RNG.poisson(1.0, n_samples), 0, 5).astype(float)

    high_risk_drug_count = np.array([
        RNG.binomial(int(dc), 0.25) if dc > 0 else 0 for dc in drug_count
    ], dtype=float)

    base_dosage = RNG.uniform(0.2, 1.0, n_samples)
    overdose_mask = RNG.uniform(0, 1, n_samples) < 0.12
    max_dosage_ratio = np.where(
        overdose_mask, RNG.uniform(1.0, 2.5, n_samples), base_dosage
    )

    interaction_prob = np.clip(0.08 * drug_count + 0.15 * high_risk_drug_count, 0, 0.9)
    has_interaction = RNG.uniform(0, 1, n_samples) < interaction_prob
    interaction_score = np.where(
        has_interaction, RNG.uniform(0.4, 1.0, n_samples), RNG.uniform(0, 0.1, n_samples)
    )

    polypharmacy = (drug_count >= 5).astype(float)

    logit = (
        -3.6
        + 0.018 * (age - 40)
        + 0.30 * drug_count
        + 0.85 * high_risk_drug_count
        + 1.15 * interaction_score
        + 0.9 * np.clip(max_dosage_ratio - 1.0, 0, None)
        + 0.45 * condition_count
        + 0.35 * polypharmacy
        + RNG.normal(0, 0.5, n_samples)
    )
    prob = 1 / (1 + np.exp(-logit))
    label = RNG.binomial(1, prob)

    df = pd.DataFrame({
        "age": age,
        "drug_count": drug_count,
        "condition_count": condition_count,
        "high_risk_drug_count": high_risk_drug_count,
        "max_dosage_ratio": max_dosage_ratio,
        "interaction_score": interaction_score,
        "polypharmacy": polypharmacy,
        "adr_label": label,
    })
    assert list(df.columns[:-1]) == FEATURE_NAMES, "Feature order must match ml_pipeline.FEATURE_NAMES"
    return df


def train_and_save(df: pd.DataFrame) -> dict:
    X = df[FEATURE_NAMES].values
    y = df["adr_label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "f1": float(f1_score(y_test, y_pred)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate": float(y.mean()),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "adr_model.joblib")

    metadata = {
        "feature_names": FEATURE_NAMES,
        "feature_importances": dict(zip(FEATURE_NAMES, model.feature_importances_.tolist())),
        "metrics": metrics,
        "model_type": "RandomForestClassifier",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_data": "synthetic, clinically-informed (see app/services/model_training.py docstring)",
    }
    with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def train_default() -> dict:
    return train_and_save(make_synthetic_dataset())
