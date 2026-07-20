"""Pydantic response models for the ADR prediction API."""

from typing import Optional

from pydantic import BaseModel


class DrugEntry(BaseModel):
    name: str
    dosage_mg: Optional[float] = None
    frequency: Optional[str] = None
    high_risk: bool = False


class InteractionEntry(BaseModel):
    drugs: list[str]
    severity: float
    reason: str


class AnalyzePrescriptionResponse(BaseModel):
    drug_list: list[DrugEntry]
    adr_risk_score: float
    severity: str
    confidence: float
    prr: float
    prr_signal: bool
    shap_values: dict[str, float]
    interactions: list[InteractionEntry]
    recommendation: str
    patient_age: int
    age_estimated: bool
    conditions: list[str]
    extraction_method: str
    warnings: list[str] = []


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
