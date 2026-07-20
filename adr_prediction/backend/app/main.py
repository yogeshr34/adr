"""FastAPI application: PDF prescription upload -> ADR risk report."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import ALLOWED_ORIGINS, MAX_UPLOAD_BYTES, MAX_UPLOAD_MB
from app.schemas import AnalyzePrescriptionResponse, DrugEntry, HealthResponse, InteractionEntry
from app.services import ml_pipeline, prescription_nlp
from app.services.pdf_extract import PDFExtractionError, extract_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    ml_pipeline.load_model()
    if not ml_pipeline.is_model_loaded():
        logger.error(
            "ADR model not found. Run `python scripts/train_model.py` inside "
            "adr_prediction/backend before starting the API."
        )
    yield


app = FastAPI(
    title="ADR Prediction API",
    description="Upload a prescription PDF and receive an adverse drug reaction (ADR) risk report.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if ml_pipeline.is_model_loaded() else "degraded",
        model_loaded=ml_pipeline.is_model_loaded(),
        model_version=ml_pipeline.model_version(),
    )


@app.post("/analyze-prescription", response_model=AnalyzePrescriptionResponse)
async def analyze_prescription(file: UploadFile = File(...)) -> AnalyzePrescriptionResponse:
    if file.content_type not in ("application/pdf", "application/x-pdf") and not (
        file.filename or ""
    ).lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are accepted.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_MB}MB limit.")

    if not ml_pipeline.is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail="ADR model is not loaded on the server. Contact the administrator.",
        )

    try:
        extraction = extract_text(file_bytes)
    except PDFExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    parsed = prescription_nlp.parse(extraction["text"])

    warnings: list[str] = []
    if not parsed.drugs:
        warnings.append(
            "No recognized drug names were found in this document. Risk score reflects "
            "patient/context features only and may be unreliable."
        )
    if parsed.age_estimated:
        warnings.append("Patient age not found in document; used population default (55).")

    result = ml_pipeline.analyze(parsed)

    drug_list = [
        DrugEntry(
            name=d.name,
            dosage_mg=d.dosage_mg,
            frequency=d.frequency,
            high_risk=ml_pipeline.is_high_risk_drug(d.key),
        )
        for d in parsed.drugs
    ]

    return AnalyzePrescriptionResponse(
        drug_list=drug_list,
        adr_risk_score=result["prediction"]["adr_risk_score"],
        severity=result["prediction"]["severity"],
        confidence=result["prediction"]["confidence"],
        prr=result["prr"]["prr"],
        prr_signal=result["prr"]["prr_signal"],
        shap_values=result["shap_values"],
        interactions=[InteractionEntry(**entry) for entry in result["interactions"]],
        recommendation=result["recommendation"],
        patient_age=parsed.age,
        age_estimated=parsed.age_estimated,
        conditions=parsed.conditions,
        extraction_method=extraction["method"],
        warnings=warnings,
    )
