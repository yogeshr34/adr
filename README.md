# ADR Prediction System (PDF → ML → SHAP → Dashboard)

Upload a prescription PDF and get back a structured adverse drug reaction
(ADR) risk report: risk score, severity, confidence, a PRR-style
pharmacovigilance signal, detected drug-drug interactions, and a SHAP
explanation of *why* the model scored it that way.

Everything the dashboard shows comes from the live API response — no mock
data.

## How it works

```
Prescription PDF
    │  PyMuPDF text extraction (OCR fallback via Tesseract for scans)
    ▼
Drug / dosage / age / condition extraction (rule-based NLP)
    │
    ▼
Feature engineering (drug count, high-risk drugs, dosage ratio,
                      interaction severity, polypharmacy, ...)
    │
    ▼
RandomForest ADR risk classifier  →  SHAP explanation
    │
    ▼
JSON report  →  Next.js dashboard
```

- **Backend**: FastAPI (`adr_prediction/backend`) — `POST /analyze-prescription`
- **Frontend**: Next.js 16 + Tailwind (`adr_prediction/frontend`)
- **Model**: scikit-learn `RandomForestClassifier`, trained on a synthetic,
  clinically-informed dataset (see
  [`app/services/model_training.py`](adr_prediction/backend/app/services/model_training.py)
  for the exact risk function). The feature *extraction* is fully rule-based
  and real; swap in a licensed FAERS/clinical dataset by replacing the
  training data — the rest of the pipeline is unaffected.
- **Explainability**: SHAP `TreeExplainer` on the trained forest.
- **Drug knowledge base**: curated reference of ~60 common drugs, dosage
  ceilings, and known dangerous interaction pairs
  ([`app/data/drug_reference.json`](adr_prediction/backend/app/data/drug_reference.json)).

## Project layout

```
adr_prediction/
  backend/
    app/
      main.py              # FastAPI app: /health, /analyze-prescription
      services/
        pdf_extract.py      # PyMuPDF + OCR fallback
        prescription_nlp.py # drug/dosage/age/condition extraction
        ml_pipeline.py      # feature engineering, prediction, SHAP, PRR proxy
        model_training.py   # synthetic dataset + RandomForest training
        prr_calculator.py   # aggregate PRR calc for real FAERS-style data
      data/                 # drug + condition reference JSON
    models/                 # trained artifact (generated, not committed)
    scripts/
      train_model.py         # CLI: retrain the model
      generate_sample_pdf.py # writes sample PDFs to tests/fixtures/
    tests/
    Dockerfile
    render.yaml
  frontend/
    app/                    # Next.js App Router pages
    components/dashboard/   # upload form + report UI
    lib/api.ts              # typed client for the backend API
```

## Local development

### Backend

```bash
cd adr_prediction/backend
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

python scripts/train_model.py   # trains and saves models/adr_model.joblib
uvicorn app.main:app --reload --port 8000
```

Visit `http://localhost:8000/health` — `{"status":"ok","model_loaded":true}`.
(If you skip the training step, the API trains a model automatically on
first startup, but running it explicitly is faster and lets you inspect the
printed metrics.)

Generate sample prescription PDFs for manual testing:

```bash
python scripts/generate_sample_pdf.py
# writes tests/fixtures/low_risk.pdf and tests/fixtures/high_risk_interaction.pdf
```

Run the test suite:

```bash
pytest
```

### Frontend

```bash
cd adr_prediction/frontend
npm install
cp .env.local.example .env.local   # NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev
```

Open `http://localhost:3000`, upload a prescription PDF, and the dashboard
renders the live risk report.

## API

### `POST /analyze-prescription`

```bash
curl -X POST "http://localhost:8000/analyze-prescription" \
  -F "file=@/path/to/prescription.pdf"
```

Response shape:

```json
{
  "drug_list": [{"name": "Warfarin", "dosage_mg": 5.0, "frequency": "once daily", "high_risk": true}],
  "adr_risk_score": 0.91,
  "severity": "High",
  "confidence": 0.91,
  "prr": 4.1,
  "prr_signal": true,
  "shap_values": {"age": 0.03, "drug_count": 0.11, "...": 0.0},
  "interactions": [{"drugs": ["Warfarin", "Ibuprofen"], "severity": 0.85, "reason": "Additive bleeding risk / GI bleed"}],
  "recommendation": "Avoid combination: Warfarin + Ibuprofen (...). Consider alternative therapy.",
  "patient_age": 74,
  "age_estimated": false,
  "conditions": ["hypertension"],
  "extraction_method": "embedded",
  "warnings": []
}
```

If the PDF is scanned and Tesseract isn't installed on the server, the API
returns a `422` explaining that OCR is unavailable.

### `GET /health`

`{"status": "ok", "model_loaded": true, "model_version": "<trained_at ISO timestamp>"}`

## Deployment

### Backend → Render

The backend needs `tesseract-ocr` as a system package, so it deploys as a
Docker web service (Render's native Python buildpack can't install system
packages).

1. Push this repo to GitHub.
2. In Render: **New → Web Service** → connect the repo.
3. Render will detect [`adr_prediction/backend/render.yaml`](adr_prediction/backend/render.yaml)
   (Blueprint deploy), or set manually:
   - Root directory: `adr_prediction/backend`
   - Environment: Docker
   - Health check path: `/health`
4. Set env var `ADR_ALLOWED_ORIGINS` to your Vercel frontend URL once you
   have it (comma-separated if you need more than one origin).
5. Deploy. The Docker build trains the model at build time, so the service
   starts serving immediately — no separate training step needed.
6. Note the resulting API URL (e.g. `https://adr-prediction-api.onrender.com`).

### Frontend → Vercel

1. In Vercel: **Add New → Project** → import the same repo.
2. Set **Root Directory** to `adr_prediction/frontend` (the root
   [`vercel.json`](vercel.json) also configures this if you deploy from the
   repo root via the CLI).
3. Add environment variable `NEXT_PUBLIC_API_URL` = your Render backend URL
   from above.
4. Deploy.

Once both are live, update `ADR_ALLOWED_ORIGINS` on Render to the real
Vercel domain (instead of `*`) to lock down CORS.

## Notes / limitations

- The ADR classifier is trained on synthetic (not real patient) data — see
  the [model_training.py](adr_prediction/backend/app/services/model_training.py)
  docstring for the exact assumptions. This is decision-support tooling, not
  a certified clinical device.
- Drug name recognition and interaction detection are limited to the curated
  reference table in `app/data/drug_reference.json`. Unrecognized drug names
  are ignored (with a warning surfaced in the API response), not guessed.
- OCR fallback requires the Tesseract binary on the host; the Docker image
  installs it automatically.
