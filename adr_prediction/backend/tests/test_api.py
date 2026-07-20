from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is True


def test_analyze_low_risk(client):
    with open(FIXTURES / "low_risk.pdf", "rb") as f:
        resp = client.post(
            "/analyze-prescription", files={"file": ("low_risk.pdf", f, "application/pdf")}
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["severity"] == "Low"
    assert len(body["drug_list"]) == 2
    assert body["interactions"] == []


def test_analyze_high_risk_interaction(client):
    with open(FIXTURES / "high_risk_interaction.pdf", "rb") as f:
        resp = client.post(
            "/analyze-prescription",
            files={"file": ("high_risk_interaction.pdf", f, "application/pdf")},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["severity"] in ("Moderate", "High")
    assert any("Warfarin" in i["drugs"] for i in body["interactions"])
    assert body["prr_signal"] is True


def test_rejects_non_pdf(client):
    resp = client.post(
        "/analyze-prescription",
        files={"file": ("note.txt", b"not a pdf", "text/plain")},
    )
    assert resp.status_code == 415
