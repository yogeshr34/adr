"""Generate sample prescription PDFs for manual/local testing."""

from pathlib import Path

import fitz

OUT_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures"

SAMPLES = {
    "low_risk.pdf": """Prescription
Patient: Jane Doe
Age: 29

Rx:
1. Amoxicillin 500 mg - three times daily
2. Acetaminophen 500 mg - as needed
""",
    "high_risk_interaction.pdf": """Prescription
Patient: John Smith
Age: 74
History: hypertension, atrial fibrillation

Rx:
1. Warfarin 5 mg - once daily
2. Ibuprofen 400 mg - twice daily
3. Lisinopril 20 mg - once daily
4. Spironolactone 25 mg - once daily
5. Digoxin 0.25 mg - once daily
6. Furosemide 40 mg - once daily
""",
}


def write_pdf(text: str, path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    doc.close()


if __name__ == "__main__":
    for filename, text in SAMPLES.items():
        write_pdf(text, OUT_DIR / filename)
        print(f"Wrote {OUT_DIR / filename}")
