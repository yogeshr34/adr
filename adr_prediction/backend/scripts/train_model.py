"""CLI entry point to (re)train and save the ADR risk classifier.

Usage (from adr_prediction/backend):
    python scripts/train_model.py

See app/services/model_training.py for the actual dataset generation and
training logic (also used as an automatic fallback if the app starts up
without a model artifact present).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.model_training import train_default  # noqa: E402

if __name__ == "__main__":
    metadata = train_default()
    print("Training complete.")
    print(json.dumps(metadata["metrics"], indent=2))
