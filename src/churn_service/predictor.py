"""Load the persisted model bundle and turn validated requests into predictions."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from churn_service.schemas import Customer, Prediction


def risk_band(probability: float, threshold: float) -> str:
    """Bucket a probability relative to the decision threshold.

    high   : at or above the threshold (customer is flagged)
    medium : at least half the threshold
    low    : everything else
    """
    if probability >= threshold:
        return "high"
    if probability >= threshold / 2:
        return "medium"
    return "low"


class ChurnPredictor:
    """Thin wrapper around the trained pipeline and its metadata."""

    def __init__(self, bundle: dict[str, Any]) -> None:
        self.pipeline = bundle["pipeline"]
        self.features: list[str] = list(bundle["features"])
        self.report: dict[str, Any] = bundle["report"]
        self.threshold: float = float(self.report["threshold"])
        self.version: str = self.report["model_version"]

    @classmethod
    def load(cls, path: str | Path) -> ChurnPredictor:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"model artifact not found: {path}")
        return cls(joblib.load(path))

    def predict(self, customers: Sequence[Customer]) -> list[Prediction]:
        # Column order comes from the bundle, so the request field order never matters.
        frame = pd.DataFrame([c.model_dump() for c in customers])[self.features]
        proba = self.pipeline.predict_proba(frame)[:, 1]
        return [
            Prediction(
                churn_probability=round(float(p), 4),
                will_churn=bool(p >= self.threshold),
                risk_band=risk_band(float(p), self.threshold),
            )
            for p in proba
        ]
