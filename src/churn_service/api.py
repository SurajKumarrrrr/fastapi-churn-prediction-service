"""FastAPI app serving the churn model.

Run locally: ``uvicorn churn_service.api:app --reload``
The model path is read from ``CHURN_MODEL_PATH`` (default ``artifacts/model.joblib``).
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status

from churn_service import __version__
from churn_service.predictor import ChurnPredictor
from churn_service.schemas import (
    BatchRequest,
    BatchResponse,
    Customer,
    Health,
    ModelInfo,
    PredictionResponse,
)

logger = logging.getLogger("churn_service")
DEFAULT_MODEL_PATH = "artifacts/model.joblib"


def create_app(model_path: str | None = None) -> FastAPI:
    """App factory: makes it easy to point tests at a temporary model."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        path = model_path or os.getenv("CHURN_MODEL_PATH", DEFAULT_MODEL_PATH)
        try:
            app.state.predictor = ChurnPredictor.load(path)
            logger.info("loaded model %s from %s", app.state.predictor.version, path)
        except FileNotFoundError:
            # Start anyway so /health can report the problem to the orchestrator.
            app.state.predictor = None
            logger.error("no model at %s; prediction endpoints return 503", path)
        yield

    app = FastAPI(
        title="Churn Prediction Service",
        version=__version__,
        description="Scores subscription customers for churn risk (trained on synthetic data).",
        lifespan=lifespan,
    )

    def get_predictor(request: Request) -> ChurnPredictor:
        predictor = request.app.state.predictor
        if predictor is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "model not loaded")
        return predictor

    @app.middleware("http")
    async def timing_header(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.2f}"
        return response

    @app.get("/health", response_model=Health)
    def health(request: Request) -> Health:
        loaded = request.app.state.predictor is not None
        return Health(status="ok" if loaded else "degraded", model_loaded=loaded)

    @app.get("/model/info", response_model=ModelInfo)
    def model_info(request: Request) -> ModelInfo:
        p = get_predictor(request)
        return ModelInfo(
            model_name=p.report["model_name"],
            model_version=p.version,
            trained_at=p.report["trained_at"],
            threshold=p.threshold,
            features=p.features,
            test_metrics=p.report["test_metrics"],
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(customer: Customer, request: Request) -> PredictionResponse:
        p = get_predictor(request)
        pred = p.predict([customer])[0]
        return PredictionResponse(
            **pred.model_dump(), model_version=p.version, threshold=p.threshold
        )

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(batch: BatchRequest, request: Request) -> BatchResponse:
        p = get_predictor(request)
        return BatchResponse(
            model_version=p.version, threshold=p.threshold, predictions=p.predict(batch.customers)
        )

    return app


app = create_app()
