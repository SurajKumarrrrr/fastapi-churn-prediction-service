from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from churn_service.api import create_app
from churn_service.train import save_bundle, train


@pytest.fixture(scope="session")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[object, object, Path]:
    """Train once on a small dataset and share the artifact across tests."""
    model, report = train(n_customers=3000, seed=7, cv_folds=3)
    path = save_bundle(model, report, tmp_path_factory.mktemp("artifacts"))
    return model, report, path


@pytest.fixture()
def client(trained) -> TestClient:
    _, _, path = trained
    with TestClient(create_app(str(path))) as c:
        yield c


@pytest.fixture()
def customer() -> dict:
    return {
        "tenure_months": 2,
        "contract": "month_to_month",
        "monthly_charges": 95.0,
        "internet_service": "fiber",
        "payment_method": "electronic_check",
        "senior_citizen": False,
        "paperless_billing": True,
        "support_tickets_90d": 3,
    }
