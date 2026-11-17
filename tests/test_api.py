from fastapi.testclient import TestClient

from churn_service.api import create_app
from churn_service.schemas import MAX_BATCH


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}
    assert "X-Process-Time-Ms" in r.headers


def test_model_info(client, trained):
    _, report, _ = trained
    body = client.get("/model/info").json()
    assert body["model_name"] == report.model_name
    assert body["threshold"] == report.threshold
    assert "contract" in body["features"]


def test_predict_single(client, customer):
    r = client.post("/predict", json=customer)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["will_churn"] == (body["churn_probability"] >= body["threshold"])
    assert body["risk_band"] in {"low", "medium", "high"}


def test_risky_customer_scores_higher_than_loyal_one(client, customer):
    loyal = {
        **customer,
        "tenure_months": 60,
        "contract": "two_year",
        "monthly_charges": 30.0,
        "internet_service": "dsl",
        "payment_method": "credit_card",
        "support_tickets_90d": 0,
    }
    risky_p = client.post("/predict", json=customer).json()["churn_probability"]
    loyal_p = client.post("/predict", json=loyal).json()["churn_probability"]
    assert risky_p > loyal_p


def test_batch_matches_single(client, customer):
    single = client.post("/predict", json=customer).json()
    batch = client.post("/predict/batch", json={"customers": [customer, customer]}).json()
    assert len(batch["predictions"]) == 2
    assert batch["predictions"][0]["churn_probability"] == single["churn_probability"]


def test_field_order_does_not_matter(client, customer):
    reordered = dict(reversed(list(customer.items())))
    a = client.post("/predict", json=customer).json()["churn_probability"]
    b = client.post("/predict", json=reordered).json()["churn_probability"]
    assert a == b


def test_validation_errors(client, customer):
    bad_cases = [
        {**customer, "tenure_months": -1},
        {**customer, "tenure_months": 100},
        {**customer, "contract": "weekly"},
        {**customer, "monthly_charges": 0},
        {**customer, "unexpected_field": 1},
        {k: v for k, v in customer.items() if k != "contract"},
    ]
    for payload in bad_cases:
        assert client.post("/predict", json=payload).status_code == 422, payload


def test_batch_limits(client, customer):
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422
    too_many = {"customers": [customer] * (MAX_BATCH + 1)}
    assert client.post("/predict/batch", json=too_many).status_code == 422


def test_missing_model_returns_503(tmp_path, customer):
    with TestClient(create_app(str(tmp_path / "missing.joblib"))) as c:
        assert c.get("/health").json() == {"status": "degraded", "model_loaded": False}
        assert c.post("/predict", json=customer).status_code == 503
        assert c.get("/model/info").status_code == 503
