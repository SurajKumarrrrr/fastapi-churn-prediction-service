import json

import numpy as np
import pytest

from churn_service.model_card import render_model_card
from churn_service.predictor import ChurnPredictor, risk_band
from churn_service.train import best_f1_threshold, classification_metrics


def test_best_f1_threshold_separable():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    t = best_f1_threshold(y, p)
    assert 0.3 < t <= 0.8
    assert classification_metrics(y, p, t)["f1"] == 1.0


def test_report_is_sane(trained):
    _, report, _ = trained
    assert report.model_name in report.cv_scores
    # The winner must be the candidate with the best CV PR-AUC.
    assert report.cv_scores[report.model_name]["pr_auc"] == max(
        s["pr_auc"] for s in report.cv_scores.values()
    )
    assert 0 < report.threshold < 1
    assert report.test_metrics["roc_auc"] > 0.70  # well above chance on this synthetic data
    assert report.n_train + report.n_test == 3000


def test_model_learns_known_drivers(trained):
    _, report, _ = trained
    top3 = list(report.permutation_importance)[:3]
    # Contract type and tenure are the strongest true drivers in the generator.
    assert "contract" in top3 or "tenure_months" in top3


def test_artifacts_round_trip(trained):
    model, report, path = trained
    predictor = ChurnPredictor.load(path)
    assert predictor.threshold == report.threshold
    metrics = json.loads((path.parent / "metrics.json").read_text())
    assert metrics["model_name"] == report.model_name


def test_load_missing_artifact(tmp_path):
    with pytest.raises(FileNotFoundError):
        ChurnPredictor.load(tmp_path / "nope.joblib")


@pytest.mark.parametrize(
    ("p", "expected"), [(0.9, "high"), (0.4, "high"), (0.25, "medium"), (0.1, "low")]
)
def test_risk_band(p, expected):
    assert risk_band(p, threshold=0.4) == expected


def test_model_card_contains_real_numbers(trained):
    _, report, _ = trained
    card = render_model_card(report)
    assert "synthetic" in card.lower()
    assert f"{report.test_metrics['roc_auc']:.3f}" in card
    assert report.model_name in card
