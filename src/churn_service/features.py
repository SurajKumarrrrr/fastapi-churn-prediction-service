"""Feature definitions and the preprocessing + model pipelines."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["tenure_months", "monthly_charges", "support_tickets_90d"]
CATEGORICAL = ["contract", "internet_service", "payment_method"]
BOOLEAN = ["senior_citizen", "paperless_billing"]
FEATURES = NUMERIC + CATEGORICAL + BOOLEAN


def _preprocessor(scale: bool) -> ColumnTransformer:
    numeric = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC),
            # handle_unknown="ignore": an unseen category at serve time becomes all-zeros
            # instead of crashing. The API schema blocks unknown values anyway.
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("bool", "passthrough", BOOLEAN),
        ],
        verbose_feature_names_out=False,
    )


def build_candidates(seed: int = 42) -> dict[str, Pipeline]:
    """Return the candidate models compared during training."""
    return {
        "logistic_regression": Pipeline(
            [
                ("prep", _preprocessor(scale=True)),
                ("clf", LogisticRegression(max_iter=1000, C=1.0)),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("prep", _preprocessor(scale=False)),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=300,
                        max_leaf_nodes=15,
                        l2_regularization=1.0,
                        early_stopping=True,
                        validation_fraction=0.15,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }
