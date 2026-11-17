"""Train, compare and persist the churn model.

Steps:
1. Generate synthetic data and hold out a stratified 20 % test set.
2. Compare candidates with 5-fold stratified CV on the training set (PR-AUC is the
   selection metric because churners are the minority class).
3. Pick the decision threshold on out-of-fold predictions (max F1), never on the test set.
4. Refit the winner on the full training set, evaluate once on the test set.
5. Save a single joblib bundle (pipeline + threshold + metadata) and a model card.

Usage: ``python -m churn_service.train --out artifacts``
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from churn_service import __version__
from churn_service.data import TARGET, generate_customers
from churn_service.features import FEATURES, build_candidates
from churn_service.model_card import render_model_card


@dataclass
class TrainingReport:
    """Everything worth knowing about one training run."""

    model_name: str
    model_version: str
    trained_at: str
    n_train: int
    n_test: int
    churn_rate_train: float
    threshold: float
    cv_scores: dict[str, dict[str, float]]
    test_metrics: dict[str, float]
    permutation_importance: dict[str, float]
    environment: dict[str, str] = field(default_factory=dict)


def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Return the probability cut-off that maximises F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # precision/recall have one more element than thresholds.
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.argmax(f1))])


def classification_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    """Threshold-free and threshold-dependent metrics, rounded for reporting."""
    pred = (proba >= threshold).astype(int)
    metrics = {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "flagged_share": float(pred.mean()),
    }
    return {k: round(float(v), 4) for k, v in metrics.items()}


def train(
    n_customers: int = 10_000, seed: int = 42, cv_folds: int = 5
) -> tuple[object, TrainingReport]:
    """Run the full training procedure and return (fitted_pipeline, report)."""
    df = generate_customers(n_customers, seed=seed)
    X, y = df[FEATURES], df[TARGET].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv_scores: dict[str, dict[str, float]] = {}
    oof: dict[str, np.ndarray] = {}
    for name, pipe in build_candidates(seed).items():
        proba = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        oof[name] = proba
        cv_scores[name] = {
            "roc_auc": round(float(roc_auc_score(y_train, proba)), 4),
            "pr_auc": round(float(average_precision_score(y_train, proba)), 4),
        }

    best = max(cv_scores, key=lambda m: cv_scores[m]["pr_auc"])
    threshold = best_f1_threshold(y_train, oof[best])

    model = build_candidates(seed)[best].fit(X_train, y_train)
    test_proba = model.predict_proba(X_test)[:, 1]

    perm = permutation_importance(
        model, X_test, y_test, scoring="average_precision", n_repeats=5, random_state=seed
    )
    importance = dict(
        sorted(
            ((f, round(float(m), 4)) for f, m in zip(FEATURES, perm.importances_mean, strict=True)),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )

    report = TrainingReport(
        model_name=best,
        model_version=f"{__version__}+seed{seed}",
        trained_at=datetime.now(UTC).isoformat(timespec="seconds"),
        n_train=len(X_train),
        n_test=len(X_test),
        churn_rate_train=round(float(y_train.mean()), 4),
        threshold=round(threshold, 4),
        cv_scores=cv_scores,
        test_metrics=classification_metrics(y_test, test_proba, threshold),
        permutation_importance=importance,
        environment={
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
        },
    )
    return model, report


def save_bundle(model: object, report: TrainingReport, out_dir: Path) -> Path:
    """Persist model + metadata as one artifact so they can never drift apart."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "model.joblib"
    joblib.dump({"pipeline": model, "features": FEATURES, "report": asdict(report)}, path)
    (out_dir / "metrics.json").write_text(json.dumps(asdict(report), indent=2))
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the churn model.")
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--n", type=int, default=10_000, help="number of synthetic customers")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-card", type=Path, default=None, help="write MODEL_CARD.md here")
    args = parser.parse_args(argv)

    model, report = train(args.n, seed=args.seed)
    path = save_bundle(model, report, args.out)
    if args.model_card:
        args.model_card.write_text(render_model_card(report))
    print(json.dumps({"artifact": str(path), **asdict(report)}, indent=2))


if __name__ == "__main__":
    main()
