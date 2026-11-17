"""Deterministic synthetic customer data for a subscription (telco-style) business.

The data is fully synthetic. Churn is drawn from a known logistic model so that
the "true" drivers are known, which makes it possible to sanity-check what the
trained model learns. Nothing here describes real customers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CONTRACTS = ("month_to_month", "one_year", "two_year")
INTERNET = ("dsl", "fiber", "none")
PAYMENT = ("electronic_check", "bank_transfer", "credit_card", "mailed_check")

TARGET = "churned"


def generate_customers(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Generate ``n`` synthetic customers with a binary ``churned`` label.

    The same ``(n, seed)`` always returns an identical frame.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)

    contract = rng.choice(CONTRACTS, size=n, p=[0.55, 0.25, 0.20])
    # Longer contracts tend to belong to longer-tenured customers.
    tenure_scale = np.select(
        [contract == "month_to_month", contract == "one_year"], [14.0, 28.0], 40.0
    )
    tenure = np.clip(rng.exponential(tenure_scale), 0, 72).round().astype(int)

    internet = rng.choice(INTERNET, size=n, p=[0.35, 0.45, 0.20])
    base_charge = np.select([internet == "fiber", internet == "dsl"], [80.0, 55.0], 25.0)
    monthly_charges = np.clip(base_charge + rng.normal(0, 12, n), 18.0, 130.0).round(2)

    payment = rng.choice(PAYMENT, size=n, p=[0.35, 0.22, 0.23, 0.20])
    senior = rng.random(n) < 0.16
    paperless = rng.random(n) < 0.6
    support_tickets = rng.poisson(np.where(internet == "fiber", 1.4, 0.9))

    logit = (
        -1.55
        + 1.10 * (contract == "month_to_month")
        - 0.90 * (contract == "two_year")
        - 0.045 * tenure
        + 0.018 * (monthly_charges - 65.0)
        + 0.45 * (internet == "fiber")
        + 0.50 * (payment == "electronic_check")
        + 0.38 * support_tickets
        + 0.30 * senior
        + 0.20 * paperless
        + rng.normal(0, 0.35, n)  # unobserved heterogeneity
    )
    churned = rng.random(n) < 1.0 / (1.0 + np.exp(-logit))

    return pd.DataFrame(
        {
            "customer_id": [f"C{seed:03d}-{i:06d}" for i in range(n)],
            "tenure_months": tenure,
            "contract": contract,
            "monthly_charges": monthly_charges,
            "internet_service": internet,
            "payment_method": payment,
            "senior_citizen": senior,
            "paperless_billing": paperless,
            "support_tickets_90d": support_tickets.astype(int),
            TARGET: churned.astype(int),
        }
    )
