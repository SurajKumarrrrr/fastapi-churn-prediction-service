import pandas as pd
import pytest

from churn_service.data import CONTRACTS, TARGET, generate_customers
from churn_service.features import FEATURES


def test_generator_is_deterministic():
    pd.testing.assert_frame_equal(generate_customers(500, seed=1), generate_customers(500, seed=1))


def test_different_seeds_differ():
    assert not generate_customers(500, seed=1).equals(generate_customers(500, seed=2))


def test_schema_and_ranges():
    df = generate_customers(2000, seed=3)
    assert set(FEATURES + [TARGET, "customer_id"]) == set(df.columns)
    assert df["customer_id"].is_unique
    assert df["tenure_months"].between(0, 72).all()
    assert df["monthly_charges"].between(18, 130).all()
    assert set(df["contract"]) <= set(CONTRACTS)
    assert df.notna().all().all()


def test_churn_rate_is_realistic_and_driven_by_contract():
    df = generate_customers(10_000, seed=42)
    assert 0.15 < df[TARGET].mean() < 0.40
    rates = df.groupby("contract")[TARGET].mean()
    assert rates["month_to_month"] > rates["one_year"] > rates["two_year"]


def test_rejects_non_positive_n():
    with pytest.raises(ValueError):
        generate_customers(0)
