import numpy as np

from pipelines.build_features import build_features
from pipelines.generate_data import generate_banking_dataset


EXPECTED_ENGINEERED_FEATURES = [
    "income_per_child",
    "balance_to_income_ratio",
    "transaction_to_income_ratio",
    "products_remaining",
    "is_young_client",
    "is_family_client",
    "is_high_income",
    "is_high_balance",
    "is_digitally_active",
    "is_long_term_client",
    "is_eligible_for_commercial_offer",
]


def test_feature_engineering_creates_expected_columns() -> None:
    data = generate_banking_dataset(
        number_of_clients=100,
        random_seed=42,
    )

    features = build_features(data)

    for column in EXPECTED_ENGINEERED_FEATURES:
        assert column in features.columns


def test_features_do_not_contain_missing_or_infinite_values() -> None:
    data = generate_banking_dataset(
        number_of_clients=100,
        random_seed=42,
    )

    features = build_features(data)

    numeric_features = features.select_dtypes(include=[np.number])

    assert not features.isna().any().any()
    assert np.isfinite(numeric_features.to_numpy()).all()


def test_commercial_eligibility_rule() -> None:
    data = generate_banking_dataset(
        number_of_clients=100,
        random_seed=42,
    )

    features = build_features(data)

    expected = (
        (features["commercial_consent"] == 1)
        & (features["complaint_open"] == 0)
    ).astype(int)

    assert (
        features["is_eligible_for_commercial_offer"] == expected
    ).all()