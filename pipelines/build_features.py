from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("data/interim/banking_clients_clean.csv")
PROCESSED_PATH = Path(
    "data/processed/banking_clients_processed.parquet"
)
FEATURES_PATH = Path("data/features/client_features.parquet")


PRODUCT_COLUMNS = [
    "has_savings_account",
    "has_premium_card",
    "has_personal_loan",
    "has_home_loan",
    "has_life_insurance",
    "has_investment_plan",
]


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    features = data.copy()

    features["income_per_child"] = (
        features["monthly_income"]
        / (features["number_of_children"] + 1)
    ).round(2)

    features["balance_to_income_ratio"] = (
        features["account_balance"]
        / features["monthly_income"].replace(0, np.nan)
    )
    features["balance_to_income_ratio"] = (
        features["balance_to_income_ratio"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .clip(0, 100)
        .round(4)
    )

    features["transaction_to_income_ratio"] = (
        features["average_transaction_amount"]
        / features["monthly_income"].replace(0, np.nan)
    )
    features["transaction_to_income_ratio"] = (
        features["transaction_to_income_ratio"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .clip(0, 10)
        .round(4)
    )

    features["products_remaining"] = (
        len(PRODUCT_COLUMNS)
        - features["number_of_products_owned"]
    )

    features["is_young_client"] = (
        features["age"] < 30
    ).astype(int)

    features["is_family_client"] = (
        features["number_of_children"] > 0
    ).astype(int)

    features["is_high_income"] = (
        features["monthly_income"] >= 7000
    ).astype(int)

    features["is_high_balance"] = (
        features["account_balance"] >= 30000
    ).astype(int)

    features["is_digitally_active"] = (
        features["digital_activity_score"] >= 0.70
    ).astype(int)

    features["is_long_term_client"] = (
        features["customer_tenure_months"] >= 60
    ).astype(int)

    features["is_eligible_for_commercial_offer"] = (
        (features["commercial_consent"] == 1)
        & (features["complaint_open"] == 0)
    ).astype(int)

    if features.isna().any().any():
        raise ValueError(
            "La table de features contient des valeurs manquantes."
        )

    return features


def main() -> None:
    data = pd.read_csv(INPUT_PATH)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(
        PROCESSED_PATH,
        index=False,
        engine="pyarrow",
    )

    features = build_features(data)

    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(
        FEATURES_PATH,
        index=False,
        engine="pyarrow",
    )

    print(f"Données transformées : {PROCESSED_PATH}")
    print(f"Table de features : {FEATURES_PATH}")
    print(f"Nombre de lignes : {len(features)}")
    print(f"Nombre de features/colonnes : {len(features.columns)}")
    print("\nNouvelles features :")
    print(
        [
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
    )


if __name__ == "__main__":
    main()