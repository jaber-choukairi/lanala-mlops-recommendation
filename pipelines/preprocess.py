from pathlib import Path

import pandas as pd
import yaml


PARAMS_PATH = Path("params.yaml")
INPUT_PATH = Path("data/raw/banking_clients.csv")
OUTPUT_PATH = Path("data/interim/banking_clients_clean.csv")

CATEGORICAL_COLUMNS = [
    "gender",
    "marital_status",
    "employment_status",
    "next_product",
]

INTEGER_COLUMNS = [
    "age",
    "number_of_children",
    "credit_score",
    "customer_tenure_months",
    "number_of_transactions",
    "number_of_products_owned",
    "has_savings_account",
    "has_premium_card",
    "has_personal_loan",
    "has_home_loan",
    "has_life_insurance",
    "has_investment_plan",
    "complaint_open",
    "commercial_consent",
]

FLOAT_COLUMNS = [
    "monthly_income",
    "account_balance",
    "average_transaction_amount",
    "digital_activity_score",
]


def load_parameters() -> dict:
    with PARAMS_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def clean_data(data: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    clean = data.copy()

    clean.columns = [
        column.strip().lower()
        for column in clean.columns
    ]

    clean = clean.drop_duplicates(subset=["client_id"])

    for column in CATEGORICAL_COLUMNS:
        clean[column] = (
            clean[column]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    for column in INTEGER_COLUMNS:
        clean[column] = pd.to_numeric(
            clean[column],
            errors="coerce",
        ).astype("Int64")

    for column in FLOAT_COLUMNS:
        clean[column] = pd.to_numeric(
            clean[column],
            errors="coerce",
        ).astype(float)

    clean["age"] = clean["age"].clip(
        parameters["minimum_age"],
        parameters["maximum_age"],
    )

    clean["monthly_income"] = clean["monthly_income"].clip(
        parameters["minimum_income"],
        parameters["maximum_income"],
    )

    numeric_columns = INTEGER_COLUMNS + FLOAT_COLUMNS

    for column in numeric_columns:
        if clean[column].isna().any():
            clean[column] = clean[column].fillna(
                clean[column].median()
            )

    for column in CATEGORICAL_COLUMNS:
        if clean[column].isna().any():
            clean[column] = clean[column].fillna("unknown")

    clean = clean.sort_values("client_id").reset_index(drop=True)

    if clean.isna().any().any():
        missing_columns = clean.columns[
            clean.isna().any()
        ].tolist()
        raise ValueError(
            f"Des valeurs manquantes subsistent : {missing_columns}"
        )

    return clean


def main() -> None:
    parameters = load_parameters()["preprocessing"]
    data = pd.read_csv(INPUT_PATH)

    clean = clean_data(data, parameters)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(OUTPUT_PATH, index=False)

    print(f"Données nettoyées : {OUTPUT_PATH}")
    print(f"Nombre de lignes : {len(clean)}")
    print(f"Doublons client_id : {clean['client_id'].duplicated().sum()}")
    print(f"Valeurs manquantes : {clean.isna().sum().sum()}")


if __name__ == "__main__":
    main()