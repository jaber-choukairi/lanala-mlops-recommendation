import json
from pathlib import Path

import pandas as pd
import yaml
from great_expectations.dataset import PandasDataset


PARAMS_PATH = Path("params.yaml")
INPUT_PATH = Path("data/raw/banking_clients.csv")
REPORT_PATH = Path("data/validation/validation_report.json")

PRODUCTS = [
    "savings_account",
    "premium_card",
    "personal_loan",
    "home_loan",
    "life_insurance",
    "investment_plan",
]

EXPECTED_COLUMNS = [
    "client_id",
    "age",
    "gender",
    "marital_status",
    "number_of_children",
    "employment_status",
    "monthly_income",
    "account_balance",
    "credit_score",
    "customer_tenure_months",
    "number_of_transactions",
    "average_transaction_amount",
    "digital_activity_score",
    "number_of_products_owned",
    "has_savings_account",
    "has_premium_card",
    "has_personal_loan",
    "has_home_loan",
    "has_life_insurance",
    "has_investment_plan",
    "complaint_open",
    "commercial_consent",
    "next_product",
]


def load_parameters() -> dict:
    with PARAMS_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_expectations(dataset: PandasDataset, minimum_rows: int) -> None:
    dataset.expect_table_columns_to_match_ordered_list(EXPECTED_COLUMNS)
    dataset.expect_table_row_count_to_be_between(
        min_value=minimum_rows,
        max_value=None,
    )

    dataset.expect_column_values_to_be_unique("client_id")

    for column in EXPECTED_COLUMNS:
        dataset.expect_column_values_to_not_be_null(column)

    dataset.expect_column_values_to_be_between(
        "age",
        min_value=18,
        max_value=85,
    )

    dataset.expect_column_values_to_be_between(
        "number_of_children",
        min_value=0,
        max_value=10,
    )

    dataset.expect_column_values_to_be_between(
        "monthly_income",
        min_value=0,
        max_value=50000,
    )

    dataset.expect_column_values_to_be_between(
        "account_balance",
        min_value=0,
        max_value=300000,
    )

    dataset.expect_column_values_to_be_between(
        "credit_score",
        min_value=300,
        max_value=850,
    )

    dataset.expect_column_values_to_be_between(
        "digital_activity_score",
        min_value=0,
        max_value=1,
    )

    dataset.expect_column_values_to_be_in_set(
        "gender",
        ["female", "male"],
    )

    dataset.expect_column_values_to_be_in_set(
        "marital_status",
        ["single", "married", "divorced", "widowed"],
    )

    dataset.expect_column_values_to_be_in_set(
        "employment_status",
        [
            "employed",
            "self_employed",
            "student",
            "retired",
            "unemployed",
        ],
    )

    binary_columns = [
        "has_savings_account",
        "has_premium_card",
        "has_personal_loan",
        "has_home_loan",
        "has_life_insurance",
        "has_investment_plan",
        "complaint_open",
        "commercial_consent",
    ]

    for column in binary_columns:
        dataset.expect_column_values_to_be_in_set(column, [0, 1])

    dataset.expect_column_values_to_be_in_set(
        "next_product",
        PRODUCTS,
    )


def validate_business_rules(data: pd.DataFrame) -> dict:
    product_columns = [
        f"has_{product}"
        for product in PRODUCTS
    ]

    calculated_product_count = data[product_columns].sum(axis=1)

    incorrect_product_count = int(
        (
            calculated_product_count
            != data["number_of_products_owned"]
        ).sum()
    )

    target_already_owned = 0

    for product in PRODUCTS:
        invalid_rows = (
            (data["next_product"] == product)
            & (data[f"has_{product}"] == 1)
        )
        target_already_owned += int(invalid_rows.sum())

    return {
        "product_count_consistent": incorrect_product_count == 0,
        "incorrect_product_count_rows": incorrect_product_count,
        "target_not_already_owned": target_already_owned == 0,
        "target_already_owned_rows": target_already_owned,
    }


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Le fichier {INPUT_PATH} n'existe pas. "
            "Exécutez d'abord la génération."
        )

    parameters = load_parameters()["data_validation"]
    data = pd.read_csv(INPUT_PATH)

    dataset = PandasDataset(data)
    build_expectations(
        dataset=dataset,
        minimum_rows=parameters["minimum_rows"],
    )

    gx_result = dataset.validate(
        result_format="SUMMARY",
        only_return_failures=False,
    )

    business_rules = validate_business_rules(data)

    successful_expectations = int(
        gx_result["statistics"]["successful_expectations"]
    )
    evaluated_expectations = int(
        gx_result["statistics"]["evaluated_expectations"]
    )

    validation_success = bool(
        gx_result["success"]
        and business_rules["product_count_consistent"]
        and business_rules["target_not_already_owned"]
    )

    report = {
        "validation_success": validation_success,
        "row_count": int(len(data)),
        "column_count": int(len(data.columns)),
        "evaluated_expectations": evaluated_expectations,
        "successful_expectations": successful_expectations,
        "unsuccessful_expectations": (
            evaluated_expectations - successful_expectations
        ),
        "success_percentage": round(
            successful_expectations
            / evaluated_expectations
            * 100,
            2,
        ),
        "business_rules": business_rules,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(json.dumps(report, indent=2))

    if not validation_success:
        raise ValueError(
            "La validation des données a échoué. "
            f"Consultez {REPORT_PATH}."
        )

    print("Validation Great Expectations réussie.")


if __name__ == "__main__":
    main()