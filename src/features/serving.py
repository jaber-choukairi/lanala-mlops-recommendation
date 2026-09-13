import numpy as np
import pandas as pd

from src.features.schema import FEATURE_COLUMNS
from src.features.schema import PRODUCT_OWNERSHIP_COLUMNS


def build_serving_features(
    client_data: dict,
) -> pd.DataFrame:
    record = client_data.copy()

    ownership_columns = list(
        PRODUCT_OWNERSHIP_COLUMNS.values()
    )

    record["number_of_products_owned"] = sum(
        int(record[column])
        for column in ownership_columns
    )

    record["income_per_child"] = round(
        record["monthly_income"]
        / (record["number_of_children"] + 1),
        2,
    )

    if record["monthly_income"] > 0:
        record["balance_to_income_ratio"] = round(
            min(
                record["account_balance"]
                / record["monthly_income"],
                100,
            ),
            4,
        )

        record["transaction_to_income_ratio"] = round(
            min(
                record["average_transaction_amount"]
                / record["monthly_income"],
                10,
            ),
            4,
        )
    else:
        record["balance_to_income_ratio"] = 0.0
        record["transaction_to_income_ratio"] = 0.0

    record["products_remaining"] = (
        len(ownership_columns)
        - record["number_of_products_owned"]
    )

    record["is_young_client"] = int(
        record["age"] < 30
    )

    record["is_family_client"] = int(
        record["number_of_children"] > 0
    )

    record["is_high_income"] = int(
        record["monthly_income"] >= 7000
    )

    record["is_high_balance"] = int(
        record["account_balance"] >= 30000
    )

    record["is_digitally_active"] = int(
        record["digital_activity_score"] >= 0.70
    )

    record["is_long_term_client"] = int(
        record["customer_tenure_months"] >= 60
    )

    record["is_eligible_for_commercial_offer"] = int(
        record["commercial_consent"] == 1
        and record["complaint_open"] == 0
    )

    features = pd.DataFrame([record])
    features = features[FEATURE_COLUMNS]

    numeric_features = features.select_dtypes(
        include=[np.number]
    )

    if not np.isfinite(
        numeric_features.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Les features contiennent une valeur non finie."
        )

    return features