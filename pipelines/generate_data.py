from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PARAMS_PATH = Path("params.yaml")
OUTPUT_PATH = Path("data/raw/banking_clients.csv")

PRODUCTS = [
    "savings_account",
    "premium_card",
    "personal_loan",
    "home_loan",
    "life_insurance",
    "investment_plan",
]


def load_parameters() -> dict:
    with PARAMS_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def softmax(values: np.ndarray) -> np.ndarray:
    shifted_values = values - np.max(values)
    exponentials = np.exp(shifted_values)
    return exponentials / exponentials.sum()


def generate_banking_dataset(
    number_of_clients: int,
    random_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)

    age = rng.integers(18, 81, number_of_clients)
    gender = rng.choice(
        ["female", "male"],
        size=number_of_clients,
        p=[0.48, 0.52],
    )

    marital_status = np.where(
        age < 25,
        rng.choice(
            ["single", "married"],
            size=number_of_clients,
            p=[0.90, 0.10],
        ),
        rng.choice(
            ["single", "married", "divorced", "widowed"],
            size=number_of_clients,
            p=[0.30, 0.55, 0.10, 0.05],
        ),
    )

    number_of_children = np.zeros(number_of_clients, dtype=int)
    married_mask = marital_status == "married"
    number_of_children[married_mask] = rng.integers(
        0,
        5,
        married_mask.sum(),
    )

    employment_status = rng.choice(
        ["employed", "self_employed", "student", "retired", "unemployed"],
        size=number_of_clients,
        p=[0.55, 0.17, 0.10, 0.10, 0.08],
    )

    income_base = {
        "employed": 4500,
        "self_employed": 5500,
        "student": 900,
        "retired": 2800,
        "unemployed": 700,
    }

    monthly_income = np.array(
        [
            max(0, rng.normal(income_base[status], income_base[status] * 0.35))
            for status in employment_status
        ]
    )

    monthly_income = np.clip(monthly_income, 0, 30000).round(2)

    customer_tenure_months = np.minimum(
        rng.integers(1, 241, number_of_clients),
        np.maximum((age - 18) * 12, 1),
    )

    account_balance = (
        monthly_income
        * rng.uniform(0.2, 8.0, number_of_clients)
        * (1 + customer_tenure_months / 240)
    )
    account_balance = np.clip(account_balance, 0, 250000).round(2)

    credit_score = (
        520
        + monthly_income / 80
        + customer_tenure_months / 3
        + rng.normal(0, 55, number_of_clients)
    )
    credit_score = np.clip(credit_score, 300, 850).astype(int)

    digital_activity_score = np.clip(
        1 - ((age - 18) / 100) + rng.normal(0, 0.18, number_of_clients),
        0,
        1,
    ).round(4)

    number_of_transactions = np.maximum(
        1,
        (
            8
            + monthly_income / 350
            + digital_activity_score * 15
            + rng.normal(0, 6, number_of_clients)
        ).astype(int),
    )

    average_transaction_amount = np.maximum(
        5,
        monthly_income
        * rng.uniform(0.03, 0.18, number_of_clients),
    ).round(2)

    complaint_open = rng.choice(
        [0, 1],
        size=number_of_clients,
        p=[0.94, 0.06],
    )

    commercial_consent = rng.choice(
        [0, 1],
        size=number_of_clients,
        p=[0.08, 0.92],
    )

    ownership_probabilities = {
        "savings_account": np.clip(
            0.20 + age / 180 + account_balance / 500000,
            0.10,
            0.80,
        ),
        "premium_card": np.clip(
            0.05 + monthly_income / 25000 + credit_score / 3000,
            0.05,
            0.65,
        ),
        "personal_loan": np.clip(
            0.08 + (monthly_income > 1800) * 0.15,
            0.05,
            0.40,
        ),
        "home_loan": np.clip(
            0.02
            + ((age >= 28) & (age <= 60)) * 0.15
            + (monthly_income > 4500) * 0.12,
            0.02,
            0.40,
        ),
        "life_insurance": np.clip(
            0.05
            + (number_of_children > 0) * 0.18
            + (age > 35) * 0.10,
            0.03,
            0.45,
        ),
        "investment_plan": np.clip(
            0.03
            + (account_balance > 30000) * 0.20
            + (monthly_income > 7000) * 0.15,
            0.02,
            0.45,
        ),
    }

    owned_products = {}

    for product in PRODUCTS:
        owned_products[product] = (
            rng.random(number_of_clients)
            < ownership_probabilities[product]
        ).astype(int)

    next_products = []

    for index in range(number_of_clients):
        scores = np.array(
            [
                0.4
                + (age[index] < 35) * 0.7
                + (account_balance[index] < 20000) * 0.5,
                -0.5
                + monthly_income[index] / 6000
                + credit_score[index] / 1000
                + digital_activity_score[index] * 0.5,
                0.1
                + (monthly_income[index] > 1800) * 0.5
                + (credit_score[index] > 580) * 0.4,
                -1.0
                + ((28 <= age[index] <= 60) * 0.7)
                + monthly_income[index] / 8000
                + credit_score[index] / 1200,
                -0.4
                + (number_of_children[index] > 0) * 1.0
                + (age[index] > 35) * 0.4,
                -1.0
                + account_balance[index] / 40000
                + monthly_income[index] / 10000,
            ],
            dtype=float,
        )

        scores += rng.normal(0, 0.35, len(PRODUCTS))

        for product_index, product in enumerate(PRODUCTS):
            if owned_products[product][index] == 1:
                scores[product_index] = -10

        if np.all(scores <= -9):
            product_to_release = rng.choice(PRODUCTS)
            owned_products[product_to_release][index] = 0
            scores[PRODUCTS.index(product_to_release)] = 0

        probabilities = softmax(scores)
        selected_product = rng.choice(PRODUCTS, p=probabilities)
        next_products.append(selected_product)

    data = pd.DataFrame(
        {
            "client_id": [
                f"CLI_{index:06d}"
                for index in range(1, number_of_clients + 1)
            ],
            "age": age,
            "gender": gender,
            "marital_status": marital_status,
            "number_of_children": number_of_children,
            "employment_status": employment_status,
            "monthly_income": monthly_income,
            "account_balance": account_balance,
            "credit_score": credit_score,
            "customer_tenure_months": customer_tenure_months,
            "number_of_transactions": number_of_transactions,
            "average_transaction_amount": average_transaction_amount,
            "digital_activity_score": digital_activity_score,
            "has_savings_account": owned_products["savings_account"],
            "has_premium_card": owned_products["premium_card"],
            "has_personal_loan": owned_products["personal_loan"],
            "has_home_loan": owned_products["home_loan"],
            "has_life_insurance": owned_products["life_insurance"],
            "has_investment_plan": owned_products["investment_plan"],
            "complaint_open": complaint_open,
            "commercial_consent": commercial_consent,
            "next_product": next_products,
        }
    )

    product_columns = [
        f"has_{product}"
        for product in PRODUCTS
    ]

    data["number_of_products_owned"] = data[product_columns].sum(axis=1)

    ordered_columns = [
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
        *product_columns,
        "complaint_open",
        "commercial_consent",
        "next_product",
    ]

    return data[ordered_columns]


def main() -> None:
    parameters = load_parameters()["data_generation"]

    data = generate_banking_dataset(
        number_of_clients=parameters["number_of_clients"],
        random_seed=parameters["random_seed"],
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)

    print(f"Dataset généré : {OUTPUT_PATH}")
    print(f"Nombre de lignes : {len(data)}")
    print(f"Nombre de colonnes : {len(data.columns)}")
    print("\nDistribution de la cible :")
    print(data["next_product"].value_counts())


if __name__ == "__main__":
    main()