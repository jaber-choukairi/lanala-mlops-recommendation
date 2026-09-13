from src.features.serving import build_serving_features


def build_client() -> dict:
    return {
        "client_id": "CLI_TEST",
        "age": 35,
        "gender": "male",
        "marital_status": "married",
        "number_of_children": 2,
        "employment_status": "employed",
        "monthly_income": 6000.0,
        "account_balance": 30000.0,
        "credit_score": 700,
        "customer_tenure_months": 72,
        "number_of_transactions": 30,
        "average_transaction_amount": 400.0,
        "digital_activity_score": 0.80,
        "has_savings_account": 1,
        "has_premium_card": 0,
        "has_personal_loan": 0,
        "has_home_loan": 0,
        "has_life_insurance": 0,
        "has_investment_plan": 0,
        "complaint_open": 0,
        "commercial_consent": 1,
    }


def test_serving_features_are_created() -> None:
    features = build_serving_features(build_client())

    assert len(features) == 1
    assert features.iloc[0]["number_of_products_owned"] == 1
    assert features.iloc[0]["products_remaining"] == 5
    assert features.iloc[0]["is_family_client"] == 1
    assert (
        features.iloc[0][
            "is_eligible_for_commercial_offer"
        ]
        == 1
    )


def test_serving_features_match_training_logic() -> None:
    features = build_serving_features(build_client())

    assert features.iloc[0]["income_per_child"] == 2000.0
    assert features.iloc[0]["balance_to_income_ratio"] == 5.0