CATEGORICAL_FEATURES = [
    "gender",
    "marital_status",
    "employment_status",
]

NUMERICAL_FEATURES = [
    "age",
    "number_of_children",
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

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

PRODUCT_OWNERSHIP_COLUMNS = {
    "savings_account": "has_savings_account",
    "premium_card": "has_premium_card",
    "personal_loan": "has_personal_loan",
    "home_loan": "has_home_loan",
    "life_insurance": "has_life_insurance",
    "investment_plan": "has_investment_plan",
}