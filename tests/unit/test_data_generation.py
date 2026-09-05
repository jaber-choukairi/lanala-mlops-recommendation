from pipelines.generate_data import PRODUCTS
from pipelines.generate_data import generate_banking_dataset


def test_generated_dataset_has_expected_size() -> None:
    data = generate_banking_dataset(
        number_of_clients=100,
        random_seed=42,
    )

    assert len(data) == 100
    assert data["client_id"].is_unique
    assert data.isna().sum().sum() == 0


def test_target_contains_known_products() -> None:
    data = generate_banking_dataset(
        number_of_clients=200,
        random_seed=42,
    )

    assert set(data["next_product"]).issubset(set(PRODUCTS))


def test_next_product_is_not_already_owned() -> None:
    data = generate_banking_dataset(
        number_of_clients=200,
        random_seed=42,
    )

    for product in PRODUCTS:
        invalid_rows = data[
            (data["next_product"] == product)
            & (data[f"has_{product}"] == 1)
        ]

        assert invalid_rows.empty