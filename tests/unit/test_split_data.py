from pipelines.generate_data import generate_banking_dataset
from pipelines.split_data import split_dataset


def test_split_sizes_and_separation() -> None:
    data = generate_banking_dataset(
        number_of_clients=1000,
        random_seed=42,
    )

    train, validation, test = split_dataset(
        data=data,
        train_size=0.70,
        validation_size=0.15,
        test_size=0.15,
        random_seed=42,
    )

    assert len(train) == 700
    assert len(validation) == 150
    assert len(test) == 150

    train_clients = set(train["client_id"])
    validation_clients = set(validation["client_id"])
    test_clients = set(test["client_id"])

    assert train_clients.isdisjoint(validation_clients)
    assert train_clients.isdisjoint(test_clients)
    assert validation_clients.isdisjoint(test_clients)


def test_split_preserves_all_target_classes() -> None:
    data = generate_banking_dataset(
        number_of_clients=1000,
        random_seed=42,
    )

    train, validation, test = split_dataset(
        data=data,
        train_size=0.70,
        validation_size=0.15,
        test_size=0.15,
        random_seed=42,
    )

    expected_classes = set(data["next_product"])

    assert set(train["next_product"]) == expected_classes
    assert set(validation["next_product"]) == expected_classes
    assert set(test["next_product"]) == expected_classes