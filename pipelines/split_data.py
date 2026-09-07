from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


PARAMS_PATH = Path("params.yaml")
INPUT_PATH = Path("data/features/client_features.parquet")
SPLITS_DIRECTORY = Path("data/splits")

TRAIN_PATH = SPLITS_DIRECTORY / "train.parquet"
VALIDATION_PATH = SPLITS_DIRECTORY / "validation.parquet"
TEST_PATH = SPLITS_DIRECTORY / "test.parquet"


def load_parameters() -> dict:
    with PARAMS_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def split_dataset(
    data: pd.DataFrame,
    train_size: float,
    validation_size: float,
    test_size: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_size = train_size + validation_size + test_size

    if abs(total_size - 1.0) > 1e-9:
        raise ValueError(
            "La somme de train_size, validation_size et test_size "
            "doit être égale à 1."
        )

    if "next_product" not in data.columns:
        raise ValueError("La colonne cible next_product est absente.")

    train_data, temporary_data = train_test_split(
        data,
        train_size=train_size,
        random_state=random_seed,
        stratify=data["next_product"],
    )

    relative_validation_size = validation_size / (
        validation_size + test_size
    )

    validation_data, test_data = train_test_split(
        temporary_data,
        train_size=relative_validation_size,
        random_state=random_seed,
        stratify=temporary_data["next_product"],
    )

    train_data = train_data.sort_values("client_id").reset_index(drop=True)
    validation_data = (
        validation_data.sort_values("client_id").reset_index(drop=True)
    )
    test_data = test_data.sort_values("client_id").reset_index(drop=True)

    return train_data, validation_data, test_data


def main() -> None:
    parameters = load_parameters()["data_split"]

    data = pd.read_parquet(INPUT_PATH)

    train_data, validation_data, test_data = split_dataset(
        data=data,
        train_size=parameters["train_size"],
        validation_size=parameters["validation_size"],
        test_size=parameters["test_size"],
        random_seed=parameters["random_seed"],
    )

    SPLITS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    train_data.to_parquet(TRAIN_PATH, index=False)
    validation_data.to_parquet(VALIDATION_PATH, index=False)
    test_data.to_parquet(TEST_PATH, index=False)

    print("Découpage terminé")
    print(f"Entraînement : {len(train_data)} lignes")
    print(f"Validation    : {len(validation_data)} lignes")
    print(f"Test          : {len(test_data)} lignes")

    print("\nDistribution entraînement :")
    print(
        train_data["next_product"]
        .value_counts(normalize=True)
        .round(4)
    )


if __name__ == "__main__":
    main()