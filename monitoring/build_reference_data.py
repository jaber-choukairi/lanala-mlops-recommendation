from pathlib import Path

import pandas as pd


SOURCE_PATH = Path("data/splits/train.parquet")
OUTPUT_PATH = Path("data/monitoring/reference.parquet")

EXCLUDED_COLUMNS = {
    "client_id",
    "next_product",
}


def main() -> None:
    dataframe = pd.read_parquet(SOURCE_PATH)

    selected_columns = [
        column
        for column in dataframe.columns
        if column not in EXCLUDED_COLUMNS
    ]

    reference = dataframe[selected_columns].copy()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    reference.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(f"Référence créée : {OUTPUT_PATH}")
    print(f"Nombre de lignes : {len(reference)}")
    print(f"Nombre de colonnes : {len(reference.columns)}")


if __name__ == "__main__":
    main()