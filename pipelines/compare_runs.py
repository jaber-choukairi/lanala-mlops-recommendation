import os
from pathlib import Path

import mlflow
import pandas as pd
import yaml


PARAMS_PATH = Path("params.yaml")


def load_parameters() -> dict:
    with PARAMS_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    parameters = load_parameters()["baseline"]

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        parameters["tracking_uri"],
    )

    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(
        parameters["experiment_name"]
    )

    if experiment is None:
        raise ValueError(
            "L'expérience MLflow n'existe pas. "
            "Lancez d'abord l'entraînement."
        )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.test_top_3_accuracy DESC"],
    )

    selected_columns = [
        "run_id",
        "tags.mlflow.runName",
        "tags.model_type",
        "metrics.validation_accuracy",
        "metrics.validation_macro_f1",
        "metrics.validation_top_3_accuracy",
        "metrics.test_accuracy",
        "metrics.test_macro_f1",
        "metrics.test_top_3_accuracy",
    ]

    available_columns = [
        column
        for column in selected_columns
        if column in runs.columns
    ]

    comparison = runs[available_columns].copy()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)

    print("\nComparaison des expériences :\n")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()