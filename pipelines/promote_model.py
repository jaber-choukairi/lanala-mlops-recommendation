import argparse
import json
import os
from pathlib import Path

import mlflow
import yaml
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException


PARAMS_PATH = Path("params.yaml")
OUTPUT_PATH = Path("reports/phase5/promotion.json")


def load_parameters() -> dict:
    with PARAMS_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)["registry"]


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--version",
        required=True,
        help="Version MLflow à promouvoir.",
    )

    parser.add_argument(
        "--to",
        choices=["staging", "production"],
        required=True,
        help="Alias cible.",
    )

    arguments = parser.parse_args()
    parameters = load_parameters()

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        parameters["tracking_uri"],
    )

    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient(
        tracking_uri=tracking_uri
    )

    model_name = parameters["model_name"]
    target_alias = arguments.to
    version = str(arguments.version)

    model_version = client.get_model_version(
        name=model_name,
        version=version,
    )

    if model_version.status != "READY":
        raise ValueError(
            f"La version {version} n'est pas READY."
        )

    previous_production_version = None

    if target_alias == "production":
        try:
            previous_production = (
                client.get_model_version_by_alias(
                    name=model_name,
                    alias="production",
                )
            )

            previous_production_version = str(
                previous_production.version
            )

            if previous_production_version != version:
                client.set_model_version_tag(
                    name=model_name,
                    version=previous_production_version,
                    key="lifecycle",
                    value="archived",
                )

                client.set_registered_model_alias(
                    name=model_name,
                    alias=(
                        "archived-v"
                        f"{previous_production_version}"
                    ),
                    version=previous_production_version,
                )

        except MlflowException:
            previous_production_version = None

    client.set_registered_model_alias(
        name=model_name,
        alias=target_alias,
        version=version,
    )

    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="lifecycle",
        value=target_alias,
    )

    result = {
        "model_name": model_name,
        "version": version,
        "alias": target_alias,
        "model_uri": (
            f"models:/{model_name}@{target_alias}"
        ),
        "previous_production_version": (
            previous_production_version
        ),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(result, file, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()