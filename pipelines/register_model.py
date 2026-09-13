import json
import os
import time
from pathlib import Path

import mlflow
import yaml
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException


PARAMS_PATH = Path("params.yaml")
PHASE4_SUMMARY = Path("reports/phase4/summary.json")
OUTPUT_PATH = Path("reports/phase5/registration.json")


def load_yaml() -> dict:
    with PARAMS_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_summary() -> dict:
    with PHASE4_SUMMARY.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def wait_until_ready(
    client: MlflowClient,
    model_name: str,
    version: str,
    timeout: int = 120,
):
    start_time = time.time()

    while time.time() - start_time < timeout:
        model_version = client.get_model_version(
            name=model_name,
            version=version,
        )

        if model_version.status == "READY":
            return model_version

        if model_version.status == "FAILED_REGISTRATION":
            raise RuntimeError(
                "L'enregistrement du modèle a échoué."
            )

        time.sleep(2)

    raise TimeoutError(
        "Le modèle n'est pas devenu READY à temps."
    )


def main() -> None:
    parameters = load_yaml()["registry"]
    summary = load_summary()

    if not summary["promotion"]["promoted"]:
        raise ValueError(
            "Le Challenger n'a pas été promu pendant "
            "la phase 4. Enregistrement refusé."
        )

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        parameters["tracking_uri"],
    )

    model_name = parameters["model_name"]
    staging_alias = parameters["staging_alias"]
    model_uri = summary["model_uri"]

    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient(
        tracking_uri=tracking_uri
    )

    try:
        client.get_registered_model(model_name)
        print(
            f"Modèle enregistré existant : {model_name}"
        )
    except MlflowException:
        client.create_registered_model(
            name=model_name,
            description=(
                "Modèle de recommandation bancaire "
                "LANALA."
            ),
        )
        print(
            f"Registered Model créé : {model_name}"
        )

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )

    model_version = wait_until_ready(
        client=client,
        model_name=model_name,
        version=str(model_version.version),
    )

    version = str(model_version.version)

    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="model_type",
        value="xgboost",
    )

    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="lifecycle",
        value="staging",
    )

    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="phase4_run_id",
        value=summary["parent_run_id"],
    )

    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="test_top_3_accuracy",
        value=str(
            summary["challenger"]["test_metrics"][
                "test_top_3_accuracy"
            ]
        ),
    )

    client.set_registered_model_alias(
        name=model_name,
        alias=staging_alias,
        version=version,
    )

    result = {
        "model_name": model_name,
        "version": version,
        "alias": staging_alias,
        "model_uri": (
            f"models:/{model_name}@{staging_alias}"
        ),
        "status": model_version.status,
        "source_model_uri": model_uri,
        "test_top_3_accuracy": (
            summary["challenger"]["test_metrics"][
                "test_top_3_accuracy"
            ]
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