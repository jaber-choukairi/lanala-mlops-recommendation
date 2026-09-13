import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from src.models.metrics import calculate_classification_metrics
from src.models.metrics import get_top_k_predictions


PARAMS_PATH = Path("params.yaml")
FEATURE_DATA_PATH = Path("data/features/client_features.parquet")
TRAIN_PATH = Path("data/splits/train.parquet")
VALIDATION_PATH = Path("data/splits/validation.parquet")
TEST_PATH = Path("data/splits/test.parquet")
REPORT_DIRECTORY = Path("reports/phase3")
SUMMARY_PATH = REPORT_DIRECTORY / "summary.json"

TARGET_COLUMN = "next_product"
IDENTIFIER_COLUMNS = ["client_id"]

CATEGORICAL_COLUMNS = [
    "gender",
    "marital_status",
    "employment_status",
]

NUMERICAL_COLUMNS = [
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

FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS


def load_parameters() -> dict:
    with PARAMS_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def calculate_file_hash(file_path: Path) -> str:
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(65536), b""):
            sha256.update(block)

    return sha256.hexdigest()


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def prepare_datasets() -> tuple:
    train_data = pd.read_parquet(TRAIN_PATH)
    validation_data = pd.read_parquet(VALIDATION_PATH)
    test_data = pd.read_parquet(TEST_PATH)

    missing_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in train_data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Features absentes du dataset : {missing_features}"
        )

    x_train = train_data[FEATURE_COLUMNS]
    y_train = train_data[TARGET_COLUMN].to_numpy()

    x_validation = validation_data[FEATURE_COLUMNS]
    y_validation = validation_data[TARGET_COLUMN].to_numpy()

    x_test = test_data[FEATURE_COLUMNS]
    y_test = test_data[TARGET_COLUMN].to_numpy()

    return (
        train_data,
        validation_data,
        test_data,
        x_train,
        y_train,
        x_validation,
        y_validation,
        x_test,
        y_test,
    )


def create_preprocessor() -> ColumnTransformer:
    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_COLUMNS,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_COLUMNS,
            ),
        ]
    )


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    plt.figure(figsize=(11, 8))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )

    plt.title(title)
    plt.xlabel("Produit prédit")
    plt.ylabel("Produit réel")
    plt.xticks(rotation=40, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_evaluation_artifacts(
    run_directory: Path,
    test_data: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    labels: np.ndarray,
    top_k: int,
    model_name: str,
) -> None:
    run_directory.mkdir(parents=True, exist_ok=True)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    with (run_directory / "classification_report.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=2)

    save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        output_path=run_directory / "confusion_matrix.png",
        title=f"Matrice de confusion — {model_name}",
    )

    top_k_results = get_top_k_predictions(
        probabilities=y_proba,
        labels=labels,
        top_k=top_k,
    )

    prediction_rows = []

    for row_index, recommendations in enumerate(top_k_results):
        row = {
            "client_id": test_data.iloc[row_index]["client_id"],
            "actual_product": y_true[row_index],
            "predicted_product": y_pred[row_index],
        }

        for recommendation in recommendations:
            rank = recommendation["rank"]
            row[f"top_{rank}_product"] = recommendation["product"]
            row[f"top_{rank}_probability"] = round(
                recommendation["probability"],
                6,
            )

        prediction_rows.append(row)

    pd.DataFrame(prediction_rows).to_csv(
        run_directory / "test_predictions.csv",
        index=False,
    )

    feature_schema = {
        "target": TARGET_COLUMN,
        "identifier_columns": IDENTIFIER_COLUMNS,
        "categorical_features": CATEGORICAL_COLUMNS,
        "numerical_features": NUMERICAL_COLUMNS,
        "all_features": FEATURE_COLUMNS,
        "classes": labels.tolist(),
    }

    with (run_directory / "feature_schema.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(feature_schema, file, indent=2)


def evaluate_model(
    model: Pipeline,
    x_data: pd.DataFrame,
    y_data: np.ndarray,
    top_k: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    predictions = model.predict(x_data)
    probabilities = model.predict_proba(x_data)
    labels = model.classes_

    metrics = calculate_classification_metrics(
        y_true=y_data,
        y_pred=predictions,
        y_proba=probabilities,
        labels=labels,
        top_k=top_k,
    )

    return metrics, predictions, probabilities, labels


def log_dataset_information(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> None:
    mlflow.log_params(
        {
            "train_rows": len(train_data),
            "validation_rows": len(validation_data),
            "test_rows": len(test_data),
            "feature_count": len(FEATURE_COLUMNS),
            "target_column": TARGET_COLUMN,
            "dataset_sha256": calculate_file_hash(
                FEATURE_DATA_PATH
            ),
        }
    )


def run_experiment(
    model: Pipeline,
    model_name: str,
    run_name: str,
    model_parameters: dict[str, Any],
    datasets: tuple,
    top_k: int,
    git_commit: str,
) -> dict[str, Any]:
    (
        train_data,
        validation_data,
        test_data,
        x_train,
        y_train,
        x_validation,
        y_validation,
        x_test,
        y_test,
    ) = datasets

    run_directory = REPORT_DIRECTORY / run_name

    with mlflow.start_run(run_name=run_name) as active_run:
        mlflow.set_tags(
            {
                "project": "LANALA",
                "phase": "phase-3",
                "model_role": "baseline",
                "model_type": model_name,
                "git_commit": git_commit,
            }
        )

        mlflow.log_params(model_parameters)
        log_dataset_information(
            train_data,
            validation_data,
            test_data,
        )

        model.fit(x_train, y_train)

        (
            validation_metrics,
            _,
            _,
            _,
        ) = evaluate_model(
            model=model,
            x_data=x_validation,
            y_data=y_validation,
            top_k=top_k,
        )

        (
            test_metrics,
            test_predictions,
            test_probabilities,
            labels,
        ) = evaluate_model(
            model=model,
            x_data=x_test,
            y_data=y_test,
            top_k=top_k,
        )

        metrics_to_log = {
            **{
                f"validation_{name}": value
                for name, value in validation_metrics.items()
            },
            **{
                f"test_{name}": value
                for name, value in test_metrics.items()
            },
        }

        mlflow.log_metrics(metrics_to_log)

        save_evaluation_artifacts(
            run_directory=run_directory,
            test_data=test_data,
            y_true=y_test,
            y_pred=test_predictions,
            y_proba=test_probabilities,
            labels=labels,
            top_k=top_k,
            model_name=model_name,
        )

        mlflow.log_artifacts(
            str(run_directory),
            artifact_path="evaluation",
        )

        signature = infer_signature(
            x_train.head(10),
            model.predict(x_train.head(10)),
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=x_train.head(3),
            serialization_format="cloudpickle",
        )

        result = {
            "run_id": active_run.info.run_id,
            "run_name": run_name,
            "model_name": model_name,
            "model_uri": model_info.model_uri,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
        }

        print(f"\nExpérience terminée : {run_name}")
        print(f"Run ID : {active_run.info.run_id}")
        print(json.dumps(metrics_to_log, indent=2))

        return result


def main() -> None:
    parameters = load_parameters()
    baseline_parameters = parameters["baseline"]
    logistic_parameters = baseline_parameters["logistic_regression"]

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        baseline_parameters["tracking_uri"],
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(
        baseline_parameters["experiment_name"]
    )

    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for child in REPORT_DIRECTORY.iterdir():
        if child.is_dir():
            shutil.rmtree(child)

    datasets = prepare_datasets()
    top_k = baseline_parameters["top_k"]
    git_commit = get_git_commit()

    dummy_model = Pipeline(
        steps=[
            (
                "classifier",
                DummyClassifier(
                    strategy="prior",
                ),
            )
        ]
    )

    dummy_result = run_experiment(
        model=dummy_model,
        model_name="DummyClassifier",
        run_name="naive-most-popular",
        model_parameters={
            "strategy": "prior",
            "top_k": top_k,
        },
        datasets=datasets,
        top_k=top_k,
        git_commit=git_commit,
    )

    logistic_model = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=logistic_parameters["max_iter"],
                    class_weight=logistic_parameters["class_weight"],
                    solver=logistic_parameters["solver"],
                    random_state=logistic_parameters["random_seed"],
                ),
            ),
        ]
    )

    logistic_result = run_experiment(
        model=logistic_model,
        model_name="LogisticRegression",
        run_name="logistic-regression-baseline",
        model_parameters={
            "max_iter": logistic_parameters["max_iter"],
            "class_weight": logistic_parameters["class_weight"],
            "solver": logistic_parameters["solver"],
            "random_seed": logistic_parameters["random_seed"],
            "top_k": top_k,
        },
        datasets=datasets,
        top_k=top_k,
        git_commit=git_commit,
    )

    summary = {
        "experiment_name": baseline_parameters["experiment_name"],
        "tracking_uri": tracking_uri,
        "dataset_sha256": calculate_file_hash(FEATURE_DATA_PATH),
        "naive_baseline": dummy_result,
        "logistic_regression": logistic_result,
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"\nRésumé sauvegardé dans {SUMMARY_PATH}")
    print("\nOuvrez MLflow : http://127.0.0.1:5000")


if __name__ == "__main__":
    main()