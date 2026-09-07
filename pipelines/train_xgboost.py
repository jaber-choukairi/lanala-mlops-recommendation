import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import yaml
from mlflow.models import infer_signature
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from pipelines.train_baselines import FEATURE_COLUMNS
from pipelines.train_baselines import create_preprocessor
from src.models.champion import evaluate_promotion
from src.models.metrics import calculate_classification_metrics
from src.models.metrics import get_top_k_predictions


PARAMS_PATH = Path("params.yaml")
FEATURES_PATH = Path("data/features/client_features.parquet")
TRAIN_PATH = Path("data/splits/train.parquet")
VALIDATION_PATH = Path("data/splits/validation.parquet")
TEST_PATH = Path("data/splits/test.parquet")

REPORT_DIRECTORY = Path("reports/phase4")
SUMMARY_PATH = REPORT_DIRECTORY / "summary.json"

TARGET_COLUMN = "next_product"


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


def load_datasets() -> tuple:
    train_data = pd.read_parquet(TRAIN_PATH)
    validation_data = pd.read_parquet(VALIDATION_PATH)
    test_data = pd.read_parquet(TEST_PATH)

    x_train = train_data[FEATURE_COLUMNS]
    x_validation = validation_data[FEATURE_COLUMNS]
    x_test = test_data[FEATURE_COLUMNS]

    label_encoder = LabelEncoder()

    y_train = label_encoder.fit_transform(
        train_data[TARGET_COLUMN]
    )
    y_validation = label_encoder.transform(
        validation_data[TARGET_COLUMN]
    )
    y_test = label_encoder.transform(
        test_data[TARGET_COLUMN]
    )

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
        label_encoder,
    )


def create_xgboost_model(
    model_parameters: dict[str, Any],
    number_of_classes: int,
    random_seed: int,
    n_jobs: int,
) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=number_of_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=random_seed,
        n_jobs=n_jobs,
        **model_parameters,
    )


def calculate_metrics(
    y_true_encoded: np.ndarray,
    probabilities: np.ndarray,
    label_encoder: LabelEncoder,
    top_k: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    predicted_encoded = np.argmax(probabilities, axis=1)

    y_true = label_encoder.inverse_transform(y_true_encoded)
    y_predicted = label_encoder.inverse_transform(predicted_encoded)
    labels = label_encoder.classes_

    metrics = calculate_classification_metrics(
        y_true=y_true,
        y_pred=y_predicted,
        y_proba=probabilities,
        labels=labels,
        top_k=top_k,
    )

    return metrics, y_true, y_predicted


def get_champion_metrics(
    experiment_id: str,
) -> tuple[dict[str, float], str]:
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=(
            "tags.model_type = 'LogisticRegression'"
        ),
        order_by=["metrics.test_top_3_accuracy DESC"],
        max_results=1,
    )

    if runs.empty:
        raise ValueError(
            "Aucun run LogisticRegression trouvé dans MLflow. "
            "Vous devez d'abord terminer la phase 3."
        )

    run = runs.iloc[0]

    metrics = {
        "test_accuracy": float(
            run["metrics.test_accuracy"]
        ),
        "test_macro_f1": float(
            run["metrics.test_macro_f1"]
        ),
        "test_weighted_f1": float(
            run["metrics.test_weighted_f1"]
        ),
        "test_top_3_accuracy": float(
            run["metrics.test_top_3_accuracy"]
        ),
    }

    return metrics, str(run["run_id"])


def save_confusion_matrix(
    y_true: np.ndarray,
    y_predicted: np.ndarray,
    labels: np.ndarray,
) -> None:
    matrix = confusion_matrix(
        y_true,
        y_predicted,
        labels=labels,
    )

    plt.figure(figsize=(11, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=labels,
        yticklabels=labels,
    )

    plt.title("Matrice de confusion — XGBoost Challenger")
    plt.xlabel("Produit prédit")
    plt.ylabel("Produit réel")
    plt.xticks(rotation=40, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        REPORT_DIRECTORY / "confusion_matrix.png",
        dpi=150,
    )
    plt.close()


def save_feature_importance(
    preprocessor,
    model: XGBClassifier,
) -> None:
    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    importance_data = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    importance_data.to_csv(
        REPORT_DIRECTORY / "feature_importance.csv",
        index=False,
    )

    top_features = importance_data.head(20).sort_values(
        "importance",
        ascending=True,
    )

    plt.figure(figsize=(10, 8))
    plt.barh(
        top_features["feature"],
        top_features["importance"],
        color="#2e8b57",
    )
    plt.title("Top 20 des features — XGBoost")
    plt.xlabel("Importance")
    plt.tight_layout()

    plt.savefig(
        REPORT_DIRECTORY / "feature_importance.png",
        dpi=150,
    )
    plt.close()


def save_optuna_history(study: optuna.Study) -> None:
    history = study.trials_dataframe()
    history.to_csv(
        REPORT_DIRECTORY / "optuna_trials.csv",
        index=False,
    )

    completed_trials = history[
        history["state"] == "COMPLETE"
    ]

    if completed_trials.empty:
        return

    plt.figure(figsize=(10, 6))
    plt.plot(
        completed_trials["number"],
        completed_trials["value"],
        marker="o",
        linestyle="none",
        alpha=0.65,
        label="Essais",
    )

    cumulative_best = completed_trials["value"].cummax()

    plt.plot(
        completed_trials["number"],
        cumulative_best,
        color="red",
        linewidth=2,
        label="Meilleur score atteint",
    )

    plt.title("Historique d’optimisation Optuna")
    plt.xlabel("Numéro de l’essai")
    plt.ylabel("Top-3 Accuracy validation")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        REPORT_DIRECTORY / "optuna_history.png",
        dpi=150,
    )
    plt.close()


def save_test_predictions(
    test_data: pd.DataFrame,
    y_true: np.ndarray,
    y_predicted: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
    top_k: int,
) -> None:
    top_k_predictions = get_top_k_predictions(
        probabilities=probabilities,
        labels=labels,
        top_k=top_k,
    )

    rows = []

    for row_index, recommendations in enumerate(
        top_k_predictions
    ):
        row = {
            "client_id": test_data.iloc[row_index]["client_id"],
            "actual_product": y_true[row_index],
            "predicted_product": y_predicted[row_index],
        }

        for recommendation in recommendations:
            rank = recommendation["rank"]
            row[f"top_{rank}_product"] = (
                recommendation["product"]
            )
            row[f"top_{rank}_probability"] = round(
                recommendation["probability"],
                6,
            )

        rows.append(row)

    pd.DataFrame(rows).to_csv(
        REPORT_DIRECTORY / "test_predictions.csv",
        index=False,
    )


def main() -> None:
    all_parameters = load_parameters()
    parameters = all_parameters["xgboost"]
    search_space = parameters["search_space"]
    promotion_parameters = parameters["promotion"]

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        parameters["tracking_uri"],
    )

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(
        parameters["experiment_name"]
    )

    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)

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
        label_encoder,
    ) = load_datasets()

    preprocessor = create_preprocessor()

    x_train_transformed = preprocessor.fit_transform(x_train)
    x_validation_transformed = preprocessor.transform(
        x_validation
    )
    x_test_transformed = preprocessor.transform(x_test)

    number_of_classes = len(label_encoder.classes_)
    top_k = parameters["top_k"]

    champion_metrics, champion_run_id = get_champion_metrics(
        experiment.experiment_id
    )

    with mlflow.start_run(
        run_name="xgboost-optuna-study"
    ) as parent_run:

        mlflow.set_tags(
            {
                "project": "LANALA",
                "phase": "phase-4",
                "model_role": "challenger",
                "optimization": "optuna",
                "git_commit": get_git_commit(),
            }
        )

        mlflow.log_params(
            {
                "number_of_trials": parameters[
                    "number_of_trials"
                ],
                "top_k": top_k,
                "optimization_metric": parameters[
                    "optimization_metric"
                ],
                "random_seed": parameters["random_seed"],
                "train_rows": len(train_data),
                "validation_rows": len(validation_data),
                "test_rows": len(test_data),
                "number_of_classes": number_of_classes,
                "dataset_sha256": calculate_file_hash(
                    FEATURES_PATH
                ),
                "champion_run_id": champion_run_id,
            }
        )

        def objective(trial: optuna.Trial) -> float:
            trial_parameters = {
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    search_space["n_estimators_min"],
                    search_space["n_estimators_max"],
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    search_space["max_depth_min"],
                    search_space["max_depth_max"],
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    search_space["learning_rate_min"],
                    search_space["learning_rate_max"],
                    log=True,
                ),
                "min_child_weight": trial.suggest_int(
                    "min_child_weight",
                    search_space["min_child_weight_min"],
                    search_space["min_child_weight_max"],
                ),
                "gamma": trial.suggest_float(
                    "gamma",
                    search_space["gamma_min"],
                    search_space["gamma_max"],
                ),
                "subsample": trial.suggest_float(
                    "subsample",
                    search_space["subsample_min"],
                    search_space["subsample_max"],
                ),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree",
                    search_space[
                        "colsample_bytree_min"
                    ],
                    search_space[
                        "colsample_bytree_max"
                    ],
                ),
                "reg_alpha": trial.suggest_float(
                    "reg_alpha",
                    search_space["reg_alpha_min"],
                    search_space["reg_alpha_max"],
                    log=True,
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda",
                    search_space["reg_lambda_min"],
                    search_space["reg_lambda_max"],
                    log=True,
                ),
            }

            model = create_xgboost_model(
                model_parameters=trial_parameters,
                number_of_classes=number_of_classes,
                random_seed=parameters["random_seed"],
                n_jobs=parameters["n_jobs"],
            )

            with mlflow.start_run(
                run_name=f"xgboost-trial-{trial.number:03d}",
                nested=True,
            ):
                mlflow.set_tags(
                    {
                        "model_type": "XGBoost",
                        "model_role": "optuna-trial",
                        "trial_number": str(trial.number),
                    }
                )

                mlflow.log_params(trial_parameters)

                model.fit(
                    x_train_transformed,
                    y_train,
                )

                validation_probabilities = (
                    model.predict_proba(
                        x_validation_transformed
                    )
                )

                validation_metrics, _, _ = calculate_metrics(
                    y_true_encoded=y_validation,
                    probabilities=validation_probabilities,
                    label_encoder=label_encoder,
                    top_k=top_k,
                )

                metrics_to_log = {
                    f"validation_{name}": value
                    for name, value
                    in validation_metrics.items()
                }

                mlflow.log_metrics(metrics_to_log)

                return validation_metrics[
                    f"top_{top_k}_accuracy"
                ]

        sampler = optuna.samplers.TPESampler(
            seed=parameters["random_seed"]
        )

        study = optuna.create_study(
            study_name=parameters["study_name"],
            direction="maximize",
            sampler=sampler,
        )

        study.optimize(
            objective,
            n_trials=parameters["number_of_trials"],
            show_progress_bar=True,
        )

        best_parameters = study.best_params

        final_model = create_xgboost_model(
            model_parameters=best_parameters,
            number_of_classes=number_of_classes,
            random_seed=parameters["random_seed"],
            n_jobs=parameters["n_jobs"],
        )

        final_model.fit(
            x_train_transformed,
            y_train,
        )

        validation_probabilities = final_model.predict_proba(
            x_validation_transformed
        )

        validation_metrics, _, _ = calculate_metrics(
            y_true_encoded=y_validation,
            probabilities=validation_probabilities,
            label_encoder=label_encoder,
            top_k=top_k,
        )

        test_probabilities = final_model.predict_proba(
            x_test_transformed
        )

        (
            test_metrics,
            y_test_labels,
            test_predictions,
        ) = calculate_metrics(
            y_true_encoded=y_test,
            probabilities=test_probabilities,
            label_encoder=label_encoder,
            top_k=top_k,
        )

        challenger_metrics = {
            f"test_{name}": value
            for name, value in test_metrics.items()
        }

        promotion_decision = evaluate_promotion(
            champion_metrics=champion_metrics,
            challenger_metrics=challenger_metrics,
            minimum_top_3_accuracy=promotion_parameters[
                "minimum_top_3_accuracy"
            ],
            minimum_top_3_improvement=promotion_parameters[
                "minimum_top_3_improvement"
            ],
            maximum_macro_f1_regression=promotion_parameters[
                "maximum_macro_f1_regression"
            ],
        )

        mlflow.log_params(
            {
                f"best_{name}": value
                for name, value in best_parameters.items()
            }
        )

        mlflow.log_metrics(
            {
                **{
                    f"best_validation_{name}": value
                    for name, value
                    in validation_metrics.items()
                },
                **challenger_metrics,
                "top_3_improvement_vs_champion": (
                    promotion_decision["differences"][
                        "top_3_accuracy"
                    ]
                ),
                "macro_f1_difference_vs_champion": (
                    promotion_decision["differences"][
                        "macro_f1"
                    ]
                ),
            }
        )

        mlflow.set_tags(
            {
                "model_type": "XGBoost",
                "promotion_decision": promotion_decision[
                    "decision"
                ],
                "selected_model": promotion_decision[
                    "selected_model"
                ],
            }
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", final_model),
            ]
        )

        encoded_predictions = pipeline.predict(
            x_test.head(10)
        )

        signature = infer_signature(
            x_test.head(10),
            encoded_predictions,
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            signature=signature,
            input_example=x_test.head(3),
            serialization_format="cloudpickle",
        )

        joblib.dump(
            label_encoder,
            REPORT_DIRECTORY / "label_encoder.joblib",
        )

        save_confusion_matrix(
            y_true=y_test_labels,
            y_predicted=test_predictions,
            labels=label_encoder.classes_,
        )

        save_feature_importance(
            preprocessor=preprocessor,
            model=final_model,
        )

        save_optuna_history(study)

        save_test_predictions(
            test_data=test_data,
            y_true=y_test_labels,
            y_predicted=test_predictions,
            probabilities=test_probabilities,
            labels=label_encoder.classes_,
            top_k=top_k,
        )

        classification_results = classification_report(
            y_test_labels,
            test_predictions,
            labels=label_encoder.classes_,
            output_dict=True,
            zero_division=0,
        )

        with (
            REPORT_DIRECTORY / "classification_report.json"
        ).open("w", encoding="utf-8") as file:
            json.dump(
                classification_results,
                file,
                indent=2,
            )

        with (
            REPORT_DIRECTORY / "promotion_decision.json"
        ).open("w", encoding="utf-8") as file:
            json.dump(
                promotion_decision,
                file,
                indent=2,
            )

        summary = {
            "experiment_name": parameters["experiment_name"],
            "parent_run_id": parent_run.info.run_id,
            "model_uri": model_info.model_uri,
            "dataset_sha256": calculate_file_hash(
                FEATURES_PATH
            ),
            "number_of_trials": len(study.trials),
            "best_trial_number": study.best_trial.number,
            "best_validation_score": float(
                study.best_value
            ),
            "best_parameters": best_parameters,
            "champion": {
                "model": "logistic_regression",
                "run_id": champion_run_id,
                "metrics": champion_metrics,
            },
            "challenger": {
                "model": "xgboost",
                "run_id": parent_run.info.run_id,
                "validation_metrics": validation_metrics,
                "test_metrics": challenger_metrics,
            },
            "promotion": promotion_decision,
        }

        with SUMMARY_PATH.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(summary, file, indent=2)

        mlflow.log_artifacts(
            str(REPORT_DIRECTORY),
            artifact_path="evaluation",
        )

        print("\nOptimisation terminée")
        print(f"Meilleur essai : {study.best_trial.number}")
        print(f"Meilleur score validation : {study.best_value:.4f}")
        print("\nMeilleurs paramètres :")
        print(json.dumps(best_parameters, indent=2))
        print("\nDécision :")
        print(json.dumps(promotion_decision, indent=2))


if __name__ == "__main__":
    main()