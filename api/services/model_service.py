import os
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient

from src.features.schema import PRODUCT_OWNERSHIP_COLUMNS
from src.features.serving import build_serving_features


class ModelService:
    def __init__(self) -> None:
        self.tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI",
            "http://127.0.0.1:5000",
        )

        self.model_name = os.getenv(
            "MODEL_NAME",
            "lanala-banking-recommender",
        )

        self.model_alias = os.getenv(
            "MODEL_ALIAS",
            "staging",
        )

        self.top_k = int(
            os.getenv("TOP_K", "3")
        )

        self.label_encoder_path = Path(
            os.getenv(
                "LABEL_ENCODER_PATH",
                "reports/phase4/label_encoder.joblib",
            )
        )

        self.model: Any | None = None
        self.label_encoder: Any | None = None
        self.model_version: str | None = None
        self.loading_error: str | None = None

    @property
    def is_loaded(self) -> bool:
        return (
            self.model is not None
            and self.label_encoder is not None
            and self.model_version is not None
        )

    def load(self) -> None:
        mlflow.set_tracking_uri(self.tracking_uri)

        client = MlflowClient(
            tracking_uri=self.tracking_uri
        )

        model_version = (
            client.get_model_version_by_alias(
                name=self.model_name,
                alias=self.model_alias,
            )
        )

        model_uri = (
            f"models:/{self.model_name}"
            f"@{self.model_alias}"
        )

        self.model = mlflow.sklearn.load_model(
            model_uri
        )

        if not self.label_encoder_path.exists():
            raise FileNotFoundError(
                "LabelEncoder introuvable : "
                f"{self.label_encoder_path}"
            )

        self.label_encoder = joblib.load(
            self.label_encoder_path
        )

        self.model_version = str(
            model_version.version
        )

        self.loading_error = None

        print(
            f"Modèle chargé : {self.model_name} "
            f"version={self.model_version} "
            f"alias={self.model_alias}"
        )

    def predict(
        self,
        client_data: dict,
    ) -> list[dict]:
        if not self.is_loaded:
            self.load()

        features = build_serving_features(
            client_data
        )

        probabilities = self.model.predict_proba(
            features
        )[0]

        encoded_classes = np.arange(
            len(probabilities)
        )

        product_names = (
            self.label_encoder.inverse_transform(
                encoded_classes
            )
        )

        owned_products = {
            product
            for product, ownership_column
            in PRODUCT_OWNERSHIP_COLUMNS.items()
            if client_data[ownership_column] == 1
        }

        candidates = [
            {
                "product": str(product),
                "probability": float(probability),
            }
            for product, probability
            in zip(product_names, probabilities)
            if product not in owned_products
        ]

        candidates.sort(
            key=lambda item: item["probability"],
            reverse=True,
        )

        recommendations = []

        for rank, candidate in enumerate(
            candidates[: self.top_k],
            start=1,
        ):
            recommendations.append(
                {
                    "rank": rank,
                    "product": candidate["product"],
                    "probability": round(
                        candidate["probability"],
                        6,
                    ),
                }
            )

        return recommendations


model_service = ModelService()