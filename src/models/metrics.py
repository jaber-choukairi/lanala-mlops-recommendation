from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import top_k_accuracy_score


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    labels: np.ndarray,
    top_k: int = 3,
) -> dict[str, float]:
    effective_k = min(top_k, len(labels))

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        f"top_{effective_k}_accuracy": top_k_accuracy_score(
            y_true,
            y_proba,
            k=effective_k,
            labels=labels,
        ),
    }

    return {
        metric_name: float(metric_value)
        for metric_name, metric_value in metrics.items()
    }


def get_top_k_predictions(
    probabilities: np.ndarray,
    labels: np.ndarray,
    top_k: int = 3,
) -> list[list[dict[str, Any]]]:
    effective_k = min(top_k, len(labels))

    descending_indices = np.argsort(
        probabilities,
        axis=1,
    )[:, ::-1][:, :effective_k]

    results = []

    for row_index, class_indices in enumerate(descending_indices):
        recommendations = []

        for rank, class_index in enumerate(class_indices, start=1):
            recommendations.append(
                {
                    "rank": rank,
                    "product": str(labels[class_index]),
                    "probability": float(
                        probabilities[row_index, class_index]
                    ),
                }
            )

        results.append(recommendations)

    return results