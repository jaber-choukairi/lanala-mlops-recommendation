import numpy as np

from src.models.metrics import calculate_classification_metrics
from src.models.metrics import get_top_k_predictions


def test_perfect_classification_metrics() -> None:
    labels = np.array(["product_a", "product_b", "product_c"])
    y_true = np.array(["product_a", "product_b", "product_c"])
    y_pred = np.array(["product_a", "product_b", "product_c"])

    probabilities = np.array(
        [
            [0.90, 0.05, 0.05],
            [0.05, 0.90, 0.05],
            [0.05, 0.05, 0.90],
        ]
    )

    metrics = calculate_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=probabilities,
        labels=labels,
        top_k=2,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["top_2_accuracy"] == 1.0


def test_top_k_predictions_are_sorted() -> None:
    labels = np.array(["a", "b", "c"])
    probabilities = np.array([[0.20, 0.70, 0.10]])

    result = get_top_k_predictions(
        probabilities=probabilities,
        labels=labels,
        top_k=2,
    )

    assert result[0][0]["product"] == "b"
    assert result[0][1]["product"] == "a"
    assert result[0][0]["rank"] == 1