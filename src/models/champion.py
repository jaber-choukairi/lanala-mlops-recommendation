from typing import Any


def evaluate_promotion(
    champion_metrics: dict[str, float],
    challenger_metrics: dict[str, float],
    minimum_top_3_accuracy: float,
    minimum_top_3_improvement: float,
    maximum_macro_f1_regression: float,
) -> dict[str, Any]:
    champion_top_3 = champion_metrics["test_top_3_accuracy"]
    challenger_top_3 = challenger_metrics["test_top_3_accuracy"]

    champion_macro_f1 = champion_metrics["test_macro_f1"]
    challenger_macro_f1 = challenger_metrics["test_macro_f1"]

    top_3_improvement = challenger_top_3 - champion_top_3
    macro_f1_difference = challenger_macro_f1 - champion_macro_f1

    gates = {
        "absolute_top_3_threshold": (
            challenger_top_3 >= minimum_top_3_accuracy
        ),
        "top_3_improvement": (
            top_3_improvement >= minimum_top_3_improvement
        ),
        "macro_f1_non_regression": (
            macro_f1_difference >= -maximum_macro_f1_regression
        ),
    }

    promoted = all(gates.values())

    if promoted:
        decision = "challenger_promoted"
        selected_model = "xgboost"
        reason = (
            "XGBoost respecte tous les seuils et devient "
            "le nouveau Champion."
        )
    else:
        decision = "challenger_rejected"
        selected_model = "logistic_regression"
        failed_gates = [
            gate_name
            for gate_name, passed in gates.items()
            if not passed
        ]
        reason = (
            "XGBoost ne respecte pas toutes les règles. "
            f"Règles échouées : {', '.join(failed_gates)}."
        )

    return {
        "promoted": promoted,
        "decision": decision,
        "selected_model": selected_model,
        "reason": reason,
        "gates": gates,
        "differences": {
            "top_3_accuracy": round(top_3_improvement, 6),
            "macro_f1": round(macro_f1_difference, 6),
        },
    }