import json
from pathlib import Path


SUMMARY_PATH = Path("reports/phase4/summary.json")


def format_metric(value: float) -> str:
    return f"{value:.4f}"


def main() -> None:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            "Le rapport de phase 4 n'existe pas. "
            "Exécutez d'abord train_xgboost."
        )

    with SUMMARY_PATH.open("r", encoding="utf-8") as file:
        summary = json.load(file)

    champion = summary["champion"]
    challenger = summary["challenger"]
    promotion = summary["promotion"]

    champion_metrics = champion["metrics"]
    challenger_metrics = challenger["test_metrics"]

    print("\nCOMPARAISON CHAMPION / CHALLENGER")
    print("=" * 72)

    print(
        f"{'Métrique':<30}"
        f"{'Logistic Regression':>20}"
        f"{'XGBoost':>15}"
    )

    print("-" * 72)

    metric_names = [
        "test_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
        "test_top_3_accuracy",
    ]

    for metric_name in metric_names:
        print(
            f"{metric_name:<30}"
            f"{format_metric(champion_metrics[metric_name]):>20}"
            f"{format_metric(challenger_metrics[metric_name]):>15}"
        )

    print("\nGates de promotion")
    print("-" * 72)

    for gate_name, passed in promotion["gates"].items():
        status = "PASS" if passed else "FAIL"
        print(f"{gate_name:<40} {status}")

    print("\nDécision finale")
    print("-" * 72)
    print(f"Décision : {promotion['decision']}")
    print(f"Modèle sélectionné : {promotion['selected_model']}")
    print(f"Explication : {promotion['reason']}")


if __name__ == "__main__":
    main()