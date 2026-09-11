from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]

SUMMARY_PATH = ROOT_DIR / "reports" / "phase4" / "summary.json"
GATES_PATH = ROOT_DIR / "config" / "quality_gates.yaml"
OUTPUT_PATH = ROOT_DIR / "reports" / "phase6" / "quality_gate_result.json"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def evaluate_quality_gate(summary: dict, configuration: dict) -> dict:
    champion = summary["champion"]["metrics"]
    challenger = summary["challenger"]["test_metrics"]
    thresholds = configuration["model"]

    top_3_accuracy = challenger["test_top_3_accuracy"]
    macro_f1 = challenger["test_macro_f1"]

    top_3_improvement = (
        top_3_accuracy - champion["test_top_3_accuracy"]
    )

    macro_f1_difference = (
        macro_f1 - champion["test_macro_f1"]
    )

    gates = {
        "minimum_test_top_3_accuracy": (
            top_3_accuracy
            >= thresholds["minimum_test_top_3_accuracy"]
        ),
        "minimum_test_macro_f1": (
            macro_f1
            >= thresholds["minimum_test_macro_f1"]
        ),
        "minimum_top_3_improvement": (
            top_3_improvement
            >= thresholds["minimum_top_3_improvement"]
        ),
        "macro_f1_non_regression": (
            macro_f1_difference
            >= -thresholds["maximum_macro_f1_regression"]
        ),
    }

    passed = all(gates.values())

    return {
        "passed": passed,
        "selected_model": (
            summary["challenger"]["model"]
            if passed
            else summary["champion"]["model"]
        ),
        "metrics": {
            "test_top_3_accuracy": top_3_accuracy,
            "test_macro_f1": macro_f1,
            "top_3_improvement": top_3_improvement,
            "macro_f1_difference": macro_f1_difference,
        },
        "thresholds": thresholds,
        "gates": gates,
    }


def main() -> None:
    try:
        summary = load_json(SUMMARY_PATH)
        configuration = load_yaml(GATES_PATH)

        result = evaluate_quality_gate(summary, configuration)

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

        with OUTPUT_PATH.open("w", encoding="utf-8") as file:
            json.dump(result, file, indent=2, ensure_ascii=False)

        print("\nQUALITY GATE")
        print("=" * 70)

        for gate_name, passed in result["gates"].items():
            status = "PASS" if passed else "FAIL"
            print(f"{gate_name:<45} {status}")

        print("-" * 70)
        print(f"Décision globale : {'PASS' if result['passed'] else 'FAIL'}")
        print(f"Modèle retenu : {result['selected_model']}")

        if not result["passed"]:
            print(
                "\nLe challenger ne respecte pas tous les seuils. "
                "Le pipeline est interrompu."
            )
            sys.exit(1)

        print("\nLe modèle respecte tous les seuils.")

    except (KeyError, TypeError, FileNotFoundError) as error:
        print(f"Erreur pendant le Quality Gate : {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()