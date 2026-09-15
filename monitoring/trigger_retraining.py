from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


SUMMARY_PATH = Path(
    "reports/monitoring/latest_drift_summary.json"
)


def should_retrain(summary: dict) -> bool:
    return bool(
        summary.get(
            "retraining_required",
            False,
        )
    )


def run_command(command: list[str]) -> None:
    print(
        "Exécution :",
        " ".join(command),
    )
    subprocess.run(
        command,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Exécuter réellement le pipeline "
            "d'entraînement."
        ),
    )
    arguments = parser.parse_args()

    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Rapport absent : {SUMMARY_PATH}"
        )

    summary = json.loads(
        SUMMARY_PATH.read_text(
            encoding="utf-8"
        )
    )

    if not should_retrain(summary):
        print(
            "Aucun réentraînement : "
            "le seuil de drift n'est pas dépassé."
        )
        return

    print(
        "Drift détecté : "
        "un réentraînement est demandé."
    )

    if not arguments.execute:
        print(
            "Mode simulation : aucune commande "
            "d'entraînement n'a été lancée."
        )
        print(
            "Utilisez --execute pour lancer "
            "le réentraînement."
        )
        return

    run_command(
        [
            sys.executable,
            "-m",
            "pipelines.train_xgboost",
        ]
    )
    run_command(
        [
            sys.executable,
            "-m",
            "pipelines.compare_champion_challenger",
        ]
    )


if __name__ == "__main__":
    main()