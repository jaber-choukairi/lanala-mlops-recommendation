from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


REFERENCE_PATH = Path(
    "data/monitoring/reference.parquet"
)
CURRENT_PATH = Path(
    "data/monitoring/current.parquet"
)
REPORT_DIRECTORY = Path(
    "reports/monitoring"
)

DRIFT_SHARE_THRESHOLD = 0.30


def find_metric_value(
    value: Any,
    searched_keys: set[str],
) -> Any:
    """Rechercher récursivement une valeur dans le JSON Evidently."""

    if isinstance(value, dict):
        for key, child_value in value.items():
            if key in searched_keys:
                return child_value

        for child_value in value.values():
            result = find_metric_value(
                child_value,
                searched_keys,
            )
            if result is not None:
                return result

    if isinstance(value, list):
        for item in value:
            result = find_metric_value(
                item,
                searched_keys,
            )
            if result is not None:
                return result

    return None


def align_datasets(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_columns = sorted(
        set(reference.columns)
        & set(current.columns)
    )

    if not common_columns:
        raise ValueError(
            "Aucune colonne commune entre référence et données récentes."
        )

    reference = reference[common_columns].copy()
    current = current[common_columns].copy()

    for column in common_columns:
        if reference[column].dtype != current[column].dtype:
            try:
                current[column] = current[column].astype(
                    reference[column].dtype
                )
            except (TypeError, ValueError):
                reference[column] = reference[column].astype(str)
                current[column] = current[column].astype(str)

    return reference, current


def main() -> None:
    reference = pd.read_parquet(REFERENCE_PATH)
    current = pd.read_parquet(CURRENT_PATH)

    reference, current = align_datasets(
        reference,
        current,
    )

    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = Report(
        [
            DataDriftPreset(
                drift_share=DRIFT_SHARE_THRESHOLD
            )
        ],
        include_tests=True,
    )

    snapshot = report.run(
        current_data=current,
        reference_data=reference,
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    html_path = (
        REPORT_DIRECTORY
        / f"data_drift_{timestamp}.html"
    )
    json_path = (
        REPORT_DIRECTORY
        / f"data_drift_{timestamp}.json"
    )
    summary_path = (
        REPORT_DIRECTORY
        / "latest_drift_summary.json"
    )

    snapshot.save_html(str(html_path))

    report_dictionary = snapshot.dict()

    json_path.write_text(
        json.dumps(
            report_dictionary,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    drift_share = find_metric_value(
        report_dictionary,
        {
            "share",
            "drift_share",
            "drifted_share",
        },
    )

    dataset_drift = find_metric_value(
        report_dictionary,
        {
            "dataset_drift",
            "drift_detected",
        },
    )

    if drift_share is None:
        print(
            "Attention : impossible d'extraire automatiquement "
            "drift_share. Consultez le rapport HTML."
        )
        drift_share = 0.0

    drift_share = float(drift_share)
    retraining_required = (
        drift_share >= DRIFT_SHARE_THRESHOLD
    )

    summary = {
        "created_at": timestamp,
        "reference_path": str(REFERENCE_PATH),
        "current_path": str(CURRENT_PATH),
        "reference_rows": len(reference),
        "current_rows": len(current),
        "total_columns": len(reference.columns),
        "drift_share": drift_share,
        "dataset_drift": bool(dataset_drift)
        if dataset_drift is not None
        else retraining_required,
        "threshold": DRIFT_SHARE_THRESHOLD,
        "retraining_required": retraining_required,
        "html_report": str(html_path),
        "json_report": str(json_path),
    }

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()