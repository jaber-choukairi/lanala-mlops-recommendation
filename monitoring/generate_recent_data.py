from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_REFERENCE_PATH = Path(
    "data/monitoring/reference.parquet"
)

DEFAULT_OUTPUT_PATH = Path(
    "data/monitoring/current.parquet"
)

DEFAULT_NUMBER_OF_ROWS = 800
DEFAULT_RANDOM_SEED = 42


def parse_arguments() -> argparse.Namespace:
    """Lire les paramètres envoyés dans le terminal."""

    parser = argparse.ArgumentParser(
        description=(
            "Générer des données récentes pour "
            "le monitoring du modèle LANALA."
        )
    )

    parser.add_argument(
        "--reference-path",
        type=Path,
        default=DEFAULT_REFERENCE_PATH,
        help="Chemin des données de référence.",
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Chemin du fichier Parquet à générer.",
    )

    parser.add_argument(
        "--rows",
        type=int,
        default=DEFAULT_NUMBER_OF_ROWS,
        help="Nombre de lignes à générer.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Graine aléatoire pour la reproductibilité.",
    )

    parser.add_argument(
        "--without-drift",
        action="store_true",
        help=(
            "Générer des données récentes "
            "sans dérive volontaire."
        ),
    )

    return parser.parse_args()


def validate_reference_data(
    dataframe: pd.DataFrame,
) -> None:
    """Vérifier que le dataset de référence est exploitable."""

    if dataframe.empty:
        raise ValueError(
            "Le dataset de référence est vide."
        )

    if len(dataframe.columns) == 0:
        raise ValueError(
            "Le dataset de référence ne contient "
            "aucune colonne."
        )

    required_columns = {
        "monthly_income",
        "account_balance",
        "digital_activity_score",
        "number_of_transactions",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        print(
            "Attention : certaines colonnes prévues "
            "pour la simulation sont absentes :"
        )

        for column in sorted(missing_columns):
            print(f"  - {column}")


def create_recent_sample(
    reference: pd.DataFrame,
    number_of_rows: int,
    random_seed: int,
) -> pd.DataFrame:
    """
    Créer un échantillon récent à partir de la référence.

    Si le nombre demandé dépasse la taille du dataset,
    l'échantillonnage avec remplacement est utilisé.
    """

    if number_of_rows <= 0:
        raise ValueError(
            "--rows doit être strictement supérieur à zéro."
        )

    use_replacement = (
        number_of_rows > len(reference)
    )

    current = reference.sample(
        n=number_of_rows,
        replace=use_replacement,
        random_state=random_seed,
    )

    return current.reset_index(drop=True)


def apply_numerical_drift(
    dataframe: pd.DataFrame,
    random_generator: np.random.Generator,
) -> pd.DataFrame:
    """Appliquer une dérive volontaire aux colonnes numériques."""

    current = dataframe.copy()

    if "monthly_income" in current.columns:
        monthly_income = pd.to_numeric(
            current["monthly_income"],
            errors="coerce",
        )

        noise = random_generator.normal(
            loc=1.65,
            scale=0.08,
            size=len(current),
        )

        current["monthly_income"] = (
            monthly_income * noise
        ).clip(
            lower=0,
            upper=50000,
        )

    if "account_balance" in current.columns:
        account_balance = pd.to_numeric(
            current["account_balance"],
            errors="coerce",
        )

        noise = random_generator.normal(
            loc=0.45,
            scale=0.05,
            size=len(current),
        )

        current["account_balance"] = (
            account_balance * noise
        ).clip(
            lower=0,
            upper=300000,
        )

    if "digital_activity_score" in current.columns:
        activity_score = pd.to_numeric(
            current["digital_activity_score"],
            errors="coerce",
        )

        activity_noise = random_generator.normal(
            loc=0.25,
            scale=0.04,
            size=len(current),
        )

        current["digital_activity_score"] = (
            activity_score + activity_noise
        ).clip(
            lower=0,
            upper=1,
        )

    if "number_of_transactions" in current.columns:
        transactions = pd.to_numeric(
            current["number_of_transactions"],
            errors="coerce",
        )

        noise = random_generator.normal(
            loc=1.50,
            scale=0.10,
            size=len(current),
        )

        current["number_of_transactions"] = (
            transactions * noise
        ).round().clip(
            lower=0,
            upper=100000,
        )

    if "average_transaction_amount" in current.columns:
        transaction_amount = pd.to_numeric(
            current["average_transaction_amount"],
            errors="coerce",
        )

        noise = random_generator.normal(
            loc=1.30,
            scale=0.08,
            size=len(current),
        )

        current["average_transaction_amount"] = (
            transaction_amount * noise
        ).clip(
            lower=0,
            upper=100000,
        )

    if "credit_score" in current.columns:
        credit_score = pd.to_numeric(
            current["credit_score"],
            errors="coerce",
        )

        score_change = random_generator.normal(
            loc=-60,
            scale=20,
            size=len(current),
        )

        current["credit_score"] = (
            credit_score + score_change
        ).round().clip(
            lower=300,
            upper=850,
        )

    return current


def apply_categorical_drift(
    dataframe: pd.DataFrame,
    random_generator: np.random.Generator,
) -> pd.DataFrame:
    """Appliquer une dérive à quelques variables catégorielles."""

    current = dataframe.copy()

    if "employment_status" in current.columns:
        number_to_change = max(
            1,
            int(len(current) * 0.40),
        )

        selected_indices = random_generator.choice(
            current.index.to_numpy(),
            size=min(
                number_to_change,
                len(current),
            ),
            replace=False,
        )

        current.loc[
            selected_indices,
            "employment_status",
        ] = "self_employed"

    if "marital_status" in current.columns:
        number_to_change = max(
            1,
            int(len(current) * 0.30),
        )

        selected_indices = random_generator.choice(
            current.index.to_numpy(),
            size=min(
                number_to_change,
                len(current),
            ),
            replace=False,
        )

        current.loc[
            selected_indices,
            "marital_status",
        ] = "single"

    return current


def preserve_reference_types(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> pd.DataFrame:
    """
    Essayer de conserver les types du dataset de référence.

    Les colonnes entières modifiées sont arrondies avant
    leur conversion.
    """

    result = current.copy()

    for column in reference.columns:
        if column not in result.columns:
            continue

        reference_type = reference[column].dtype

        try:
            if pd.api.types.is_integer_dtype(
                reference_type
            ):
                result[column] = (
                    pd.to_numeric(
                        result[column],
                        errors="coerce",
                    )
                    .round()
                    .astype(reference_type)
                )

            elif pd.api.types.is_float_dtype(
                reference_type
            ):
                result[column] = (
                    pd.to_numeric(
                        result[column],
                        errors="coerce",
                    )
                    .astype(reference_type)
                )

            else:
                result[column] = result[column].astype(
                    reference_type
                )

        except (TypeError, ValueError):
            print(
                "Attention : impossible de restaurer "
                f"le type de la colonne '{column}'."
            )

    return result


def print_comparison(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> None:
    """Afficher une comparaison rapide dans le terminal."""

    monitored_columns = [
        "monthly_income",
        "account_balance",
        "digital_activity_score",
        "number_of_transactions",
        "average_transaction_amount",
        "credit_score",
    ]

    print()
    print("COMPARAISON RAPIDE")
    print("=" * 72)

    for column in monitored_columns:
        if (
            column not in reference.columns
            or column not in current.columns
        ):
            continue

        reference_values = pd.to_numeric(
            reference[column],
            errors="coerce",
        )

        current_values = pd.to_numeric(
            current[column],
            errors="coerce",
        )

        reference_mean = (
            reference_values.mean()
        )

        current_mean = (
            current_values.mean()
        )

        print(
            f"{column:<32} "
            f"référence={reference_mean:>12.3f} "
            f"récent={current_mean:>12.3f}"
        )


def generate_recent_data(
    reference_path: Path,
    output_path: Path,
    number_of_rows: int,
    random_seed: int,
    simulate_drift: bool,
) -> pd.DataFrame:
    """Exécuter toute la génération des données récentes."""

    if not reference_path.exists():
        raise FileNotFoundError(
            "Fichier de référence introuvable : "
            f"{reference_path}. "
            "Exécutez d'abord : "
            "python -m monitoring.build_reference_data"
        )

    if not reference_path.is_file():
        raise IsADirectoryError(
            "Le chemin de référence est un dossier : "
            f"{reference_path}"
        )

    reference = pd.read_parquet(
        reference_path
    )

    validate_reference_data(reference)

    current = create_recent_sample(
        reference=reference,
        number_of_rows=number_of_rows,
        random_seed=random_seed,
    )

    if simulate_drift:
        random_generator = (
            np.random.default_rng(
                random_seed
            )
        )

        current = apply_numerical_drift(
            dataframe=current,
            random_generator=random_generator,
        )

        current = apply_categorical_drift(
            dataframe=current,
            random_generator=random_generator,
        )

    current = preserve_reference_types(
        reference=reference,
        current=current,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    current.to_parquet(
        output_path,
        index=False,
    )

    print()
    print("DONNÉES RÉCENTES GÉNÉRÉES")
    print("=" * 72)
    print(
        f"Référence          : {reference_path}"
    )
    print(
        f"Sortie             : {output_path}"
    )
    print(
        f"Nombre de lignes   : {len(current)}"
    )
    print(
        f"Nombre de colonnes : {len(current.columns)}"
    )
    print(
        "Drift simulé       : "
        f"{'oui' if simulate_drift else 'non'}"
    )
    print(
        f"Graine aléatoire   : {random_seed}"
    )

    print_comparison(
        reference=reference,
        current=current,
    )

    return current


def main() -> None:
    arguments = parse_arguments()

    generate_recent_data(
        reference_path=arguments.reference_path,
        output_path=arguments.output_path,
        number_of_rows=arguments.rows,
        random_seed=arguments.seed,
        simulate_drift=not arguments.without_drift,
    )


if __name__ == "__main__":
    main()