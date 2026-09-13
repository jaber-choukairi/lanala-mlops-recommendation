from pipelines.check_quality_gate import evaluate_quality_gate


CONFIGURATION = {
    "model": {
        "minimum_test_top_3_accuracy": 0.84,
        "minimum_test_macro_f1": 0.33,
        "minimum_top_3_improvement": 0.01,
        "maximum_macro_f1_regression": 0.005,
    }
}


def make_summary(
    champion_top_3: float = 0.82,
    champion_macro_f1: float = 0.34,
    challenger_top_3: float = 0.86,
    challenger_macro_f1: float = 0.35,
) -> dict:
    return {
        "champion": {
            "model": "logistic_regression",
            "metrics": {
                "test_top_3_accuracy": champion_top_3,
                "test_macro_f1": champion_macro_f1,
            },
        },
        "challenger": {
            "model": "xgboost",
            "test_metrics": {
                "test_top_3_accuracy": challenger_top_3,
                "test_macro_f1": challenger_macro_f1,
            },
        },
    }


def test_quality_gate_passes_for_good_challenger() -> None:
    result = evaluate_quality_gate(
        make_summary(),
        CONFIGURATION,
    )

    assert result["passed"] is True
    assert result["selected_model"] == "xgboost"
    assert all(result["gates"].values())


def test_quality_gate_fails_below_top_3_threshold() -> None:
    result = evaluate_quality_gate(
        make_summary(challenger_top_3=0.81),
        CONFIGURATION,
    )

    assert result["passed"] is False
    assert result["selected_model"] == "logistic_regression"
    assert result["gates"]["minimum_test_top_3_accuracy"] is False


def test_quality_gate_fails_without_improvement() -> None:
    result = evaluate_quality_gate(
        make_summary(
            champion_top_3=0.85,
            challenger_top_3=0.855,
        ),
        CONFIGURATION,
    )

    assert result["passed"] is False
    assert result["gates"]["minimum_top_3_improvement"] is False


def test_quality_gate_fails_on_macro_f1_regression() -> None:
    result = evaluate_quality_gate(
        make_summary(
            champion_macro_f1=0.35,
            challenger_macro_f1=0.33,
        ),
        CONFIGURATION,
    )

    assert result["passed"] is False
    assert result["gates"]["macro_f1_non_regression"] is False