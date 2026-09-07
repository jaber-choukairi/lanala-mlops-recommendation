from src.models.champion import evaluate_promotion


def test_challenger_is_promoted_when_all_gates_pass() -> None:
    champion = {
        "test_top_3_accuracy": 0.80,
        "test_macro_f1": 0.60,
    }

    challenger = {
        "test_top_3_accuracy": 0.84,
        "test_macro_f1": 0.63,
    }

    result = evaluate_promotion(
        champion_metrics=champion,
        challenger_metrics=challenger,
        minimum_top_3_accuracy=0.70,
        minimum_top_3_improvement=0.01,
        maximum_macro_f1_regression=0.01,
    )

    assert result["promoted"] is True
    assert result["selected_model"] == "xgboost"
    assert all(result["gates"].values())


def test_challenger_is_rejected_without_improvement() -> None:
    champion = {
        "test_top_3_accuracy": 0.84,
        "test_macro_f1": 0.63,
    }

    challenger = {
        "test_top_3_accuracy": 0.845,
        "test_macro_f1": 0.64,
    }

    result = evaluate_promotion(
        champion_metrics=champion,
        challenger_metrics=challenger,
        minimum_top_3_accuracy=0.70,
        minimum_top_3_improvement=0.01,
        maximum_macro_f1_regression=0.01,
    )

    assert result["promoted"] is False
    assert result["selected_model"] == "logistic_regression"
    assert result["gates"]["top_3_improvement"] is False


def test_challenger_is_rejected_on_macro_f1_regression() -> None:
    champion = {
        "test_top_3_accuracy": 0.80,
        "test_macro_f1": 0.65,
    }

    challenger = {
        "test_top_3_accuracy": 0.85,
        "test_macro_f1": 0.60,
    }

    result = evaluate_promotion(
        champion_metrics=champion,
        challenger_metrics=challenger,
        minimum_top_3_accuracy=0.70,
        minimum_top_3_improvement=0.01,
        maximum_macro_f1_regression=0.01,
    )

    assert result["promoted"] is False
    assert (
        result["gates"]["macro_f1_non_regression"]
        is False
    )