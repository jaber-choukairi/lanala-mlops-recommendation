from monitoring.trigger_retraining import (
    should_retrain,
)


def test_retraining_is_triggered() -> None:
    summary = {
        "retraining_required": True,
    }

    assert should_retrain(summary) is True


def test_retraining_is_not_triggered() -> None:
    summary = {
        "retraining_required": False,
    }

    assert should_retrain(summary) is False


def test_missing_value_does_not_trigger() -> None:
    assert should_retrain({}) is False