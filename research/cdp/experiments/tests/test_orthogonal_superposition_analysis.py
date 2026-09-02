from analyze_orthogonal_superposition import classify_ratio


def test_zero_accuracy_loss_passes_despite_low_order_logit_drift() -> None:
    row = {
        "R": 128,
        "evaluated_regimes": 128,
        "baseline_accuracy": 0.9142,
        "maximum_absolute_accuracy_gap": 0.0,
        "maximum_absolute_logit_difference": 1.1e-5,
        "accuracy_zero_loss": True,
        "runtime": {
            "rejection_probe": {
                "all_rejected": True,
                "unauthorized_model_calls": 0,
            }
        },
    }
    assert classify_ratio(row)["status"] == "pass"


def test_accuracy_loss_still_fails() -> None:
    row = {
        "R": 128,
        "evaluated_regimes": 128,
        "baseline_accuracy": 0.9142,
        "maximum_absolute_accuracy_gap": 1 / 7600,
        "maximum_absolute_logit_difference": 1.1e-5,
        "accuracy_zero_loss": False,
        "runtime": {
            "rejection_probe": {
                "all_rejected": True,
                "unauthorized_model_calls": 0,
            }
        },
    }
    assert classify_ratio(row)["status"] == "failure"
