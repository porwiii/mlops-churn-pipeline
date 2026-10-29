import math

def validate_model(metrics: dict) -> bool:
    thresholds = {
        "f1": 0.55,
        "roc_auc": 0.80,
    }

    for name, threshold in thresholds.items():
        value = metrics.get(name)

        if value is None:
            return False

        if not math.isfinite(value):
            return False

        if value < threshold:
            return False

    return True