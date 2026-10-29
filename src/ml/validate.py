def validate_model(metrics: dict) -> bool:
    thresholds = {
        "f1": 0.55,
        "roc_auc": 0.80,
    }

    for metric_name, threshold in thresholds.items():
        if metrics[metric_name] < threshold:
            return False

    return True
