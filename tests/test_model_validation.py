from src.ml.validate import validate_model

def test_model_accepted():
    metrics = {
        "f1": 0.65,
        "roc_auc": 0.85
    }

    assert validate_model(metrics) is True

def test_model_rejected():
    metrics = {
        "f1": 0.40,
        "roc_auc": 0.90
    }

    assert validate_model(metrics) is False

def test_model_border():
    metrics = {
        "f1": 0.55,
        "roc_auc": 0.80
    }

    assert validate_model(metrics) is True

def test_model_nan():
    metrics = {
        "f1": float("nan"),
        "roc_auc": 0.80
    }

    assert validate_model(metrics) is False