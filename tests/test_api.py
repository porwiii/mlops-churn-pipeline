import numpy as np
import pytest

from fastapi.testclient import TestClient
from src.api.main import app


class FakeModel:
    def predict(self, X):
        return np.array([1] * len(X))

    def predict_proba(self, X):
        return np.array([[0.25, 0.75]] * len(X))


@pytest.fixture
def client():
    app.state.model = FakeModel()

    test_client = TestClient(app)
    yield test_client

    test_client.close()
    app.state.model = None

def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "model_loaded": True
    }


@pytest.fixture
def valid_customer():
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 65.5,
        "TotalCharges": 786.0
    }


def test_predict(client, valid_customer):
    response = client.post(
        "/predict",
        json=valid_customer
    )

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] == 1
    assert result["probability"] == pytest.approx(0.75)

def test_negative_tenure(client, valid_customer):
    invalid_customer = valid_customer.copy()
    invalid_customer["tenure"] = -5
    
    response = client.post(
            "/predict",
            json=invalid_customer
        )
    
    assert response.status_code == 422

def test_negative_tenure(client, valid_customer):
    invalid_customer = valid_customer.copy()
    invalid_customer.pop('MonthlyCharges', None)

    response = client.post(
            "/predict",
            json=invalid_customer
        )
    
    assert response.status_code == 422
    
class BrokenModel:
    def predict(self, X):
        raise RuntimeError("Inference failed")

def test_prediction_failure(client, valid_customer):
    app.state.model = BrokenModel()

    test_client = TestClient(
        app,
        raise_server_exceptions=False
    )

    response = test_client.post(
        "/predict",
        json=valid_customer
    )

    assert response.status_code == 500
    test_client.close()
