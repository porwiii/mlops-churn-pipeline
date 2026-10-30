import os
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
from fastapi import FastAPI
import pandas as pd
from fastapi import Request

from src.api.schemas import CustomerInput, PredictionResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(
        os.getenv(
            "MLFLOW_TRACKING_URI",
            "http://127.0.0.1:5000"
        )
    )

    # Load champion model at application startup
    app.state.model = mlflow.sklearn.load_model(
        "models:/churn-classifier@champion"
    )

    yield

    # Release reference on shutdown
    app.state.model = None


app = FastAPI(
    title="Churn Prediction API",
    version="0.1.0",
    lifespan=lifespan
)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": app.state.model is not None
    }

@app.get("/")
def root():
    return {"message": "Churn Prediction API"}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput, request: Request):
    # Convert validated Pydantic object into a DataFrame
    X = pd.DataFrame([customer.model_dump()])

    # Use the model loaded at application startup
    model = request.app.state.model

    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0, 1])

    return PredictionResponse(
        prediction=prediction,
        probability=probability
    )
