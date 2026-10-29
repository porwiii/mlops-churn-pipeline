import mlflow
import mlflow.sklearn
import pandas as pd

mlflow.set_tracking_uri("http://127.0.0.1:5000")

model = mlflow.sklearn.load_model(
    "models:/churn-classifier@champion"
)

df = pd.read_parquet("data/processed/churn")
X = df.drop(columns=["customerID", "Churn"])

predictions = model.predict(X.head(5))
probabilities = model.predict_proba(X.head(5))[:, 1]

print("Predictions:", predictions)
print("Probabilities:", probabilities)
