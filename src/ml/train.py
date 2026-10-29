import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)
import mlflow
import mlflow.sklearn

from src.ml.validate import validate_model


df = pd.read_parquet("data/processed/churn")

print(df.shape)
print(df.dtypes)

def train_model(df: pd.DataFrame):
    # Separate X and y
    y = df["Churn"]
    X = df.drop(["customerID", "Churn"], axis=1)

    # Train/test split
    # 80% train+validation, 20% test
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 75% z pozostałych 80% = 60% całego zbioru
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev,
        test_size=0.25,
        random_state=42,
        stratify=y_dev
    )

    # Create preprocessing pipeline
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    ct = ColumnTransformer(
    [("scale", StandardScaler(), numeric_features),
     ("ohe", OneHotEncoder(handle_unknown='ignore'), categorical_features)]
     )
    # Train LogisticRegression
    model = Pipeline(
        [("preprocessor", ct), 
         ("classifier", LogisticRegression(max_iter=1000, random_state=42, C=1.0))]
         )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    # Calculate metrics
    metrics = {"accuracy": accuracy_score(y_test, y_val),
               "precision": precision_score(y_test, y_val),
               "recall": recall_score(y_test, y_val),
               "f1": f1_score(y_test, y_val),
               "roc_auc": roc_auc_score(y_test, y_val)
               }

    return model, metrics


def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("churn-prediction")

    df = pd.read_parquet("data/processed/churn")

    with mlflow.start_run():
        model, metrics = train_model(df)

        mlflow.log_params({
            "model_type": "LogisticRegression",
            "C": model.named_steps["classifier"].C,
            "random_state": 42,
            "train_ratio": 0.6,
            "validation_ratio": 0.2,
            "test_ratio": 0.2,
        })

        mlflow.log_metrics(metrics)

        # Evaluate quality gate
        accepted = validate_model(metrics)

        mlflow.set_tag(
            "validation_status",
            "accepted" if accepted else "rejected"
        )

        print("Metrics:", metrics)
        print("Accepted:", accepted)

        if not accepted:
            raise ValueError(
                "Model failed validation quality gate"
            )

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="churn-model",
            input_example=df.drop(
                columns=["customerID", "Churn"]
            ).head(3)
        )

        print("Metrics:", metrics)
        print("Model URI:", model_info.model_uri)

        # loaded_model = mlflow.sklearn.load_model(
        #     model_info.model_uri
        # )

        # X_sample = df.drop(columns=["customerID", "Churn"]).head(5)

        # predictions = loaded_model.predict(X_sample)

        # print("Predictions from loaded model:", predictions)


if __name__ == "__main__":
    main()
