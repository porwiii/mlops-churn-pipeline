import mlflow
from mlflow import MlflowClient


def promote_model(version: str):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    client = MlflowClient()
    model_name = "churn-classifier"

    # Verify model was accepted
    model_version = client.get_model_version(
        name=model_name,
        version=version
    )

    if model_version.tags.get("validation_status") != "accepted":
        raise ValueError(
            "Cannot promote a model that failed validation"
        )

    # Assign champion alias
    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=version
    )

    print(f"Promoted {model_name} v{version} to champion")


if __name__ == "__main__":
    promote_model("2")
