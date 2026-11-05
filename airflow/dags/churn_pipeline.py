from datetime import datetime, timedelta, timezone
import os
import subprocess
import sys

from airflow.sdk import dag, task


@dag(
    dag_id="churn_data_pipeline",
    start_date=datetime(2026, 11, 1, tzinfo=timezone.utc),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "data-engineering"],
)
def churn_data_pipeline():

    @task(retries=2, retry_delay=timedelta(minutes=1))
    def process_data():
        subprocess.run(
            [sys.executable, "-m", "src.data.ingest"],
            cwd="/opt/project",
            check=True
        )
        return "/opt/project/data/processed/churn"

    @task(retries=1)
    def train_model_task(data_path: str):
        import mlflow
        import mlflow.sklearn
        import pandas as pd

        # Make project imports available
        sys.path.insert(0, "/opt/project")
        from src.ml.train import train_model

        mlflow.set_tracking_uri(
            os.environ["MLFLOW_TRACKING_URI"]
        )
        mlflow.set_experiment("churn-prediction")

        df = pd.read_parquet(data_path)

        with mlflow.start_run(
            run_name="airflow-churn-training"
        ) as run:
            model, metrics = train_model(df)

            mlflow.log_metrics(metrics)
            mlflow.log_params({
                "model_type": "LogisticRegression",
                "C": model.named_steps["classifier"].C,
                "source": "airflow",
            })

            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                name="churn-model",
                input_example=df.drop(
                    columns=["customerID", "Churn"]
                ).head(3),
            )

            return {
                "run_id": run.info.run_id,
                "model_uri": model_info.model_uri,
                "metrics": {
                    name: float(value)
                    for name, value in metrics.items()
                },
            }

    @task(retries=0)
    def validate_model_task(training_result: dict):
        sys.path.insert(0, "/opt/project")
        from src.ml.validate import validate_model

        metrics = training_result["metrics"]
        accepted = validate_model(metrics)

        if not accepted:
            raise ValueError(
                f"Model failed quality gate: {metrics}"
            )

        return training_result

    @task(retries=0)
    def register_model_task(training_result: dict):
        import mlflow
        from mlflow import MlflowClient

        mlflow.set_tracking_uri(
            os.environ["MLFLOW_TRACKING_URI"]
        )

        registered = mlflow.register_model(
            model_uri=training_result["model_uri"],
            name="churn-classifier",
        )
        
        client = MlflowClient()
        model_name = "churn-classifier"
        run_id = training_result["run_id"]

        # Check existing versions
        versions = client.search_model_versions(
            filter_string=f"name = '{model_name}'"
        )

        for version in versions:
            if version.tags.get("source_run_id") == run_id:
                print(
                    f"Run {run_id} already registered "
                    f"as version {version.version}"
                )
                return str(version.version)

        # Register only if no matching version exists
        registered = mlflow.register_model(
            model_uri=training_result["model_uri"],
            name=model_name,
        )

        client.set_model_version_tag(
            name="churn-classifier",
            version=registered.version,
            key="source_run_id",
            value=training_result["run_id"]
        )

        print(f"Registered version: {registered.version}")
        return str(registered.version)

    data_path = process_data()
    training_result = train_model_task(data_path)
    validated = validate_model_task(training_result)
    register_model_task(validated)


churn_data_pipeline()
