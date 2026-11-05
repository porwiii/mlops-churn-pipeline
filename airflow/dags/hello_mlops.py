from datetime import datetime, timedelta, timezone
from airflow.sdk import dag, task

@dag(
    dag_id="hello_mlops",
    start_date=datetime(2026, 11, 1, tzinfo=timezone.utc),
    schedule=None,
    catchup=False,
    tags=["mlops", "learning"],
)
def hello_mlops():

    @task(retries=2, retry_delay=timedelta(seconds=30))
    def check_data():
        print("Checking input data...")
        return 7043

    @task
    def report(row_count: int):
        print(f"Dataset contains {row_count} rows")
        print("Data validation completed!")

    report(check_data())


hello_mlops()
