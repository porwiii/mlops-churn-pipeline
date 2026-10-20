from pyspark.sql import SparkSession

from src.data.transform import transform_data
from src.data.validate import validate_data


def main():
    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName("churn-pipeline")
        .getOrCreate()
    )

    try:
        df = spark.read.csv(
            "data/raw/Telco-Customer-Churn.csv",
            header=True,
            inferSchema=True
        )

        df = transform_data(df)
        validate_data(df)

        output_path = "data/processed/churn"

        df.write.mode("overwrite").parquet(output_path)

        print("Pipeline completed successfully!")

        df_loaded = spark.read.parquet(output_path)

        df_loaded.printSchema()
        print(f"Loaded rows: {df_loaded.count()}")
        

    finally:
        spark.stop()


if __name__ == "__main__":
    main()