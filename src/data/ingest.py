from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def main():
    spark = (
        SparkSession.builder
        .appName("churn-pipeline")
        .master("local[*]")
        .getOrCreate()
    )

    try:
        df = spark.read.csv(
            "data/raw/Telco-Customer-Churn.csv",
            header=True,
            inferSchema=True
        )

        df.show(5)

        df.printSchema()

        print("Number of rows:", df.count())

        df.select("customerID", "TotalCharges", "tenure", "MonthlyCharges", "Churn").show(5)

        df.groupBy("Churn").count().show()

        expressions = [F.count(
                            F.when(F.col(col_name).isNull(), 1)
                        ).alias(f"missing_{col_name}") 
                        for col_name in df.columns]

        df.select(*expressions).show()

        df.groupBy("customerID") \
            .count() \
            .filter(F.col("count") > 1) \
            .show()

        invalid_tenure = df.filter(F.col("tenure") < 0).count()
        print(f"Invalid tenure values: {invalid_tenure}")

        df.filter(
            F.col("MonthlyCharges") < 0
        ).show()

        df.filter(
            ~F.col("Churn").isin("Yes", "No")
        ).show()

        df.filter(
            F.trim(F.col("TotalCharges")) == ""
        ).show()

        df = df.withColumn(
            "TotalCharges",
            F.expr("try_cast(TotalCharges AS DOUBLE)")
        )

        df.select(
            F.count(
                F.when(F.col("TotalCharges").isNull(), 1)
                ).alias("missing_TotalCharges")
        ).show()

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
