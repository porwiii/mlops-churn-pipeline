from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def transform_data(df: DataFrame) -> DataFrame:
    # Convert TotalCharges to numeric
    df = df.withColumn(
        "TotalCharges",
        F.expr("try_cast(TotalCharges AS DOUBLE)")
    )

    # Fill missing charges for new customers
    df = df.withColumn(
        "TotalCharges",
        F.when(
            (F.col("tenure") == 0) &
            F.col("TotalCharges").isNull(),
            0.0
        ).otherwise(F.col("TotalCharges"))
    )

    # Encode target variable
    df = df.withColumn(
        "Churn",
        F.when(F.col("Churn") == "Yes", 1)
         .when(F.col("Churn") == "No", 0)
         .otherwise(None)
    )

    return df
