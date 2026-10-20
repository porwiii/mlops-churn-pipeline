from pyspark.sql import DataFrame, functions as F


def validate_data(df: DataFrame) -> None:
    invalid_tenure = df.filter(
        F.col("tenure").isNull() |
        (F.col("tenure") < 0)
    ).count()

    if invalid_tenure > 0:
        raise ValueError(
            f"Invalid tenure values: {invalid_tenure}"
        )

    invalid_customer_id = df.filter(
        F.col("customerID").isNull() |
        (F.trim(F.col("customerID")) == "")
    ).count()

    multiple_id_count = (
        df.groupBy("customerID")
          .count()
          .filter(F.col("count") > 1)
          .count()
    )

    if invalid_customer_id > 0 or multiple_id_count > 0:
        raise ValueError(
            f"Invalid customer IDs: {invalid_customer_id}, "
            f"duplicated IDs: {multiple_id_count}"
        )

    invalid_monthly_charges = df.filter(
        F.col("MonthlyCharges").isNull() |
        F.isnan(F.col("MonthlyCharges")) |
        (F.col("MonthlyCharges") < 0)
    ).count()

    if invalid_monthly_charges > 0:
        raise ValueError(
            f"Invalid MonthlyCharges values: "
            f"{invalid_monthly_charges}"
        )

    invalid_total_charges = df.filter(
        F.col("TotalCharges").isNull() |
        F.isnan(F.col("TotalCharges")) |
        (F.col("TotalCharges") < 0)
    ).count()

    if invalid_total_charges > 0:
        raise ValueError(
            f"Invalid TotalCharges values: "
            f"{invalid_total_charges}"
        )

    invalid_churn = df.filter(
        F.col("Churn").isNull() |
        (~F.col("Churn").isin(0, 1))
    ).count()

    if invalid_churn > 0:
        raise ValueError(
            f"Invalid Churn values: {invalid_churn}"
        )
