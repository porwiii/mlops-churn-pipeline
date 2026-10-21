from src.data.transform import transform_data
from src.data.validate import validate_data
import pytest


def test_full_pipeline(spark, tmp_path):
    # Arrange
    csv_path = tmp_path / "input.csv"
    parquet_path = tmp_path / "processed"

    csv_path.write_text(
        "customerID,tenure,MonthlyCharges,TotalCharges,Churn\n"
        "001,0,25.0,,No\n"
        "002,12,50.0,600.0,Yes\n"
        "003,5,30.0,150.0,No\n"
    )

    # Act
    df = spark.read.csv(
        str(csv_path),
        header=True,
        inferSchema=True
    )

    df = transform_data(df)
    validate_data(df)

    df.write.mode("overwrite").parquet(
        str(parquet_path)
    )

    result = spark.read.parquet(
        str(parquet_path)
    )

    # Assert
    assert result.count() == 3

    assert result.schema["TotalCharges"].dataType.simpleString() == "double"
    assert result.schema["Churn"].dataType.simpleString() == "int"

    rows = result.orderBy("customerID").collect()

    assert rows[0]["TotalCharges"] == 0.0
    assert rows[1]["TotalCharges"] == 600.0

    assert [row["Churn"] for row in rows] == [0, 1, 0]


def test_pipeline_rejects_invalid_data(spark, tmp_path):
    # Arrange
    csv_path = tmp_path / "input.csv"
    parquet_path = tmp_path / "processed"

    csv_path.write_text(
        "customerID,tenure,MonthlyCharges,TotalCharges,Churn\n"
        "001,0,-25.0,,No\n"
        "002,12,50.0,600.0,Yes\n"
        "003,5,30.0,150.0,No\n"
    )

    # Act
    df = spark.read.csv(
        str(csv_path),
        header=True,
        inferSchema=True
    )

    with pytest.raises(ValueError, match="Invalid MonthlyCharges"):
        run_pipeline(df, str(parquet_path))

    assert not parquet_path.exists()


def run_pipeline(df, output_path):
    df = transform_data(df)
    validate_data(df)
    df.write.mode("overwrite").parquet(output_path)