from src.data.validate import validate_data

import pytest


def test_valid_data(spark):
    data = [
        ("001", 5, 100.0, 1, 10.0),
        ("002", 7, 50.0, 0, 45.0),
    ]

    columns = [
        "customerID",
        "tenure",
        "TotalCharges",
        "Churn",
        "MonthlyCharges",
    ]

    df = spark.createDataFrame(data, columns)

    validate_data(df)

def test_invalid_MonthlyCharges(spark):
    # Arrange — prepare input
    data = [
        ("001", 5, 100.0, 1, 10.0),
        ("002", 7, 50.0, 0, -45.0),
    ]

    columns = [
        "customerID",
        "tenure",
        "TotalCharges",
        "Churn",
        "MonthlyCharges",
    ]

    df = spark.createDataFrame(data, columns)

    # Assert — verify results
    with pytest.raises(ValueError,match="Invalid MonthlyCharges"):
        validate_data(df)

def test_duplicate_customer_id(spark):
    # Arrange — prepare input
    data = [
        ("002", 5, 100.0, 1, 10.0),
        ("002", 7, 50.0, 0, -45.0),
    ]

    columns = [
        "customerID",
        "tenure",
        "TotalCharges",
        "Churn",
        "MonthlyCharges",
    ]

    df = spark.createDataFrame(data, columns)

    # Assert — verify results
    with pytest.raises(ValueError, match="duplicated IDs:"):
        validate_data(df)
