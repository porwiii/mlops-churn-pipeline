from src.data.transform import transform_data


def test_churn_encoding(spark):
    # Arrange — prepare input
    data = [
        ("001", 12, "120.50", "Yes"),
        ("002", 5, "50.00", "No"),
    ]

    columns = [
        "customerID",
        "tenure",
        "TotalCharges",
        "Churn",
    ]

    df = spark.createDataFrame(data, columns)

    # Act — execute transformation
    result = transform_data(df)

    # Assert — verify results
    actual = [
        row["Churn"]
        for row in result.orderBy("customerID").collect()
    ]

    assert actual == [1, 0]

def test_missing_TotalCharges(spark):
    # Arrange — prepare input
    data = [
        ("001", 0, "", "No"),
        ("002", 12, "", "Yes"),
    ]

    columns = [
        "customerID",
        "tenure",
        "TotalCharges",
        "Churn"
    ]

    df = spark.createDataFrame(data, columns)
    
    # Act — execute transformation
    result = transform_data(df)

    # Assert — verify results
    actual = [
        row["TotalCharges"]
        for row in result.orderBy("customerID").collect()
    ]

    assert actual == [0.0, None]

