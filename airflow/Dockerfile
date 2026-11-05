FROM apache/airflow:3.3.1-python3.12

USER root

# Java required by PySpark
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless && \
    rm -rf /var/lib/apt/lists/*

USER airflow

# Spark dependencies
RUN pip install --no-cache-dir \
    "apache-airflow==3.3.1" \
    "pyspark==4.2.0" \
    "mlflow==3.17.2" \
    "numpy==2.5.4" \
    "pandas==3.1.0" \
    "scikit-learn==1.9.1" \
    "scipy==1.18.1"