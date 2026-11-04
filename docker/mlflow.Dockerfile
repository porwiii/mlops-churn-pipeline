FROM python:3.12-slim

WORKDIR /app

COPY requirements-mlflow.lock.txt .

RUN pip install --no-cache-dir -r requirements-mlflow.lock.txt

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
