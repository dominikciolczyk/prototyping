`zenml login --local`

`docker start mlflow-postgres`

`mlflow server \
  --backend-store-uri postgresql://mlflow_user:mlflow_pass@localhost:5432/mlflow_db \
  --default-artifact-root s3://cloud-resource-prediction-artifacts/mlflow \
  --host 0.0.0.0 \
  --port 5000`

`conda activate PD1`

`python run.py`