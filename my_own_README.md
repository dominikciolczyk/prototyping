`conda activate zenml-env`

`zenml login --local`

`docker start mlflow-postgres`

`mlflow server \
  --backend-store-uri postgresql://mlflow_user:mlflow_pass@localhost:5432/mlflow_db \
  --default-artifact-root s3://cloud-resource-prediction-artifacts/mlflow \
  --host 0.0.0.0 \
  --port 5000`

`python run.py`


conda create -n zenml-env python=3.9 -y
conda activate zenml-env
pip install "zenml[server]"
zenml integration install pandas sklearn pytorch mlflow evidently s3 -y
#pip install statsmodels sktime boto3 s3fs
pip install statsmodels sktime openpyxl optuna psycopg2-binary datetime
