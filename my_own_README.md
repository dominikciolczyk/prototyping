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
pip install statsmodels sktime openpyxl optuna psycopg2-binary datetime river


Alfa 9:
 |  model: 0.1886  |  baseline: 0.3880
Alfa 10:
 model: 0.2037  |  baseline: 0.3918

Alfa 8:
AsymmetricL1  |  model: 0.1782  |  baseline: 0.3841

Alfa 10 beta 0.5
symmetricL1  |  model: 0.4028  |  baseline: 0.5654

Alfa 10 beta 0.2 0 - super
AsymmetricL1  |  model: 0.5560  |  baseline: 0.7014

Alfa 10 beta 0.21
AsymmetricL1  |  model: 0.5515  |  baseline: 0.6964

Alfa 10 beta 0.2125
AsymmetricL1  |  model: 0.5606  |  baseline: 0.6952

Alfa 10 beta 0.65
AsymmetricL1  |  model: 0.3715  |  baseline: 0.5070

Alfa 10 beta 0.65 bez weekend
AsymmetricL1  |  model: 0.3281  |  baseline: 0.5070

Alfa 10 beta 0.65 bez day of week - najlepszy wizualnie
model: 0.3360  |  baseline: 0.5070


Można odrzucić kolumnę DISC.


Z kolumną MEMORY:
alfa 10 beta 0.58

0.19 > 0.20 lepsze od 0.21 - za duże loss

beta    = 0.17

beta    0.39

0.75????
0.83



===================

beta 1.29

1.11




========== General model =================
Saved best result: {'best_params': {'n_cnn_layers': 3, 'ch1': 8, 'k1': 11, 'ch2': 72, 'k2': 17, 'ch3': 8, 'k3': 9, 'hidden_lstm': 256, 'lstm_layers': 1}, 'best_value': 0.7900648713111877, 'best_trial_number': 56, 'timestamp': '2025-06-30T04:30:18.681781'}


==================== KD model ===================
Best trial: #66 → value=0.7668, params={'kd_kind': 'AsymmetricSmoothL1', 'distill_alpha': 0.2, 'alpha': 10, 'beta': 1}