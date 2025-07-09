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
pip install statsmodels sktime openpyxl optuna psycopg2-binary datetime river imageio seaborn


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



Teacher:
AsymmetricSmoothL1  |  model: 2.0404  |  max baseline: 2.2802

Student:
 AsymmetricSmoothL1  |  model: 2.1258  |  max baseline: 2.2802


[online_evaluator] Average loss for model: 1.3855
[online_evaluator] Average loss for baseline: 2.0800


Po powrocie punkt wyjcia teachera:
AsymmetricSmoothL1  |  model: 0.9416  |  max baseline: 1.2622


New best:
AsymmetricSmoothL1  |  model: 0.8874  |  max baseline: 1.2622
  batch: 64
  cnn_channels: [64]
  kernels: [13]
  hidden_lstm: 126
  lstm_layers: 1
  dropout_rate: 0.1
  alpha: 10
  beta: 3.0
  lr: 0.001


AsymmetricSmoothL1  |  model: 0.8605  |  max baseline: 1.2622
  batch: 64
  cnn_channels: [ 64 ]
  kernels: [ 13 ]
  hidden_lstm: 512
  lstm_layers: 1
  dropout_rate: 0.1
  alpha: 10
  beta: 3.0
  lr: 0.001



AsymmetricSmoothL1  |  model: 0.8495  |  max baseline: 1.2622
  batch: 64
  cnn_channels: [ 64 ]
  kernels: [ 12 ]
  hidden_lstm: 512
  lstm_layers: 1
  dropout_rate: 0.1
  alpha: 10
  beta: 3.0
  lr: 0.001


AsymmetricSmoothL1  |  model: 0.8407  |  max baseline: 1.2622
  batch: 64
  cnn_channels: [ 32, 64, 128 ]
  kernels: [ 3, 5, 7 ]
  hidden_lstm: 512
  lstm_layers: 1
  dropout_rate: 0.1
  alpha: 10
  beta: 3.0
  lr: 0.001


AsymmetricSmoothL1  |  model: 0.8258  |  max baseline: 1.2622
  batch: 64
  cnn_channels: [ 64, 128, 256 ]
  kernels: [ 3, 5, 7 ]
  hidden_lstm: 512
  lstm_layers: 1
  dropout_rate: 0.1
  alpha: 10
  beta: 3.0
  lr: 0.001

AsymmetricSmoothL1  |  model: 0.7855  |  max baseline: 1.2622
  batch: 64
  cnn_channels: [ 64, 128, 256, 512 ]
  kernels: [ 3, 5, 7, 9 ]
  hidden_lstm: 256
  lstm_layers: 1
  dropout_rate: 0.1
  alpha: 10
  beta: 3.0
  lr: 0.001



Saved best result: {'best_params': {'min_strength': 0.8896879145148959, 'correlation_threshold': 0.946155123922659, 'threshold_strategy': 'std', 'threshold': 3.30948408750141, 'q':
 0.954166045314732, 'reduction_method': 'ffill_bfill', 'use_hour_features': False, 'use_day_of_week_features': False, 'is_weekend_mode': 'none'}, 'best_value': 0.6897233128547668, 'best_trial_number': 78, 'timestamp': '2025-07-07T04:49:04.700292'}


Saved best result: {'best_params': {'min_strength': 0.8418052233367769, 'correlation_threshold': 0.9569165272783219, 'threshold': 4.797380848863149, 'reduction_method': 'interpolat
e_spline', 'interpolation_order': 2, 'use_hour_features': False, 'use_day_of_week_features': False, 'is_weekend_mode': 'none'}, 'best_value': 0.6887093186378479, 'best_trial_number': 147, 'timestamp': '2025-07-07T11:53:37.181835'}