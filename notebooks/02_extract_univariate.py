#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


from pathlib import Path
from importlib import reload
import pandas as pd
import matplotlib
import numpy as np
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from src.constants import PROCESSED_POLCOM_DATA_2022_M_PATH, PROCESSED_POLCOM_DATA_2022_Y_PATH, PROCESSED_AZURE_DATA_PATH


# In[3]:


dfs_M = {file.name: pd.read_parquet(file) for file in Path(PROCESSED_POLCOM_DATA_2022_M_PATH).glob("*.parquet")}
dfs_Y = {file.name: pd.read_parquet(file) for file in Path(PROCESSED_POLCOM_DATA_2022_Y_PATH).glob("*.parquet")}


# In[4]:


def filter_cpu_absolute(cols):
    return [col for col in cols if ("CPU" in col) and "PERCENT" not in col]

# Apply filtering to keep only absolute CPU usage (MHz) metrics
cpu_absolute_only_dfs_M = {file: df[filter_cpu_absolute(df.columns)] for file, df in dfs_M.items()}
cpu_absolute_only_dfs_Y = {file: df[filter_cpu_absolute(df.columns)] for file, df in dfs_Y.items()}


# In[5]:


remaining_columns_M = {file: df.columns.tolist() for file, df in cpu_absolute_only_dfs_M.items()}
remaining_columns_Y = {file: df.columns.tolist() for file, df in cpu_absolute_only_dfs_Y.items()}

remaining_columns_M, remaining_columns_Y


# In[6]:


cpu_absolute_only_dfs_M


# In[7]:


split_datasets = {}

# Iterate over each VM dataset
for vm, df in cpu_absolute_only_dfs_M.items():
    for col in df.columns:
        # Create a new dataframe for each column and rename the column to a predefined name
        split_df = df[[col]].rename(columns={col: 'METRIC_VALUE'})
        split_datasets[f"{vm}_{col}"] = split_df


# In[8]:


cpu_absolute_only_dfs_M = split_datasets


# In[9]:


#min_date = max(df.index.min() for df in cpu_absolute_only_dfs_M.values())
#max_date = min(df.index.max() for df in cpu_absolute_only_dfs_M.values())

#print(f"Common Time Range: {min_date} to {max_date}")


# In[10]:


#df_combined = pd.concat(cpu_absolute_only_dfs_M.values(), axis=1, keys=cpu_absolute_only_dfs_M.keys())

# Rename columns to show which VM they belong to
#df_combined.columns = [f"{vm}_{col}" for vm, col in df_combined.columns]

# Display result
#df_combined


# In[11]:


# Updated function to remove both NaNs and leading/trailing zeros while preserving gaps in the middle
def trim_nan_and_zero_edges(df):
    """Removes NaN and zero values only at the beginning and end of a DataFrame while preserving gaps in the middle."""
    first_valid_idx = df[(df.notna()) & (df != 0)].first_valid_index()
    last_valid_idx = df[(df.notna()) & (df != 0)].last_valid_index()
    return df.loc[first_valid_idx:last_valid_idx]

# Apply trimming to each VM dataset
cpu_trimmed_dfs_M = {vm: trim_nan_and_zero_edges(df) for vm, df in cpu_absolute_only_dfs_M.items()}

# Verify results: Check if NaN and zero values at the edges were removed
zero_nan_summary = {vm: (df.isin([0]).sum().sum(), df.isna().sum().sum(), df.index.min(), df.index.max()) for vm, df in cpu_trimmed_dfs_M.items()}
zero_nan_summary


# In[12]:


len(cpu_trimmed_dfs_M)


# In[13]:


# remove dataset if it contains any NaN values

cpu_trimmed_dfs_M = {vm: df for vm, df in cpu_trimmed_dfs_M.items() if df.isna().sum().sum() == 0}



# In[14]:


len(cpu_trimmed_dfs_M)


# In[15]:


cpu_trimmed_dfs_M


# In[ ]:


# Plot each VM's CPU usage separately
for vm, df in cpu_absolute_only_dfs_M.items():
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["METRIC_VALUE"], label=f"CPU Usage - {vm}", color="blue")
    plt.xlabel("Time")
    plt.ylabel("CPU Usage (MHz)")
    plt.title(f"CPU Usage Over Time for {vm}")
    plt.legend()
    plt.show()


# In[17]:


from sklearn.ensemble import IsolationForest

def detect_anomalies(df, column_name, contamination=0.01):
    """Detect anomalies in a time series using Isolation Forest."""
    clf = IsolationForest(contamination=contamination, random_state=42)
    df['anomaly'] = clf.fit_predict(df[[column_name]])  # -1 for anomaly, 1 for normal
    return df

# Apply anomaly detection to each VM dataset
cpu_anomaly_dfs_M = {vm: detect_anomalies(df, column_name="METRIC_VALUE") for vm, df in cpu_trimmed_dfs_M.items()}

# Plot CPU usage with anomalies marked
for vm, df in cpu_anomaly_dfs_M.items():
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["METRIC_VALUE"], label=f"CPU Usage - {vm}", color="blue")
    
    # Highlight anomalies
    anomalies = df[df['anomaly'] == -1]
    plt.scatter(anomalies.index, anomalies["METRIC_VALUE"], color='red', label='Anomalies', marker='x')
    
    plt.xlabel("Time")
    plt.ylabel("CPU Usage (MHz)")
    plt.title(f"CPU Usage with Anomaly Detection for {vm}")
    plt.legend()
    plt.show()


# In[20]:


def create_multi_step_sequences(data, column_name, seq_length=10, forecast_horizon=5):
    """Convert time series into sequences for multi-step forecasting with anomaly flags."""
    X, y = [], []
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        seq_values = data.iloc[i:i+seq_length][[column_name, 'anomaly']].values  # Include anomaly flag
        target_values = data.iloc[i+seq_length:i+seq_length+forecast_horizon][column_name].values
        X.append(seq_values)
        y.append(target_values)
    return np.array(X), np.array(y)

seq_length = 50
forecast_horizon = 50

X_train_list, y_train_list = [], []

for vm, df in cpu_anomaly_dfs_M.items():
    scaler = MinMaxScaler()
    df['METRIC_VALUE'] = scaler.fit_transform(df[['METRIC_VALUE']])

    # Convert anomalies to binary (0 for normal, 1 for anomalies)
    df['anomaly'] = df['anomaly'].apply(lambda x: 1 if x == -1 else 0)

    X, y = create_multi_step_sequences(df, seq_length=seq_length, forecast_horizon=forecast_horizon, column_name="METRIC_VALUE")

    X_train_list.append(X)
    y_train_list.append(y)

# Combine all VM datasets
X_train = np.vstack(X_train_list)
y_train = np.vstack(y_train_list)

print("Final Training Shape:", X_train.shape, y_train.shape)


# In[21]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Flatten

def build_model(seq_length, forecast_horizon):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation="relu", input_shape=(seq_length, 2)),  # Two input features now
        LSTM(50, return_sequences=False),
        Flatten(),
        Dense(forecast_horizon)  # Predict multiple future steps
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# Train model
model = build_model(seq_length=seq_length, forecast_horizon=forecast_horizon)
model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=1)


# In[20]:


# Read the CSV file with timestamp parsing.
df = pd.read_csv(f'{PROCESSED_AZURE_DATA_PATH}azure.csv', parse_dates=['timestamp'])

# Clean column names: e.g., "avg cpu" becomes "avg_cpu"
df.columns = df.columns.str.strip().str.replace(' ', '_')

# Set the timestamp as the index.
df.set_index('timestamp', inplace=True)

# Resample the data to 2-hour intervals (using mean aggregation) to match your training data.
df_resampled = df.resample('2H').mean().dropna()

# Detect anomalies in the resampled data
df_resampled = detect_anomalies(df_resampled, column_name="avg_cpu")

# Use only the avg_cpu column and anomaly column.
data = df_resampled[['avg_cpu', 'anomaly']].values  # shape: (num_timesteps, 2)

# Scale data
scaler = MinMaxScaler()
azure_scaled = scaler.fit_transform(data)

azure_scaled_df = pd.DataFrame(azure_scaled, index=df_resampled.index, columns=["avg_cpu", "anomaly"])

# Create sequences including anomaly information
X_test, _ = create_multi_step_sequences(azure_scaled_df, column_name="avg_cpu", seq_length=seq_length, forecast_horizon=forecast_horizon)

print("Azure Test Data Shape:", X_test.shape)


# In[21]:


y_pred = model.predict(X_test)

print(f"Predicted CPU usage (normalized) for next {forecast_horizon} steps:", y_pred)


# In[23]:


# Select the last sequence's prediction
y_pred_last = y_pred[-1]  # Take the last predicted sequence (shape: forecast_horizon,)

# Select the corresponding last actual values
y_actual_last = azure_scaled_df["avg_cpu"].iloc[-forecast_horizon:].values  # Match forecast_horizon

# Plot predictions vs actual values
plt.figure(figsize=(10, 5))
plt.plot(range(1, forecast_horizon + 1), y_actual_last, label="Actual (Normalized)", marker="o")
plt.plot(range(1, forecast_horizon + 1), y_pred_last, label="Predicted (Normalized)", marker="x")
plt.xlabel("Time Steps Ahead")
plt.ylabel("Normalized CPU Usage")
plt.title("Azure Data - Multi-Step Prediction")
plt.legend()
plt.show()


# In[99]:


get_ipython().system('jupyter nbconvert --to script 02_extract_univariate.ipynb')


# In[100]:


get_ipython().system('pwd')

