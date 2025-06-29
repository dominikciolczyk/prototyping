import pandas as pd
import numpy as np
from river import preprocessing

# --- 1. Przykładowy DataFrame ---
df = pd.DataFrame({
    'x': [10, 12, 14, 16],
    'y': [20, 22, 24, 26]
})

# --- 2. Inicjalizacja skalera ---
scaler = preprocessing.StandardScaler()

# --- 3. Uczenie skalera na danych ---
for row in df.to_dict(orient="records"):
    scaler.learn_one(row)

# --- 4. Transformacja danych ---
scaled_rows = [scaler.transform_one(row) for row in df.to_dict(orient="records")]
scaled_df = pd.DataFrame(scaled_rows)
print("Scaled Data:\n", scaled_df)

# --- 5. Odwrócenie transformacji ---
restored = []

for row in scaled_rows:
    restored_row = {}
    for col in row:
        mean = scaler.means[col]
        std = np.sqrt(scaler.vars[col]) + 1e-8
        restored_row[col] = row[col] * std + mean
    restored.append(restored_row)

restored_df = pd.DataFrame(restored)
print("\nRestored Data:\n", restored_df)

# --- 6. Porównanie z oryginałem ---
print("\nOriginal Data:\n", df)
print("\nDifference:\n", df - restored_df)
