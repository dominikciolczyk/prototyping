from kafka import KafkaConsumer
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Load the trained model
model = tf.keras.models.load_model("../models/cpu_usage_model")  # Update with actual path

def detect_anomaly(value, X_seq):
    """Detect if the incoming value is an anomaly using Isolation Forest."""
    if len(X_seq) < 10:  # Ensure we have enough data to train the model
        return 1  # Default to normal if not enough data

    clf = IsolationForest(contamination=0.01, random_state=42)
    clf.fit(np.array(X_seq).reshape(-1, 1))  # Fit on existing sequence
    return clf.predict(np.array(value).reshape(1, -1))[0]  # -1 for anomaly, 1 for normal


# Kafka Consumer Configuration
consumer = KafkaConsumer(
    "avg_cpu_topic",  # Topic name
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))  # Deserialize JSON messages
)

print("Listening for messages on topic: avg_cpu_topic...")

losses = []  # Store loss values for plotting
X_seq, y_seq = [], []  # Store streaming sequences
seq_length = 36
forecast_horizon = 36

# Consume messages
for message in consumer:
    data = message.value  # Extract the message payload

    # Convert new data point to NumPy array
    new_X = np.array([data["new_X"]]).reshape(1, 1)  # Single time point
    new_y = np.array([data["new_y"]]).reshape(1, 1)  # Single target value

    print("Received new data point")

    # Detect anomalies
    anomaly_flag = detect_anomaly(new_X, X_seq)  # Compare with existing sequence

    # Append anomaly flag to new_X
    new_X = np.append(new_X, anomaly_flag).reshape(1, -1, 2)

    # Update dataset with new point
    X_seq.append(new_X)
    y_seq.append(new_y)

    # Ensure sequences match expected length
    if len(X_seq) > seq_length:
        X_seq.pop(0)
        y_seq.pop(0)

    if len(X_seq) == seq_length:
        # Convert to NumPy arrays for training
        X_train = np.array(X_seq).reshape(1, seq_length, 2)
        y_train = np.array(y_seq).reshape(1, forecast_horizon)

        # Make predictions
        predictions = model.predict(X_train)
        print("Predictions:", predictions)

        # Online weight update
        history = model.fit(X_train, y_train, epochs=1, verbose=1, batch_size=1)

        # Store loss for visualization
        losses.append(history.history['loss'][0])

        # Save updated model (optional, if using continual learning)
        model.save("path/to/your_model_updated.h5")  # Update with actual path

        print("Model updated and saved.")
        print("---")

        # Plot training loss
        plt.figure(figsize=(8, 4))
        plt.plot(losses, label='Training Loss', color='blue')
        plt.xlabel("Batch Updates")
        plt.ylabel("Loss")
        plt.title("Model Training Loss Over Time")
        plt.legend()
        plt.show()
