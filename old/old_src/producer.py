from kafka import KafkaProducer
import json
import pandas as pd
from time import sleep

from src.constants import PROCESSED_AZURE_DATA_PATH
# Read the CSV file with timestamp parsing.
df = pd.read_csv(f'{PROCESSED_AZURE_DATA_PATH}azure.csv', parse_dates=['timestamp'])

# Kafka Producer Configuration
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

topic = "avg_cpu_topic"

print("Starting to stream single time points to Kafka...")

seq_length = 36
forecast_horizon = 36

# Iterate over each time point in the dataset
for i in range(len(df) - forecast_horizon):
    new_X = df.iloc[i][['avg_cpu']].values.tolist()  # Single time point
    new_y = df.iloc[i + forecast_horizon]['avg_cpu'].tolist()  # Future target value

    # Prepare the message payload
    message = {
        "new_X": new_X,  # Single time point
        "new_y": new_y  # Single target value
    }

    # Send the message to the Kafka topic
    producer.send(topic, value=message)
    producer.flush()  # Ensure the message is sent immediately

    print(f"Sent new data point at index {i}")

    # Simulate real-time streaming with a delay
    sleep(1)

print("Finished streaming all data points.")
