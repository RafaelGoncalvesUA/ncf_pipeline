# KAFKA PRODUCER TO GENERATE A STREAM OF DATA

from kafka.admin import KafkaAdminClient, NewTopic
from kafka import KafkaProducer
import json
import pandas as pd
import time

KAFKA_TOPIC = "movielens"
KAFKA_BOOTSTRAP_SERVER = "localhost:9092"

admin_client = KafkaAdminClient(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVER, client_id="test"
)

if "movielens" not in admin_client.list_topics():
    topic = NewTopic(name="movielens", num_partitions=1, replication_factor=1)
    admin_client.create_topics(new_topics=[topic], validate_only=False)

print("Loading data...")
ratings = pd.read_csv("../../data/ratings_batches.csv")
ratings = ratings.to_dict(orient="records")
print("Data loaded successfully")

kafka_producer_object = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVER,
    value_serializer=lambda x: json.dumps(x).encode("utf-8"),
)

ctr = 0

for rating in ratings:
    if ctr == 0:
        should_continue = input("Continue?: ")

    print("Message to be send : ", rating)
    kafka_producer_object.send(KAFKA_TOPIC, rating)

    ctr += 1

    time.sleep(0.0001)

    if ctr % 100000 == 0:
        print(f"Sent {ctr} messages")
        time.sleep(300)