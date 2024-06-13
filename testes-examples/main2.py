from pyspark.sql import SparkSession
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import horovod.spark.torch as hvd
from horovod.spark.common.backend import SparkBackend
from horovod.spark.common.store import Store
from pyspark.sql.functions import col
from pyspark.ml.evaluation import RegressionEvaluator

# Configure Spark session
spark = SparkSession.builder \
    .appName("DistributedTraining") \
    .config("spark.executor.memory", "60g") \
    .config("spark.executor.cores", 12) \
    .config("spark.executor.instances", 3) \
    .config("spark.rapids.memory.gpu.reserve", "10g") \
    .config("spark.executorEnv.TF_FORCE_GPU_ALLOW_GROWTH", "true") \
    .config("spark.kryoserializer.buffer.max", "2000m") \
    .getOrCreate()

# Load the dataset using Pandas
ratings = pd.read_csv('../data/ratings.csv')[:10000]

# Map users and items to unique ids
user_ids = ratings['userId'].unique()
item_ids = ratings['movieId'].unique()

user2idx = {user: idx for idx, user in enumerate(user_ids)}
item2idx = {item: idx for idx, item in enumerate(item_ids)}

ratings['userId'] = ratings['userId'].map(user2idx)
ratings['movieId'] = ratings['movieId'].map(item2idx)

# Convert to Spark DataFrame
df = spark.createDataFrame(ratings)

# Repartition the DataFrame for parallel processing
df = df.repartition(3)  # assuming 3 executors

# Split into training and test sets
train_df, test_df = df.randomSplit([0.8, 0.2])

class CollaborativeFilteringModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(CollaborativeFilteringModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        x = torch.cat([user_embeds, item_embeds], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Set up the Horovod store
prefix_path = '/tmp/horovod_store'  # Specify your path here

# Create the store with the required prefix path
store = Store.create(
    prefix_path=prefix_path,
    save_runs=True  # To save each run
)

# Initialize the model and optimizer
embedding_dim = 50
num_users = len(user_ids)
num_items = len(item_ids)

num_executors = 3 # Number of executors

model = CollaborativeFilteringModel(num_users, num_items, embedding_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001 * num_executors)
loss_fn = nn.MSELoss()

# Set up Horovod's SparkBackend
backend = SparkBackend(num_proc=3)

# Create the Horovod Torch Estimator
torch_estimator = hvd.TorchEstimator(
    backend=backend,
    store=store,
    model=model,
    optimizer=optimizer,
    loss=lambda input, target: loss_fn(input, target.float()),
    input_shapes=[[-1], [-1]],
    feature_cols=['userId', 'movieId'],
    label_cols=['rating'],
    batch_size=64,
    epochs=10,
    validation=0.1,
    verbose=2
)

# Train the model
torch_model = torch_estimator.fit(train_df)

# Predict using the trained model
pred_df = torch_model.transform(test_df)

# Evaluate the predictions
evaluator = RegressionEvaluator(
    labelCol='rating', 
    predictionCol='prediction', 
    metricName='rmse'
)

rmse = evaluator.evaluate(pred_df)
print(f"Test RMSE: {rmse}")