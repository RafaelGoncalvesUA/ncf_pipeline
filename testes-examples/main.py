import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import wandb


wandb.login(key='a6beca469e4665eb0bc333c6a761f5503f5554d4', relogin=True)

wandb.init(
    project='caa-proj2', 
    config={
        'embedding_dim': 50,
        'lr': 0.001,
        'batch_size': 64,
        'num_epochs': 10
    }
)

wandb.log({'loss': 0.5,})
exit(0)

# Load the MovieLens dataset
# Assuming the dataset is in the file 'ratings.csv'
ratings = pd.read_csv('data/ml-1m/ratings.csv')

# Map users and items to unique ids
user_ids = ratings['userId'].unique()
item_ids = ratings['movieId'].unique()

user2idx = {user: idx for idx, user in enumerate(user_ids)}
item2idx = {item: idx for idx, item in enumerate(item_ids)}

ratings['userId'] = ratings['userId'].map(user2idx)
ratings['movieId'] = ratings['movieId'].map(item2idx)

num_users = len(user_ids)
num_items = len(item_ids)
embedding_dim = 50  # Dimension of the embeddings

# Define a PyTorch dataset
class MovieLensDataset(Dataset):
    def __init__(self, ratings):
        self.users = ratings['userId'].values
        self.items = ratings['movieId'].values
        self.ratings = ratings['rating'].values

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return (self.users[idx], self.items[idx], self.ratings[idx])

dataset = MovieLensDataset(ratings)

# Split the dataset into training and test sets
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the neural network model
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

# Initialize the model, loss function, and optimizer
model = CollaborativeFilteringModel(num_users, num_items, embedding_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for users, items, ratings in train_loader:
        users = users.long()
        items = items.long()
        ratings = ratings.float().view(-1, 1)
        
        optimizer.zero_grad()
        predictions = model(users, items)
        loss = criterion(predictions, ratings)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}')

# Evaluate the model
model.eval()
total_loss = 0
with torch.no_grad():
    for users, items, ratings in test_loader:
        users = users.long()
        items = items.long()
        ratings = ratings.float().view(-1, 1)
        
        predictions = model(users, items)
        loss = criterion(predictions, ratings)
        total_loss += loss.item()
    
    avg_loss = total_loss / len(test_loader)
    print(f'Test Loss: {avg_loss}')

print('---')

K = 10
# show random K predictions (user, item, true rating, predicted rating)
for i in range(K):
    idx = np.random.randint(len(test_dataset))
    user, item, rating = test_dataset[idx]
    user = torch.tensor([user])
    item = torch.tensor([item])
    rating = torch.tensor([rating])
    
    prediction = model(user, item)
    print(f'User: {user.item()}, Item: {item.item()}, True Rating: {rating.item()}, Predicted Rating: {prediction.item()}')