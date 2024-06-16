import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

ratings = pd.read_csv('data/ml-1m/ratings.csv')

user_ids = ratings['userId'].unique()
item_ids = ratings['movieId'].unique()

user2idx = {user: idx for idx, user in enumerate(user_ids)}
item2idx = {item: idx for idx, item in enumerate(item_ids)}

ratings['userId'] = ratings['userId'].map(user2idx)
ratings['movieId'] = ratings['movieId'].map(item2idx)

num_users = len(user_ids)
num_items = len(item_ids)

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

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

batch_size = 1024
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class NonLinearModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, dropout):
        super(NonLinearModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.fc1 = nn.Linear(embedding_dim * 2, 128)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        x = torch.cat([user_embeds, item_embeds], dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x.squeeze()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embedding_dim = 50
dropout = 0.1

learning_rate = 1e-3

num_epochs = 10

model = NonLinearModel(num_users, num_items, embedding_dim, dropout).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss
    
criterion = RMSELoss()

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    for user_ids, item_ids, ratings in tqdm(dataloader, desc="Training"):
        user_ids, item_ids, ratings = user_ids.to(device), item_ids.to(device), ratings.to(device).float()
        optimizer.zero_grad()
        outputs = model(user_ids, item_ids)
        loss = criterion(outputs, ratings)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for user_ids, item_ids, ratings in tqdm(dataloader, desc="Evaluating"):
            user_ids, item_ids, ratings = user_ids.to(device), item_ids.to(device), ratings.to(device).float()
            outputs = model(user_ids, item_ids)
            loss = criterion(outputs, ratings)
            eval_loss += loss.item()
    return eval_loss / len(dataloader)

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion, device)
    eval_loss = evaluate(model, test_loader, criterion, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}")

def predict(model, user_id, item_id, device):
    model.eval()
    with torch.no_grad():
        user_tensor = torch.tensor([user_id]).to(device)
        item_tensor = torch.tensor([item_id]).to(device)
        rating = model(user_tensor, item_tensor)
        return rating.item()
    
K = 10
for i in range(K):
    idx = np.random.randint(len(test_dataset))
    user, item, rating = test_dataset[idx]
    prediction = predict(model, user, item, device)
    print(f"User: {user}, Item: {item}, True Rating: {rating}, Predicted Rating: {prediction:.2f}")