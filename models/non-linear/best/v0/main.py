# BEST MODEL v1 (CPU):
# Epoch 1/10, Train Loss: 1.1368, Eval Loss: 0.9279
# Epoch 2/10, Train Loss: 0.9923, Eval Loss: 0.8693
# Epoch 3/10, Train Loss: 0.9273, Eval Loss: 0.8218
# Epoch 4/10, Train Loss: 0.8800, Eval Loss: 0.7877
# Epoch 5/10, Train Loss: 0.8439, Eval Loss: 0.7615
# Epoch 6/10, Train Loss: 0.8139, Eval Loss: 0.7415
# Epoch 7/10, Train Loss: 0.7888, Eval Loss: 0.7102
# Epoch 8/10, Train Loss: 0.7639, Eval Loss: 0.6813
# Epoch 9/10, Train Loss: 0.7380, Eval Loss: 0.6536
# Epoch 10/10, Train Loss: 0.7106, Eval Loss: 0.6224
# Elapsed time: 6225.63s

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from time import perf_counter
import pickle

class MovieLensDataset(Dataset):
    def __init__(self, ratings):
        self.users = ratings['userId'].values
        self.items = ratings['movieId'].values
        self.ratings = ratings['rating'].values

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return (self.users[idx], self.items[idx], self.ratings[idx])

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss
    
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

        self.optim = optim.Adam(self.parameters(), lr=0.0010887937927239) # best value
        self.criterion = RMSELoss()

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
    
    def train_(self, dataloader, device):
        self.train()
        train_loss = 0
        for user_ids, item_ids, ratings in dataloader:
            user_ids, item_ids, ratings = user_ids.to(device), item_ids.to(device), ratings.to(device).float()
            self.optim.zero_grad()
            outputs = self(user_ids, item_ids)
            loss = self.criterion(outputs, ratings)
            loss.backward()
            self.optim.step()
            train_loss += loss.item()
        return train_loss / len(dataloader)
    
    def evaluate(self, dataloader, device):
        self.eval()
        eval_loss = 0
        with torch.no_grad():
            for user_ids, item_ids, ratings in dataloader:
                user_ids, item_ids, ratings = user_ids.to(device), item_ids.to(device), ratings.to(device).float()
                outputs = self(user_ids, item_ids)
                loss = self.criterion(outputs, ratings)
                eval_loss += loss.item()
        return eval_loss / len(dataloader)

    def fit(self, train_loader, test_loader, device, num_epochs):
        for epoch in range(num_epochs):
            train_loss = self.train_(train_loader, device)
            eval_loss = self.evaluate(test_loader, device)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}")

    def predict(self, user_id, item_id, device):
        self.eval()
        with torch.no_grad():
            user_tensor = torch.tensor([user_id]).to(device)
            item_tensor = torch.tensor([item_id]).to(device)
            rating = self(user_tensor, item_tensor)
            return rating.item()

idx = 0

os.makedirs(f'non-linear/best/v{idx}', exist_ok=True)

ratings = pd.read_csv('data/ml-1m/ratings_small.csv')

user_ids = ratings['userId'].unique()
item_ids = ratings['movieId'].unique()

user2idx = {user: idx for idx, user in enumerate(user_ids)}
item2idx = {item: idx for idx, item in enumerate(item_ids)}


with open(f'non-linear/best/v{idx}/user2idx.pkl', 'wb') as f:
    pickle.dump(user2idx, f)

with open(f'non-linear/best/v{idx}/item2idx.pkl', 'wb') as f:
    pickle.dump(item2idx, f)

ratings['userId'] = ratings['userId'].map(user2idx)
ratings['movieId'] = ratings['movieId'].map(item2idx)

scaler = MinMaxScaler()
scaler.fit_transform(ratings[['rating']])

with open(f'non-linear/best/v{idx}/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

num_users = 162541
num_items = 59047

train_dataset = MovieLensDataset(ratings)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NonLinearModel(num_users, num_items, 512, 0.2)

gtime = perf_counter()
model.fit(train_loader, train_loader, device, 10)
gtime = perf_counter() - gtime

print(f"Elapsed time: {gtime:.2f}s")

torch.save(model.state_dict(), f'non-linear/best/v{idx}/model.pth')