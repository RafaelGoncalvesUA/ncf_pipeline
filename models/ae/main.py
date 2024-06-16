import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from tqdm import tqdm

ratings = pd.read_csv('data/ml-1m/ratings.csv')

user_ids = ratings['userId'].unique()
item_ids = ratings['movieId'].unique()

user2idx = {user: idx for idx, user in enumerate(user_ids)}
item2idx = {item: idx for idx, item in enumerate(item_ids)}

ratings['userId'] = ratings['userId'].map(user2idx)
ratings['movieId'] = ratings['movieId'].map(item2idx)

num_users = len(user2idx)
num_items = len(item2idx)

class MovieLensDataset(Dataset):
    def __init__(self, ratings):
        self.users = ratings['userId'].values
        self.items = ratings['movieId'].values
        self.ratings = ratings['rating'].values

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

dataset = MovieLensDataset(ratings)

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

batch_size = 1024
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class AE(nn.Module):
    def __init__(self, num_users, num_items, hidden_dim, latent_dim, dropout):
        super(AE, self).__init__()
        self.user_embedding = nn.Embedding(num_users, latent_dim)
        self.item_embedding = nn.Embedding(num_items, latent_dim)
        
        self.encoder = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, user_ids, item_ids):
        user_vec = self.user_embedding(user_ids)
        item_vec = self.item_embedding(item_ids)
        combined_vec = torch.cat([user_vec, item_vec], dim=1)
        encoded = self.encoder(combined_vec)
        decoded = self.decoder(encoded)
        return decoded.squeeze()
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = 20
hidden_dim = 64
dropout = 0.1

learning_rate = 1e-3

num_epochs = 10

model = AE(num_users, num_items, hidden_dim, latent_dim, dropout).to(device)

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