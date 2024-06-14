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

class VAE(nn.Module):
    def __init__(self, num_users, num_items, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.user_embedding = nn.Embedding(num_users, latent_dim)
        self.item_embedding = nn.Embedding(num_items, latent_dim)
        
        self.encoder = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def encode(self, user_vec, item_vec):
        combined_vec = torch.cat([user_vec, item_vec], dim=1)
        hidden = self.encoder(combined_vec)
        mu = self.mu(hidden)
        log_var = self.log_var(hidden)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, user_ids, item_ids):
        user_vec = self.user_embedding(user_ids)
        item_vec = self.item_embedding(item_ids)
        mu, log_var = self.encode(user_vec, item_vec)
        z = self.reparameterize(mu, log_var)
        pred_rating = self.decode(z)
        return pred_rating.squeeze(), mu, log_var
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = 20
hidden_dim = 64

learning_rate = 1e-3

num_epochs = 10

model = VAE(num_users, num_items, hidden_dim, latent_dim)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def vae_loss(reconstructed_ratings, true_ratings, mu, log_var, beta):
    mse_loss = F.mse_loss(reconstructed_ratings, true_ratings, reduction='mean')
    kld_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    total_loss = mse_loss + beta * kld_loss
    return total_loss

criterion = vae_loss

def train(model, dataloader, optimizer, criterion, beta, device):
    model.train()
    train_loss = 0
    for user_ids, item_ids, ratings in tqdm(dataloader, desc="Training"):
        user_ids, item_ids, ratings = user_ids.to(device), item_ids.to(device), ratings.to(device).float()
        optimizer.zero_grad()
        outputs, mu, log_var = model(user_ids, item_ids)
        loss = criterion(outputs, ratings, mu, log_var, beta)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(dataloader)

def evaluate(model, dataloader, criterion, beta, device):
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for user_ids, item_ids, ratings in tqdm(dataloader, desc="Evaluating"):
            user_ids, item_ids, ratings = user_ids.to(device), item_ids.to(device), ratings.to(device).float()
            outputs, mu, log_var = model(user_ids, item_ids)
            loss = criterion(outputs, ratings, mu, log_var, beta)
            eval_loss += loss.item()
    return eval_loss / len(dataloader)

for epoch in range(num_epochs):
    beta = min(1.0, epoch / (num_epochs / 2.0))
    train_loss = train(model, train_loader, optimizer, criterion, beta, device)
    eval_loss = evaluate(model, test_loader, criterion, beta, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}")

def predict(model, user_id, item_id, device):
    model.eval()
    with torch.no_grad():
        user_tensor = torch.tensor([user_id]).to(device)
        item_tensor = torch.tensor([item_id]).to(device)
        rating, _, _ = model(user_tensor, item_tensor)
        return rating.item()
    
K = 10
for i in range(K):
    idx = np.random.randint(len(test_dataset))
    user, item, rating = test_dataset[idx]
    prediction = predict(model, user, item, device)
    print(f"User: {user}, Item: {item}, True Rating: {rating}, Predicted Rating: {prediction:.2f}")
    
        