# BEST MODEL DISCOVERED IN CROSS-VALIDATION

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import wandb

wandb.login(key="1bf6d96598e920a3fe32392d71154f5e9011cdbd", relogin=True)
wandb.init(project="proj-caa-2")


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
    
    def train_(self, dataloader, criterion, device):
        self.train()
        train_loss = 0
        for user_ids, item_ids, ratings in dataloader:
            user_ids, item_ids, ratings = user_ids.to(device), item_ids.to(device), ratings.to(device).float()
            self.optim.zero_grad()
            outputs = self(user_ids, item_ids)
            loss = criterion(outputs, ratings)
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
