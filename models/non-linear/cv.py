import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
from time import process_time
from tqdm import tqdm

ray_results = pd.read_csv('models/non-linear/ray_results/ray_tune_results.csv').sort_values('eval_loss', ascending=True)

idx = 0

print(ray_results)

best_config = ray_results.iloc[idx]

print(best_config)

embedding_dim = best_config['config/embedding_dim']
dropout = best_config['config/dropout']
learning_rate = best_config['config/learning_rate']
num_epochs = best_config['config/num_epochs']

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

batch_size = 1024

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

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss
    
criterion = nn.MSELoss()
metric = RMSELoss()

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

def evaluate(model, dataloader, criterion, metric, device):
    model.eval()
    eval_loss = 0
    eval_metric = 0
    with torch.no_grad():
        for user_ids, item_ids, ratings in tqdm(dataloader, desc="Evaluating"):
            user_ids, item_ids, ratings = user_ids.to(device), item_ids.to(device), ratings.to(device).float()
            outputs = model(user_ids, item_ids)
            eval_loss += criterion(outputs, ratings).item()
            eval_metric += metric(outputs, ratings).item()
    return eval_loss / len(dataloader), eval_metric / len(dataloader)

K = 10

kfold = KFold(n_splits=K, shuffle=True, random_state=42)

train_losses = []
eval_losses = []
eval_metrics = []

times = []

for fold, (train_idx, eval_idx) in enumerate(kfold.split(dataset)):
    print(f"Fold {fold + 1}")

    gtime = process_time()

    fold_train_losses = []
    fold_eval_losses = []
    fold_eval_metrics = []
    
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
    eval_subsampler = torch.utils.data.SubsetRandomSampler(eval_idx)
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
    eval_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=eval_subsampler)
    
    model = NonLinearModel(num_users, num_items, embedding_dim, dropout).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        eval_loss, eval_metric = evaluate(model, eval_loader, criterion, metric, device)

        fold_train_losses.append(train_loss)
        fold_eval_losses.append(eval_loss)
        fold_eval_metrics.append(eval_metric)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}, Eval Metric: {eval_metric:.4f}")
    
    train_losses.append(fold_train_losses)
    eval_losses.append(fold_eval_losses)
    eval_metrics.append(fold_eval_metrics)

    times.append(process_time() - gtime)

results = []

for fold, (train_losses_fold, eval_losses_fold, eval_metrics_fold, time) in enumerate(zip(train_losses, eval_losses, eval_metrics, times)):
    for epoch, (train_loss, eval_loss, eval_metric) in enumerate(zip(train_losses_fold, eval_losses_fold, eval_metrics_fold)):
        results.append({
            'config/embedding_dim': embedding_dim,
            'config/dropout': dropout,
            'config/learning_rate': learning_rate,
            'config/num_epochs': num_epochs,
            'fold': fold,
            'epoch': epoch,
            'train_loss': train_loss,
            'eval_loss': eval_loss,
            'eval_metric': eval_metric,
            'elapsed_time': time,
            'num_users': num_users,
            'num_items': num_items,
            'num_ratings': len(ratings)
        })

results_df = pd.DataFrame(results)

os.makedirs('non-linear/cv_results', exist_ok=True)

results_df.to_csv(f'non-linear/cv_results/{idx}.csv', index=False)

