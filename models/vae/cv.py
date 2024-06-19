import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
from time import process_time
from tqdm import tqdm

ray_results = pd.read_csv('models/vae/ray_results/ray_tune_results.csv').sort_values('eval_loss', ascending=True)

idx = 0

print(ray_results)

best_config = ray_results.iloc[idx]

print(best_config)

hidden_dim = best_config['config/hidden_dim']
latent_dim = best_config['config/latent_dim']
dropout = best_config['config/dropout']
learning_rate = best_config['config/learning_rate']

class BetaLinearAnnealer:
    def __init__(self, start_value, end_value, num_epochs):
        self.start_value = start_value
        self.end_value = end_value
        self.num_epochs = num_epochs

    def get_beta(self, epoch):
        progress = epoch / self.num_epochs
        return self.start_value + progress * (self.end_value - self.start_value)
    
class BetaExponentialAnnealer:
    def __init__(self, start_value, end_value, num_epochs):
        self.start_value = start_value
        self.end_value = end_value
        self.num_epochs = num_epochs

    def get_beta(self, epoch):
        progress = epoch / self.num_epochs
        return self.start_value + progress * (self.end_value - self.start_value)

class BetaCosineAnnealer:
    def __init__(self, start_value, end_value, num_epochs):
        self.start_value = start_value
        self.end_value = end_value
        self.num_epochs = num_epochs

    def get_beta(self, epoch):
        cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch / self.num_epochs))
        return self.end_value + (self.start_value - self.end_value) * cosine_decay

num_epochs = best_config['config/num_epochs']
start_beta = best_config['config/start_beta']
offset_beta = best_config['config/offset_beta']
annealer = best_config['config/annealer']
end_beta = start_beta + offset_beta

if annealer == "linear":
    beta_annealer = BetaLinearAnnealer(start_beta, end_beta, num_epochs)
elif annealer == "exponential":
    beta_annealer = BetaExponentialAnnealer(start_beta, end_beta, num_epochs)
elif annealer == "cosine":
    beta_annealer = BetaCosineAnnealer(start_beta, end_beta, num_epochs)
elif annealer == "constant":
    beta_annealer = BetaLinearAnnealer(start_beta, start_beta, num_epochs)
elif annealer == "zero":
    beta_annealer = BetaLinearAnnealer(0.0, 0.0, num_epochs)

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

class VAE(nn.Module):
    def __init__(self, num_users, num_items, hidden_dim, latent_dim, dropout):
        super(VAE, self).__init__()
        self.user_embedding = nn.Embedding(num_users, latent_dim)
        self.item_embedding = nn.Embedding(num_items, latent_dim)
        
        self.encoder = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        self._initialize_weights()


    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.01)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

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

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss
    
def vae_loss(reconstructed_ratings, true_ratings, mu, log_var, beta):
    mse_loss = nn.MSELoss()(reconstructed_ratings, true_ratings)
    kld_loss = -0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))
    return mse_loss + beta * kld_loss

criterion = vae_loss
metric = RMSELoss()

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

def evaluate(model, dataloader, criterion, beta, metric, device):
    model.eval()
    eval_loss = 0
    eval_metric = 0
    with torch.no_grad():
        for user_ids, item_ids, ratings in tqdm(dataloader, desc="Evaluating"):
            user_ids, item_ids, ratings = user_ids.to(device), item_ids.to(device), ratings.to(device).float()
            outputs, mu, log_var = model(user_ids, item_ids)
            eval_loss += criterion(outputs, ratings, mu, log_var, beta).item()
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
    
    model = VAE(num_users, num_items, hidden_dim=hidden_dim, latent_dim=latent_dim, dropout=dropout).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        beta = beta_annealer.get_beta(epoch)
        train_loss = train(model, train_loader, optimizer, criterion, beta, device)
        eval_loss, eval_metric = evaluate(model, eval_loader, criterion, beta, metric, device)

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
            'config/hidden_dim': hidden_dim,
            'config/latent_dim': latent_dim,
            'config/dropout': dropout,
            'config/learning_rate': learning_rate,
            'config/start_beta': start_beta,
            'config/offset_beta': offset_beta,
            'config/annealer': annealer,
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

os.makedirs('vae/cv_results', exist_ok=True)

results_df.to_csv(f'vae/cv_results/{idx}.csv', index=False)

