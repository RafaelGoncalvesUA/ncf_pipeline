import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

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

def tune_vae(config):
    model = VAE(num_users, num_items, config['hidden_dim'], config['latent_dim'], config['dropout']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    start_beta = config['start_beta']
    end_beta = config['start_beta'] + config['offset_beta']
    num_epochs = config['num_epochs']

    if config['annealer'] == "linear":
        beta_annealer = BetaLinearAnnealer(start_beta, end_beta, num_epochs)
    elif config['annealer'] == "exponential":
        beta_annealer = BetaExponentialAnnealer(start_beta, end_beta, num_epochs)
    elif config['annealer'] == "cosine":
        beta_annealer = BetaCosineAnnealer(start_beta, end_beta, num_epochs)
    elif config['annealer'] == "constant":
        beta_annealer = BetaLinearAnnealer(start_beta, start_beta, num_epochs)
    elif config['annealer'] == "zero":
        beta_annealer = BetaLinearAnnealer(0.0, 0.0, num_epochs)

    for epoch in range(num_epochs):
        beta = beta_annealer.get_beta(epoch)
        train_loss = train(model, train_loader, optimizer, criterion, beta, device)
        eval_loss, eval_metric = evaluate(model, test_loader, criterion, beta, metric, device)
        
        ray.train.report(dict(train_loss=train_loss, eval_loss=eval_loss, eval_metric=eval_metric))

search_space = {
    "hidden_dim": tune.choice([32, 64, 128, 256, 512, 1024]),
    "latent_dim": tune.choice([8, 16, 32, 64, 128, 256]),
    "dropout": tune.choice([0.0] * 3 + [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]),
    "learning_rate": tune.loguniform(1e-4, 1e-2),
    "start_beta": tune.uniform(0.0001, 0.01),
    "offset_beta": tune.uniform(0.0, 1.0),
    "annealer": tune.choice(["linear", "exponential", "cosine", "constant", "zero"]),
    "num_epochs": 10
}

reporter = CLIReporter(
    parameter_columns=["hidden_dim", "latent_dim", "learning_rate", "dropout", "start_beta", "offset_beta", "annealer"],
    metric_columns=["train_loss", "eval_loss", "eval_metric"],
)

scheduler = ASHAScheduler(
    metric="eval_metric",
    mode="min",
    max_t=10,
    grace_period=1,
    reduction_factor=2
)

def trial_dirname_creator(trial):
    return f"trial_{trial.trial_id}"

storage_path = os.path.abspath("./vae/ray_results")

analysis = tune.run(
    tune_vae,
    config=search_space,
    num_samples=500,
    scheduler=scheduler,
    progress_reporter=reporter,
    storage_path=storage_path,
    trial_dirname_creator=trial_dirname_creator,
)

df = analysis.results_df
df.to_csv("vae/ray_tune_results.csv", index=False)