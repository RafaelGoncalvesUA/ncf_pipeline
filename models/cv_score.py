import os
import glob
import pandas as pd
import numpy as np

cv_results = sorted(glob.glob(f'*/cv_results/*.csv'))

print("Using last epoch eval loss:")

for cv_result in cv_results:
    df = pd.read_csv(cv_result)
    last_eval_losses = df[df['epoch'] == df['epoch'].max()]['eval_loss']
    print(f"- {cv_result}: {np.mean(last_eval_losses):.4f} ± {np.std(last_eval_losses):.4f}")

print("Using best eval loss")

for cv_result in cv_results:
    df = pd.read_csv(cv_result)
    best_eval_losses = df[df['eval_loss'] == df['eval_loss'].min()]['eval_loss']
    print(f"- {cv_result}: {np.mean(best_eval_losses):.4f} ± {np.std(best_eval_losses):.4f}")

print("Using best epoch eval loss:")

for cv_result in cv_results:
    df = pd.read_csv(cv_result)
    # get epoch with best avg eval loss
    best_epoch = df.groupby('epoch')['eval_loss'].mean().idxmin()
    best_eval_losses = df[df['epoch'] == best_epoch]['eval_loss']
    print(f"- {cv_result} epoch {best_epoch}: {np.mean(best_eval_losses):.4f} ± {np.std(best_eval_losses):.4f}")
