import os
import glob
import pandas as pd
import numpy as np

cv_results = sorted(glob.glob(f'*/cv_results/*.csv'))

for cv_result in cv_results:
    df = pd.read_csv(cv_result)
    last_eval_losses = df[df['epoch'] == df['epoch'].max()]['eval_loss']
    print(f"{cv_result}: {np.mean(last_eval_losses):.4f} ± {np.std(last_eval_losses):.4f}")