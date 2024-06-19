import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_whisker(model):
    cv_results = pd.read_csv(f'{model}/cv_results/best.csv')
    
    plt.figure(figsize=(10, 6))
    
    boxplot = sns.boxplot(x='fold', y='eval_loss', data=cv_results, showmeans=True,
                          meanprops={"marker": "^", "markerfacecolor": "green", "markeredgecolor": "black", "markersize": 12},
                          boxprops=dict(facecolor="None", edgecolor="black"),
                          whiskerprops=dict(color="black"),
                          capprops=dict(color="black"),
                          medianprops={"color": "orange", "linewidth": 2})
    
    # sns.stripplot(x='fold', y='eval_loss', data=cv_results, color='red', alpha=0.5, jitter=True)
    
    plt.xlabel('Fold')
    plt.ylabel('Validation Loss')
    plt.grid(True, which='major', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    os.makedirs(f'images/{model}/', exist_ok=True)
    plt.savefig(f'images/{model}/cv_whisker.png')
    
    plt.show()

def plot_loss(model, fold):
    cv_results = pd.read_csv(f'{model}/cv_results/best.csv')
    
    cv_results = cv_results[cv_results['fold'] == fold]
    
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(x='epoch', y='train_loss', data=cv_results, label='Train Loss', color='blue')
    sns.lineplot(x='epoch', y='eval_loss', data=cv_results, label='Validation Loss', color='red')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, which='major', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    os.makedirs(f'images/{model}/', exist_ok=True)
    plt.savefig(f'images/{model}/cv_fold_loss_{fold}.png')
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize the model')
    parser.add_argument('model', type=str, choices=['ae', 'vae', 'non-linear'], help='Model to visualize')
    parser.add_argument('config', type=int, help='Configuration number')
    parser.add_argument('-w', '--cv-whisker', action='store_true', help='Visualize the whisker plot')
    parser.add_argument('-l', '--cv-fold-loss', type=int, help='Visualize the CV training and validation loss')
    args = parser.parse_args()

    print(args)

    if args.cv_whisker:
        plot_whisker(args.model)
    elif args.cv_fold_loss is not None:
        plot_loss(args.model, args.cv_fold_loss)
    else:
        print('Please specify the type of visualization to generate')

    

if __name__ == '__main__':
    main()