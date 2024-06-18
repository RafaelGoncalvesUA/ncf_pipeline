import matplotlib.pyplot as plt
import seaborn as sns

def plot_ratings_count(df, set_limit = False, scale = ('M', 1000000)):
    plt.figure(figsize= (8, 5))

    ax = sns.countplot(data = df, x ="rating", palette="viridis")

    if set_limit:
        ax.set_ylim(0, 8000000)

    ylabels = ['{:.0f}'.format(x) + f' {scale[0]}' for x in ax.get_yticks()/scale[1]]
    ax.set_yticklabels(ylabels)

    plt.xlabel("Ratings")

    plt.tight_layout()