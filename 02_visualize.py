"""
Step 2: Visualize the genres
------------------------------
Goal: see which audio features actually look different across genres,
using pictures instead of just numbers.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("spotify_songs.csv")

# These are the features we think might differ between genres,
# based on the averages we already looked at.
features = ["danceability", "energy", "tempo", "acousticness", "speechiness"]

# Set up a grid of 5 small plots, one per feature
fig, axes = plt.subplots(1, len(features), figsize=(22, 5))

for ax, feature in zip(axes, features):
    sns.boxplot(data=df, x="playlist_genre", y=feature, ax=ax)
    ax.set_title(feature)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("genre_features_boxplot.png", dpi=150)
print("Saved genre_features_boxplot.png - open it to see the results!")
