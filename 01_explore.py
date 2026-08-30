"""
Step 1: Explore the Spotify dataset
------------------------------------
Goal: understand what data we have before doing anything clever with it.
"""

import pandas as pd

# Load the dataset
df = pd.read_csv("spotify_songs.csv")

print("Shape of the dataset (rows, columns):")
print(df.shape)
print()

print("Column names:")
print(list(df.columns))
print()

print("First 5 rows:")
print(df.head())
print()

print("How many tracks per genre?")
print(df["playlist_genre"].value_counts())
print()

print("Any missing values?")
print(df.isnull().sum()[df.isnull().sum() > 0])
