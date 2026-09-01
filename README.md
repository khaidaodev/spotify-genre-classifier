# Spotify Genre Classifier

My first proper end-to-end ML project. Can you actually predict a song's genre just from its audio characteristics (danceability, energy, tempo, that sort of thing), with zero info about the artist or title?

## The data

32,833 real Spotify tracks across 6 genres (EDM, rap, pop, R&B, latin, rock). Each track comes with 12 numeric audio features straight from Spotify's own API: danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration.

## How it's put together

| File | What it does |
|---|---|
| `01_explore.py` | loads the data, checks the shape, missing values, whether genres are balanced |
| `02_visualize.py` | boxplots comparing the audio features across genres |
| `03_train_model.py` | trains the model and checks how good it actually is |

Run them in order:

```bash
pip install -r requirements.txt
python3 01_explore.py
python3 02_visualize.py
python3 03_train_model.py
```

## The model

A Random Forest classifier (basically a big group of decision trees that each vote on the answer), trained on an 80/20 split, so the score below is on songs the model never saw while training.

Also ran 5-fold cross-validation on the training set before touching the test set at all. The problem with just one fixed 80/20 split is the accuracy you get depends partly on luck, which songs happened to land in the test set that time round. Cross-validation splits the training data 5 ways and rotates which chunk gets held out, so you get 5 accuracy readings instead of 1: 55.2% on average, +/- 0.6% between folds, using the library's default settings (200 trees, no depth limit, etc).

Then used that cross-validation to actually do something useful: a random search over Random Forest hyperparameters (number of trees, tree depth, how many features each split considers, minimum samples per split/leaf), 6 random combinations checked across all 5 folds each, picking whichever combination scored best on average rather than just going with the defaults. Winner: 250 trees, max depth 30, `max_features="log2"`, `min_samples_split=10`, `min_samples_leaf=2`. That nudged cross-validation accuracy from 55.2% to 55.7%, a small but real gain, tuning a Random Forest that's already close to its ceiling on this data only buys you so much, which is itself a useful thing to know rather than assume.

Test accuracy (tuned model): 55.7%. Doesn't sound huge on its own, but random guessing across 6 genres would only get you about 16.7%, so this is roughly 3.3x better than just guessing.

![Confusion matrix](confusion_matrix.png)

### What it's good and bad at

Best at spotting rock (71% F1, F1 is basically a combined precision and recall score, one number that punishes a model for being wrong either by missing tracks or wrongly flagging them) and EDM (67% F1). Makes sense, both have a pretty consistent audio fingerprint: rock leans lower danceability with more energy swings, EDM is just high energy in a specific tempo range.

Worst at pop, only 37% F1, mostly getting confused with R&B and latin. Honestly this tracks though. Pop isn't really its own sound, it's more of a commercial label that borrows bits from whatever genre is popular at the time, so there isn't one clean audio signature for the model to latch onto. Kind of a nice thing to have found out by accident rather than something I set out to prove.

![Feature importance](feature_importance.png)

Tempo, speechiness, and danceability came out as the most important features for the tuned model, before tuning it was danceability, energy, and speechiness. The ranking shifting a bit after tuning (`max_features="log2"` means each split only gets to consider a smaller random slice of features rather than most of them) was a good reminder that "feature importance" isn't some fixed property of the data, it's a property of the specific model you trained.

## What I'd try next

- Use the finer-grained `playlist_subgenre` labels instead, harder problem
- Try XGBoost or LightGBM and see if it beats the tuned Random Forest
- Widen the hyperparameter search (more combinations, wider ranges) now that the pipeline for it exists, this run only checked 6 combinations to keep it running in a couple of minutes

## Testing and git

I'll be honest, this one doesn't have a test suite, it's the smallest of my three projects and I leaned on cross-validation and just checking the numbers by hand instead of writing pytest tests for it. The commits still track real steps though, baseline model, then cross-validation as a sanity check, then the hyperparameter search. Testing's the first thing I'd add if I kept building this one out.

## Tools used

Python, pandas, scikit-learn, matplotlib, seaborn.
