"""
Step 3: Train a genre classifier
----------------------------------
Goal: predict a track's genre (pop, rap, rock, EDM, R&B, latin) purely from
its audio features (danceability, energy, tempo, etc) using a Random Forest.

Why Random Forest?
- Works well on tabular numeric data straight out of the box
- Doesn't need feature scaling
- Gives us feature importances for free, which helps explain *why* the
  model makes the decisions it does (useful for explaining this in interviews!)
"""

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 1. Load the data
df = pd.read_csv("spotify_songs.csv")

# 2. Pick our input features (X) and the thing we're predicting (y)
FEATURES = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo", "duration_ms",
]
X = df[FEATURES]
y = df["playlist_genre"]

# 3. Split into training data (80%) and test data (20%)
#    The model never sees the test set during training or cross-validation below -
#    that's how we get an honest score of how well it generalises to new songs.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3.5 Cross-validate on the training set first, before touching the test set at all.
#    One fixed 80/20 split only gives one accuracy number, and that number depends
#    partly on which songs happened to land in the test set that time, a lucky or
#    unlucky split can make the model look better or worse than it really is.
#    5-fold cross-validation instead splits the training data into 5 chunks, trains on
#    4 of them and evaluates on the 5th, five times over with a different chunk held out
#    each time, so every training song gets used for evaluation exactly once. Averaging
#    across folds gives a steadier estimate, and the spread between folds shows how much
#    that estimate can actually be trusted.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1,
)
print("5-fold cross-validation accuracy (training set only):")
print(f"  per fold: {[round(s, 3) for s in cv_scores]}")
print(f"  mean: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
print()

# 4. Train the final model on the full training set
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 5. Evaluate on the held-out test set - songs neither training nor cross-validation
#    above ever saw, this is the real "how does it do on brand new songs" number.
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Held-out test accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
print()
print("Full classification report:")
print(classification_report(y_test, y_pred))

# 6. Confusion matrix - which genres does it mix up?
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predicted genre")
plt.ylabel("Actual genre")
plt.title(f"Confusion Matrix (accuracy = {accuracy*100:.1f}%)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("\nSaved confusion_matrix.png")

# 7. Feature importance - which audio features mattered most?
importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
plt.figure(figsize=(8, 6))
importances.plot(kind="barh")
plt.title("Which audio features matter most for predicting genre?")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
print("Saved feature_importance.png")

# 8. Save the trained model so it can be reused without retraining
joblib.dump(model, "genre_classifier_model.pkl")
print("Saved genre_classifier_model.pkl")
