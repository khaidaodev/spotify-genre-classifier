"""
Step 3: Train a genre classifier
----------------------------------
Goal: predict a track's genre (pop, rap, rock, EDM, R&B, latin) purely from
its audio features (danceability, energy, tempo, etc) using a Random Forest.

Why Random Forest?
- Works well on tabular numeric data straight out of the box
- Doesn't need feature scaling
- Gives us feature importances for free, which helps explain *why* the
  model makes the decisions it does, not just what it decided
"""

import pandas as pd
from sklearn.model_selection import train_test_split
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
#    The model never sees the test set while training - that's how we get
#    an honest score of how well it generalises to new songs.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train the model
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 5. Evaluate on the held-out test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Test accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
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
