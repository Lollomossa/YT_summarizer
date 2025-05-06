# classifier.py – Training e salvataggio del modello + vectorizer

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------
# 1. uploading the dataset
# ------------------------
df = pd.read_csv("dataset.csv")
print(" Dataset uploaded. First lines:")
print(df.head())

# ------------------------
# 2. TF-IDF
# ------------------------
print("Creating the TF-IDF matrix...")
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(df["transcript"])
y = df["label"]

# ------------------------
# 3. Split train/test
# ------------------------
print("Divide in training and test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------
# 4. model training
# ------------------------
print(" Training the Naive Bayes model...")
model = MultinomialNB()
model.fit(X_train, y_train)

# ------------------------
# 5. evaluation
# ------------------------
print("Valutazione del modello:")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Real")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ------------------------
# 6. saving
# ------------------------
joblib.dump(model, "classifier.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("Model and vectorizer saved correctly!")