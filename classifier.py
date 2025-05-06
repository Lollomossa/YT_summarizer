# classifier.py – ONLY FOR INFERENCE

import joblib
from sklearn.exceptions import NotFittedError

# ➔ Function to upload the classifier
def load_classifier(path="classifier.pkl"):
    try:
        classifier = joblib.load(path)
        return classifier
    except FileNotFoundError as e:
        raise RuntimeError("❌ Unable to upload the classifier.pkl.") from e

# ➔ Function to predict the topic given X (vettore TF-IDF)
def predict_topic(classifier, X):
    if X is None or X.shape[0] == 0:
        return "N/A"
    try:
        prediction = classifier.predict(X)
        return prediction[0]
    except ValueError as ve:
        raise ValueError("❌ Error during prediction analysis: check the vectorizer and the input format.") from ve
    except NotFittedError:
        raise RuntimeError("❌ The classifier wasn't trained.")