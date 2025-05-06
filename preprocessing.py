
import re
import nltk
import spacy
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# (only the first time) download of the necessary resources
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")

# function to clena  single text
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    doc = nlp(" ".join(tokens))
    lemmatized = [token.lemma_ for token in doc]
    return " ".join(lemmatized)

# function to upload the already saved vectorizer
def load_vectorizer(path="vectorizer.pkl"):
    with open(path, "rb") as f:
        vectorizer = pickle.load(f)
    return vectorizer

# ➔ (Extra)function to create and save a new vectorizer, if needed
def create_and_save_vectorizer(texts, path="vectorizer.pkl"):
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    vectorizer.fit(texts)
    with open(path, "wb") as f:
        pickle.dump(vectorizer, f)
    return vectorizer