import numpy as np
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

nltk.download('punkt')

# ➔ Function to upload the alreay saved vectorizer
def load_vectorizer(path="vectorizer.pkl"):
    with open(path, "rb") as f:
        vectorizer = pickle.load(f)
    return vectorizer

# ➔ Principal function to summarize the content
def summarize_text(text, n_sentences=5):
    if not text or len(text.strip()) == 0:
        return "No content available for summary."

    # Upload the vectorizer
    vectorizer = load_vectorizer()

    # Use the tokenizer
    tokenizer = PunktSentenceTokenizer()
    sentences = tokenizer.tokenize(text)

    if len(sentences) <= n_sentences:
        return text  # if you're dealing with only a few phrases, return everything

    # vectorize the sentences
    tfidf_matrix = vectorizer.transform(sentences)
    scores = tfidf_matrix.sum(axis=1).A1

    # Select the phrases with a higher score
    top_n_idx = np.argsort(scores)[-n_sentences:]
    top_n_idx.sort()

    summary = " ".join([sentences[i] for i in top_n_idx])
    return summary

from transformers import pipeline

# Load abstractive model (only once)
abstractive_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def generate_abstractive_summary(text, min_length=30, max_length=130):
    if len(text) > 1024:
        text = text[:1024]  # Optionally truncate if too long
    summary = abstractive_summarizer(text, min_length=min_length, max_length=max_length, do_sample=False)
    return summary[0]['summary_text']