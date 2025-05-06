# YouTube Educational Video Summarizer and Classifier

This project implements an end-to-end pipeline to **download**, **transcribe**, **classify** and **summarize** educational YouTube videos.
It supports both **extractive** and **abstractive summarization** through a simple **Streamlit** user interface.

---

##  Features

-  Download and transcribe audio from YouTube videos (OpenAI Whisper)
-  Text preprocessing (tokenization, stopwords removal, lemmatization)
-  Topic classification (Naive Bayes Classifier on TF-IDF vectors)
-  Summarization modes:
  - **Extractive summarization** (based on TF-IDF scores)
  - **Abstractive summarization** (via HuggingFace `facebook/bart-large-cnn`)
-  Evaluation using ROUGE metrics
-  Simple and interactive UI with Streamlit

---

##  Installation Guide (MacOS)

>  **Important:**  
> Some packages must be installed **manually** before installing the full project requirements!

### 1. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 2. Install manual dependencies

Install OpenAI Whisper (from GitHub):

```bash
pip install git+https://github.com/openai/whisper.git
```

Install missing libraries:

```bash
pip install nltk scikit-learn spacy
```

Download the SpaCy English model:

```bash
python -m spacy download en_core_web_sm
```

---

### 3. Install remaining project requirements

```bash
pip install -r requirements.txt
```

---

### 4. Launch the application

```bash
streamlit run app.py
```

 The app will open automatically in your default browser!

---

##  Project Structure

```plaintext
├── app.py                  # Streamlit app interface
├── stt.py                   # Speech-to-text (Whisper) module
├── preprocessing.py         # Text cleaning and vectorization
├── classifier.py            # Topic classifier (Naive Bayes)
├── summarizer.py            # Summarization (Extractive and Abstractive)
├── run_pipeline.py          # Full pipeline runner script
├── cleaned_dataset.csv      # Preprocessed dataset
├── tfidf_matrix.npy         # Saved TF-IDF matrix
├── vectorizer.pkl           # Saved TF-IDF vectorizer
├── classifier.pkl           # Trained topic classifier
├── evaluation_report.md     # Manual evaluation report (ROUGE metrics)
├── requirements.txt         # Project dependencies
├── README.md                # This file
```

---

##  Important Notes

- `Whisper` must be installed **manually** via GitHub.
- `en_core_web_sm` (SpaCy model) must be **downloaded manually**.
- **Long input texts** are automatically truncated to avoid token limits in the abstractive summarizer.

---

##  Future Improvements

- Train a **deep learning classifier** for more complex topic detection
- Implement full **abstractive summarization** with larger models (T5, Pegasus)
- Expand the dataset to include hundreds of videos
- Add multi-language support

---

##  Author

**Lorenzo Mossini**  
Project completed for academic purposes (hephaestus ML student association), april 2025

---

