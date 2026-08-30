"""
final_metrics.py
Computes coherence + diversity for your FINAL trained LDA and NMF models
(the ones saved after you set N_TOPICS in modeling.py and re-ran it).

Usage (from PyCharm terminal, project root):
    python src/final_metrics.py

Reads: data/complaints_clean.csv, data/lda_model.joblib, data/nmf_model.joblib,
       data/count_vectorizer.joblib, data/tfidf_vectorizer.joblib
Writes: data/final_model_metrics.csv - the Model Comparison page in app.py
        reads this file automatically to fill in its summary table.
"""

import joblib
import pandas as pd
from gensim.corpora import Dictionary

from modeling import get_top_words
from evaluation import compute_coherence, compute_diversity

if __name__ == "__main__":
    df = pd.read_csv("data/complaints_clean.csv")
    docs = df["clean_text"].astype(str).tolist()
    tokenized_docs = [doc.split() for doc in docs]
    dictionary = Dictionary(tokenized_docs)

    tfidf_vectorizer = joblib.load("data/tfidf_vectorizer.joblib")
    count_vectorizer = joblib.load("data/count_vectorizer.joblib")
    lda_model = joblib.load("data/lda_model.joblib")
    nmf_model = joblib.load("data/nmf_model.joblib")

    lda_topics = get_top_words(lda_model, count_vectorizer.get_feature_names_out())
    nmf_topics = get_top_words(nmf_model, tfidf_vectorizer.get_feature_names_out())

    lda_coherence = compute_coherence(lda_topics, tokenized_docs, dictionary)
    nmf_coherence = compute_coherence(nmf_topics, tokenized_docs, dictionary)
    lda_diversity = compute_diversity(lda_topics)
    nmf_diversity = compute_diversity(nmf_topics)

    print(f"LDA  - coherence: {lda_coherence:.4f}, diversity: {lda_diversity:.4f}")
    print(f"NMF  - coherence: {nmf_coherence:.4f}, diversity: {nmf_diversity:.4f}")

    n_topics = lda_model.components_.shape[0]
    winner = "NMF" if nmf_coherence >= lda_coherence else "LDA"
    print(f"\nn_topics = {n_topics}")
    print(f"Higher coherence: {winner}")

    results = pd.DataFrame({
        "Metric": ["Topic Coherence (c_v)", "Topic Diversity"],
        "LDA": [round(lda_coherence, 4), round(lda_diversity, 4)],
        "NMF": [round(nmf_coherence, 4), round(nmf_diversity, 4)],
    })
    results.to_csv("data/final_model_metrics.csv", index=False)
    print("\nSaved to data/final_model_metrics.csv")
