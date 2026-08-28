"""
modeling.py
Builds TF-IDF / Count representations and fits LDA + NMF topic models.

Usage (from PyCharm terminal, project root):
    python src/modeling.py
Reads data/complaints_clean.csv (produced by preprocessing.py).
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import joblib


def build_vectorizers(docs, max_features=5000, min_df=5, max_df=0.9, ngram_range=(1, 2)):
    """
    Two vectorizers, on purpose:
    - TF-IDF feeds NMF (matrix factorization on weighted terms)
    - Raw counts feed LDA (a generative model over word COUNTS, not TF-IDF weights -
      this is the textbook-correct pairing and worth stating explicitly in your report)
    """
    tfidf_vectorizer = TfidfVectorizer(
        max_df=max_df, min_df=min_df, max_features=max_features, ngram_range=ngram_range
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(docs)

    count_vectorizer = CountVectorizer(
        max_df=max_df, min_df=min_df, max_features=max_features, ngram_range=ngram_range
    )
    count_matrix = count_vectorizer.fit_transform(docs)

    return tfidf_vectorizer, tfidf_matrix, count_vectorizer, count_matrix


def fit_lda(count_matrix, n_topics, random_state=42):
    lda = LatentDirichletAllocation(
        n_components=n_topics, random_state=random_state, learning_method="online", max_iter=20
    )
    lda_doc_topic = lda.fit_transform(count_matrix)
    return lda, lda_doc_topic


def fit_nmf(tfidf_matrix, n_topics, random_state=42):
    nmf = NMF(n_components=n_topics, random_state=random_state, init="nndsvd", max_iter=500)
    nmf_doc_topic = nmf.fit_transform(tfidf_matrix)
    return nmf, nmf_doc_topic


def get_top_words(model, feature_names, n_top=10):
    """Returns a list of lists: top-N words per topic, ordered by weight."""
    topics = []
    for topic_weights in model.components_:
        top_indices = topic_weights.argsort()[: -n_top - 1 : -1]
        topics.append([feature_names[i] for i in top_indices])
    return topics


if __name__ == "__main__":
    df = pd.read_csv("data/complaints_clean.csv")
    docs = df["clean_text"].astype(str).tolist()

    tfidf_vectorizer, tfidf_matrix, count_vectorizer, count_matrix = build_vectorizers(docs)

    N_TOPICS = 11

    lda_model, lda_doc_topic = fit_lda(count_matrix, N_TOPICS)
    nmf_model, nmf_doc_topic = fit_nmf(tfidf_matrix, N_TOPICS)

    lda_topics = get_top_words(lda_model, count_vectorizer.get_feature_names_out())
    nmf_topics = get_top_words(nmf_model, tfidf_vectorizer.get_feature_names_out())

    print("\n--- LDA Topics ---")
    for i, words in enumerate(lda_topics):
        print(f"Topic {i}: {', '.join(words)}")

    print("\n--- NMF Topics ---")
    for i, words in enumerate(nmf_topics):
        print(f"Topic {i}: {', '.join(words)}")

    # Persist everything the Streamlit app and evaluation script will need
    joblib.dump(tfidf_vectorizer, "data/tfidf_vectorizer.joblib")
    joblib.dump(count_vectorizer, "data/count_vectorizer.joblib")
    joblib.dump(lda_model, "data/lda_model.joblib")
    joblib.dump(nmf_model, "data/nmf_model.joblib")
    np.save("data/lda_doc_topic.npy", lda_doc_topic)
    np.save("data/nmf_doc_topic.npy", nmf_doc_topic)
    print("\nSaved models and doc-topic matrices to data/")
