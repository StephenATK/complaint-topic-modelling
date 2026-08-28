"""
evaluation.py
Topic coherence, topic diversity, topic distribution, and an n_topics tuning sweep.

Usage (from PyCharm terminal, project root):
    python src/evaluation.py
Reads data/complaints_clean.csv. Prints a coherence-vs-n_topics table you can
screenshot/plot for your report, plus final diversity/distribution numbers.
"""

import pandas as pd
import numpy as np
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

from modeling import build_vectorizers, fit_lda, fit_nmf, get_top_words


def _filter_for_coherence(topics, dictionary):
    """Coherence scoring needs single tokens that exist in the gensim dictionary.
    Our vectorizers use ngram_range=(1,2), so some top words are bigrams like
    'debt collector' (a space-joined phrase) that won't match any single token
    in the dictionary. Drop those here - this only affects the coherence score,
    not the topic words shown in the app or report."""
    filtered = []
    for topic in topics:
        words = [w for w in topic if w in dictionary.token2id]
        if not words:
            # Fallback: split any bigram into its individual words instead of
            # dropping the topic entirely
            words = [tok for w in topic for tok in w.split() if tok in dictionary.token2id]
        filtered.append(words)
    return filtered


def compute_coherence(topics, tokenized_docs, dictionary):
    """topics: list of lists of top words per topic (from get_top_words)."""
    safe_topics = _filter_for_coherence(topics, dictionary)
    cm = CoherenceModel(
        topics=safe_topics, texts=tokenized_docs, dictionary=dictionary, coherence="c_v"
    )
    return cm.get_coherence()


def compute_diversity(topics):
    """% of unique words among all top-N words across all topics.
    Low diversity = topics are redundant/overlapping."""
    all_words = [word for topic in topics for word in topic]
    unique_words = set(all_words)
    return len(unique_words) / len(all_words)


def topic_distribution(doc_topic_matrix):
    """Document count assigned to each topic (argmax of the doc-topic matrix)."""
    assignments = np.argmax(doc_topic_matrix, axis=1)
    counts = pd.Series(assignments).value_counts().sort_index()
    return counts


def sweep_n_topics(count_matrix, tfidf_matrix, count_vectorizer, tfidf_vectorizer,
                    tokenized_docs, dictionary, topic_range=range(4, 13)):
    """Fit LDA + NMF across a range of n_topics and score each on coherence.
    Use this to justify your final n_topics choice with a plot in the report."""
    results = []
    for n in topic_range:
        lda_model, _ = fit_lda(count_matrix, n)
        nmf_model, _ = fit_nmf(tfidf_matrix, n)

        lda_topics = get_top_words(lda_model, count_vectorizer.get_feature_names_out())
        nmf_topics = get_top_words(nmf_model, tfidf_vectorizer.get_feature_names_out())

        lda_coherence = compute_coherence(lda_topics, tokenized_docs, dictionary)
        nmf_coherence = compute_coherence(nmf_topics, tokenized_docs, dictionary)

        results.append({"n_topics": n, "lda_coherence": lda_coherence, "nmf_coherence": nmf_coherence})
        print(f"n_topics={n}: LDA coherence={lda_coherence:.4f}, NMF coherence={nmf_coherence:.4f}")

    return pd.DataFrame(results)


if __name__ == "__main__":
    df = pd.read_csv("data/complaints_clean.csv")
    docs = df["clean_text"].astype(str).tolist()
    tokenized_docs = [doc.split() for doc in docs]
    dictionary = Dictionary(tokenized_docs)

    tfidf_vectorizer, tfidf_matrix, count_vectorizer, count_matrix = build_vectorizers(docs)

    print("Running n_topics sweep (this fits many models - grab a coffee)...")
    sweep_df = sweep_n_topics(
        count_matrix, tfidf_matrix, count_vectorizer, tfidf_vectorizer,
        tokenized_docs, dictionary, topic_range=range(4, 13)
    )
    sweep_df.to_csv("data/n_topics_sweep.csv", index=False)
    print("\nSaved sweep results to data/n_topics_sweep.csv - plot this for your report.")

    best_n_lda = sweep_df.loc[sweep_df["lda_coherence"].idxmax(), "n_topics"]
    best_n_nmf = sweep_df.loc[sweep_df["nmf_coherence"].idxmax(), "n_topics"]
    print(f"\nBest n_topics by coherence -> LDA: {best_n_lda}, NMF: {best_n_nmf}")