"""
build_topic_maps.py
Pre-builds the interactive pyLDAvis topic maps for NMF and LDA and saves them
as HTML files, so the deployed app loads them instantly instead of computing
them live on every visit.

Usage (right-click -> Run in PyCharm, or from the terminal):
    python src/build_topic_maps.py

Run it again whenever you re-run modeling.py, so the maps match the models.

Reads:  data/complaints_clean.csv, the saved models and vectorizers,
        data/lda_doc_topic.npy, data/nmf_doc_topic.npy
Writes: data/topic_map_nmf.html, data/topic_map_lda.html
"""

import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

# Always work from the project root, whichever folder PyCharm launches from.
# This avoids the "No such file or directory: 'data/...'" error.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from topic_map import build_topic_map_html  # noqa: E402

# Library deprecation notices and log(0) warnings from pyLDAvis don't affect the output.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main():
    df = pd.read_csv("data/complaints_clean.csv")
    docs = df["clean_text"].astype(str)

    tfidf_vectorizer = joblib.load("data/tfidf_vectorizer.joblib")
    count_vectorizer = joblib.load("data/count_vectorizer.joblib")
    nmf_model = joblib.load("data/nmf_model.joblib")
    lda_model = joblib.load("data/lda_model.joblib")
    nmf_doc_topic = np.load("data/nmf_doc_topic.npy")
    lda_doc_topic = np.load("data/lda_doc_topic.npy")

    jobs = [
        ("nmf", nmf_model, tfidf_vectorizer, nmf_doc_topic),
        ("lda", lda_model, count_vectorizer, lda_doc_topic),
    ]

    for name, model, vectorizer, doc_topic in jobs:
        if doc_topic.shape[0] != len(docs):
            print(f"Skipping {name.upper()}: {doc_topic.shape[0]} topic rows but {len(docs)} "
                  f"complaints. Re-run modeling.py so they match, then run this again.")
            continue

        print(f"Building the {name.upper()} topic map (can take a minute)...")
        html = build_topic_map_html(
            model,
            vectorizer.get_feature_names_out(),
            doc_topic,
            vectorizer.transform(docs),
        )
        path = f"data/topic_map_{name}.html"
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  Saved {path} ({os.path.getsize(path) / 1024:.0f} KB)")

    print("\nDone. Commit the new HTML files in data/ so the deployed app can use them.")


if __name__ == "__main__":
    main()
