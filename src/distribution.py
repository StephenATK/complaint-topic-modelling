"""
distribution.py
Shared helper for the "topic distribution" evaluation criterion.
Used by src/distribution_metrics.py and by the app, so both report the same numbers.

Each complaint is counted under the topic it scores highest on.
"""

import numpy as np
import pandas as pd

NEAR_EMPTY_SHARE = 0.02   # a topic holding under 2% of complaints counts as near-empty


def topic_counts(doc_topic):
    """Complaints per topic, using each complaint's highest-scoring topic."""
    doc_topic = np.asarray(doc_topic)
    n_topics = doc_topic.shape[1]
    assigned = doc_topic.argmax(axis=1)
    return pd.Series(assigned).value_counts().reindex(range(n_topics), fill_value=0).sort_index()


def summarize(counts):
    """Single-number summaries of how evenly complaints spread across topics.

    evenness: Shannon entropy of the topic shares divided by its maximum (log of the
    number of topics). 1.0 means every topic holds the same number of complaints;
    values near 0 mean nearly everything sits in one topic.
    """
    counts = pd.Series(counts).astype(float)
    total = counts.sum()
    shares = counts / total if total else counts
    nonzero = shares[shares > 0]
    k = len(counts)
    evenness = float(-(nonzero * np.log(nonzero)).sum() / np.log(k)) if k > 1 else 1.0
    return {
        "complaints": int(total),
        "largest_topic": int(shares.idxmax()),
        "largest_share": float(shares.max()),
        "smallest_topic": int(shares.idxmin()),
        "smallest_share": float(shares.min()),
        "evenness": evenness,
        "topics_under_2pct": int((shares < NEAR_EMPTY_SHARE).sum()),
    }
