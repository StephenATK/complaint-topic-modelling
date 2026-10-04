"""
distribution_metrics.py
Measures topic distribution for both final models: how the complaints spread across the topics.

Usage (right-click > Run in PyCharm, or from the terminal):
    python src/distribution_metrics.py

Reads:  data/nmf_doc_topic.npy, data/lda_doc_topic.npy, data/topic_labels.json
Writes: data/topic_distribution.csv   one row per topic per model (counts and shares)
        data/distribution_metrics.csv one row per model (summary numbers)
"""

import json
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from distribution import topic_counts, summarize  # noqa: E402


def topic_names(model):
    try:
        with open("data/topic_labels.json", encoding="utf-8") as f:
            entries = json.load(f).get(model, {}) or {}
        return {int(k): v.get("name", f"Topic {k}") for k, v in entries.items()}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def main():
    per_topic, summary = [], []
    for model in ["NMF", "LDA"]:
        doc_topic = np.load(f"data/{model.lower()}_doc_topic.npy")
        counts = topic_counts(doc_topic)
        names = topic_names(model)
        total = counts.sum()
        for t, n in counts.items():
            per_topic.append({"model": model, "topic": t, "topic_name": names.get(t, f"Topic {t}"),
                              "complaints": int(n), "share_pct": round(100 * n / total, 2)})
        s = summarize(counts)
        summary.append({"model": model,
                        "complaints": s["complaints"],
                        "largest_topic": names.get(s["largest_topic"], f"Topic {s['largest_topic']}"),
                        "largest_share_pct": round(100 * s["largest_share"], 1),
                        "smallest_topic": names.get(s["smallest_topic"], f"Topic {s['smallest_topic']}"),
                        "smallest_share_pct": round(100 * s["smallest_share"], 1),
                        "evenness": round(s["evenness"], 3),
                        "topics_under_2pct": s["topics_under_2pct"]})

    pd.DataFrame(per_topic).to_csv("data/topic_distribution.csv", index=False)
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv("data/distribution_metrics.csv", index=False)

    print(summary_df.to_string(index=False))
    print("\nEvenness runs from 0 (everything in one topic) to 1 (every topic the same size).")
    print("Saved data/topic_distribution.csv and data/distribution_metrics.csv")


if __name__ == "__main__":
    main()
