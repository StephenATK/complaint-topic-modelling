"""
topic_map.py
Builds the interactive pyLDAvis topic map as a self-contained HTML string.

Shared by two callers:
  - src/build_topic_maps.py  (pre-builds the maps once and saves them to data/)
  - app.py                   (fallback: builds live if a pre-built file is missing)
"""

import os

import numpy as np
import pyLDAvis
from pyLDAvis._prepare import js_MMDS

# pyLDAvis ships the JavaScript and CSS it needs inside the package. Inlining them
# makes each saved map fully self-contained: no downloads from d3js.org or
# jsdelivr when the page is viewed, so the map still works on a slow or
# restricted network (e.g. classroom Wi-Fi during a demo).
_JS_DIR = os.path.join(os.path.dirname(pyLDAvis.__file__), "js")


def _asset(name):
    with open(os.path.join(_JS_DIR, name), encoding="utf-8") as f:
        return f.read()


def _self_contained_page(vis_json):
    """Builds a standalone HTML page around pyLDAvis's JSON data. Written with
    string concatenation rather than an f-string, because the inlined
    JavaScript is full of curly braces."""
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<style>" + _asset("ldavis.v1.0.0.css") +
        " body{margin:0;background:#ffffff;font-family:Arial,Helvetica,sans-serif;}</style>"
        "</head><body>"
        "<div id='ldavis_map'></div>"
        "<script>" + _asset("d3.v5.min.js") + "</script>"
        "<script>" + _asset("ldavis.v3.0.0.js") + "</script>"
        "<script>var ldavisData = " + vis_json + ";"
        "new LDAvis('#ldavis_map', ldavisData);"
        # pyLDAvis draws at a fixed ~1,200px width. Scale the whole page down to
        # fit narrower frames (like the app's content column) instead of cutting
        # off the term chart on the right.
        "function fitMap(){var b=document.body;b.style.zoom=1;"
        "var w=document.getElementById('ldavis_map').scrollWidth||1;"
        "b.style.zoom=Math.min(1,(window.innerWidth-8)/w);}"
        "window.addEventListener('resize',fitMap);setTimeout(fitMap,50);setTimeout(fitMap,500);"
        "</script>"
        "</body></html>"
    )


def build_topic_map_html(model, feature_names, doc_topic, doc_term_matrix):
    """
    Works for both LDA and NMF.

    - NMF's outputs are not probability distributions, so topic-term and
      doc-topic weights are normalised to sum to 1 (pyLDAvis requires this).
    - Documents with no weight on any topic, or no known vocabulary, are dropped
      before normalising. Dividing by a zero row sum is what makes pyLDAvis
      reject the input.
    - mds=js_MMDS positions the bubbles with an iterative method. The default
      (PCoA) can produce complex numbers on real data, which crashes the
      JSON export.
    - start_index=0 makes bubble numbers match the app's topic numbers.
    - n_jobs=1 avoids spawning extra processes, which can exhaust memory on
      Windows laptops and on Streamlit Community Cloud.
    - The returned page has d3 and the LDAvis script inlined, so it needs no
      internet connection to display.
    """
    topic_term = np.asarray(model.components_, dtype=float)
    topic_term_dists = topic_term / topic_term.sum(axis=1, keepdims=True)

    doc_topic = np.asarray(doc_topic, dtype=float)
    doc_lengths = np.asarray(doc_term_matrix.sum(axis=1)).ravel()
    row_sums = doc_topic.sum(axis=1)

    keep = (row_sums > 0) & (doc_lengths > 0)
    doc_topic_dists = doc_topic[keep] / row_sums[keep][:, None]
    doc_lengths = doc_lengths[keep]

    term_frequency = np.asarray(doc_term_matrix.sum(axis=0)).ravel()

    vis_data = pyLDAvis.prepare(
        topic_term_dists=topic_term_dists,
        doc_topic_dists=doc_topic_dists,
        doc_lengths=doc_lengths,
        vocab=list(feature_names),
        term_frequency=term_frequency,
        sort_topics=False,
        start_index=0,
        mds=js_MMDS,
        n_jobs=1,
    )
    return _self_contained_page(vis_data.to_json())
