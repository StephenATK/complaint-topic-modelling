# Group 8 — Customer Complaint Topic Modelling

Topic modelling (LDA and NMF) on CFPB debt collection complaints, with a Streamlit app.
NMF is the final model: 11 topics, coherence 0.616, diversity 0.927.

## Setup in PyCharm

1. Open this folder as a PyCharm project and make sure an interpreter is selected
   (bottom-right corner shows a Python version).
2. In the **Terminal** tab: `pip install -r requirements.txt`

Versions in `requirements.txt` are pinned to the ones the deployed app runs on.
You don't need to reinstall locally if your existing setup already works.

## Pipeline (run in this order)

```
python src/preprocessing.py      # data/complaints_raw.csv -> data/complaints_clean.csv
python src/modeling.py           # trains LDA + NMF, saves models to data/
python src/evaluation.py         # coherence sweep -> data/n_topics_sweep.csv
python src/final_metrics.py      # final coherence + diversity -> data/final_model_metrics.csv
python src/build_topic_maps.py   # pre-builds the interactive topic maps -> data/topic_map_*.html
```

Re-run `build_topic_maps.py` any time you re-run `modeling.py`, so the maps match the models.

## Topic names

Topic names live in `data/topic_labels.json`, stored separately for NMF and LDA.
Each NMF topic also has a writing-style tag (Complaint issue, Legal template, or
Narrative account).

To rename topics, open **Topic Explorer → Name your topics**, edit the names, then
click **Save names**. Commit `data/topic_labels.json` so the deployed app shows them
to every visitor. On the deployed app itself, saved changes only last until the app
restarts, so always save locally and commit.

## Run the app locally

Use the Terminal tab, not the green Run button:

```
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)

Commit and push these so the deployed app has everything it needs:

- `app.py`, `requirements.txt`, everything in `src/`
- `data/complaints_clean.csv`, the four `.joblib` files, the two `.npy` files
- `data/n_topics_sweep.csv`, `data/final_model_metrics.csv`
- `data/topic_labels.json`, `data/topic_map_nmf.html`, `data/topic_map_lda.html`

The app redeploys automatically on every push to `main`. It was deployed with
Python 3.12 (set under Advanced settings at deploy time).

The saved topic maps include all their JavaScript, so they display without any
internet access beyond the app itself.
