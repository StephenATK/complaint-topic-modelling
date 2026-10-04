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
python src/distribution_metrics.py  # topic distribution for both models -> data/distribution_metrics.csv
```

Re-run `build_topic_maps.py` any time you re-run `modeling.py`, so the maps match the models.

## Rating topic interpretability

Interpretability is judged by people, so it needs the group. The rating is blind: raters
see 22 shuffled topics (11 from each model) without knowing which model each came from.

1. `pip install openpyxl` (once, local only; the live app doesn't need it)
2. `python src/make_rating_sheet.py` creates `ratings/rating_sheet_TEMPLATE.xlsx` and `data/rating_key.csv`.
   Don't share the key until everyone has finished.
3. Each member saves a copy as `ratings/<their name>.xlsx` and fills in the yellow cells.
4. `python src/score_ratings.py` writes `data/interpretability_scores.csv` and `data/interpretability_by_topic.csv`.
5. Commit `data/interpretability_scores.csv` so the live app's Model Comparison page shows the result.

## Voice Complaints page

Record a complaint with the microphone, or upload recordings (WAV, MP3, M4A, OGG, FLAC, WEBM, AAC or MP4 audio).
The app turns speech into text with faster-whisper (OpenAI's Whisper `base.en` model, running on the CPU),
then classifies the text the same way as a typed complaint. Uploads get a results table, a topic chart and a
CSV download. Recordings are not saved.

- The Whisper model (about 145 MB) downloads automatically the first time the page transcribes something,
  so the first voice complaint after the app wakes up can take up to a minute.
- It works best on recordings of the customer speaking. On two-person calls the agent's words are
  transcribed too, which can blur the topic.
- Versions are pinned in `requirements.txt`. Keep `av==16.0.1`: faster-whisper 1.2.1 fails with PyAV 19.
- If `pip install -r requirements.txt` fails on your own computer's Python version, every other page still
  works. The live app runs Python 3.12, where the pinned versions install.

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
