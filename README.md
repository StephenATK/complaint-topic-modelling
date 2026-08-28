# Group 8 — Customer Complaint Topic Modelling

## Setup in PyCharm

1. Open this folder as a PyCharm project.
2. Make sure PyCharm's interpreter is set to a Python environment (bottom-right corner
   should show a Python version, not "No interpreter").
3. Open the **Terminal** tab at the bottom of PyCharm (not the Python Console) and run:
   ```
   pip install -r requirements.txt
   ```

## Get the data

1. Go to https://www.consumerfinance.gov/data-research/consumer-complaints/
2. Download the complaints CSV.
3. Place it at `data/complaints_raw.csv` (this file is gitignored - too large for GitHub).
4. Open `src/preprocessing.py` and set `product_filter` to whichever `Product` category
   your group is focusing on (check the unique values in the `Product` column first).

## Run the pipeline (in order, from the Terminal tab)

```
python src/preprocessing.py     # cleans data/complaints_raw.csv -> data/complaints_clean.csv
python src/modeling.py          # trains LDA + NMF, saves models to data/
python src/evaluation.py        # runs the n_topics coherence sweep -> data/n_topics_sweep.csv
```

After the sweep, update `N_TOPICS` in `src/modeling.py` to the best value and re-run
`modeling.py` so the saved models use your tuned topic count.

## Run the Streamlit app

**Important:** use the Terminal tab, not the green Run button.

```
streamlit run app.py
```

This opens the app in your browser at `localhost:8501`.

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (add `data/complaints_raw.csv` to `.gitignore` — it's too big).
2. Go to https://share.streamlit.io, connect your GitHub, point it at this repo's `app.py`.
3. Make sure a small processed sample of `data/complaints_clean.csv` (and the saved
   `.joblib`/`.npy` model files) IS committed to the repo, since the cloud app needs
   them and won't have access to your local `data/complaints_raw.csv`.
