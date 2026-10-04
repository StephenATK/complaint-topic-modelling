"""
score_ratings.py
Scores the "topic interpretability" criterion from the group's filled-in rating sheets.

Usage (after everyone has saved ratings/<name>.xlsx):
    python src/score_ratings.py

Reads:  ratings/*.xlsx (every file except the TEMPLATE), data/rating_key.csv
Writes: data/interpretability_scores.csv    one row per model
        data/interpretability_by_topic.csv  one row per topic, with the names raters suggested
"""

import glob
import os

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)


def main():
    key = pd.read_csv("data/rating_key.csv")
    files = [f for f in sorted(glob.glob("ratings/*.xlsx")) if "TEMPLATE" not in os.path.basename(f).upper()
             and not os.path.basename(f).startswith("~$")]
    if not files:
        raise SystemExit("No filled-in sheets found. Save each rater's copy as ratings/<name>.xlsx first.")

    rows, problems = [], []
    for f in files:
        rater = os.path.splitext(os.path.basename(f))[0]
        sheet = pd.read_excel(f, sheet_name="Ratings")
        sheet = sheet[sheet["Code"].astype(str).str.match(r"^T\d{2}$")]
        for _, r in sheet.iterrows():
            scores = []
            for col in ["Word clarity (1-5)", "Example fit (1-5)"]:
                v = pd.to_numeric(r.get(col), errors="coerce")
                scores.append(v if pd.notna(v) and 1 <= v <= 5 else None)
            if None in scores:
                problems.append(f"{rater} {r['Code']}")
                continue
            name = r.get("Suggested name")
            rows.append({"rater": rater, "code": r["Code"], "word_clarity": scores[0], "example_fit": scores[1],
                         "overall": sum(scores) / 2, "suggested_name": "" if pd.isna(name) else str(name).strip()})
    if problems:
        print(f"Skipped {len(problems)} incomplete or invalid rows:", ", ".join(problems[:15]), "..." if len(problems) > 15 else "")
    data = pd.DataFrame(rows).merge(key, on="code")

    by_topic = (data.groupby(["model", "topic", "code"])
                .agg(raters=("rater", "nunique"), word_clarity=("word_clarity", "mean"),
                     example_fit=("example_fit", "mean"), overall=("overall", "mean"), spread=("overall", "std"),
                     suggested_names=("suggested_name", lambda s: "; ".join(sorted({x for x in s if x}))))
                .reset_index().round(2))
    by_model = (data.groupby("model")
                .agg(raters=("rater", "nunique"), ratings=("overall", "size"), word_clarity=("word_clarity", "mean"),
                     example_fit=("example_fit", "mean"), overall=("overall", "mean"))
                .reset_index())
    spread = by_topic.groupby("model")["spread"].mean().rename("rater_spread")
    by_model = by_model.merge(spread, on="model", how="left").round(2)

    by_topic.sort_values(["model", "topic"]).to_csv("data/interpretability_by_topic.csv", index=False)
    by_model.to_csv("data/interpretability_scores.csv", index=False)
    print(by_model.to_string(index=False))
    print("\nOverall is the average of word clarity and example fit, on a 1 to 5 scale.")
    print("rater_spread is the average disagreement between raters per topic (lower means more agreement).")
    print("Saved data/interpretability_scores.csv and data/interpretability_by_topic.csv")


if __name__ == "__main__":
    main()
