"""
preprocessing.py
Data cleaning + text preprocessing for the CFPB Customer Complaint dataset.

Usage (from PyCharm terminal, project root):
    python src/preprocessing.py
This will read data/complaints_raw.csv, clean it, and write data/complaints_clean.csv.
"""

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --- One-time NLTK downloads (safe to call every run; NLTK skips if already present) ---
for resource in ["stopwords", "wordnet", "omw-1.4", "punkt", "punkt_tab"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception as e:
        print(f"Could not download NLTK resource '{resource}': {e}")

LEMMATIZER = WordNetLemmatizer()

# Standard English stopwords + domain-specific noise words that show up in
# almost every complaint regardless of topic (add to this list as you explore the data)
BASE_STOPWORDS = set(stopwords.words("english"))
DOMAIN_STOPWORDS = {
    "company", "consumer", "account", "would", "told", "said",
    "also", "get", "got", "im", "ive", "us", "please", "even",
    "still", "back", "make", "made", "much", "since"
}
ALL_STOPWORDS = BASE_STOPWORDS.union(DOMAIN_STOPWORDS)


def clean_text(text: str) -> str:
    """Regex-based cleaning: lowercase, strip redactions/PII/noise, keep letters only."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"x{2,}", " ", text)              # CFPB redaction placeholders (XXXX)
    text = re.sub(r"\$[\d,]+\.?\d*", " ", text)      # dollar amounts
    text = re.sub(r"http\S+|www\.\S+", " ", text)    # URLs
    text = re.sub(r"\S+@\S+", " ", text)             # emails
    text = re.sub(r"[^a-z\s]", " ", text)            # anything that isn't a letter/space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(text: str, min_token_len: int = 3) -> str:
    """Tokenize, remove stopwords, lemmatize. Returns a cleaned, space-joined string
    ready for the vectorizers (TfidfVectorizer / CountVectorizer both accept raw strings)."""
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    tokens = [
        LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in ALL_STOPWORDS and len(tok) >= min_token_len
    ]
    return " ".join(tokens)


def load_and_prepare(
    raw_path: str = "data/complaints_raw.csv",
    text_column: str = "Consumer complaint narrative",
    product_filter: str | None = None,
    product_column: str = "Product",
    sample_size: int | None = 30000,
    min_words: int = 15,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load the raw CFPB CSV, filter to a single product category (recommended -
    keeps topics coherent instead of mixing e.g. mortgages with credit cards),
    drop empty/short narratives, sample down to a manageable size, and clean.

    Returns a DataFrame with columns: raw_text, clean_text
    """
    df = pd.read_csv(raw_path, low_memory=False)
    df = df[df[text_column].notna()].copy()

    if product_filter:
        df = df[df[product_column] == product_filter].copy()
        print(f"Filtered to product = '{product_filter}': {len(df)} rows")

    df["word_count"] = df[text_column].str.split().str.len()
    df = df[df["word_count"] >= min_words].copy()

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    print("Cleaning and preprocessing text (this can take a few minutes on 20k+ rows)...")
    df["clean_text"] = df[text_column].apply(preprocess_text)
    df = df[df["clean_text"].str.len() > 0].copy()  # drop rows that became empty after cleaning

    result = df[[text_column, "clean_text"]].rename(columns={text_column: "raw_text"})
    return result.reset_index(drop=True)


if __name__ == "__main__":
    # Adjust product_filter to whatever category your group chose to focus on
    cleaned_df = load_and_prepare(
        raw_path="data/complaints_raw.csv",
        product_filter=None,
        sample_size=30000,
    )
    cleaned_df.to_csv("data/complaints_clean.csv", index=False)
    print(f"Saved {len(cleaned_df)} cleaned complaints to data/complaints_clean.csv")
    print(cleaned_df.head())
