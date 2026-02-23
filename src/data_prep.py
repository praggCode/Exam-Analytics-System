import pandas as pd
from bs4 import BeautifulSoup
import re
import os
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import ssl

# Fix SSL for NLTK downloads on macOS
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Initialize lemmatizer
def setup_nltk():
    resources = ['wordnet', 'punkt', 'omw-1.4', 'punkt_tab']
    for res in resources:
        try:
            nltk.data.find(res)
        except LookupError:
            nltk.download(res)

setup_nltk()
lemmatizer = WordNetLemmatizer()

# ---------------- CONFIGURATION ----------------
INPUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/raw/posts_questions.csv"))
OUTPUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/processed/processed_data.csv"))
TARGET_ROWS = 1000000
CHUNK_SIZE = 50000

def clean_html(text):
    if pd.isna(text):
        return ""

    try:
        # Using lxml parser for speed as verified in requirements
        soup = BeautifulSoup(text, "lxml")
        for tag in soup.find_all(["code", "pre", "script", "style"]):
            tag.decompose()

        text = soup.get_text(separator=" ")
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'\s+', ' ', text)

        # Remove stray surrogate characters
        text = text.encode("utf-8", "ignore").decode("utf-8").strip()

        # Lemmatization (optional/graceful failure)
        try:
            tokens = word_tokenize(text.lower())
            text = " ".join([lemmatizer.lemmatize(token) for token in tokens])
        except Exception:
            pass

        return text.strip()

    except Exception:
        return ""

def process_data():
    print(f"Starting processing to collect {TARGET_ROWS} rows...")
    print(f"Input: {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")

    collected_chunks = []
    total_collected = 0

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # ---------------- LOAD DATA IN CHUNKS ----------------
    for i, chunk in enumerate(pd.read_csv(INPUT_PATH, chunksize=CHUNK_SIZE)):
        # Filter valid rows
        valid_chunk = chunk[
            (chunk["answer_count"] > 0) &
            (chunk["view_count"] > 50) &
            (chunk["title"].notna()) &
            (chunk["body"].notna())
        ].copy()

        if valid_chunk.empty:
            continue

        # Clean HTML (Title + Body)
        # Doing this per chunk to keep progress visible
        valid_chunk["question"] = valid_chunk["title"].astype(str) + " " + valid_chunk["body"].astype(str)
        valid_chunk["question"] = valid_chunk["question"].apply(clean_html)

        collected_chunks.append(valid_chunk)
        total_collected += len(valid_chunk)
        
        print(f"Chunk {i+1}: Collected {len(valid_chunk)} rows (Total: {total_collected})")

        if total_collected >= TARGET_ROWS:
            break

    if not collected_chunks:
        print("No valid rows found.")
        return

    # Combine and trim to exact TARGET_ROWS
    df = pd.concat(collected_chunks, ignore_index=True)
    if len(df) > TARGET_ROWS:
        df = df.iloc[:TARGET_ROWS]

    print(f"\nFinal dataset size: {len(df)}")

    # ---------------- DIFFICULTY METRIC ----------------
    df["answer_rate"] = df["answer_count"] / df["view_count"]
    df["difficulty_metric"] = df["answer_rate"]

    # Quantile-based labeling
    q1 = df["difficulty_metric"].quantile(0.33)
    q2 = df["difficulty_metric"].quantile(0.66)

    def label_difficulty(x):
        if x <= q1:
            return "Hard"
        elif x <= q2:
            return "Medium"
        else:
            return "Easy"

    df["difficulty"] = df["difficulty_metric"].apply(label_difficulty)

    # ---------------- FINAL DATASET ----------------
    clean_df = df[[
        "question",
        "difficulty",
        "answer_count",
        "view_count",
        "score",
        "tags"
    ]]

    print("\nClass distribution:")
    print(clean_df["difficulty"].value_counts())

    clean_df.to_csv(OUTPUT_PATH, index=False)
    print("\nCleaned dataset saved to:", OUTPUT_PATH)

if __name__ == "__main__":
    process_data()

