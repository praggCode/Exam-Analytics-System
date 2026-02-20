import pandas as pd
from bs4 import BeautifulSoup
import re


INPUT_PATH = "../data/posts_questions.csv"
OUTPUT_PATH = "../data/cleaned_posts_questions2.csv"

def clean_html(text):
    if pd.isna(text):
        return ""

    try:
        soup = BeautifulSoup(text, "lxml")

        # Remove code-heavy and non-content elements
        for tag in soup.find_all(["code", "pre", "script", "style"]):
            tag.decompose()

        text = soup.get_text(separator=" ")

        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove stray surrogate characters
        text = text.encode("utf-8", "ignore").decode("utf-8")

        return text.strip()

    except Exception:
        return ""


# ---------------- LOAD DATA ----------------
df = pd.read_csv(INPUT_PATH)
df = df[(df["answer_count"] > 0) &
    (df["view_count"] > 50) &
    (df["title"].notna()) &
    (df["body"].notna())]
df = df.sample(frac=1,random_state=42)


# Combine title + body
df["question"] = df["title"].astype(str) + " " + df["body"].astype(str)
df["question"] = df["question"].apply(clean_html)


# ---------------- DIFFICULTY METRIC ----------------

df["answer_rate"] = df["answer_count"] / df["view_count"]

df["difficulty_metric"] = df["answer_rate"]

print(df["difficulty_metric"].describe())



 
print(df["answer_count"].describe())

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

print("\nClass distribution:\n")
print(clean_df["difficulty"].value_counts())

clean_df.to_csv(OUTPUT_PATH, index=False)
print("\nCleaned dataset saved to:", OUTPUT_PATH)
