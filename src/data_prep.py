import pandas as pd
from bs4 import BeautifulSoup
import re
import numpy as np


INPUT_PATH = "../data/posts_questions.csv"
OUTPUT_PATH = "../data/cleaned_posts_questions_data.csv"

def clean_html(text):
    if pd.isna(text):
        return ""

    try:
        soup = BeautifulSoup(text, "lxml")
        for tag in soup.find_all(["code", "pre", "script", "style"]):
            tag.decompose()

        text = soup.get_text(separator=" ")
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'\s+', ' ', text)
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

df["log_views"] = np.log1p(df["view_count"])
df["score_norm"] = (df["score"] - df["score"].min()) / (df["score"].max() - df["score"].min())

df["difficulty_metric"] = (
    0.5 * df["answer_rate"] +
    0.3 * df["score_norm"] +
    0.2 * df["log_views"]
)

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
    "score"
]]

print("\nClass distribution:\n")
print(clean_df["difficulty"].value_counts())

clean_df.to_csv(OUTPUT_PATH, index=False)
print("\nCleaned dataset saved to:", OUTPUT_PATH)
