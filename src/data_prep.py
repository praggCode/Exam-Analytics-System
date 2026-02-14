import pandas as pd
from bs4 import BeautifulSoup
import re
from sklearn.preprocessing import MinMaxScaler

INPUT_PATH = "data/posts_questions.csv"
OUTPUT_PATH = "data/cleaned_posts_questions_data.csv"

def clean_html(text):
    if pd.isna(text):
        return ""
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup.find_all(["code", "pre", "script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

df = pd.read_csv(INPUT_PATH, nrows=10000)

# Combine title + body
df["question"] = df["title"].astype(str) + " " + df["body"].astype(str)
df["question"] = df["question"].apply(clean_html)

# ---------------- DIFFICULTY SCORING ----------------
scaler = MinMaxScaler()

df[["score_n", "answer_n", "view_n"]] = scaler.fit_transform(df[["score", "answer_count", "view_count"]])

df["difficulty_score"] = (
    0.5 * df["view_n"] +      
    0.3 * df["answer_n"] +   
    0.2 * df["score_n"]  
)

q1 = df["difficulty_score"].quantile(0.33)
q2 = df["difficulty_score"].quantile(0.66)

def label_difficulty(x):
    if x <= q1:
        return "Hard"
    elif x <= q2:
        return "Medium"
    else:
        return "Easy"

df["difficulty"] = df["difficulty_score"].apply(label_difficulty)
clean_df = df[["question", "difficulty"]]
clean_df.to_csv(OUTPUT_PATH, index=False)
