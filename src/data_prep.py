import pandas as pd
from bs4 import BeautifulSoup
import re

INPUT_PATH = "data/posts_questions.csv"
OUTPUT_PATH = "data/cleaned_posts_questions_data.csv"

def clean_html(text):
    if pd.isna(text):
        return ""
    soup = BeautifulSoup(text, "lxml")
    text = soup.get_text()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df = pd.read_csv(INPUT_PATH, nrows=10000)

# "Combining title + body"
df["question"] = df["title"].astype(str) + " " + df["body"].astype(str)

df["question"] = df["question"].apply(clean_html)

# Easy = score >= 2
df["difficulty"] = df["score"].apply(lambda x: "Easy" if x >= 2 else "Hard")

clean_df = df[["question", "difficulty"]]

print("Saving cleaned dataset")
clean_df.to_csv(OUTPUT_PATH, index=False)

print("Cleaned data saved at", OUTPUT_PATH)