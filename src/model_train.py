import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import numpy as np
from scipy.sparse import hstack


DATA_PATH = "../data/cleaned_posts_questions2.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.sample(n=300000, random_state=42)
    print("Dataset shape:", df.shape)
    print("\nClass distribution:\n")
    print(df["difficulty"].value_counts())
    return df


def create_features(df):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), stop_words="english", max_features=50000, min_df=5
    )

    X_tfidf = vectorizer.fit_transform(df["question"])
    scaler = StandardScaler()

    X_numeric = scaler.fit_transform(
        df[["score", "question_length","tag_count"]]
    )
    
    X = hstack([X_tfidf, X_numeric])
    y = df["difficulty"]

    print("\nFeature matrix shape:", X.shape)

    return X, y, vectorizer


if __name__ == "__main__":
    df = load_data()
    df = df.dropna(subset=["question"])
    df["question"] = df["question"].astype(str)
    df["question_length"] = df["question"].apply(len)
    df["tag_count"] = df["tags"].apply(lambda x: x.count("|"))

    X, y, vectorizer = create_features(df)

    # print("This is X:",X)
    # print("This is y:",y)

    print("This is vectorizer:", vectorizer)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=69, stratify=y
    )

    # print("\nTrain shape:", X_train.shape)
    # print("Test shape:", X_test.shape)

    model = LogisticRegression(max_iter=5000, solver="saga", class_weight="balanced")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:\n")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
