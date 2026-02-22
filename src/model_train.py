import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from scipy.sparse import hstack
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

DATA_PATH = "../data/cleaned_posts_questions_data.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)
    if len(df) > 300000:
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
        df[["score", "question_length"]]
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

    X, y, vectorizer = create_features(df)

    # print("This is X:",X)
    # print("This is y:",y)

    print("This is vectorizer:", vectorizer)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=69, stratify=y
    )

    # print("\nTrain shape:", X_train.shape)
    # print("Test shape:", X_test.shape)

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import LinearSVC
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=5000, solver="saga", class_weight="balanced", random_state=42),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
        "Linear SVC": LinearSVC(class_weight="balanced", max_iter=5000, random_state=42)
    }

    for name, model in models.items():
        print(f"\n============= {name} =============")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")

        print("\nClassification Report:\n")
        print(classification_report(y_test, y_pred))

        print("\nConfusion Matrix:\n")
        print(confusion_matrix(y_test, y_pred))
