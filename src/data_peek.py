import pandas as pd
file_path = '../data/posts_questions.csv'
df_peek = pd.read_csv(file_path, nrows=5)
df_cleaned = pd.read_csv("../data/cleaned_posts_questions2.csv",nrows=5)
print(df_peek.columns.tolist())

print(df_peek.sample(n=5))
print(df_cleaned.sample(n=5))