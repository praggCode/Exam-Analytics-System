import pandas as pd
file_path = 'data/posts_questions.csv'
df_peek = pd.read_csv(file_path, nrows=5)
print(df_peek.columns.tolist())