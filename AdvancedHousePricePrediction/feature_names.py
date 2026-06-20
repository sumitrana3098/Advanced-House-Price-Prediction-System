import pandas as pd

df = pd.read_csv("Dataset.csv")

df = pd.get_dummies(df, drop_first=True)

X = df.drop("Price", axis=1)

print(X.columns.tolist())