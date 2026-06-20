import pandas as pd

df = pd.read_csv("Dataset.csv")

# Convert categorical columns
df = pd.get_dummies(df, drop_first=True)

print(df.corr()["Price"].sort_values(ascending=False))