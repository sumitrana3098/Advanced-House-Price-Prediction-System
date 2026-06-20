import pandas as pd

# Load dataset
df = pd.read_csv("Dataset.csv")

print("\nFIRST 10 ROWS")
print(df.head(10))

print("\nDATASET INFO")
print(df.info())

print("\nMISSING VALUES")
print(df.isnull().sum())