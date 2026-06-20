import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("Dataset.csv")

print(df.head())

# Dataset Shape
print("\nDataset Shape:")
print(df.shape)

# Basic Statistics
print("\nBasic Statistics:")
print(df.describe())

# ---------------------------
# 1. Price Distribution
# ---------------------------
plt.figure(figsize=(8,5))
sns.histplot(df["Price"], kde=True)
plt.title("Distribution of House Prices")
plt.xlabel("Price")
plt.ylabel("Count")
plt.show()

# ---------------------------
# 2. Area vs Price
# ---------------------------
plt.figure(figsize=(8,5))
sns.scatterplot(x="Area", y="Price", data=df)
plt.title("Area vs Price")
plt.show()

# ---------------------------
# 3. Bedrooms vs Price
# ---------------------------
plt.figure(figsize=(8,5))
sns.boxplot(x="Bedrooms", y="Price", data=df)
plt.title("Bedrooms vs Price")
plt.show()

# ---------------------------
# 4. Bathrooms vs Price
# ---------------------------
plt.figure(figsize=(8,5))
sns.boxplot(x="Bathrooms", y="Price", data=df)
plt.title("Bathrooms vs Price")
plt.show()

# ---------------------------
# 5. City vs Price
# ---------------------------
plt.figure(figsize=(10,5))
sns.boxplot(x="City", y="Price", data=df)
plt.title("City vs Price")
plt.xticks(rotation=45)
plt.show()

# ---------------------------
# 6. Furnishing vs Price
# ---------------------------
plt.figure(figsize=(8,5))
sns.boxplot(x="Furnishing", y="Price", data=df)
plt.title("Furnishing vs Price")
plt.show()

# ---------------------------
# 7. Locality Rating vs Price
# ---------------------------
plt.figure(figsize=(8,5))
sns.boxplot(x="Locality Rating", y="Price", data=df)
plt.title("Locality Rating vs Price")
plt.show()

# ---------------------------
# 8. Correlation Heatmap
# ---------------------------
numeric_df = df.select_dtypes(include=["int64", "float64"])

plt.figure(figsize=(10,6))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()