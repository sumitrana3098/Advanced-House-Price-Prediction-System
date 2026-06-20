import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("Dataset.csv")

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Features and target
X = df.drop("Price", axis=1)
y = df["Price"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Create dataframe
importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

importance["Absolute"] = importance["Coefficient"].abs()

importance = importance.sort_values(
    by="Absolute",
    ascending=False
)

print(importance.head(15))

# Plot
plt.figure(figsize=(10,6))

sns.barplot(
    x="Absolute",
    y="Feature",
    data=importance.head(15)
)

plt.title("Top 15 Features Affecting House Price")

plt.show()