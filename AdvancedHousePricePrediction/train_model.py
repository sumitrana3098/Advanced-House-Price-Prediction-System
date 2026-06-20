import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# ==================================
# Load Dataset
# ==================================

df = pd.read_csv("Dataset.csv")

print("Dataset Loaded Successfully")
print(df.head())

# ==================================
# Convert Categorical Columns
# ==================================

df = pd.get_dummies(df, drop_first=True)

print("\nDataset Shape After Encoding:")
print(df.shape)

# ==================================
# Features and Target
# ==================================

X = df.drop("Price", axis=1)
y = df["Price"]

# ==================================
# Train-Test Split
# ==================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# ==================================
# Linear Regression
# ==================================

lr = LinearRegression()

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

print("\n==============================")
print("LINEAR REGRESSION")
print("==============================")

print("R2 Score:",
      r2_score(y_test, lr_pred))

print("MAE:",
      mean_absolute_error(y_test, lr_pred))

print("RMSE:",
      mean_squared_error(y_test, lr_pred) ** 0.5)

# ==================================
# Decision Tree
# ==================================

dt = DecisionTreeRegressor(
    random_state=42
)

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

print("\n==============================")
print("DECISION TREE")
print("==============================")

print("R2 Score:",
      r2_score(y_test, dt_pred))

print("MAE:",
      mean_absolute_error(y_test, dt_pred))

print("RMSE:",
      mean_squared_error(y_test, dt_pred) ** 0.5)

# ==================================
# Random Forest
# ==================================

rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("\n==============================")
print("RANDOM FOREST")
print("==============================")

print("R2 Score:",
      r2_score(y_test, rf_pred))

print("MAE:",
      mean_absolute_error(y_test, rf_pred))

print("RMSE:",
      mean_squared_error(y_test, rf_pred) ** 0.5)

# ==================================
# Save Model
# ==================================

joblib.dump(lr, "house_price_model.pkl")

print("\nModel Saved Successfully")
print("File Name: house_price_model.pkl")