import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df = pd.read_excel("Sales_Data.xlsx")
print(df.head())
print(df.columns.tolist())
print(df.select_dtypes(include="number").columns.tolist())
df.columns = df.columns.str.strip()

X = df[["Units Sold", "Sale Price", "Discount"]]
y = df["Sales"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape)
print(X_test.shape)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(y_pred[:10])
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)
comparison = pd.DataFrame({
    "Actual Sales": y_test.values,
    "Predicted Sales": y_pred
})

print(comparison.head(10))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()
results = pd.DataFrame({
    "Actual Sales": y_test,
    "Predicted Sales": y_pred
})

print(results.head(10))
results.to_excel("Regression_Results.xlsx", index=False)

print("Regression results saved successfully!")
results.to_excel("Regression_Results.xlsx", index=False)

print("Regression results saved successfully!")
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()
print("\n--- Regression Model Summary ---")
print("Target Variable: Sales")
print("Features: Units Sold, Sale Price, Discount")
print("Train-Test Split: 80%-20%")
print("R2 Score:", r2)
print("MAE:", mae)
print("MSE:", mse)
print("\n--- EDA: Statistical Summary ---")
print(df.describe())
print("\n--- EDA: Missing Values ---")
print(df.isnull().sum())
print("\n--- EDA: Duplicate Rows ---")
print("Duplicate rows:", df.duplicated().sum())
print("\n--- EDA: Sales Analysis ---")
print("Total Sales:", df["Sales"].sum())
print("Average Sales:", df["Sales"].mean())
print("Minimum Sales:", df["Sales"].min())
print("Maximum Sales:", df["Sales"].max())
print("\n--- EDA: Product-wise Sales ---")
product_sales = df.groupby("Product")["Sales"].sum().sort_values(ascending=False)
print(product_sales)
print("\n--- EDA: Country-wise Sales ---")
country_sales = df.groupby("Country")["Sales"].sum().sort_values(ascending=False)
print(country_sales)
print("\n--- EDA: Segment-wise Sales ---")
segment_sales = df.groupby("Segment")["Sales"].sum().sort_values(ascending=False)
print(segment_sales)
print("\n--- EDA: Year-wise Sales ---")
year_sales = df.groupby("Year")["Sales"].sum()
print(year_sales)
import matplotlib.pyplot as plt

plt.plot(year_sales.index, year_sales.values, marker="o")
plt.xlabel("Year")
plt.ylabel("Total Sales")
plt.title("Year-wise Sales Trend")
plt.show()
plt.bar(product_sales.index, product_sales.values)
plt.xlabel("Product")
plt.ylabel("Total Sales")
plt.title("Product-wise Sales")
plt.xticks(rotation=45)
plt.show()
print("\n--- EDA: Sales vs Profit Correlation ---")
print(df[["Sales", "Profit"]].corr())
