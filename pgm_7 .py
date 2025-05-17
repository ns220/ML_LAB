import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# Load dataset
data = pd.read_csv(r"C:\Users\dell\Downloads\auto-mpg.csv")

# Convert 'horsepower' to numeric (handle '?')
data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')

# Data overview
print(data.info())
print(data.describe().T)

# Data cleaning
print("Missing values:\n", data.isnull().sum())
print("Duplicate rows:", data.duplicated().sum())

# Fill missing values in 'horsepower'
df = data.copy()
df['mpg'].fillna(df['mpg'].mean(), inplace=True)
df['cylinders'].fillna(df['cylinders'].mean(), inplace=True)
df['displacement'].fillna(df['displacement'].mode()[0], inplace=True)
df['horsepower'].fillna(df['horsepower'].mean(), inplace=True)
df['weight'].fillna(df['weight'].median(), inplace=True)
df['acceleration'].fillna(df['acceleration'].median(), inplace=True)

# Drop non-numeric or categorical columns for correlation analysis
if 'origin' in df.columns and 'name' in df.columns:
    corr = df.drop(columns=['origin', 'name']).corr()
else:
    corr = df.select_dtypes(include=[np.number]).corr()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# Features and target
X = df[['horsepower']]
y = df['mpg']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Polynomial features (degree = 2)
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

# Model training
model = LinearRegression()
model.fit(X_poly_train, y_train)
y_pred = model.predict(X_poly_test)

# Plot test set predictions
plt.figure(figsize=(8,6))
plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.scatter(X_test, y_pred, color='red', label='Predicted Data', alpha=0.7)
plt.title('Polynomial Regression (degree=2): Horsepower vs MPG')
plt.xlabel('Horsepower')
plt.ylabel('Miles Per Gallon (MPG)')
plt.legend()
plt.grid(True)
plt.show()

# Smooth polynomial curve
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_plot_poly = poly.transform(X_plot)
y_plot = model.predict(X_plot_poly)

plt.figure(figsize=(8,6))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X_plot, y_plot, color='red', linewidth=2, label='Polynomial Regression Curve')
plt.title('Polynomial Regression: Horsepower vs MPG')
plt.xlabel('Horsepower')
plt.ylabel('Miles Per Gallon (MPG)')
plt.legend()
plt.grid(True)
plt.show()

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.2f}")
