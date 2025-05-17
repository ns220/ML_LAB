import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Load California Housing dataset
data = fetch_california_housing()

# Convert to DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target

# Variable meaning mapping
variable_meaning = {
    "MedInc": "Median income in block group",
    "HouseAge": "Median house age in block group",
    "AveRooms": "Average number of rooms per household",
    "AveBedrms": "Average number of bedrooms per household",
    "Population": "Population of block group",
    "AveOccup": "Average number of household members",
    "Latitude": "Latitude of block group",
    "Longitude": "Longitude of block group",
    "Target": "Median house value (in $100,000s)"
}

# Create variable meaning table
variable_df = pd.DataFrame(list(variable_meaning.items()), columns=["Feature", "Description"])
print("\nVariable Meaning Table:")
print(variable_df)

# Basic dataset info
print("\nBasic Information about Dataset:")
print(df.info())

print("\nFirst Five Rows of Dataset:")
print(df.head())

# Summary Statistics
print("\nSummary Statistics:")
print(df.describe())

summary_explanation = """
The summary statistics table provides key percentiles and other descriptive metrics:
- 25% (First Quartile - Q1): This represents the value below which 25% of the data falls.
- 50% (Median - Q2): This is the middle value when the data is sorted. It provides a central tendency measure.
- 75% (Third Quartile - Q3): This represents the value below which 75% of the data falls.
These percentiles are useful for detecting skewness, data distribution, and identifying potential outliers.
"""
print("\nSummary Statistics Explanation:")
print(summary_explanation)

# Missing Values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# Histogram plots
plt.figure(figsize=(12, 8))
df.hist(figsize=(12, 8), bins=30, edgecolor='black')
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()

# Boxplots for outlier detection
plt.figure(figsize=(12, 6))
sns.boxplot(data=df)
plt.xticks(rotation=45)
plt.title("Boxplots of Features to Identify Outliers")
plt.show()

# Correlation Matrix
plt.figure(figsize=(10, 6))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()

# Pairplot (subset)
sns.pairplot(df[['MedInc', 'HouseAge', 'AveRooms', 'Target']], diag_kind='kde')
plt.show()
