import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv(r'C:\Users\vijay\Desktop\Machine Learning Course Batches\FDP_ML_6th\your_filename.csv')

# Display the first 5 rows
print("First 5 rows:")
print(df.head())

# Display data info
print("\nDataset Info:")
print(df.info())

# Unique values
print("\nNumber of unique values per column:")
print(df.nunique())

# Missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Duplicate rows
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

# Fill missing values in 'total_bedrooms' with median
if 'total_bedrooms' in df.columns:
    median_value = df['total_bedrooms'].median()
    df['total_bedrooms'].fillna(median_value, inplace=True)
    print("\nMissing values in 'total_bedrooms' filled with median:", median_value)

# Convert selected columns to int
for col in df.columns[2:7]:
    try:
        df[col] = df[col].astype('int')
        print(f"Converted {col} to integer type.")
    except:
        print(f"Could not convert {col} to integer.")

# Show updated DataFrame head and summary
print("\nUpdated DataFrame:")
print(df.head())

print("\nDescriptive Statistics:")
print(df.describe().T)

# Numerical columns
Numerical = df.select_dtypes(include=[np.number]).columns
print("\nNumerical Columns:")
print(Numerical)

# Plot histograms
for col in Numerical:
    plt.figure(figsize=(10, 6))
    df[col].plot(kind='hist', title=col, bins=60, edgecolor='black')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# Plot boxplots
for col in Numerical:
    plt.figure(figsize=(6, 6))
    sns.boxplot(x=df[col], color='blue')
    plt.title(col)
    plt.xlabel(col)
    plt.show()
