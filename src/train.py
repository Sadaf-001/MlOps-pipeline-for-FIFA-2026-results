import pandas as pd

# Load dataset
df = pd.read_csv('data/results.csv')

# Display first rows
print("FIRST 5 ROWS:")
print(df.head())

# Dataset shape
print("\nDATASET SHAPE:")
print(df.shape)

# Column names
print("\nCOLUMN NAMES:")
print(df.columns)

# Missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# Dataset information
print("\nDATASET INFO:")
print(df.info())
