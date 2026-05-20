import pandas as pd

# Load dataset
df = pd.read_csv('data/results.csv')

# Display first rows
print(df.head())

# Dataset shape
print("Dataset shape:", df.shape)

# Column names
print("Columns:")
print(df.columns)
