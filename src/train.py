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


# Create target column
def get_result(row):
    if row['home_score'] > row['away_score']:
        return 'Home Win'
    elif row['home_score'] < row['away_score']:
        return 'Away Win'
    else:
        return 'Draw'

df['result'] = df.apply(get_result, axis=1)

# Display results
print("\nRESULT COLUMN:")
print(df[['home_team', 'away_team', 'home_score', 'away_score', 'result']].head())

# Result distribution
print("\nRESULT DISTRIBUTION:")
print(df['result'].value_counts())

from sklearn.preprocessing import LabelEncoder

# Initialize encoders
home_encoder = LabelEncoder()
away_encoder = LabelEncoder()
tournament_encoder = LabelEncoder()

# Encode categorical columns
df['home_team_encoded'] = home_encoder.fit_transform(df['home_team'])

df['away_team_encoded'] = away_encoder.fit_transform(df['away_team'])

df['tournament_encoded'] = tournament_encoder.fit_transform(df['tournament'])

# Display encoded columns
print("\nENCODED FEATURES:")
print(df[['home_team',
          'home_team_encoded',
          'away_team',
          'away_team_encoded']].head())

from sklearn.model_selection import train_test_split

# Select features
X = df[['home_team_encoded',
        'away_team_encoded',
        'tournament_encoded',
        'neutral']]

# Target column
y = df['result']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTRAIN TEST SPLIT COMPLETE")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

from sklearn.ensemble import RandomForestClassifier

# Initialize model
model = RandomForestClassifier()

# Train model
model.fit(X_train, y_train)

print("\nMODEL TRAINED SUCCESSFULLY")

# Make predictions
predictions = model.predict(X_test)

print("\nPREDICTIONS:")
print(predictions[:10])

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)

print("\nMODEL ACCURACY:")
print(accuracy)

import joblib

# Save trained model
joblib.dump(model, 'models/fifa_model.pkl')

print("\nMODEL SAVED SUCCESSFULLY")
