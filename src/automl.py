from pycaret.classification import *

import pandas as pd

# Load dataset
df = pd.read_csv('data/results.csv')

# Create target variable
def get_result(row):
    if row['home_score'] > row['away_score']:
        return 'Home Win'
    elif row['home_score'] < row['away_score']:
        return 'Away Win'
    else:
        return 'Draw'

df['result'] = df.apply(get_result, axis=1)

# Select features
df = df[['home_team',
         'away_team',
         'tournament',
         'neutral',
         'result']]

# Setup AutoML
setup(
    data=df,
    target='result',
    session_id=42
)

# Compare models automatically
best_model = compare_models()

print(best_model)
