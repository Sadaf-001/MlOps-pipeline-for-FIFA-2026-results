import joblib
import pandas as pd

# Load trained model
model = joblib.load('models/fifa_model.pkl')

# Example match input
sample_match = pd.DataFrame({
    'home_team_encoded': [10],
    'away_team_encoded': [25],
    'tournament_encoded': [3],
    'neutral': [False]
})

# Predict result
prediction = model.predict(sample_match)

print("MATCH PREDICTION:")
print(prediction)
