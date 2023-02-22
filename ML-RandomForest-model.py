import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the data
data = pd.read_csv('football_data.csv')

# Train a random forest classifier on the data
X = data.drop(['result'], axis=1)
y = data['result']

rf_model = RandomForestClassifier().fit(X, y)

# Use the model to predict the outcome of a match
home_team = 'Manchester United'
away_team = 'Liverpool'
home_shots_on_target = 7
away_shots_on_target = 5
home_possession = 0.6
away_possession = 0.4

prediction = rf_model.predict([[home_team, away_team, home_shots_on_target, away_shots_on_target, home_possession, away_possession]])
