import pandas as pd
import pomegranate as pg

# Load the data
data = pd.read_csv('football_data.csv')

# Define the Bayesian network structure
model = pg.BayesianNetwork.from_structure(data, state_names=data.columns.tolist())

# Train the model on the data
model.fit(data)

# Use the model to predict the outcome of a match
home_team = 'Manchester United'
away_team = 'Liverpool'
home_shots_on_target = 7
away_shots_on_target = 5
home_possession = 0.6
away_possession = 0.4

prediction = model.predict_proba([[home_team, away_team, home_shots_on_target, away
