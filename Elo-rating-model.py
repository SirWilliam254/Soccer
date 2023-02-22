import pandas as pd

# Load the data
data = pd.read_csv('football_data.csv')

# Calculate the Elo ratings for each team
elo_ratings = {}

for i, row in data.iterrows():
    home_team = row['home_team']
    away_team = row['away_team']
    home_score = row['goals_home']
    away_score = row['goals_away']
    
    if home_team not in elo_ratings:
        elo_ratings[home_team] = 1500
    if away_team not in elo_ratings:
        elo_ratings[away_team] = 1500
        
    home_expected = 1 / (1 + 10**((elo_ratings[away_team] - elo_ratings[home_team]) / 400))
    away_expected = 1 - home_expected
    
    elo_ratings[home_team] += 32 * (home_score - home_expected)
    elo_ratings[away_team] += 32 * (away_score - away_expected)

# Use the Elo ratings to predict the outcome of a match
home_team = 'Manchester United'
away_team = 'Liverpool'

home_rating = elo_ratings[home_team]
away_rating = elo_ratings[away_team]

home_win_prob = 1 / (1 + 10**((away_rating - home_rating) / 400))
away_win_prob = 1 - home_win_prob
