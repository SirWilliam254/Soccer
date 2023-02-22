import pandas as pd
import statsmodels.api as sm

# Load the data
data = pd.read_csv('football_data.csv')

# Create the Poisson regression model
poisson_model = sm.GLM(data[['goals_home', 'goals_away']], data[['home_team', 'away_team']],
                      family=sm.families.Poisson()).fit()

# Use the model to predict the expected number of goals for each team
home_goals = poisson_model.predict(exog=data[['home_team', 'away_team']])
away_goals = poisson_model.predict(exog=data[['away_team', 'home_team']])

