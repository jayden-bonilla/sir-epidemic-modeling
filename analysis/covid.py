import pandas as pd
import numpy as np

#COVID Dataset External Validation 
covid = pd.read_csv(r'data/covid.csv')
usa = covid[covid['location_key']=='US']
usa = usa.drop(labels = ['new_tested', 'cumulative_tested', 'cumulative_recovered', 'new_recovered'], axis=1)
usa['infections'] = usa['cumulative_confirmed'] - usa['cumulative_deceased']


# Standardize # of cases to proportion of population infected 
N = 345_000_000
usa['standardized_infections'] = usa['infections'] / N

# Filter to include days after threshold 
usa = usa[usa['standardized_infections'] > 1e-5]
usa['Day'] = np.arange(1, len(usa)+1)

# Prepare COVID data to feed into ML
usa_first20 = usa.iloc[0:20]
usa_long = usa_first20.pivot(index = 'location_key', columns = 'Day', values = 'standardized_infections')


