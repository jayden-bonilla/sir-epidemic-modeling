import pandas as pd
import numpy as np

# -----
# Machine Learning Prep
# -----

def build_ml_dataset(long_df, beta_values, gamma_values, R0_values):

    # Convert vertical df to horizontal df to be usable for ML
    ml_df = long_df.pivot(index="Outbreak_ID", columns="Day", values="Cumulative_Infected")

    # Select raw days 0-19 to use as 20 observation days, relabeled 1-20 below
    ml_df = ml_df[np.arange(0, 20, 1.0)]
    ml_df['beta'] = beta_values
    ml_df['gamma'] = gamma_values
    ml_df['R0'] = R0_values

    # Rename raw day 0-19 columns to observation days 1-20
    ml_df = ml_df.rename(
        columns={day: int(day) + 1 for day in range(20)}
    )
    return ml_df