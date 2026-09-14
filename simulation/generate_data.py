import numpy as np
import pandas as pd
from models.sir_model import run_simulation, sir_model
# -----
# Generate beta, gamma, R0, I0
# -----

def generate_outbreaks():
    number_of_outbreaks = 1500
    rng = np.random.default_rng(seed=7)

    R0_values = rng.uniform(0.5, 5.5, number_of_outbreaks)

    gamma_values = rng.uniform(1/10, 1/3,  number_of_outbreaks)

    beta_values = R0_values * gamma_values

    I0_values = rng.uniform(1e-6, 1e-4, number_of_outbreaks)

    # -----
    # Initial Values
    # -----

    outbreak_parameters = zip(beta_values, gamma_values, R0_values, I0_values)

    all_results = []
    outbreak_id = 0

    for beta, gamma, R0, I0 in outbreak_parameters:
        S0 = 1.0 - I0
        initial_recovered = 0.0

        y0 = np.array([S0, I0, initial_recovered])


        mini_df = run_simulation(
    lambda t, y: sir_model(t, y, beta, gamma), 0, y0, 0.1, 20)
        #mini_df = add_observation_noise(mini_df, N = 1_000_000, r=10, rng = rng)
        mini_df["Outbreak_ID"] = outbreak_id
        mini_df["beta"] = beta
        mini_df["gamma"] = gamma
        mini_df["R0"] = R0



        all_results.append(mini_df)

        outbreak_id += 1

    long_df = pd.concat(all_results, ignore_index=True)
    return long_df, beta_values, gamma_values, R0_values

