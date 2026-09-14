# SIR Epidemic Modeling

Estimating the basic reproduction number (R0) of an epidemic from early
outbreak data, using a simulated-data machine learning approach validated
against real US COVID-19 case data.

## Motivation

Early in an outbreak, knowing R0 (how many people each infected person
goes on to infect) matters for public health response — but it's hard to
estimate directly from limited early data. This project trains machine
learning models on thousands of simulated outbreaks to learn the
relationship between early case trajectories and R0, then tests whether
that relationship holds up against a real epidemic.

## Approach

1. **Simulate outbreaks** (`simulation/generate_data.py`, `models/sir_model.py`) —
   a deterministic SIR (Susceptible-Infected-Recovered) compartmental model,
   integrated with 4th-order Runge-Kutta. 1,500 synthetic outbreaks are
   generated with randomly sampled parameters (R0 ~ Uniform(0.5, 5.5),
   recovery rate γ ~ Uniform(1/10, 1/3), initial infected proportion I0 ~
   Uniform(1e-6, 1e-4)).

2. **Build a training set** (`simulation/preprocess.py`) — each simulated
   outbreak's cumulative-infected trajectory over its first 20 days becomes
   one row of training data, labeled with the true R0 used to generate it.

3. **Train models** (`ml/linear_regression.py`, `ml/random_forest.py`) — a
   linear regression and a random forest are each trained to predict R0
   from the 20-day trajectory. Random forest considerably outperforms
   linear regression, especially with fewer observed days.

4. **Validate against reality** (`analysis/covid.py`) — the same 20-day
   window is extracted from real US COVID-19 case data (standardized to
   proportion of population infected) and fed to the trained random forest
   to produce a real-world R0 estimate.

## Result

The model estimates R0 ≈ 1.95 for the US COVID-19 outbreak, compared to
published estimates of roughly 3-5. The gap suggests the model
underestimates R0 in this range — plausibly because it's trained on
idealized simulations without observation noise or reporting delays, both
of which are present in real surveillance data. Run `reproduce_covid_estimate.py`
to regenerate this result end to end.

## Status

This reflects the project as of September 2026. I'm currently extending it
toward a stochastic SIR model (adding observation noise to the simulations)
and possibly an SEIR model, as part of my undergraduate honors thesis — the
public-facing pipeline here is a frozen snapshot rather than the actively
evolving version.

## Running this

Scripts assume they're run from this folder (the repo root), since data
paths are relative (e.g., `data/covid.csv`). To reproduce the headline
result:

\
python3 reproduce_covid_estimate.py
\

## Repository structure

- `models/` — the SIR ODE and RK4 numerical solver
- `simulation/` — synthetic outbreak generation and ML dataset preparation
- `ml/` — model training (linear regression, random forest)
- `analysis/` — real-world COVID-19 data preparation
- `data/` — US COVID-19 case data (subset of a larger public dataset)
- `reproduce_covid_estimate.py` — trains the models and reproduces the R0 estimate above
