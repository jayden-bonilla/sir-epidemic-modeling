"""
Reproduces the COVID-19 R0 estimate cited in my PhD applications.

Trains both models on simulated outbreak data, then applies the
random forest (the stronger performer) to real US COVID-19 case
data to estimate R0.
"""

from simulation.generate_data import generate_outbreaks
from simulation.preprocess import build_ml_dataset
from ml.linear_regression import train_linear_regression
from ml.random_forest import train_randomforest_regression
from analysis.covid import usa_long

days = 20
feature_cols = list(range(1, days + 1))

# ----- Train on simulated outbreaks -----
long_df, beta_values, gamma_values, R0_values = generate_outbreaks()
ml_df = build_ml_dataset(long_df, beta_values, gamma_values, R0_values)

rf_model, rf_r2, rf_mse, _, _ = train_randomforest_regression(ml_df, days)

# ----- Apply to real COVID-19 data -----
X_covid = usa_long[feature_cols]
predicted_r0 = rf_model.predict(X_covid)[0]

print(f"\nPredicted R0 for US COVID-19 outbreak (Random Forest): {predicted_r0:.2f}")