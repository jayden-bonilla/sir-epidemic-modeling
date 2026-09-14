import numpy as np
import pandas as pd

# -----
# Numerical Solver
# -----

def rk4_step(ode_function, t, y, h):
    """Advance the system one RK4 step. Returns (t_next, y_next)."""
    slope1 = h * ode_function(t, y)
    slope2 = h * ode_function(t + h/2, y + slope1/2)
    slope3 = h * ode_function(t + h/2, y + slope2/2)
    slope4 = h * ode_function(t + h, y + slope3)

    y_next = y + (slope1 + 2*slope2 + 2*slope3 + slope4) / 6
    t_next = round(t + h, 10)  # guards against float drift throwing off the day-detection below

    return t_next, y_next


def run_simulation(ode_function, t0, y0, h, t_max):
    """
    Run RK4 from t0 to t_max, recording S/I/R at each whole day.
    Assumes h divides evenly into 1 (whole-day) units.
    """
    data = {
        "Day": [],
        "Susceptible": [],
        "Infected": [],
        "Recovered": []
    }

    t, y = t0, y0
    while t < t_max + 0.1:  # small pad so the final day still gets recorded
        if np.isclose(t % 1, 0, atol=1e-8):
            data["Day"].append(t)
            data["Susceptible"].append(y[0])
            data["Infected"].append(y[1])
            data["Recovered"].append(y[2])

        t, y = rk4_step(ode_function, t, y, h)

    df = pd.DataFrame(data)
    df["Cumulative_Infected"] = 1 - df["Susceptible"]
    return df

# -----
# Compartmental Model
# -----

def sir_model(t, y, beta, gamma):
    S, I, R = y

    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return np.array([dSdt, dIdt, dRdt])