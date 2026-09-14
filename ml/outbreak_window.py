import pandas as pd

def evaluate_outbreak_windows(df, train_model, windows=(5, 10, 15, 20)):
    results = []

    for days in windows:
        _, r2, mse, _, _ = train_model(df, days)

        results.append({
            "Outbreak_Window": days,
            "R2": r2,
            "MSE": mse
        })

    return pd.DataFrame(results)