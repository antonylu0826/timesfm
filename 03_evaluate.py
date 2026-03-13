"""
Step 3: 計算 MAE / MAPE / RMSE，並與 Naive Baseline 比較
"""

import numpy as np
import pandas as pd

RESULTS_FILE = "data/forecast_results.csv"
DATA_FILE = "data/taiex_3y.csv"
SPLIT_RATIO = 0.9


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def naive_forecast(series, split_idx, horizon):
    """Naive baseline：用 context 最後一個值重複預測"""
    last_value = series[split_idx - 1]
    return np.full(horizon, last_value)


def evaluate():
    df_results = pd.read_csv(RESULTS_FILE)
    df_data = pd.read_csv(DATA_FILE, index_col="Date", parse_dates=True)
    series = df_data["Close"].values.astype(np.float32)
    split_idx = int(len(series) * SPLIT_RATIO)

    print("=" * 60)
    print(f"{'Horizon':>8} | {'Model':>6} | {'MAE':>8} | {'MAPE':>7} | {'RMSE':>8}")
    print("-" * 60)

    for horizon in sorted(df_results["horizon"].unique()):
        subset = df_results[df_results["horizon"] == horizon].copy()
        n = len(subset)

        y_true = subset["actual"].values
        y_pred = subset["predicted"].values
        y_naive = naive_forecast(series, split_idx, n)

        model_mae  = mae(y_true, y_pred)
        model_mape = mape(y_true, y_pred)
        model_rmse = rmse(y_true, y_pred)

        naive_mae  = mae(y_true, y_naive)
        naive_mape = mape(y_true, y_naive)
        naive_rmse = rmse(y_true, y_naive)

        print(f"{horizon:>7}d | {'TimesFM':>6} | {model_mae:>8.1f} | {model_mape:>6.2f}% | {model_rmse:>8.1f}")
        print(f"{'':>8} | {'Naive':>6} | {naive_mae:>8.1f} | {naive_mape:>6.2f}% | {naive_rmse:>8.1f}")
        print("-" * 60)

    print("=" * 60)


if __name__ == "__main__":
    evaluate()
