"""
Step 4: 繪圖 — 實際值 vs TimesFM 預測值（三種 horizon）
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

RESULTS_FILE = "data/forecast_results.csv"
DATA_FILE = "data/taiex_3y.csv"
SPLIT_RATIO = 0.9

plt.rcParams["font.family"] = ["Microsoft JhengHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_forecasts():
    df_data = pd.read_csv(DATA_FILE, index_col="Date", parse_dates=True)
    df_results = pd.read_csv(RESULTS_FILE, parse_dates=["date"])
    series = df_data["Close"].values
    dates = df_data.index
    split_idx = int(len(series) * SPLIT_RATIO)

    horizons = sorted(df_results["horizon"].unique())
    fig, axes = plt.subplots(len(horizons), 1, figsize=(14, 5 * len(horizons)), sharex=False)
    if len(horizons) == 1:
        axes = [axes]

    fig.suptitle("TimesFM 2.5 — 台灣加權指數預測 (^TWII)", fontsize=16, fontweight="bold", y=1.01)

    # 顯示最近 120 天 context + 預測區段
    context_window = 120

    for ax, horizon in zip(axes, horizons):
        subset = df_results[df_results["horizon"] == horizon].copy()
        n = len(subset)

        ctx_start = max(0, split_idx - context_window)
        ctx_dates = dates[ctx_start:split_idx]
        ctx_values = series[ctx_start:split_idx]

        pred_dates = subset["date"].values
        pred_values = subset["predicted"].values
        actual_values = subset["actual"].values

        # Context
        ax.plot(ctx_dates, ctx_values, color="#2196F3", linewidth=1.5, label="歷史資料（Context）")
        # 實際值（hold-out）
        ax.plot(pred_dates, actual_values, color="#4CAF50", linewidth=1.5, linestyle="--", label="實際值（Hold-out）")
        # 預測值
        ax.plot(pred_dates, pred_values, color="#F44336", linewidth=1.5, label=f"TimesFM 預測（{horizon}天）")

        # 切割線
        ax.axvline(x=dates[split_idx], color="gray", linestyle=":", linewidth=1, alpha=0.7)
        ax.text(dates[split_idx], ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else series.min(),
                " 預測起點", color="gray", fontsize=9, va="bottom")

        mape_val = np.mean(np.abs((actual_values - pred_values) / actual_values)) * 100
        mae_val = np.mean(np.abs(actual_values - pred_values))
        ax.set_title(f"Horizon = {horizon} 天  |  MAPE = {mape_val:.2f}%  |  MAE = {mae_val:.0f} 點", fontsize=12)
        ax.set_ylabel("指數點位")
        ax.legend(loc="upper left", fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = "data/forecast_plot.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"圖表已儲存至 {output_path}")
    plt.show()


if __name__ == "__main__":
    plot_forecasts()
