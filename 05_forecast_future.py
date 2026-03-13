"""
Step 5: 用全部歷史資料預測未來 30 / 60 / 90 個交易日
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.tseries.offsets import BDay
import timesfm
from timesfm import ForecastConfig

DATA_FILE = "data/taiex_3y.csv"
OUTPUT_FILE = "data/future_forecast.csv"
PLOT_FILE = "data/future_forecast_plot.png"

HORIZONS = [30, 60, 90]
MAX_CONTEXT = 512

plt.rcParams["font.family"] = ["Microsoft JhengHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HORIZON_COLORS = {30: "#E53935", 60: "#FB8C00", 90: "#8E24AA"}


def load_data():
    df = pd.read_csv(DATA_FILE, index_col="Date", parse_dates=True)
    return df["Close"].values.astype(np.float32), df.index


def build_model():
    print("載入 TimesFM 2.5 模型 ...")
    tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch",
        torch_compile=False,
    )
    tfm.compile(ForecastConfig(
        max_context=MAX_CONTEXT,
        max_horizon=max(HORIZONS),
        per_core_batch_size=1,
        normalize_inputs=True,
    ))
    print("模型載入完成")
    return tfm


def generate_future_dates(last_date, n_days):
    """產生 n_days 個交易日（跳過週末）"""
    return pd.date_range(start=last_date + BDay(1), periods=n_days, freq=BDay())


def run_future_forecast(tfm, series, dates):
    last_date = dates[-1]
    print(f"\n以 {last_date.date()} 為起點，預測未來交易日 ...")

    results = []
    for horizon in HORIZONS:
        print(f"預測 horizon = {horizon} 天 ...")
        point_forecast, _ = tfm.forecast(
            horizon=horizon,
            inputs=[series],
        )
        pred = point_forecast[0, :horizon]
        future_dates = generate_future_dates(last_date, horizon)

        for i, (d, v) in enumerate(zip(future_dates, pred)):
            results.append({
                "horizon": horizon,
                "step": i + 1,
                "date": d.date(),
                "predicted": float(v),
            })

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n未來預測已儲存至 {OUTPUT_FILE}")
    return df


def plot_future(df_future, series, dates):
    # 顯示最近 90 天歷史 + 預測
    context_window = 90
    ctx_dates = dates[-context_window:]
    ctx_values = series[-context_window:]

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("TimesFM 2.5 — 台灣加權指數未來預測 (^TWII)", fontsize=15, fontweight="bold")

    # 歷史資料
    ax.plot(ctx_dates, ctx_values, color="#2196F3", linewidth=2, label="歷史資料（近90天）")
    ax.axvline(x=dates[-1], color="gray", linestyle=":", linewidth=1)
    ax.text(dates[-1], ctx_values.min(), "  預測起點", color="gray", fontsize=9)

    # 90 天完整預測線
    s90 = df_future[df_future["horizon"] == 90]
    pred_dates_90 = pd.to_datetime(s90["date"])
    pred_values_90 = s90["predicted"].values
    ax.plot(pred_dates_90, pred_values_90,
            color="#9E9E9E", linewidth=1.5, linestyle="--", label="_nolegend_")

    # 各 horizon 端點標記
    for horizon in HORIZONS:
        subset = df_future[df_future["horizon"] == horizon]
        last = subset.iloc[-1]
        d = pd.to_datetime(last["date"])
        v = last["predicted"]
        ax.axvline(x=d, color=HORIZON_COLORS[horizon], linestyle=":", linewidth=1, alpha=0.6)
        ax.scatter([d], [v], color=HORIZON_COLORS[horizon], s=60, zorder=5)
        ax.annotate(f"{horizon}天\n{v:,.0f}",
                    xy=(d, v), xytext=(6, 6), textcoords="offset points",
                    color=HORIZON_COLORS[horizon], fontsize=9, fontweight="bold")

    # 補 legend patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#2196F3", linewidth=2, label="歷史資料（近90天）"),
        Line2D([0], [0], color="#9E9E9E", linewidth=1.5, linestyle="--", label="預測曲線（90天）"),
    ] + [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=HORIZON_COLORS[h],
               markersize=8, label=f"{h} 天端點")
        for h in HORIZONS
    ]
    ax.set_ylabel("指數點位")
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
    print(f"圖表已儲存至 {PLOT_FILE}")
    plt.show()


if __name__ == "__main__":
    series, dates = load_data()
    tfm = build_model()
    df_future = run_future_forecast(tfm, series, dates)

    print("\n未來預測摘要：")
    for horizon in HORIZONS:
        subset = df_future[df_future["horizon"] == horizon]
        last = subset.iloc[-1]
        print(f"  {horizon:2d} 天後 ({last['date']})：預測 {last['predicted']:,.0f} 點")

    plot_future(df_future, series, dates)
