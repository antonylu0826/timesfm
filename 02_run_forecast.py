"""
Step 2: 載入 TimesFM 2.5 (timesfm-2.5-200m-pytorch)，對 TAIEX 做 zero-shot 預測
輸出預測結果至 data/forecast_results.csv
"""

import numpy as np
import pandas as pd
import timesfm
from timesfm import ForecastConfig

DATA_FILE = "data/taiex_3y.csv"
OUTPUT_FILE = "data/forecast_results.csv"

# 預測 horizon（交易日）
HORIZONS = [30, 60, 90]
# 資料切割：前 90% context，後 10% hold-out
SPLIT_RATIO = 0.9
# context 長度（取最近 512 天，需是 patch_size=32 的倍數）
MAX_CONTEXT = 512


def load_data():
    df = pd.read_csv(DATA_FILE, index_col="Date", parse_dates=True)
    series = df["Close"].values.astype(np.float32)
    dates = df.index
    return series, dates


def build_model():
    print("載入 TimesFM 2.5 模型 (google/timesfm-2.5-200m-pytorch) ...")
    tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch",
        torch_compile=False,  # CPU 上 compile 較慢，先關閉
    )
    tfm.compile(ForecastConfig(
        max_context=MAX_CONTEXT,
        max_horizon=max(HORIZONS),
        per_core_batch_size=1,
        normalize_inputs=True,
    ))
    print("模型載入完成")
    return tfm


def run_forecast(tfm, series, dates):
    split_idx = int(len(series) * SPLIT_RATIO)
    context = series[:split_idx]
    actual = series[split_idx:]
    context_dates = dates[:split_idx]
    forecast_dates = dates[split_idx:]

    print(f"\n資料總長度：{len(series)} 天")
    print(f"Context：{len(context)} 天 ({dates[0].date()} ~ {context_dates[-1].date()})")
    print(f"Hold-out：{len(actual)} 天 ({forecast_dates[0].date()} ~ {dates[-1].date()})")

    results = []

    for horizon in HORIZONS:
        print(f"\n預測 horizon = {horizon} 天 ...")
        point_forecast, _ = tfm.forecast(
            horizon=horizon,
            inputs=[context],
        )
        # point_forecast shape: (1, horizon)
        pred = point_forecast[0, :horizon]
        actual_h = actual[:horizon]
        n = min(len(pred), len(actual_h))

        for i in range(n):
            results.append({
                "horizon": horizon,
                "step": i + 1,
                "date": forecast_dates[i].date() if i < len(forecast_dates) else None,
                "actual": float(actual_h[i]),
                "predicted": float(pred[i]),
            })

    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_FILE, index=False)
    print(f"\n預測結果已儲存至 {OUTPUT_FILE}")
    return df_results


if __name__ == "__main__":
    series, dates = load_data()
    tfm = build_model()
    df_results = run_forecast(tfm, series, dates)
    print("\n預測結果預覽：")
    print(df_results.groupby("horizon").head(3).to_string(index=False))
