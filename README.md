# TimesFM 台灣加權指數預測

用 Google [TimesFM 2.5](https://huggingface.co/google/timesfm-2.5-200m-pytorch) 對台灣加權指數（TAIEX, `^TWII`）近三年日線資料做時序預測與評估。

## 環境需求

- Python **3.12**（PyPI 版 timesfm 不支援 3.13+，需從 GitHub 安裝）
- CPU 推論即可（GPU 2GB VRAM 不足）

## 安裝

```bash
# 建立虛擬環境
py -3.12 -m venv .venv
.venv\Scripts\activate

# 安裝依賴
pip install yfinance pandas numpy matplotlib seaborn scikit-learn huggingface_hub
pip install "timesfm[torch] @ git+https://github.com/google-research/timesfm.git"
```

> 模型 `google/timesfm-2.5-200m-pytorch` 為 gated repo，需先至 HuggingFace 接受使用條款，再執行 `hf auth login`。

## 執行順序

```bash
python 01_fetch_data.py        # 下載台股近三年資料
python 02_run_forecast.py      # 回測預測（前 90% 訓練，後 10% 測試）
python 03_evaluate.py          # 計算 MAE / MAPE / RMSE
python 04_visualize.py         # 繪製回測結果圖
python 05_forecast_future.py   # 預測未來 30 / 60 / 90 個交易日
```

## 資料切割

| 用途 | 範圍 |
|------|------|
| Context window | 前 90%（約 700 個交易日） |
| Hold-out test | 後 10%（約 80 個交易日） |

## 輸出

- `data/forecast_results.csv` — 回測預測結果
- `data/forecast_plot.png` — 回測結果圖
- `data/future_forecast.csv` — 未來預測結果
- `data/future_forecast_plot.png` — 未來預測圖
