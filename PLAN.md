# TimesFM 2.5 台灣股市指數預測專案計劃

## 專案目標
用 Google TimesFM 2.5 對台灣加權指數（TAIEX, `^TWII`）近三年日線資料做時序預測與評估。

## 專案架構

```
timesfm/
├── PLAN.md                   # 本計劃文件
├── requirements.txt          # 依賴套件
├── 01_fetch_data.py          # 抓取台股近3年資料 (yfinance)
├── 02_run_forecast.py        # 載入 TimesFM 2.5，做預測
├── 03_evaluate.py            # MAE / MAPE / RMSE 評估
├── 04_visualize.py           # 繪圖：實際 vs 預測
└── data/
    └── taiex_3y.csv          # 存放下載的指數資料
```

## 實作流程

| 步驟 | 檔案 | 說明 |
|------|------|------|
| 1 | `01_fetch_data.py` | yfinance 抓 `^TWII`，時間範圍 2023-01-01 ~ 2026-03-13，約 780 個交易日，存為 CSV |
| 2 | `02_run_forecast.py` | 載入 `timesfm-2-0-500m`（HuggingFace），zero-shot 推論，預測 30 / 60 / 90 天 horizon |
| 3 | `03_evaluate.py` | 計算 MAE、MAPE、RMSE，與 naive baseline（上一值）比較 |
| 4 | `04_visualize.py` | matplotlib 繪出完整曲線 + 預測結果 |

## 資料切割策略

- **Context window**：前 90%（約 700 個交易日）
- **Hold-out test**：後 10%（約 80 個交易日）
- 預測 horizon：30 / 60 / 90 天

## 主要依賴套件

```
timesfm[torch]      # Google TimesFM（PyTorch 版）
yfinance            # 台股資料來源
pandas
numpy
matplotlib
seaborn
scikit-learn        # 評估指標
huggingface_hub     # 模型下載
```

## 待確認事項

- [ ] 執行環境：JAX 或 PyTorch（影響 timesfm 安裝參數）
- [ ] GPU 是否可用（影響推論速度）
- [ ] 預測 horizon 是否需要調整

## 執行順序

```bash
pip install -r requirements.txt
python 01_fetch_data.py
python 02_run_forecast.py
python 03_evaluate.py
python 04_visualize.py
```
