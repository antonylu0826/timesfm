# 環境設定紀錄

## 系統環境
- OS: Windows 11
- GPU: NVIDIA 2GB VRAM（不足，改用 CPU 推論）
- Python: 3.12（需 3.12+ 才能跑 timesfm 2.5 新版套件）

## 踩過的坑

| 問題 | 原因 | 解法 |
|------|------|------|
| `timesfm[torch]` 裝到 1.0.0（JAX 版） | 系統 Python 3.13，最新相容版只到 1.0.0 | 改用 Python 3.12 |
| `ModuleNotFoundError` | venv 未啟動，裝到全域 | 確認 `(.venv)` 前綴再裝 |
| `401 Unauthorized` | HuggingFace 未登入或未接受 gated repo 條款 | `hf auth login` |
| `FileNotFoundError: torch_model.ckpt` | timesfm 1.3.0 hardcode 舊格式，2.5 模型改用 safetensors | 需要新版 timesfm（Python 3.12） |
| `RuntimeError: Missing key(s) in state_dict` | timesfm 1.3.0 架構與 2.5 模型 weights 不相容 | 需要新版 timesfm（Python 3.12） |
| API 變更（`TimesFmHparams` 等找不到） | timesfm 2.0.0 完全重寫 API | 改用 `TimesFM_2p5_200M_torch.from_pretrained()` + `compile()` + `forecast()` |
| 未來預測圖只顯示一條線 | 三個 horizon 前段預測值完全相同互相覆蓋 | 改為單條 90 天線 + 各 horizon 端點標記 |

## 安裝步驟

### 1. 安裝 Python 3.12
從 python.org 下載 Windows installer (64-bit)，安裝時勾選「Add Python to PATH」。

### 2. 建立 venv
```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
```

### 3. 安裝依賴
```bash
pip install yfinance pandas numpy matplotlib seaborn scikit-learn huggingface_hub
pip install "timesfm[torch] @ git+https://github.com/google-research/timesfm.git"
```

> 安裝結果：`timesfm-2.0.0` + `torch-2.10.0` + `safetensors-0.7.0`
> PyPI 上的 timesfm 最新版不支援 Python 3.12，需從 GitHub 安裝。

### 4. HuggingFace 認證
模型 `google/timesfm-2.5-200m-pytorch` 為 gated repo，需先：
1. 登入 HuggingFace，至模型頁面接受使用條款
2. 建立 Access Token（Read 權限）
3. 執行登入：
```bash
hf auth login
```

### 5. 執行順序
```bash
python 01_fetch_data.py    # 下載台灣加權指數近三年資料
python 02_run_forecast.py  # hold-out 回測預測（前90%訓練，後10%測試）
python 03_evaluate.py      # 計算 MAE / MAPE / RMSE，與 Naive baseline 比較
python 04_visualize.py     # 繪製回測結果圖
python 05_forecast_future.py  # 用全部資料預測未來 30/60/90 個交易日
```
