"""
Step 1: 抓取台灣加權指數 (^TWII) 近三年日線資料，存為 CSV
"""

import os
import yfinance as yf
import pandas as pd

TICKER = "^TWII"
START_DATE = "2023-01-01"
END_DATE = "2026-03-13"
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "taiex_3y.csv")


def fetch_taiex():
    print(f"下載 {TICKER} {START_DATE} ~ {END_DATE} ...")
    df = yf.download(TICKER, start=START_DATE, end=END_DATE, auto_adjust=True)

    if df.empty:
        raise ValueError("下載失敗，請確認網路連線或 ticker 名稱")

    df = df[["Close"]].copy()
    df.index.name = "Date"
    df.columns = ["Close"]
    df.dropna(inplace=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE)

    print(f"共 {len(df)} 個交易日")
    print(f"日期範圍：{df.index[0].date()} ~ {df.index[-1].date()}")
    print(f"收盤價範圍：{df['Close'].min():.0f} ~ {df['Close'].max():.0f}")
    print(f"已儲存至 {OUTPUT_FILE}")
    return df


if __name__ == "__main__":
    fetch_taiex()
