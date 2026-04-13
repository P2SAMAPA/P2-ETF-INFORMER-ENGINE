from huggingface_hub import hf_hub_download
import pandas as pd
import os
from config import HF_DATASET_INPUT, OPTION_A_ETFS, OPTION_B_ETFS

def load_dataset(option: str = "both", include_benchmarks: bool = True):
    """Load ETF price data from master.parquet."""
    print(f"Downloading dataset: {HF_DATASET_INPUT}")
    master_path = hf_hub_download(
        repo_id=HF_DATASET_INPUT,
        filename="data/master.parquet",
        repo_type="dataset",
        token=os.getenv("HF_TOKEN")
    )
    df = pd.read_parquet(master_path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')
        df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)

    if option == "a":
        tickers = OPTION_A_ETFS.copy()
        if include_benchmarks:
            tickers.insert(0, "AGG")
    elif option == "b":
        tickers = OPTION_B_ETFS.copy()
        if include_benchmarks:
            tickers.insert(0, "SPY")
    else:  # both
        tickers = OPTION_A_ETFS + OPTION_B_ETFS
        if include_benchmarks:
            tickers = ["AGG", "SPY"] + tickers

    data = {}
    for ticker in tickers:
        for suffix in ["_Close", "_close", "Close_"]:
            col = f"{ticker}{suffix}" if suffix.startswith("_") else f"{suffix}{ticker}"
            if col in df.columns:
                series = df[col].dropna()
                if len(series) > 0:
                    data[ticker] = pd.DataFrame({'close': series})
                    print(f"✅ Loaded {ticker}: {len(series)} rows")
                break
        else:
            print(f"⚠️ No close column for {ticker}")
    return data

def load_macro_data():
    """Load macro variables from master.parquet."""
    master_path = hf_hub_download(
        repo_id=HF_DATASET_INPUT,
        filename="data/master.parquet",
        repo_type="dataset",
        token=os.getenv("HF_TOKEN")
    )
    df = pd.read_parquet(master_path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')
        df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    macro_cols = {}
    for col in df.columns:
        if 'VIX' in col.upper():
            macro_cols['VIX'] = df[col]
        if 'T10Y2Y' in col.upper():
            macro_cols['T10Y2Y'] = df[col]
        if 'HY' in col.upper() and 'SPREAD' in col.upper():
            macro_cols['HY_SPREAD'] = df[col]
    if macro_cols:
        macro_df = pd.DataFrame(macro_cols).ffill().bfill()
        print(f"Loaded macro: {list(macro_cols.keys())}")
        return macro_df
    else:
        print("No macro columns found, returning None")
        return None
