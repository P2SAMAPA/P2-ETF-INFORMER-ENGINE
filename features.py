import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame, macro_df: pd.DataFrame = None) -> pd.DataFrame:
    """Engineer features, then replace inf/NaN with 0."""
    df = df.copy()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['vol_20'] = df['log_return'].rolling(20).std() * np.sqrt(252)
    df['mom_10'] = df['close'] / df['close'].shift(10) - 1
    df['mom_60'] = df['close'] / df['close'].shift(60) - 1

    price_cols = ['vol_20', 'mom_10', 'mom_60']
    X = df[price_cols].copy()
    # Replace inf/-inf with NaN, then forward fill, then fill remaining NaN with 0
    X = X.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

    if macro_df is not None:
        macro_aligned = macro_df.reindex(df.index, method='ffill')
        for col in macro_aligned.columns:
            X[f'macro_{col}'] = macro_aligned[col]
            X[f'macro_{col}_chg5'] = macro_aligned[col].pct_change(5)
        X = X.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    return X
