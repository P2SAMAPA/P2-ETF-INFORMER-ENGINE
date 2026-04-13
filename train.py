import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os
import json
import pandas as pd
from datetime import datetime
from config import *
from loader import load_dataset, load_macro_data
from features import engineer_features
from model import InformerModel
from huggingface_hub import upload_file
from sklearn.preprocessing import StandardScaler

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_sequences(data_dict, macro_df, seq_len, pred_len, scaler=None):
    X_enc_list, X_dec_list, y_list = [], [], []
    # First collect all feature arrays, clean them
    all_features = []
    for ticker, df in data_dict.items():
        feat = engineer_features(df, macro_df)
        values = feat.values
        # Clean: replace inf, -inf, nan with 0
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        all_features.append(values)
    all_features = np.vstack(all_features)
    # Remove constant columns (std=0) to avoid scaler warning
    stds = all_features.std(axis=0)
    non_const = stds > 1e-8
    all_features = all_features[:, non_const]
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(all_features)
    # Now process each ticker with the same scaler, using only non-constant columns
    for ticker, df in data_dict.items():
        feat = engineer_features(df, macro_df)
        values = feat.values
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        values = values[:, non_const]  # keep same columns as scaler
        targets = df['close'].pct_change().shift(-pred_len).values
        for i in range(seq_len, len(values) - pred_len):
            x_enc = values[i-seq_len:i]
            x_dec = np.zeros((seq_len + pred_len, x_enc.shape[-1]))
            x_dec[:seq_len] = x_enc
            y = targets[i+pred_len-1] if pred_len==1 else targets[i:i+pred_len]
            X_enc_list.append(x_enc)
            X_dec_list.append(x_dec)
            y_list.append(y)
    X_enc = torch.tensor(np.array(X_enc_list), dtype=torch.float32)
    X_dec = torch.tensor(np.array(X_dec_list), dtype=torch.float32)
    y = torch.tensor(np.array(y_list), dtype=torch.float32).unsqueeze(1)
    return X_enc, X_dec, y, scaler, non_const

def train_model(model, loader, epochs, lr, device):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_enc, x_dec, y in loader:
            x_enc, x_dec, y = x_enc.to(device), x_dec.to(device), y.to(device)
            pred = model(x_enc, x_dec)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

def generate_signals(option, model, device, macro_df, seq_len, pred_len, scaler, non_const):
    if option == 'A':
        tickers = OPTION_A_ETFS
    else:
        tickers = OPTION_B_ETFS
    raw_data = load_dataset(option.lower(), include_benchmarks=False)
    forecasts = {}
    for ticker in tickers:
        if ticker not in raw_data:
            continue
        df = raw_data[ticker]
        feat = engineer_features(df, macro_df)
        values = feat.values[-seq_len:].astype(np.float32)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        values = values[:, non_const]
        if len(values) < seq_len:
            continue
        values_scaled = scaler.transform(values)
        x_enc = torch.tensor(values_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        x_dec = torch.zeros(1, seq_len+pred_len, values_scaled.shape[-1], device=device)
        x_dec[:, :seq_len] = x_enc
        with torch.no_grad():
            pred = model(x_enc, x_dec)
        mu = pred[0, -1].item()
        if np.isnan(mu):
            mu = 0.0
        sigma = 0.01  # fixed uncertainty
        confidence = 1 - 2 * sigma / (abs(mu) + sigma + 1e-8)
        forecasts[ticker] = {'mu': mu, 'sigma': sigma, 'confidence': confidence}
    if forecasts:
        top_pick = max(forecasts, key=lambda x: forecasts[x]['mu'])
        top_mu = forecasts[top_pick]['mu']
    else:
        top_pick = None
        top_mu = 0.0
    return {
        "generated_at": datetime.utcnow().isoformat(),
        "forecasts": forecasts,
        "top_pick": top_pick,
        "top_mu": top_mu
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--option", default="both", choices=["a","b","both"])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_data = load_dataset(args.option, include_benchmarks=False)
    macro_df = load_macro_data()
    if macro_df is None:
        dummy_idx = next(iter(raw_data.values())).index
        macro_df = pd.DataFrame(index=dummy_idx, data={'dummy':0.0})

    # Determine feature dimension after cleaning (non_const will be used)
    X_enc, X_dec, y, scaler, non_const = create_sequences(raw_data, macro_df, LOOKBACK, 1)
    print(f"X_enc shape: {X_enc.shape}, X_dec shape: {X_dec.shape}, y shape: {y.shape}")
    print(f"Feature dimension after cleaning: {X_enc.shape[-1]}")

    dataset = TensorDataset(X_enc, X_dec, y)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    INFORMER_CONFIG['enc_in'] = X_enc.shape[-1]
    INFORMER_CONFIG['dec_in'] = X_enc.shape[-1]
    INFORMER_CONFIG['seq_len'] = LOOKBACK
    INFORMER_CONFIG['label_len'] = LOOKBACK // 2
    INFORMER_CONFIG['pred_len'] = 1

    model = InformerModel(INFORMER_CONFIG).to(device)
    train_model(model, loader, args.epochs, args.lr, device)

    # Save model, scaler, and non_const mask
    model_path = "informer_model.pth"
    torch.save(model.state_dict(), model_path)
    import joblib
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(non_const, "non_const.pkl")
    print("Model, scaler, and feature mask saved.")

    token = os.getenv("HF_TOKEN")
    if token:
        upload_file(path_or_fileobj=model_path, path_in_repo=model_path,
                    repo_id=HF_DATASET_OUTPUT, repo_type="dataset", token=token)
        upload_file(path_or_fileobj="scaler.pkl", path_in_repo="scaler.pkl",
                    repo_id=HF_DATASET_OUTPUT, repo_type="dataset", token=token)
        upload_file(path_or_fileobj="non_const.pkl", path_in_repo="non_const.pkl",
                    repo_id=HF_DATASET_OUTPUT, repo_type="dataset", token=token)
        print("✅ Model and preprocessors uploaded")

    model.eval()
    signal_A = generate_signals('A', model, device, macro_df, LOOKBACK, 1, scaler, non_const)
    signal_B = generate_signals('B', model, device, macro_df, LOOKBACK, 1, scaler, non_const)

    os.makedirs("signals", exist_ok=True)
    with open("signals/signal_A.json", "w") as f:
        json.dump(signal_A, f, indent=2)
    with open("signals/signal_B.json", "w") as f:
        json.dump(signal_B, f, indent=2)

    if token:
        for opt in ["A","B"]:
            upload_file(path_or_fileobj=f"signals/signal_{opt}.json",
                        path_in_repo=f"signals/signal_{opt}.json",
                        repo_id=HF_DATASET_OUTPUT, repo_type="dataset", token=token)
        print("✅ Signals uploaded")

if __name__ == "__main__":
    main()
