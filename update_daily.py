import json, os, torch, pandas as pd, numpy as np
from datetime import datetime
from huggingface_hub import hf_hub_download
from config import *
from loader import load_dataset, load_macro_data
from features import engineer_features
from model import InformerModel
import joblib

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = hf_hub_download(repo_id=HF_DATASET_OUTPUT, filename="informer_model.pth",
                                 repo_type="dataset", token=os.getenv("HF_TOKEN"))
    feature_scaler = joblib.load(hf_hub_download(repo_id=HF_DATASET_OUTPUT, filename="feature_scaler.pkl",
                                                 repo_type="dataset", token=os.getenv("HF_TOKEN")))
    target_scaler = joblib.load(hf_hub_download(repo_id=HF_DATASET_OUTPUT, filename="target_scaler.pkl",
                                                repo_type="dataset", token=os.getenv("HF_TOKEN")))
    non_const = joblib.load(hf_hub_download(repo_id=HF_DATASET_OUTPUT, filename="non_const.pkl",
                                            repo_type="dataset", token=os.getenv("HF_TOKEN")))

    sample_data = load_dataset("both", include_benchmarks=False)
    sample_ticker = next(iter(sample_data.keys()))
    macro_df = load_macro_data()
    if macro_df is None:
        dummy_idx = sample_data[sample_ticker].index
        macro_df = pd.DataFrame(index=dummy_idx, data={'dummy':0.0})
    sample_feat = engineer_features(sample_data[sample_ticker], macro_df)
    sample_clean = np.nan_to_num(sample_feat.values, nan=0.0, posinf=0.0, neginf=0.0)
    feature_dim = sample_clean[:, non_const].shape[1]

    INFORMER_CONFIG['enc_in'] = feature_dim
    INFORMER_CONFIG['dec_in'] = feature_dim
    INFORMER_CONFIG['seq_len'] = LOOKBACK
    INFORMER_CONFIG['pred_len'] = 1

    model = InformerModel(INFORMER_CONFIG).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    def gen(option):
        tickers = OPTION_A_ETFS if option=='A' else OPTION_B_ETFS
        data = load_dataset(option.lower(), include_benchmarks=False)
        forecasts = {}
        for ticker in tickers:
            if ticker not in data: continue
            df = data[ticker]
            feat = engineer_features(df, macro_df)
            values = feat.values[-LOOKBACK:].astype(np.float32)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            values = values[:, non_const]
            if len(values) < LOOKBACK: continue
            values_scaled = feature_scaler.transform(values)
            x_enc = torch.tensor(values_scaled, dtype=torch.float32).unsqueeze(0).to(device)
            x_dec = torch.zeros(1, LOOKBACK+1, feature_dim, device=device)
            x_dec[:, :LOOKBACK] = x_enc
            with torch.no_grad():
                mu_scaled, log_sigma_scaled = model(x_enc, x_dec)
            mu = target_scaler.inverse_transform(mu_scaled.cpu().numpy())[0, -1]
            sigma_scaled = torch.exp(log_sigma_scaled).cpu().numpy()[0, -1]
            sigma = sigma_scaled * target_scaler.scale_[0]
            confidence = 1 - 2*sigma/(abs(mu)+sigma+1e-8)
            forecasts[ticker] = {'mu': float(mu), 'sigma': float(sigma), 'confidence': float(confidence)}
        top_pick = max(forecasts, key=lambda x: forecasts[x]['mu']) if forecasts else None
        return {"generated_at": datetime.utcnow().isoformat(), "forecasts": forecasts,
                "top_pick": top_pick, "top_mu": forecasts[top_pick]['mu'] if top_pick else 0.0}

    signal_A = gen('A')
    signal_B = gen('B')
    os.makedirs("signals", exist_ok=True)
    with open("signals/signal_A.json","w") as f: json.dump(signal_A, f, indent=2)
    with open("signals/signal_B.json","w") as f: json.dump(signal_B, f, indent=2)
    print("Signals saved.")

if __name__ == "__main__":
    main()
