# Hugging Face datasets
HF_DATASET_INPUT = "P2SAMAPA/p2-etf-deepm-data"          # same raw data (ETF prices + macro)
HF_DATASET_OUTPUT = "P2SAMAPA/p2-etf-informer-results"   # new output dataset

# Option A – Fixed Income & Alternatives (tradable, no AGG)
OPTION_A_ETFS = ["TLT", "LQD", "HYG", "VNQ", "GLD", "SLV", "PFF", "MBB"]

# Option B – Equity Sectors (tradable, no SPY)
OPTION_B_ETFS = ["QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "IWM", "IWF", "XSD", "XBI", "GDX", "XME"]

# Informer hyperparameters
INFORMER_CONFIG = {
    "seq_len": 20,           # lookback window (days)
    "label_len": 10,         # start token length for decoder
    "pred_len": 1,           # forecast horizon (1 day ahead)
    "enc_in": None,          # will be set dynamically (number of price + macro features)
    "dec_in": None,
    "c_out": 1,              # output dimension (next day return)
    "d_model": 128,
    "n_heads": 8,
    "e_layers": 2,
    "d_layers": 1,
    "d_ff": 256,
    "factor": 5,             # ProbSparsity factor
    "padding": 0,
    "distil": True,
    "dropout": 0.1,
    "output_attention": False,
}

# Training hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 50
LOOKBACK = 20   # must match seq_len
