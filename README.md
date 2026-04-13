# P2-ETF-INFORMER-ENGINE

ProbSparse Self-Attention Transformer (INFORMER) for next-day ETF return forecasting.

- **Input dataset**: `P2SAMAPA/p2-etf-deepm-data` (same as NSDE engine)
- **Output dataset**: `P2SAMAPA/p2-etf-informer-results`
- **Universes**: Option A (fixed income & alternatives) and Option B (equity sectors). Benchmarks AGG/SPY excluded from trading.

## Features
- Point forecast for next trading day
- Gaussian uncertainty estimation (heuristic)
- Macro variables (VIX, yield curve, HY spread) as exogenous inputs
- Streamlit dashboard with hero box, metrics, bar chart

## Run locally
```bash
pip install -r requirements.txt
export HF_TOKEN=your_token
python train.py --option both
streamlit run app.py
