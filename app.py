import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
from datetime import datetime
import numpy as np
from huggingface_hub import hf_hub_download
import config as cfg
from trading_calendar import format_next_trading_day
from loader import load_dataset

st.set_page_config(page_title="P2 Informer Engine", layout="wide", page_icon="📈", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.hero-box {
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
    border-radius: 20px; padding: 1.5rem; margin-bottom: 1rem; color: white;
}
.top-pick-label { font-size: 0.9rem; text-transform: uppercase; letter-spacing: 2px; opacity: 0.8; }
.top-pick-ticker { font-size: 3.5rem; font-weight: 800; line-height: 1; margin: 0.25rem 0; }
.expected-return { font-size: 1.8rem; font-weight: 600; margin: 0.5rem 0; }
.small-date { font-size: 0.8rem; opacity: 0.8; margin-top: 1rem; border-top: 1px solid rgba(255,255,255,0.2); padding-top: 0.75rem; }
.metrics-container { background: #f8fafc; border-radius: 16px; padding: 1rem; margin: 1rem 0; display: flex; justify-content: space-around; border: 1px solid #e2e8f0; }
.metric-card { text-align: center; }
.metric-label { font-size: 0.8rem; color: #475569; text-transform: uppercase; }
.metric-value { font-size: 1.4rem; font-weight: 700; color: #000000; }
</style>
""", unsafe_allow_html=True)

st.title("📈 P2 ETF INFORMER Engine")
st.markdown("**ProbSparse Self-Attention Transformer** — Next-day ETF return forecasts")

@st.cache_data(ttl=60)
def load_signal(opt):
    try:
        path = hf_hub_download(repo_id=cfg.HF_DATASET_OUTPUT, filename=f"signals/signal_{opt}.json", repo_type="dataset")
        with open(path) as f: return json.load(f)
    except: return None

@st.cache_data(ttl=3600)
def load_historical_prices():
    data = load_dataset("both", include_benchmarks=False)
    return {t: df['close'] for t, df in data.items()}

def compute_metrics(price_series):
    if price_series is None or len(price_series)<2: return None,None,None
    daily_returns = price_series.pct_change().dropna()
    ann_return = (price_series.iloc[-1]/price_series.iloc[0])**(252/len(daily_returns))-1
    ann_vol = daily_returns.std()*np.sqrt(252)
    sharpe = ann_return/ann_vol if ann_vol else 0
    cum = (1+daily_returns).cumprod()
    dd = (cum/cum.expanding().max())-1
    return ann_return, sharpe, dd.min()

historical = load_historical_prices()
signal_A = load_signal("A")
signal_B = load_signal("B")
tabA, tabB = st.tabs(["Option A — Fixed Income & Alternatives", "Option B — Equity Sectors"])

def render_tab(signal, label, ticker_list):
    if not signal:
        st.info("No signal data. Run training.")
        return
    fc = signal.get("forecasts", {})
    top = signal.get("top_pick")
    top_mu = signal.get("top_mu", 0.0)
    gen_time = signal.get("generated_at", "")[:19].replace("T"," ")
    top_mu_pct = top_mu * 100
    st.markdown(f"""
    <div class="hero-box">
        <div class="top-pick-label">TOP PICK</div>
        <div class="top-pick-ticker">{top}</div>
        <div class="expected-return">Expected Return: {top_mu_pct:.2f}% for next trading day</div>
        <div class="small-date">📅 US Markets Next Trading Day: {format_next_trading_day()} • Generated: {gen_time}</div>
    </div>
    """, unsafe_allow_html=True)
    if top and top in historical:
        ar, sr, mdd = compute_metrics(historical[top])
        if ar is not None:
            st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
            c1,c2,c3 = st.columns(3)
            c1.markdown(f'<div class="metric-card"><div class="metric-label">Annual Return</div><div class="metric-value">{ar:.2%}</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="metric-card"><div class="metric-label">Sharpe Ratio</div><div class="metric-value">{sr:.2f}</div></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="metric-card"><div class="metric-label">Max Drawdown</div><div class="metric-value">{mdd:.2%}</div></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    data = [{"Ticker": t, "μ": fc[t]["mu"], "σ": fc[t]["sigma"], "Confidence": fc[t]["confidence"]} for t in ticker_list if t in fc]
    df = pd.DataFrame(data).sort_values("μ", ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=df["Ticker"], x=df["μ"], orientation='h', error_x=dict(type='data', array=df["σ"]),
                         marker_color=['#3b82f6' if t==top else '#64748b' for t in df["Ticker"]]))
    fig.update_layout(title="Next-Day Return Forecasts (μ ± 1σ)", xaxis_title="Expected Return", height=620, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("All Forecasts")
    st.dataframe(df.style.format({"μ":"{:.4f}","σ":"{:.4f}","Confidence":"{:.1%}"}).background_gradient(subset=["μ"], cmap="Blues"), use_container_width=True)
    st.caption(f"Option {label} • {len(df)} ETFs • Model: INFORMER (ProbSparse)")

with tabA:
    render_tab(signal_A, "A", cfg.OPTION_A_ETFS)
with tabB:
    render_tab(signal_B, "B", cfg.OPTION_B_ETFS)

st.caption("**P2-ETF-INFORMER-ENGINE** • ProbSparse Self-Attention Transformer • Research only")
