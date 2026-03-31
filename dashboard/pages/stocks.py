"""FinSight AI — Stock Analysis Page (US + Indian NSE)"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dashboard.api_client import get

US_SYMBOLS    = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL", "AMZN", "META", "JPM", "NFLX", "SPY"]
INDIA_SYMBOLS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
                 "WIPRO.NS", "SBIN.NS", "BAJFINANCE.NS", "ADANIENT.NS", "^NSEI"]

COMPANY_NAMES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA",
    "TSLA": "Tesla", "GOOGL": "Google", "AMZN": "Amazon",
    "META": "Meta", "JPM": "JPMorgan", "NFLX": "Netflix", "SPY": "S&P 500",
    "RELIANCE.NS": "Reliance", "TCS.NS": "TCS", "INFY.NS": "Infosys",
    "HDFCBANK.NS": "HDFC Bank", "ICICIBANK.NS": "ICICI Bank",
    "WIPRO.NS": "Wipro", "SBIN.NS": "SBI", "BAJFINANCE.NS": "Bajaj Finance",
    "ADANIENT.NS": "Adani", "^NSEI": "Nifty 50",
}


def show():
    st.title("📊 Stock Analysis")

    # ─── Market + Symbol Selector ─────────────────────────────
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        market = st.selectbox("Market", ["🇺🇸 US", "🇮🇳 NSE India"])
    with col2:
        symbols = US_SYMBOLS if "US" in market else INDIA_SYMBOLS
        symbol  = st.selectbox("Symbol", symbols,
                               format_func=lambda s: f"{s.replace('.NS','')} — {COMPANY_NAMES.get(s, s)}")
    with col3:
        limit = st.slider("Data Points", 50, 500, 200)

    # Currency symbol
    cur = "₹" if ".NS" in symbol or symbol == "^NSEI" else "$"

    # ─── Live Quote ───────────────────────────────────────────
    quote_data = get(f"/stocks/{symbol}/quote")
    if quote_data and quote_data.get("data"):
        q = quote_data["data"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price",      f"{cur}{q.get('price', 0):,.2f}")
        c2.metric("Change",     f"{q.get('change_pct', 0):+.2f}%")
        c3.metric("Prev Close", f"{cur}{q.get('prev_close', 0):,.2f}")
        c4.metric("Volume",     f"{q.get('volume', 0):,}")

    st.divider()

    # ─── Price Chart ──────────────────────────────────────────
    hist = get(f"/stocks/{symbol}/history?limit={limit}")
    if hist and hist.get("data"):
        df = pd.DataFrame(hist["data"])
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.03, row_heights=[0.7, 0.3])

        fig.add_trace(go.Candlestick(
            x=df["time"],
            open=df["open"], high=df["high"],
            low=df["low"],   close=df["close"],
            name=symbol,
            increasing_line_color="#00b894",
            decreasing_line_color="#e17055",
        ), row=1, col=1)

        colors = ["#00b894" if c >= o else "#e17055"
                  for c, o in zip(df["close"], df["open"])]
        fig.add_trace(go.Bar(
            x=df["time"], y=df["volume"],
            name="Volume", marker_color=colors, opacity=0.7
        ), row=2, col=1)

        company = COMPANY_NAMES.get(symbol, symbol)
        fig.update_layout(
            title=f"{company} ({symbol.replace('.NS','')}) — Price & Volume ({cur})",
            xaxis_rangeslider_visible=False,
            height=600, template="plotly_dark", showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ─── Technical Indicators ─────────────────────────────────
    st.subheader("📐 Technical Indicators")
    ind_data = get(f"/stocks/{symbol}/indicators")
    if ind_data and ind_data.get("indicators"):
        ind = ind_data["indicators"]

        c1, c2, c3, c4 = st.columns(4)
        rsi = float(ind.get("rsi_14") or 50)
        c1.metric("RSI (14)", f"{rsi:.1f}",
                  "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral")

        macd = float(ind.get("macd") or 0)
        sig  = float(ind.get("macd_signal") or 0)
        c2.metric("MACD", f"{macd:.4f}", "Bullish" if macd > sig else "Bearish")

        ema20 = float(ind.get("ema_20") or 0)
        ema50 = float(ind.get("ema_50") or 0)
        c3.metric("EMA 20", f"{cur}{ema20:,.2f}")
        c4.metric("EMA 50", f"{cur}{ema50:,.2f}")

        # RSI Gauge
        fig_rsi = go.Figure(go.Indicator(
            mode="gauge+number",
            value=rsi,
            title={"text": "RSI (14)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#0984e3"},
                "steps": [
                    {"range": [0, 30],   "color": "#00b894"},
                    {"range": [30, 70],  "color": "#fdcb6e"},
                    {"range": [70, 100], "color": "#e17055"},
                ],
            }
        ))
        fig_rsi.update_layout(height=250, template="plotly_dark")
        st.plotly_chart(fig_rsi, use_container_width=True)

    # ─── NSE specific info ────────────────────────────────────
    if ".NS" in symbol or symbol == "^NSEI":
        st.info("🇮🇳 **NSE Market Hours:** 9:15 AM – 3:30 PM IST | Currency: Indian Rupee (₹)")
