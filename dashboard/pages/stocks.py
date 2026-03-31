"""FinSight AI — Stock Analysis Page"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dashboard.api_client import get


def show():
    st.title("📊 Stock Analysis")

    # ─── Symbol Selector ──────────────────────────────────────────────
    col1, col2 = st.columns([2, 1])
    with col1:
        symbol = st.selectbox("Select Symbol", [
            "AAPL", "MSFT", "NVDA", "TSLA", "GOOGL",
            "AMZN", "META", "JPM", "NFLX", "SPY"
        ])
    with col2:
        limit = st.slider("Data Points", 50, 500, 200)

    # ─── Live Quote ───────────────────────────────────────────────────
    quote_data = get(f"/stocks/{symbol}/quote")
    if quote_data and quote_data.get("data"):
        q = quote_data["data"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price",   f"${q.get('price', 0):.2f}")
        c2.metric("Change",  f"{q.get('change_pct', 0):+.2f}%")
        c3.metric("Prev Close", f"${q.get('prev_close', 0):.2f}")
        c4.metric("Volume",  f"{q.get('volume', 0):,}")

    st.divider()

    # ─── Price Chart ──────────────────────────────────────────────────
    hist = get(f"/stocks/{symbol}/history?limit={limit}")
    if hist and hist.get("data"):
        df = pd.DataFrame(hist["data"])
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time")

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3]
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df["time"],
            open=df["open"], high=df["high"],
            low=df["low"],   close=df["close"],
            name=symbol,
            increasing_line_color="#00b894",
            decreasing_line_color="#e17055",
        ), row=1, col=1)

        # Volume bars
        colors = ["#00b894" if c >= o else "#e17055"
                  for c, o in zip(df["close"], df["open"])]
        fig.add_trace(go.Bar(
            x=df["time"], y=df["volume"],
            name="Volume", marker_color=colors, opacity=0.7
        ), row=2, col=1)

        fig.update_layout(
            title=f"{symbol} — Price & Volume",
            xaxis_rangeslider_visible=False,
            height=600,
            template="plotly_dark",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ─── Technical Indicators ─────────────────────────────────────────
    st.subheader("📐 Technical Indicators")
    ind_data = get(f"/stocks/{symbol}/indicators")
    if ind_data and ind_data.get("indicators"):
        ind = ind_data["indicators"]

        c1, c2, c3, c4 = st.columns(4)
        rsi = ind.get("rsi_14", 0) or 0
        c1.metric("RSI (14)", f"{float(rsi):.1f}",
                  "Overbought" if float(rsi) > 70 else "Oversold" if float(rsi) < 30 else "Neutral")

        macd = ind.get("macd", 0) or 0
        sig  = ind.get("macd_signal", 0) or 0
        c2.metric("MACD", f"{float(macd):.4f}",
                  "Bullish" if float(macd) > float(sig) else "Bearish")

        ema20 = ind.get("ema_20", 0) or 0
        ema50 = ind.get("ema_50", 0) or 0
        c3.metric("EMA 20", f"${float(ema20):.2f}")
        c4.metric("EMA 50", f"${float(ema50):.2f}")

        # RSI Gauge
        fig_rsi = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(rsi),
            title={"text": "RSI (14)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#0984e3"},
                "steps": [
                    {"range": [0, 30],  "color": "#00b894"},
                    {"range": [30, 70], "color": "#fdcb6e"},
                    {"range": [70, 100],"color": "#e17055"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 2},
                    "thickness": 0.75,
                    "value": float(rsi)
                }
            }
        ))
        fig_rsi.update_layout(height=250, template="plotly_dark")
        st.plotly_chart(fig_rsi, use_container_width=True)
