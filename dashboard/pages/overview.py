"""FinSight AI — Overview Page"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dashboard.api_client import get


def show():
    st.title("🏠 FinSight AI — Market Overview")
    st.caption("Real-time stock analysis powered by ML + RAG + LangGraph")

    # ─── Live Quotes Grid ─────────────────────────────────────────────
    st.subheader("📡 Live Watchlist")

    symbols = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL", "AMZN", "META", "JPM", "NFLX", "SPY"]

    cols = st.columns(5)
    for i, symbol in enumerate(symbols):
        with cols[i % 5]:
            data = get(f"/stocks/{symbol}/quote")
            if data and data.get("data"):
                q = data["data"]
                price  = q.get("price", 0)
                change = q.get("change_pct", 0)
                color  = "🟢" if change >= 0 else "🔴"
                st.metric(
                    label=symbol,
                    value=f"${price:.2f}",
                    delta=f"{change:+.2f}%"
                )
            else:
                st.metric(label=symbol, value="—", delta="N/A")

    st.divider()

    # ─── Quick Predictions ────────────────────────────────────────────
    st.subheader("🤖 ML Signal Scanner")

    pred_cols = st.columns(5)
    for i, symbol in enumerate(symbols[:5]):
        with pred_cols[i]:
            pred = get(f"/predictions/{symbol}")
            if pred and pred.get("prediction"):
                p = pred["prediction"]
                direction  = p.get("direction", "—")
                confidence = p.get("confidence", 0)
                emoji = "⬆️" if direction == "UP" else "⬇️" if direction == "DOWN" else "➡️"
                st.metric(
                    label=symbol,
                    value=f"{emoji} {direction}",
                    delta=f"{confidence*100:.0f}% conf"
                )

    st.divider()

    # ─── Platform Stats ───────────────────────────────────────────────
    st.subheader("📊 Platform Statistics")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Symbols Tracked", "10")
    with c2:
        st.metric("ML Models", "3 (XGBoost, Prophet, LSTM)")
    with c3:
        st.metric("AI Agents", "5 (Router, News, ML, Risk, Portfolio)")
    with c4:
        st.metric("Data Pipeline", "✅ Kafka → TimescaleDB")
