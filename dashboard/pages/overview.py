"""FinSight AI — Overview Page (US + Indian Stocks)"""

import streamlit as st
import pandas as pd
from dashboard.api_client import get


def show():
    st.title("🏠 FinSight AI — Market Overview")
    st.caption("Real-time US + Indian NSE stock analysis powered by ML + RAG + LangGraph")

    # ─── Market Toggle ────────────────────────────────────────
    market_tab = st.radio("Market", ["🇺🇸 US Markets", "🇮🇳 Indian NSE", "🌍 All"], horizontal=True)

    US_SYMBOLS     = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL", "AMZN", "META", "JPM", "NFLX", "SPY"]
    INDIA_SYMBOLS  = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
                      "WIPRO.NS", "SBIN.NS", "BAJFINANCE.NS", "ADANIENT.NS", "^NSEI"]

    if market_tab == "🇺🇸 US Markets":
        symbols = US_SYMBOLS
        currency = "USD"
    elif market_tab == "🇮🇳 Indian NSE":
        symbols = INDIA_SYMBOLS
        currency = "INR"
    else:
        symbols = US_SYMBOLS + INDIA_SYMBOLS
        currency = "MIXED"

    # ─── Live Quotes Grid ─────────────────────────────────────
    st.subheader(f"📡 Live Prices ({currency})")

    cols = st.columns(5)
    for i, symbol in enumerate(symbols):
        with cols[i % 5]:
            data = get(f"/stocks/{symbol}/quote")
            if data and data.get("data"):
                q      = data["data"]
                price  = q.get("price", 0)
                change = q.get("change_pct", 0)
                cur    = "₹" if ".NS" in symbol or symbol == "^NSEI" else "$"
                display_name = symbol.replace(".NS", "")
                st.metric(
                    label=display_name,
                    value=f"{cur}{price:,.2f}",
                    delta=f"{change:+.2f}%"
                )
            else:
                st.metric(label=symbol.replace(".NS",""), value="—", delta="N/A")

    st.divider()

    # ─── ML Signal Scanner ────────────────────────────────────
    st.subheader("🤖 ML Signal Scanner")
    pred_cols = st.columns(5)
    for i, symbol in enumerate(symbols[:5]):
        with pred_cols[i]:
            pred = get(f"/predictions/{symbol}")
            if pred and pred.get("prediction"):
                p    = pred["prediction"]
                dir  = p.get("direction", "—")
                conf = p.get("confidence", 0)
                emoji = "⬆️" if dir == "UP" else "⬇️" if dir == "DOWN" else "➡️"
                st.metric(
                    label=symbol.replace(".NS", ""),
                    value=f"{emoji} {dir}",
                    delta=f"{conf*100:.0f}% conf"
                )
            else:
                st.metric(label=symbol.replace(".NS",""), value="—", delta="No model")

    for i, symbol in enumerate(display_symbols):
        with pred_cols[i]:
            pred = get(f"/predictions/{symbol}")
            if pred and pred.get("prediction"):
                p   = pred["prediction"]
                dir = p.get("direction", "—")
                conf = p.get("confidence", 0)
                emoji = "⬆️" if dir == "UP" else "⬇️" if dir == "DOWN" else "➡️"
                st.metric(
                    label=symbol.replace(".NS", ""),
                    value=f"{emoji} {dir}",
                    delta=f"{conf*100:.0f}% conf"
                )

    st.divider()

    # ─── Market Hours Info ────────────────────────────────────
    st.subheader("🕐 Market Hours (IST)")
    c1, c2, c3 = st.columns(3)
    c1.info("🇺🇸 **US Markets**\n\nOpen: 7:00 PM IST\nClose: 1:30 AM IST")
    c2.info("🇮🇳 **NSE India**\n\nOpen: 9:15 AM IST\nClose: 3:30 PM IST")
    c3.success("💡 **Tip**\n\nUS + India give you\n24hr market coverage!")

    st.divider()

    # ─── Platform Stats ───────────────────────────────────────
    st.subheader("📊 Platform Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Symbols", "20 (10 US + 10 NSE)")
    c2.metric("ML Models", "3 per symbol")
    c3.metric("AI Agents", "5 LangGraph agents")
    c4.metric("Data Pipeline", "✅ Kafka → TimescaleDB")
