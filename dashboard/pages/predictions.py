"""FinSight AI — Predictions Page"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dashboard.api_client import get


def show():
    st.title("🤖 AI Predictions")
    st.caption("ML-powered price direction predictions using XGBoost + Prophet + LSTM")

    symbol = st.selectbox("Select Symbol", [
        "AAPL", "MSFT", "NVDA", "TSLA", "GOOGL",
        "AMZN", "META", "JPM", "NFLX", "SPY"
    ])

    col1, col2 = st.columns(2)

    # ─── Direction Prediction ─────────────────────────────────────────
    with col1:
        st.subheader("📈 Direction Signal")
        pred = get(f"/predictions/{symbol}")
        if pred and pred.get("prediction"):
            p = pred["prediction"]
            direction  = p.get("direction", "NEUTRAL")
            confidence = p.get("confidence", 0.5)
            score      = p.get("score", 0)

            if direction == "UP":
                st.success(f"## ⬆️ BULLISH — {confidence*100:.0f}% confidence")
            elif direction == "DOWN":
                st.error(f"## ⬇️ BEARISH — {confidence*100:.0f}% confidence")
            else:
                st.warning(f"## ➡️ NEUTRAL — {confidence*100:.0f}% confidence")

            # Confidence gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=confidence * 100,
                title={"text": "Confidence %"},
                delta={"reference": 50},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": "#0984e3"},
                    "steps": [
                        {"range": [0, 40],   "color": "#e17055"},
                        {"range": [40, 60],  "color": "#fdcb6e"},
                        {"range": [60, 100], "color": "#00b894"},
                    ],
                }
            ))
            fig.update_layout(height=250, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📋 Signals")
            for signal in p.get("signals", []):
                if "⚠️" in signal:
                    st.warning(signal)
                elif "✅" in signal:
                    st.success(signal)
                else:
                    st.info(signal)

    # ─── Prophet Forecast ─────────────────────────────────────────────
    with col2:
        st.subheader("📅 7-Day Prophet Forecast")
        with st.spinner("Running Prophet forecast..."):
            forecast = get(f"/predictions/{symbol}/prophet?days=7", timeout=30)

        if forecast and forecast.get("forecast"):
            df = pd.DataFrame(forecast["forecast"])
            df["ds"] = pd.to_datetime(df["ds"])

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["ds"], y=df["yhat"],
                mode="lines+markers",
                name="Forecast",
                line=dict(color="#0984e3", width=2)
            ))
            fig.add_trace(go.Scatter(
                x=pd.concat([df["ds"], df["ds"][::-1]]),
                y=pd.concat([df["yhat_upper"], df["yhat_lower"][::-1]]),
                fill="toself",
                fillcolor="rgba(9,132,227,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Confidence Band"
            ))
            fig.update_layout(
                title=f"{symbol} — 7 Day Forecast",
                template="plotly_dark",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Forecast table
            df_display = df[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
            df_display.columns = ["Date", "Forecast", "Lower", "Upper"]
            df_display["Date"] = df_display["Date"].dt.strftime("%Y-%m-%d")
            for col in ["Forecast", "Lower", "Upper"]:
                df_display[col] = df_display[col].apply(lambda x: f"${x:.2f}")
            st.dataframe(df_display, use_container_width=True)
