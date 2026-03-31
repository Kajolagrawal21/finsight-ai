"""
FinSight AI — Streamlit Dashboard
Interactive dashboard for stock analysis, predictions, and AI chat.
"""

import streamlit as st

st.set_page_config(
    page_title="FinSight AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 FinSight AI")
    st.caption("Intelligent Stock Analysis Platform")
    st.divider()

    page = st.selectbox("Navigate", [
        "🏠 Overview",
        "📊 Stock Analysis",
        "🤖 AI Predictions",
        "💬 AI Chat",
        "🏥 System Health",
    ])

    st.divider()
    st.caption("Built with LangGraph + RAG + MLflow")

# ─── Page Router ─────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    from dashboard.pages import overview
    overview.show()

elif page == "📊 Stock Analysis":
    from dashboard.pages import stocks
    stocks.show()

elif page == "🤖 AI Predictions":
    from dashboard.pages import predictions
    predictions.show()

elif page == "💬 AI Chat":
    from dashboard.pages import chat
    chat.show()

elif page == "🏥 System Health":
    from dashboard.pages import health
    health.show()
