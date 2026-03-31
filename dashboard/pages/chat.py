"""FinSight AI — AI Chat Page"""

import streamlit as st
from dashboard.api_client import post


def show():
    st.title("💬 AI Chat — Ask Anything About Stocks")
    st.caption("Powered by RAG + LangGraph agents + Local LLM (Llama 3)")

    # Symbol filter
    col1, col2 = st.columns([1, 3])
    with col1:
        symbol = st.selectbox("Focus on symbol (optional)", [
            "None", "AAPL", "MSFT", "NVDA", "TSLA",
            "GOOGL", "AMZN", "META", "JPM", "NFLX", "SPY"
        ])
        symbol = None if symbol == "None" else symbol

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": "👋 Hi! I'm FinSight AI. Ask me anything about stocks — "
                       "I'll use real news and technical analysis to answer!"
        })

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📰 {len(msg['sources'])} sources used"):
                    for s in msg["sources"]:
                        st.caption(f"**{s.get('title', '')}** — {s.get('source', '')} ({s.get('date', '')})")

    # Chat input
    if prompt := st.chat_input("Ask about any stock..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing news and data..."):
                result = post("/chat", {
                    "question": prompt,
                    "symbol":   symbol
                }, timeout=120)

            if result:
                answer  = result.get("answer", "Sorry, I couldn't generate an answer.")
                sources = result.get("sources", [])
                st.write(answer)
                if sources:
                    with st.expander(f"📰 {len(sources)} sources used"):
                        for s in sources:
                            st.caption(f"**{s.get('title', '')}** — {s.get('source', '')} ({s.get('date', '')})")

                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": answer,
                    "sources": sources
                })
            else:
                st.error("Failed to get response. Is Ollama running?")

    # Suggested questions
    st.divider()
    st.caption("💡 Try asking:")
    suggestions = [
        "Why is NVDA stock moving?",
        "What are analysts saying about AAPL?",
        "Is TSLA a good buy right now?",
        "What's the outlook for the S&P 500?",
    ]
    cols = st.columns(4)
    for i, s in enumerate(suggestions):
        with cols[i]:
            if st.button(s, use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": s})
                st.rerun()
