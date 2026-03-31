"""FinSight AI — System Health Page"""

import streamlit as st
from dashboard.api_client import get


def show():
    st.title("🏥 System Health")
    st.caption("Monitor all FinSight AI services")

    if st.button("🔄 Refresh"):
        st.rerun()

    health = get("/health")
    if health:
        overall = health.get("status", "unknown")
        if overall == "healthy":
            st.success(f"## ✅ System Status: HEALTHY")
        else:
            st.warning(f"## ⚠️ System Status: DEGRADED")

        st.subheader("Services")
        services = health.get("services", {})
        for svc, status in services.items():
            if "✅" in status:
                st.success(f"**{svc.upper()}**: {status}")
            else:
                st.error(f"**{svc.upper()}**: {status}")

    st.divider()
    st.subheader("🔗 Service Links")
    c1, c2, c3, c4 = st.columns(4)
    c1.link_button("Kafka UI",   "http://localhost:8080")
    c2.link_button("MLflow",     "http://localhost:5002")
    c3.link_button("Grafana",    "http://localhost:3000")
    c4.link_button("Prometheus", "http://localhost:9090")

    st.divider()
    st.subheader("📋 Quick Stats")
    stats = get("/stocks")
    if stats:
        symbols = stats.get("symbols", [])
        st.metric("Tracked Symbols", len(symbols))
        st.dataframe(symbols, use_container_width=True)
