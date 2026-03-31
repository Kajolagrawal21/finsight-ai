"""
FinSight AI — Dashboard API Helper
Connects Streamlit dashboard to FastAPI backend.
"""

import requests
import streamlit as st

API_BASE = "http://localhost:8001/api/v1"


def get(endpoint: str, timeout: int = 10):
    """Make GET request to API with error handling."""
    try:
        r = requests.get(f"{API_BASE}{endpoint}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ API not running. Start with: uvicorn api.main:app --port 8001")
        return None
    except Exception as e:
        st.error(f"❌ API error: {e}")
        return None


def post(endpoint: str, data: dict, timeout: int = 60):
    """Make POST request to API with error handling."""
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=data, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ API not running. Start with: uvicorn api.main:app --port 8001")
        return None
    except Exception as e:
        st.error(f"❌ API error: {e}")
        return None
