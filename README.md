# 🚀 FinSight AI — Intelligent Stock Analysis & Prediction Platform

> **Production-grade, end-to-end ML platform** for real-time stock analysis, price prediction,
> and AI-powered financial intelligence using LangGraph multi-agents, RAG, and MLOps.

[![CI/CD](https://github.com/Kajolagrawal21/finsight-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/Kajolagrawal21/finsight-ai/actions)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![MLflow](https://img.shields.io/badge/MLflow-2.19-orange)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA LAYER                              │
│  yfinance → Kafka → TimescaleDB │ NewsAPI → Qdrant (RAG)   │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                     ML LAYER                                │
│  XGBoost │ Prophet │ LSTM │ MLflow │ Drift Detection        │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                   AI AGENT LAYER                            │
│  LangGraph: Router→News→ML→Risk→Portfolio→Synthesizer       │
│  RAG: LangChain + Qdrant + Ollama (local LLM, FREE)        │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                   SERVING LAYER                             │
│  FastAPI (REST) │ Redis cache │ Streamlit dashboard         │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                   DEVOPS LAYER                              │
│  Docker Compose │ GitHub Actions CI/CD │ Prometheus+Grafana │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Data Ingestion** | yfinance, NewsAPI, Apache Kafka |
| **Storage** | TimescaleDB, Redis, Qdrant |
| **ML Models** | XGBoost, Prophet, LSTM (PyTorch) |
| **Experiment Tracking** | MLflow |
| **LLM / Agents** | LangGraph, LangChain, Ollama (Llama 3 — local, free) |
| **API** | FastAPI + WebSockets |
| **Dashboard** | Streamlit + Plotly |
| **Monitoring** | Prometheus + Grafana |
| **CI/CD** | GitHub Actions |
| **Containers** | Docker + Docker Compose |

---

## ⚡ Quick Start

```bash
# Clone & setup
git clone https://github.com/Kajolagrawal21/finsight-ai.git
cd finsight-ai
cp .env.example .env
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Start everything
docker compose up -d
make producer     # terminal 1
make consumer     # terminal 2
make features     # terminal 3
python -m ml.training.train_pipeline
make api          # terminal 4
make dashboard    # terminal 5

# Install free local LLM
ollama pull llama3
```

---

## 🌐 Service URLs

| Service | URL |
|---|---|
| Dashboard | http://localhost:8501 |
| API Docs | http://localhost:8001/docs |
| MLflow | http://localhost:5002 |
| Kafka UI | http://localhost:8080 |
| Grafana | http://localhost:3000 |
| Qdrant | http://localhost:6333/dashboard |

---

## 🗺️ Build Phases

- [x] Phase 1 — Real-time data pipeline (Kafka → TimescaleDB)
- [x] Phase 2 — ML models (XGBoost, Prophet, LSTM + MLflow)
- [x] Phase 3 — RAG knowledge base (News → Qdrant)
- [x] Phase 4 — LangGraph multi-agent system
- [x] Phase 5 — FastAPI backend
- [x] Phase 6 — Streamlit dashboard
- [x] Phase 7 — MLOps (drift detection, auto-retrain, monitoring)
- [x] Phase 8 — CI/CD polish

---

## 📄 License

MIT — built as a portfolio project by Kajol Agrawal
