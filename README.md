# 🚀 FinSight AI — Intelligent Stock Analysis & Prediction Platform

> **Production-grade end-to-end ML platform** for real-time stock analysis, price prediction,
> and AI-powered financial intelligence using LangGraph multi-agents, RAG, and MLOps.

---

## 🏗️ Architecture

```
Data Layer      → yfinance → Kafka → TimescaleDB + Qdrant
ML Layer        → XGBoost + Prophet + LSTM + TFT → MLflow
Agent Layer     → LangGraph multi-agent → RAG (LangChain + Ollama)
API Layer       → FastAPI + WebSockets + Redis
Dashboard       → Streamlit + Plotly
DevOps          → Docker Compose + GitHub Actions + Prometheus + Grafana
```

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Data ingestion | yfinance, NewsAPI, Kafka |
| Storage | TimescaleDB (PostgreSQL), Redis, Qdrant |
| ML Models | XGBoost, Prophet, LSTM, Temporal Fusion Transformer |
| Experiment tracking | MLflow |
| LLM/Agents | LangGraph, LangChain, Ollama (local — free) |
| API | FastAPI, WebSockets |
| Dashboard | Streamlit, Plotly |
| Monitoring | Prometheus, Grafana |
| CI/CD | GitHub Actions |
| Containers | Docker, Docker Compose |

---

## ⚡ Quick Start

### Prerequisites
- Docker Desktop installed and running
- Python 3.11+
- Git
- 8GB RAM recommended

### 1. Clone & Setup

```bash
git clone https://github.com/yourusername/finsight-ai.git
cd finsight-ai
cp .env.example .env        # fill in your API keys
make setup                  # install Python deps
```

### 2. Start Infrastructure

```bash
make up
```

This starts: Kafka, TimescaleDB, Redis, Qdrant, MLflow, Prometheus, Grafana.

Wait ~30 seconds for all services to initialize.

### 3. Run the Data Pipeline

Open 3 terminal tabs:

```bash
# Terminal 1 — Kafka Producer (fetches & publishes stock data)
make producer

# Terminal 2 — Kafka Consumer (reads from Kafka, writes to TimescaleDB)
make consumer

# Terminal 3 — Feature Engineering (computes indicators)
make features
```

### 4. Install Ollama (free local LLM)

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3          # downloads ~4GB model — fully free, runs locally
```

### 5. Access the Services

| Service | URL | Credentials |
|---|---|---|
| Kafka UI | http://localhost:8080 | — |
| MLflow | http://localhost:5000 | — |
| Grafana | http://localhost:3000 | admin / finsight123 |
| Prometheus | http://localhost:9090 | — |
| Qdrant | http://localhost:6333/dashboard | — |

---

## 📁 Project Structure

```
finsight-ai/
├── .github/workflows/          # CI/CD pipelines
├── data/
│   ├── sql/init.sql            # TimescaleDB schema
│   ├── ingestion/
│   │   ├── producer.py         # Kafka producer
│   │   └── consumer.py         # Kafka → TimescaleDB consumer
│   └── processing/
│       └── feature_engineering.py   # Technical indicators
├── ml/                         # Phase 2 — ML models (coming soon)
├── agents/                     # Phase 4 — LangGraph agents (coming soon)
├── rag/                        # Phase 3 — RAG pipeline (coming soon)
├── api/                        # Phase 5 — FastAPI backend (coming soon)
├── dashboard/                  # Phase 6 — Streamlit UI (coming soon)
├── monitoring/
│   └── prometheus.yml
├── docker-compose.yml
├── requirements.txt
├── Makefile
└── .env.example
```

---

## 🗺️ Build Roadmap

- [x] **Phase 1** — Infrastructure & Data Pipeline ← *You are here*
- [ ] **Phase 2** — Feature Engineering & ML Models (XGBoost, Prophet, LSTM, TFT)
- [ ] **Phase 3** — RAG Knowledge Base (News + SEC filings → Qdrant)
- [ ] **Phase 4** — LangGraph Multi-Agent System
- [ ] **Phase 5** — FastAPI Backend (REST + WebSocket)
- [ ] **Phase 6** — Streamlit Dashboard
- [ ] **Phase 7** — MLOps & Model Monitoring
- [ ] **Phase 8** — CI/CD Polish & Demo

---

## 🤝 Contributing

Built as a portfolio project. PRs welcome!

---

## 📄 License

MIT
