# ============================================================
# FinSight AI — Makefile
# ============================================================

.PHONY: help setup up down logs producer consumer features clean

help:
	@echo ""
	@echo "  FinSight AI — Available Commands"
	@echo "  ─────────────────────────────────────────────────"
	@echo "  make setup      Install Python dependencies"
	@echo "  make up         Start all Docker services"
	@echo "  make down       Stop all Docker services"
	@echo "  make logs       Tail all container logs"
	@echo "  make producer   Run Kafka data producer"
	@echo "  make consumer   Run Kafka → TimescaleDB consumer"
	@echo "  make features   Run feature engineering pipeline"
	@echo "  make status     Show status of all services"
	@echo "  make clean      Remove all volumes and containers"
	@echo ""

# ─── Environment ──────────────────────────────────────────────
setup:
	@echo "📦 Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "✅ Setup complete!"

# ─── Docker ───────────────────────────────────────────────────
up:
	@echo "🚀 Starting FinSight AI infrastructure..."
	docker compose up -d
	@echo ""
	@echo "  Services running:"
	@echo "  ┌─────────────────────────────────────────┐"
	@echo "  │  Kafka UI      → http://localhost:8080  │"
	@echo "  │  MLflow        → http://localhost:5000  │"
	@echo "  │  Grafana       → http://localhost:3000  │"
	@echo "  │  Prometheus    → http://localhost:9090  │"
	@echo "  │  Qdrant        → http://localhost:6333  │"
	@echo "  │  TimescaleDB   → localhost:5432         │"
	@echo "  │  Redis         → localhost:6379         │"
	@echo "  └─────────────────────────────────────────┘"

down:
	docker compose down

logs:
	docker compose logs -f

status:
	docker compose ps

# ─── Data Pipeline ────────────────────────────────────────────
producer:
	@echo "📤 Starting Kafka producer..."
	python data/ingestion/producer.py

consumer:
	@echo "👂 Starting Kafka consumer..."
	python data/ingestion/consumer.py

features:
	@echo "🔧 Running feature engineering..."
	python data/processing/feature_engineering.py

# ─── Full pipeline in one shot (3 separate terminals recommended) ──
pipeline: up
	@echo "⏳ Waiting for services to be ready..."
	sleep 15
	@echo "▶ Run these in separate terminals:"
	@echo "  Terminal 1: make producer"
	@echo "  Terminal 2: make consumer"
	@echo "  Terminal 3: make features"

# ─── Cleanup ──────────────────────────────────────────────────
clean:
	@echo "🧹 Cleaning up all data volumes..."
	docker compose down -v --remove-orphans
	@echo "✅ Clean complete!"
