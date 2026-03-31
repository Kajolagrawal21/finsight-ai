"""
FinSight AI — Health Check Routes
"""

import psycopg2
import redis as redis_lib
from fastapi import APIRouter
from qdrant_client import QdrantClient

router = APIRouter()

DB_CONFIG = {
    "host": "localhost", "port": 5432,
    "dbname": "finsight", "user": "finsight_user", "password": "finsight_pass",
}


@router.get("/health")
async def health_check():
    """Check health of all services."""
    status = {}

    # TimescaleDB
    try:
        conn = psycopg2.connect(**DB_CONFIG, connect_timeout=3)
        conn.close()
        status["timescaledb"] = "✅ healthy"
    except Exception as e:
        status["timescaledb"] = f"❌ {e}"

    # Redis
    try:
        r = redis_lib.Redis(host="localhost", port=6379, socket_timeout=3)
        r.ping()
        status["redis"] = "✅ healthy"
    except Exception as e:
        status["redis"] = f"❌ {e}"

    # Qdrant
    try:
        client = QdrantClient(host="localhost", port=6333, timeout=3)
        client.get_collections()
        status["qdrant"] = "✅ healthy"
    except Exception as e:
        status["qdrant"] = f"❌ {e}"

    overall = "healthy" if all("✅" in v for v in status.values()) else "degraded"
    return {"status": overall, "services": status}
