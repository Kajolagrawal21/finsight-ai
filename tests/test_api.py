"""
FinSight AI — API Tests
Basic tests to verify API endpoints work correctly.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


@pytest.fixture
def client():
    """Create test client with mocked dependencies."""
    with patch("psycopg2.connect") as mock_db, \
         patch("redis.Redis") as mock_redis:

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = lambda s: s
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = [
            {"symbol": "AAPL", "company": "Apple Inc.", "sector": "Technology"}
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_db.return_value = mock_conn

        mock_redis_inst = MagicMock()
        mock_redis_inst.get.return_value = None
        mock_redis.return_value = mock_redis_inst

        from api.main import app
        yield TestClient(app)


def test_root(client):
    """Test root endpoint returns correct info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "FinSight AI"
    assert data["status"] == "running"


def test_health_endpoint_exists(client):
    """Test health endpoint is accessible."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200


def test_stocks_endpoint_exists(client):
    """Test stocks endpoint is accessible."""
    response = client.get("/api/v1/stocks")
    assert response.status_code == 200


def test_openapi_docs(client):
    """Test OpenAPI docs are available."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_openapi_json(client):
    """Test OpenAPI JSON schema is available."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert data["info"]["title"] == "FinSight AI"
