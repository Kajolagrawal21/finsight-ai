"""
FinSight AI — Stocks Routes
Live prices and historical OHLCV data.
"""

import redis
import json
import psycopg2
import psycopg2.extras
import yfinance as yf
from fastapi import APIRouter, HTTPException
from typing import Optional

router = APIRouter()

DB_CONFIG = {
    "host": "localhost", "port": 5432,
    "dbname": "finsight", "user": "finsight_user", "password": "finsight_pass",
}

CACHE = redis.Redis(host="localhost", port=6379, decode_responses=True)
CACHE_TTL = 60  # seconds


@router.get("/stocks")
async def get_watchlist():
    """Get all tracked symbols."""
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT symbol, company, sector FROM watchlist WHERE active = TRUE")
        symbols = cur.fetchall()
    conn.close()
    return {"symbols": [dict(s) for s in symbols]}


@router.get("/stocks/{symbol}/quote")
async def get_quote(symbol: str):
    """Get live quote for a symbol. Cached for 60 seconds."""
    symbol = symbol.upper()
    cache_key = f"quote:{symbol}"

    # Check cache first
    cached = CACHE.get(cache_key)
    if cached:
        return {"source": "cache", "data": json.loads(cached)}

    try:
        ticker = yf.Ticker(symbol)
        info   = ticker.fast_info
        data = {
            "symbol":     symbol,
            "price":      round(float(info.last_price), 2),
            "prev_close": round(float(info.previous_close), 2),
            "change_pct": round((info.last_price - info.previous_close) / info.previous_close * 100, 2),
            "volume":     int(info.three_month_average_volume or 0),
        }
        CACHE.setex(cache_key, CACHE_TTL, json.dumps(data))
        return {"source": "live", "data": data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stocks/{symbol}/history")
async def get_history(symbol: str, limit: int = 100):
    """Get historical OHLCV from TimescaleDB."""
    symbol = symbol.upper()
    conn   = psycopg2.connect(**DB_CONFIG)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT time, open, high, low, close, volume
            FROM stock_prices WHERE symbol = %s
            ORDER BY time DESC LIMIT %s
        """, (symbol, limit))
        rows = cur.fetchall()
    conn.close()
    return {"symbol": symbol, "count": len(rows), "data": [dict(r) for r in rows]}


@router.get("/stocks/{symbol}/indicators")
async def get_indicators(symbol: str):
    """Get latest technical indicators for a symbol."""
    symbol = symbol.upper()
    conn   = psycopg2.connect(**DB_CONFIG)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT * FROM technical_indicators
            WHERE symbol = %s ORDER BY time DESC LIMIT 1
        """, (symbol,))
        row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail=f"No indicators for {symbol}")
    return {"symbol": symbol, "indicators": dict(row)}
