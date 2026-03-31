"""
FinSight AI — Predictions Routes
Runs ML models and returns price predictions.
"""

import redis
import json
import psycopg2
import psycopg2.extras
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

DB_CONFIG = {
    "host": "localhost", "port": 5432,
    "dbname": "finsight", "user": "finsight_user", "password": "finsight_pass",
}

CACHE = redis.Redis(host="localhost", port=6379, decode_responses=True)


class PredictionRequest(BaseModel):
    symbol: str
    horizon: int = 1   # days ahead to predict


@router.get("/predictions/{symbol}")
async def get_prediction(symbol: str):
    """
    Get ML prediction for a symbol.
    Uses latest technical indicators to predict price direction.
    """
    symbol    = symbol.upper()
    cache_key = f"prediction:{symbol}"

    cached = CACHE.get(cache_key)
    if cached:
        return {"source": "cache", "prediction": json.loads(cached)}

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Get latest indicators
            cur.execute("""
                SELECT i.*, s.close, s.volume
                FROM technical_indicators i
                JOIN stock_prices s ON i.time = s.time AND i.symbol = s.symbol
                WHERE i.symbol = %s
                ORDER BY i.time DESC LIMIT 1
            """, (symbol,))
            row = cur.fetchone()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")

        row = dict(row)

        # Rule-based prediction from indicators
        rsi        = float(row.get("rsi_14") or 50)
        macd       = float(row.get("macd") or 0)
        macd_sig   = float(row.get("macd_signal") or 0)
        close      = float(row.get("close") or 0)
        ema20      = float(row.get("ema_20") or close)
        ema50      = float(row.get("ema_50") or close)

        # Score: positive = bullish, negative = bearish
        score = 0
        signals = []

        if rsi < 30:
            score += 2
            signals.append("RSI oversold — strong buy signal")
        elif rsi > 70:
            score -= 2
            signals.append("RSI overbought — sell signal")
        else:
            signals.append(f"RSI neutral ({rsi:.1f})")

        if macd > macd_sig:
            score += 1
            signals.append("MACD bullish crossover")
        else:
            score -= 1
            signals.append("MACD bearish")

        if close > ema20:
            score += 1
            signals.append("Price above EMA20 — bullish")
        else:
            score -= 1
            signals.append("Price below EMA20 — bearish")

        if ema20 > ema50:
            score += 1
            signals.append("EMA20 > EMA50 — uptrend")
        else:
            score -= 1
            signals.append("EMA20 < EMA50 — downtrend")

        # Final direction
        if score >= 2:
            direction   = "UP"
            confidence  = min(0.5 + score * 0.1, 0.95)
        elif score <= -2:
            direction   = "DOWN"
            confidence  = min(0.5 + abs(score) * 0.1, 0.95)
        else:
            direction   = "NEUTRAL"
            confidence  = 0.5

        prediction = {
            "symbol":     symbol,
            "direction":  direction,
            "confidence": round(confidence, 3),
            "score":      score,
            "signals":    signals,
            "price":      round(close, 2),
            "rsi":        round(rsi, 2),
        }

        CACHE.setex(cache_key, 300, json.dumps(prediction))
        return {"source": "model", "prediction": prediction}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predictions/{symbol}/prophet")
async def get_prophet_forecast(symbol: str, days: int = 7):
    """Get Prophet forecast for next N days."""
    symbol = symbol.upper()
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT time, close FROM stock_prices
                WHERE symbol = %s ORDER BY time DESC LIMIT 500
            """, (symbol,))
            rows = cur.fetchall()
        conn.close()

        if len(rows) < 50:
            raise HTTPException(status_code=400, detail="Not enough data for forecast")

        df = pd.DataFrame([dict(r) for r in rows])
        df = df.rename(columns={"time": "ds", "close": "y"})
        df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
        df = df.sort_values("ds")

        from prophet import Prophet
        model  = Prophet(daily_seasonality=True, weekly_seasonality=True)
        model.fit(df)

        future   = model.make_future_dataframe(periods=days, freq="1D")
        forecast = model.predict(future)
        result   = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(days)

        return {
            "symbol":   symbol,
            "horizon":  days,
            "forecast": result.to_dict(orient="records")
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
