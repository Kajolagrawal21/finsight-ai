"""
FinSight AI — Analysis Routes
Runs the full LangGraph multi-agent analysis.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import redis
import json

router = APIRouter()
CACHE  = redis.Redis(host="localhost", port=6379, decode_responses=True)


class AnalysisRequest(BaseModel):
    symbol:   str
    question: Optional[str] = None


@router.post("/analyze")
async def analyze_stock(request: AnalysisRequest):
    """
    Run full multi-agent analysis for a stock.
    Uses Router → News → ML → Risk → Portfolio → Synthesizer agents.
    """
    symbol    = request.symbol.upper()
    question  = request.question or f"Give me a complete analysis of {symbol} stock"
    cache_key = f"analysis:{symbol}:{question[:50]}"

    cached = CACHE.get(cache_key)
    if cached:
        return {"source": "cache", "analysis": json.loads(cached)}

    try:
        from agents.graph import run_agent
        result = run_agent(question, symbol)

        analysis = {
            "symbol":        symbol,
            "question":      question,
            "final_answer":  result.get("final_answer", ""),
            "news_context":  result.get("news_context", ""),
            "ml_analysis":   result.get("ml_analysis", ""),
            "risk_analysis": result.get("risk_analysis", ""),
            "agents_run":    result.get("agents_done", []),
            "sources_count": len(result.get("sources", [])),
        }

        # Cache for 5 minutes
        CACHE.setex(cache_key, 300, json.dumps(analysis))
        return {"source": "agents", "analysis": analysis}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analyze/{symbol}/quick")
async def quick_analysis(symbol: str):
    """Quick technical analysis without LLM (instant response)."""
    symbol = symbol.upper()
    try:
        import psycopg2
        import psycopg2.extras

        conn = psycopg2.connect(
            host="localhost", port=5432, dbname="finsight",
            user="finsight_user", password="finsight_pass"
        )
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT i.rsi_14, i.macd, i.macd_signal, i.bb_upper,
                       i.bb_lower, i.ema_20, i.ema_50, i.atr_14, s.close
                FROM technical_indicators i
                JOIN stock_prices s ON i.time = s.time AND i.symbol = s.symbol
                WHERE i.symbol = %s ORDER BY i.time DESC LIMIT 1
            """, (symbol,))
            row = cur.fetchone()
        conn.close()

        if not row:
            raise HTTPException(404, f"No data for {symbol}")

        row   = dict(row)
        close = float(row["close"])
        rsi   = float(row["rsi_14"] or 50)

        signals = []
        if rsi > 70:   signals.append({"type": "warning", "msg": "RSI Overbought"})
        elif rsi < 30: signals.append({"type": "bullish", "msg": "RSI Oversold - Buy Signal"})
        else:          signals.append({"type": "neutral", "msg": f"RSI Neutral ({rsi:.1f})"})

        if float(row["macd"] or 0) > float(row["macd_signal"] or 0):
            signals.append({"type": "bullish", "msg": "MACD Bullish Crossover"})
        else:
            signals.append({"type": "bearish", "msg": "MACD Bearish"})

        return {
            "symbol":  symbol,
            "price":   round(close, 2),
            "rsi":     round(rsi, 2),
            "signals": signals,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
