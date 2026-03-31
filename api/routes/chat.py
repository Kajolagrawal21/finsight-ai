"""
FinSight AI — Chat Routes
RAG-powered Q&A about stocks using news context.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter()


class ChatRequest(BaseModel):
    question: str
    symbol:   Optional[str] = None


@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Ask any question about stocks.
    Uses RAG to find relevant news and LLM to generate answer.
    """
    try:
        from rag.retriever import answer_question
        result = answer_question(
            question=request.question,
            symbol=request.symbol.upper() if request.symbol else None
        )
        return {
            "question":    result["question"],
            "answer":      result["answer"],
            "num_sources": result["num_sources"],
            "sources": [
                {
                    "title":  s["title"],
                    "source": s["source"],
                    "date":   s["published_at"][:10] if s["published_at"] else "",
                    "score":  s["score"],
                }
                for s in result["sources"]
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/history/{symbol}")
async def get_news_summary(symbol: str):
    """Get a quick news summary for a symbol."""
    try:
        from rag.retriever import retrieve_relevant_news
        symbol   = symbol.upper()
        articles = retrieve_relevant_news(
            f"latest news about {symbol} stock",
            symbol=symbol,
            top_k=5
        )
        return {
            "symbol":   symbol,
            "articles": articles,
            "count":    len(articles)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
