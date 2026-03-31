"""
FinSight AI — FastAPI Backend
Main application entry point.
"""

import time
import redis
import psycopg2
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from api.routes import stocks, predictions, analysis, chat, health

app = FastAPI(
    title="FinSight AI",
    description="Intelligent Stock Analysis & Prediction Platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS ────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Prometheus metrics ───────────────────────────────────────────────────────
Instrumentator().instrument(app).expose(app)

# ─── Request timing middleware ────────────────────────────────────────────────
@app.middleware("http")
async def add_process_time(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = str(time.time() - start)
    return response

# ─── Routers ─────────────────────────────────────────────────────────────────
app.include_router(health.router,       prefix="/api/v1", tags=["Health"])
app.include_router(stocks.router,       prefix="/api/v1", tags=["Stocks"])
app.include_router(predictions.router,  prefix="/api/v1", tags=["Predictions"])
app.include_router(analysis.router,     prefix="/api/v1", tags=["Analysis"])
app.include_router(chat.router,         prefix="/api/v1", tags=["Chat"])

@app.get("/")
async def root():
    return {
        "name":    "FinSight AI",
        "version": "1.0.0",
        "docs":    "/docs",
        "status":  "running"
    }
