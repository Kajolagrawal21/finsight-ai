"""
FinSight AI — Individual Agents
Each agent is a focused specialist. They read the shared state,
do their job, and write their findings back to state.

Agents:
- RouterAgent:     decides which agents to call based on the question
- NewsAgent:       retrieves and summarizes relevant news
- MLAgent:         fetches ML model predictions from TimescaleDB
- RiskAgent:       assesses risk based on volatility and indicators
- PortfolioAgent:  gives portfolio-level suggestions
- SynthesizerAgent: combines all agent outputs into a final answer
"""

import logging
import psycopg2
import psycopg2.extras
from langchain_ollama import OllamaLLM
from agents.state import AgentState
from rag.retriever import retrieve_relevant_news

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("finsight.agents")

OLLAMA_MODEL = "llama3"

DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "dbname":   "finsight",
    "user":     "finsight_user",
    "password": "finsight_pass",
}


def get_llm():
    return OllamaLLM(model=OLLAMA_MODEL, temperature=0.1)


# ─── Router Agent ─────────────────────────────────────────────────────────────

def router_agent(state: AgentState) -> AgentState:
    """
    Decides which agents to call based on the question.
    Like a dispatcher — reads the question and routes to right specialists.
    """
    logger.info("🔀 Router Agent: analyzing question...")
    question = state["question"].lower()

    agents = []

    # Always get news context
    agents.append("news")

    # ML predictions needed?
    if any(w in question for w in ["predict", "price", "forecast", "target", "direction", "up", "down", "buy", "sell"]):
        agents.append("ml")

    # Risk assessment needed?
    if any(w in question for w in ["risk", "volatile", "safe", "dangerous", "loss", "drawdown", "worry"]):
        agents.append("risk")

    # Portfolio advice needed?
    if any(w in question for w in ["portfolio", "invest", "allocate", "position", "weight", "hold", "diversif"]):
        agents.append("portfolio")

    # Default: if question is about a specific stock, run ml + risk
    if state.get("symbol") and len(agents) == 1:
        agents.extend(["ml", "risk"])

    logger.info(f"🔀 Router decided: {agents}")
    return {**state, "agents_to_run": agents, "agents_done": []}


# ─── News Agent ───────────────────────────────────────────────────────────────

def news_agent(state: AgentState) -> AgentState:
    """
    Retrieves relevant news and generates a summary using RAG.
    """
    logger.info("📰 News Agent: retrieving articles...")

    question = state["question"]
    symbol   = state.get("symbol")

    try:
        articles = retrieve_relevant_news(question, symbol=symbol, top_k=5)

        if not articles:
            news_context = "No recent news found for this query."
        else:
            # Summarize with LLM
            context = "\n\n".join([
                f"[{a['source']}] {a['title']}\n{a['summary']}"
                for a in articles[:3]
            ])

            llm    = get_llm()
            prompt = f"""Summarize these news articles about {symbol or 'the market'} in 3 bullet points.
Focus on what's most relevant to: "{question}"

Articles:
{context}

Summary (3 bullet points):"""

            summary      = llm.invoke(prompt)
            news_context = f"📰 News Summary:\n{summary}"

        agents_done = state.get("agents_done", []) + ["news"]
        logger.info("✅ News Agent complete")

        return {
            **state,
            "news_context": news_context,
            "sources":      articles,
            "agents_done":  agents_done,
        }

    except Exception as e:
        logger.error(f"❌ News Agent error: {e}")
        return {**state, "news_context": f"News unavailable: {e}",
                "agents_done": state.get("agents_done", []) + ["news"]}


# ─── ML Agent ─────────────────────────────────────────────────────────────────

def ml_agent(state: AgentState) -> AgentState:
    """
    Fetches latest technical indicators and generates ML-based analysis.
    """
    logger.info("🤖 ML Agent: fetching indicators...")

    symbol = state.get("symbol")
    if not symbol:
        return {**state, "ml_analysis": "No symbol specified for ML analysis.",
                "agents_done": state.get("agents_done", []) + ["ml"]}

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:

            # Get latest indicators
            cur.execute("""
                SELECT rsi_14, macd, macd_signal, bb_upper, bb_lower,
                       ema_20, ema_50, atr_14
                FROM technical_indicators
                WHERE symbol = %s
                ORDER BY time DESC
                LIMIT 1
            """, (symbol,))
            indicators = cur.fetchone()

            # Get latest price
            cur.execute("""
                SELECT close, volume, time
                FROM stock_prices
                WHERE symbol = %s
                ORDER BY time DESC
                LIMIT 1
            """, (symbol,))
            price_row = cur.fetchone()

        conn.close()

        if not indicators or not price_row:
            return {**state,
                    "ml_analysis": f"No data available for {symbol}",
                    "agents_done": state.get("agents_done", []) + ["ml"]}

        # Interpret indicators
        rsi   = float(indicators["rsi_14"] or 50)
        macd  = float(indicators["macd"] or 0)
        macd_signal = float(indicators["macd_signal"] or 0)
        close = float(price_row["close"])
        ema20 = float(indicators["ema_20"] or close)
        ema50 = float(indicators["ema_50"] or close)

        # Simple rule-based signals from indicators
        signals = []
        if rsi > 70:
            signals.append("⚠️ RSI overbought (>70) — potential pullback")
        elif rsi < 30:
            signals.append("✅ RSI oversold (<30) — potential bounce")
        else:
            signals.append(f"✅ RSI neutral ({rsi:.1f})")

        if macd > macd_signal:
            signals.append("✅ MACD bullish crossover — upward momentum")
        else:
            signals.append("⚠️ MACD bearish — downward momentum")

        if close > ema20 > ema50:
            signals.append("✅ Price above EMA20 > EMA50 — strong uptrend")
        elif close < ema20 < ema50:
            signals.append("⚠️ Price below EMA20 < EMA50 — downtrend")
        else:
            signals.append("➡️ Mixed EMA signals — sideways trend")

        ml_analysis = f"""📊 Technical Analysis for {symbol}:
Current Price: ${close:.2f}
RSI(14): {rsi:.1f}
MACD: {macd:.4f} vs Signal: {macd_signal:.4f}
EMA20: ${ema20:.2f} | EMA50: ${ema50:.2f}

Signals:
""" + "\n".join(signals)

        agents_done = state.get("agents_done", []) + ["ml"]
        logger.info("✅ ML Agent complete")

        return {**state, "ml_analysis": ml_analysis, "agents_done": agents_done}

    except Exception as e:
        logger.error(f"❌ ML Agent error: {e}")
        return {**state, "ml_analysis": f"ML analysis error: {e}",
                "agents_done": state.get("agents_done", []) + ["ml"]}


# ─── Risk Agent ───────────────────────────────────────────────────────────────

def risk_agent(state: AgentState) -> AgentState:
    """
    Assesses risk using ATR, Bollinger Bands, and price volatility.
    """
    logger.info("⚠️ Risk Agent: assessing risk...")

    symbol = state.get("symbol")
    if not symbol:
        return {**state, "risk_analysis": "No symbol for risk analysis.",
                "agents_done": state.get("agents_done", []) + ["risk"]}

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT i.atr_14, i.bb_upper, i.bb_lower, i.bb_middle,
                       s.close
                FROM technical_indicators i
                JOIN stock_prices s ON i.time = s.time AND i.symbol = s.symbol
                WHERE i.symbol = %s
                ORDER BY i.time DESC
                LIMIT 1
            """, (symbol,))
            row = cur.fetchone()

            # Get 5-day price range for volatility
            cur.execute("""
                SELECT MAX(high) as week_high, MIN(low) as week_low,
                       STDDEV(close) as price_stddev
                FROM stock_prices
                WHERE symbol = %s
                AND time > NOW() - INTERVAL '5 days'
            """, (symbol,))
            vol_row = cur.fetchone()

        conn.close()

        if not row:
            return {**state, "risk_analysis": f"No risk data for {symbol}",
                    "agents_done": state.get("agents_done", []) + ["risk"]}

        close     = float(row["close"])
        atr       = float(row["atr_14"] or 0)
        bb_upper  = float(row["bb_upper"] or close * 1.02)
        bb_lower  = float(row["bb_lower"] or close * 0.98)
        bb_middle = float(row["bb_middle"] or close)

        atr_pct   = (atr / close) * 100
        bb_width  = ((bb_upper - bb_lower) / bb_middle) * 100

        # Risk level
        if atr_pct > 3:
            risk_level = "🔴 HIGH RISK"
        elif atr_pct > 1.5:
            risk_level = "🟡 MEDIUM RISK"
        else:
            risk_level = "🟢 LOW RISK"

        # BB position
        if close > bb_upper * 0.98:
            bb_signal = "⚠️ Near upper Bollinger Band — overbought zone"
        elif close < bb_lower * 1.02:
            bb_signal = "✅ Near lower Bollinger Band — oversold zone"
        else:
            bb_signal = "✅ Price within Bollinger Bands — normal range"

        week_high = float(vol_row["week_high"] or close) if vol_row else close
        week_low  = float(vol_row["week_low"] or close) if vol_row else close

        risk_analysis = f"""⚠️ Risk Assessment for {symbol}:
Risk Level: {risk_level}
ATR(14): ${atr:.2f} ({atr_pct:.2f}% of price)
Bollinger Band Width: {bb_width:.2f}%
5-Day Range: ${week_low:.2f} — ${week_high:.2f}
{bb_signal}

Stop Loss Suggestion: ${(close - 2 * atr):.2f} (2x ATR below current price)
Take Profit Suggestion: ${(close + 3 * atr):.2f} (3x ATR above current price)"""

        agents_done = state.get("agents_done", []) + ["risk"]
        logger.info("✅ Risk Agent complete")

        return {**state, "risk_analysis": risk_analysis, "agents_done": agents_done}

    except Exception as e:
        logger.error(f"❌ Risk Agent error: {e}")
        return {**state, "risk_analysis": f"Risk analysis error: {e}",
                "agents_done": state.get("agents_done", []) + ["risk"]}


# ─── Portfolio Agent ───────────────────────────────────────────────────────────

def portfolio_agent(state: AgentState) -> AgentState:
    """
    Gives portfolio-level context and position sizing suggestions.
    """
    logger.info("💼 Portfolio Agent: generating suggestions...")

    symbol = state.get("symbol", "the market")

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Get all symbols performance for context
            cur.execute("""
                SELECT symbol,
                       FIRST(close, time) as open_price,
                       LAST(close, time) as close_price
                FROM stock_prices
                WHERE time > NOW() - INTERVAL '1 day'
                GROUP BY symbol
                ORDER BY symbol
            """)
            perf = cur.fetchall()
        conn.close()

        if perf:
            performers = []
            for p in perf:
                chg = ((float(p["close_price"]) - float(p["open_price"])) /
                       float(p["open_price"]) * 100)
                performers.append(f"{p['symbol']}: {chg:+.2f}%")

            portfolio_tips = f"""💼 Portfolio Context:
Today's Performance:
{chr(10).join(performers)}

Suggestions for {symbol}:
• Consider position sizing: risk no more than 1-2% of portfolio per trade
• Diversify across sectors — don't over-concentrate in tech
• Use the stop loss levels from risk analysis to protect capital
• Review correlation with SPY before adding new positions"""
        else:
            portfolio_tips = f"""💼 Portfolio Suggestions for {symbol}:
• Risk no more than 1-2% of total portfolio per position
• Consider dollar-cost averaging for volatile stocks
• Balance individual stock picks with SPY/index exposure"""

        agents_done = state.get("agents_done", []) + ["portfolio"]
        logger.info("✅ Portfolio Agent complete")

        return {**state, "portfolio_tips": portfolio_tips, "agents_done": agents_done}

    except Exception as e:
        logger.error(f"❌ Portfolio Agent error: {e}")
        return {**state, "portfolio_tips": f"Portfolio analysis error: {e}",
                "agents_done": state.get("agents_done", []) + ["portfolio"]}


# ─── Synthesizer Agent ────────────────────────────────────────────────────────

def synthesizer_agent(state: AgentState) -> AgentState:
    """
    Combines all agent outputs into one coherent final answer.
    Like a chief analyst who reads all reports and writes the summary.
    """
    logger.info("🧠 Synthesizer Agent: combining insights...")

    # Collect all available context
    sections = []

    if state.get("news_context"):
        sections.append(state["news_context"])
    if state.get("ml_analysis"):
        sections.append(state["ml_analysis"])
    if state.get("risk_analysis"):
        sections.append(state["risk_analysis"])
    if state.get("portfolio_tips"):
        sections.append(state["portfolio_tips"])

    combined_context = "\n\n".join(sections)
    symbol   = state.get("symbol", "the market")
    question = state["question"]

    try:
        llm    = get_llm()
        prompt = f"""You are FinSight AI, a senior financial analyst.
Using the analysis below, answer the user's question clearly and concisely.
Structure your answer with key insights, be direct, and end with a clear recommendation.

Symbol: {symbol}
Question: {question}

Analysis from our systems:
{combined_context}

Final Answer (be direct, structured, and actionable):"""

        final_answer = llm.invoke(prompt)
        logger.info("✅ Synthesizer Agent complete")

    except Exception as e:
        logger.error(f"❌ Synthesizer error: {e}")
        final_answer = f"Analysis Summary:\n\n{combined_context}"

    return {**state, "final_answer": final_answer}
