"""
FinSight AI — LangGraph Agent Graph
Connects all agents into a directed workflow graph.

Flow:
  START → Router → [News, ML, Risk, Portfolio] → Synthesizer → END

LangGraph is like a flowchart for AI agents.
Each node is an agent, each edge is a connection.
The graph decides which path to take based on the router's decision.
"""

import logging
from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.agents import (
    router_agent, news_agent, ml_agent,
    risk_agent, portfolio_agent, synthesizer_agent
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("finsight.graph")


def should_run_ml(state: AgentState) -> str:
    """Conditional edge — run ML agent only if router decided to."""
    if "ml" in state.get("agents_to_run", []):
        return "ml_agent"
    return "risk_check"


def should_run_risk(state: AgentState) -> str:
    """Conditional edge — run Risk agent only if router decided to."""
    if "risk" in state.get("agents_to_run", []):
        return "risk_agent"
    return "portfolio_check"


def should_run_portfolio(state: AgentState) -> str:
    """Conditional edge — run Portfolio agent only if router decided to."""
    if "portfolio" in state.get("agents_to_run", []):
        return "portfolio_agent"
    return "synthesizer"


def build_graph() -> StateGraph:
    """
    Build the LangGraph agent workflow.
    
    Think of it like building a flowchart:
    - Add nodes (agents)
    - Add edges (connections between agents)
    - Add conditional edges (if/else routing)
    """
    graph = StateGraph(AgentState)

    # ── Add all agent nodes ──────────────────────────────────
    graph.add_node("router",       router_agent)
    graph.add_node("news_agent",   news_agent)
    graph.add_node("ml_agent",     ml_agent)
    graph.add_node("risk_agent",   risk_agent)
    graph.add_node("portfolio_agent", portfolio_agent)
    graph.add_node("synthesizer",  synthesizer_agent)

    # ── Entry point ──────────────────────────────────────────
    graph.set_entry_point("router")

    # ── Router always goes to news first ─────────────────────
    graph.add_edge("router", "news_agent")

    # ── After news: conditionally run ML ─────────────────────
    graph.add_conditional_edges(
        "news_agent",
        should_run_ml,
        {
            "ml_agent":   "ml_agent",
            "risk_check": "risk_agent" if True else "synthesizer",
        }
    )

    # ── After ML: conditionally run Risk ─────────────────────
    graph.add_conditional_edges(
        "ml_agent",
        should_run_risk,
        {
            "risk_agent":     "risk_agent",
            "portfolio_check": "portfolio_agent",
        }
    )

    # ── After Risk: conditionally run Portfolio ───────────────
    graph.add_conditional_edges(
        "risk_agent",
        should_run_portfolio,
        {
            "portfolio_agent": "portfolio_agent",
            "synthesizer":     "synthesizer",
        }
    )

    # ── Portfolio always goes to synthesizer ─────────────────
    graph.add_edge("portfolio_agent", "synthesizer")

    # ── Synthesizer is the end ───────────────────────────────
    graph.add_edge("synthesizer", END)

    return graph.compile()


def run_agent(question: str, symbol: str = None) -> dict:
    """
    Run the full agent graph for a question.
    Returns the complete state with final answer and all agent outputs.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"🚀 Running FinSight AI Agents")
    logger.info(f"❓ Question: {question}")
    logger.info(f"📈 Symbol: {symbol}")
    logger.info(f"{'='*60}")

    # Initial state
    initial_state: AgentState = {
        "question":       question,
        "symbol":         symbol,
        "news_context":   None,
        "ml_analysis":    None,
        "risk_analysis":  None,
        "portfolio_tips": None,
        "agents_to_run":  [],
        "agents_done":    [],
        "final_answer":   None,
        "sources":        [],
        "error":          None,
    }

    graph  = build_graph()
    result = graph.invoke(initial_state)

    logger.info(f"\n✅ Agents completed: {result.get('agents_done', [])}")
    return result


if __name__ == "__main__":
    # Test the full agent graph
    test_cases = [
        ("Should I buy NVDA right now?", "NVDA"),
        ("What is the risk of holding TSLA?", "TSLA"),
        ("How is the overall market doing?", "SPY"),
    ]

    for question, symbol in test_cases:
        result = run_agent(question, symbol)
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        print(f"\n🤖 FINAL ANSWER:\n{result['final_answer']}")
        print(f"\n📰 Sources used: {len(result.get('sources', []))}")
        print(f"✅ Agents run: {result.get('agents_done', [])}")
