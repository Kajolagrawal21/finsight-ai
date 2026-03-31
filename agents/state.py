"""
FinSight AI — Agent State
Defines the shared state that flows through the LangGraph agent network.
Think of it like a shared whiteboard all agents can read from and write to.
"""

from typing import TypedDict, List, Optional, Annotated
import operator


class AgentState(TypedDict):
    """
    Shared state passed between all agents in the graph.
    Each agent reads what it needs and adds its findings.
    """
    # Input
    question:       str             # user's original question
    symbol:         Optional[str]   # stock symbol if detected

    # Agent outputs (each agent fills its section)
    news_context:   Optional[str]   # news agent findings
    ml_analysis:    Optional[str]   # ML prediction agent findings
    risk_analysis:  Optional[str]   # risk agent findings
    portfolio_tips: Optional[str]   # portfolio agent suggestions

    # Routing
    agents_to_run:  List[str]       # which agents router decided to call
    agents_done:    List[str]       # which agents have completed

    # Final output
    final_answer:   Optional[str]   # synthesizer's combined answer
    sources:        List[dict]      # news sources used

    # Metadata
    error:          Optional[str]   # any error that occurred
