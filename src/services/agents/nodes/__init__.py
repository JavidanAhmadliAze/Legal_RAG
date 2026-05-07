"""
LangGraph nodes for the legal-RAG agent.

Each node is a small async function that takes the AgentState and returns
either a Command(goto=..., update={...}) or a dict update. The current graph
([../graph.py](../graph.py)) wires:

    START → guardian → write_research_brief → retrieval_supervisor → generator → END

`clarify_with_user` exists but is not currently wired into the graph; it is
exported here so it can be re-introduced as a conditional edge without
moving code.
"""

from src.services.agents.nodes.clarify import clarify_with_user
from src.services.agents.nodes.generator import generator
from src.services.agents.nodes.guardrail import guardian
from src.services.agents.nodes.research_brief import write_research_brief
from src.services.agents.nodes.retrieval_supervisor import retrieval_supervisor

__all__ = [
    "clarify_with_user",
    "generator",
    "guardian",
    "retrieval_supervisor",
    "write_research_brief",
]
