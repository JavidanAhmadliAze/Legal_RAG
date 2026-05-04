from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.services.agents.nodes import (
    generator,
    guardian,
    retrieval_supervisor,
    write_research_brief,
)
from src.services.agents.state import AgentState


def build_graph(checkpointer):
    """
    Pipeline:
      START → guardian ──(blocked)──► END
                       └──(pass)──► write_research_brief
                                    → retrieval_supervisor (fan-out N sub-agents)
                                    → generator → END
    """
    graph = StateGraph(AgentState)

    graph.add_node("guardian", guardian)
    graph.add_node("write_research_brief", write_research_brief)
    graph.add_node("retrieval_supervisor", retrieval_supervisor)
    graph.add_node("generator", generator)

    graph.add_edge(START, "guardian")
    graph.add_edge("write_research_brief", "retrieval_supervisor")
    graph.add_edge("retrieval_supervisor", "generator")
    graph.add_edge("generator", END)

    return graph.compile(checkpointer=checkpointer)
