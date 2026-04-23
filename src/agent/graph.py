from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.agent.nodes import generator, guardian, retriever, write_research_brief
from src.agent.state import AgentState


def build_graph(checkpointer):
    """
    Pipeline:
      START → guardian ──(blocked)──► END
                       └──(pass)──► write_research_brief → retriever → generator → END
    """
    graph = StateGraph(AgentState)

    graph.add_node("guardian", guardian)
    graph.add_node("write_research_brief", write_research_brief)
    graph.add_node("retriever", retriever)
    graph.add_node("generator", generator)

    graph.add_edge(START, "guardian")
    graph.add_edge("write_research_brief", "retriever")
    graph.add_edge("retriever", "generator")
    graph.add_edge("generator", END)

    return graph.compile(checkpointer=checkpointer)
