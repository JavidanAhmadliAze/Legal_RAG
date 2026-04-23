"""
Phoenix Arize observability setup.

Starts a local Phoenix server and registers an OpenTelemetry tracer provider
so that LangChain chain calls, retrieval stages, and cross-encoder scores
are all captured as spans visible at http://localhost:6006.

Metrics surfaced per query:
  - bm25_hits / knn_hits / post_filter_hits counts
  - cross-encoder score for every reranked chunk  (proxy for precision)
  - full retrieved context sent to the LLM        (manual recall review)
  - LLM latency, token counts (via LangChain auto-instrumentation)
"""

from __future__ import annotations

import phoenix as px
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor
from phoenix.otel import register


def init(project_name: str = "legal-rag", port: int = 6006) -> None:
    """Start Phoenix server and instrument LangChain + OpenAI. Call once at startup."""
    import os
    os.environ.setdefault("PHOENIX_PORT", str(port))
    px.launch_app()

    tracer_provider = register(
        project_name=project_name,
        endpoint=f"http://localhost:{port}/v1/traces",
    )

    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
    print(f"Phoenix UI → http://localhost:{port}")


def get_tracer(name: str = "legal-rag.retriever"):
    """Return an OTel tracer for manual span instrumentation."""
    from opentelemetry import trace
    return trace.get_tracer(name)
