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

import os

import phoenix as px
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter


def init(project_name: str = "legal-rag", port: int = 6006) -> None:
    """Instrument LangChain + OpenAI and connect to Phoenix.

    If PHOENIX_HOST is set (e.g. in Docker), connect to that external Phoenix
    instance.  Otherwise launch a local Phoenix server on *port*.
    """
    phoenix_host = os.environ.get("PHOENIX_HOST", "")
    if phoenix_host:
        endpoint = f"http://{phoenix_host}:{port}/v1/traces"
        print(f"Phoenix → {endpoint}")
    else:
        os.environ.setdefault("PHOENIX_PORT", str(port))
        px.launch_app()
        endpoint = f"http://localhost:{port}/v1/traces"
        print(f"Phoenix UI → http://localhost:{port}")

    exporter = OTLPSpanExporter(endpoint=endpoint)
    tracer_provider = TracerProvider()
    # BatchSpanProcessor exports in a background thread — never blocks the event loop.
    tracer_provider.add_span_processor(BatchSpanProcessor(exporter))

    from opentelemetry import trace as otel_trace
    otel_trace.set_tracer_provider(tracer_provider)

    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)


def get_tracer(name: str = "legal-rag.retriever"):
    """Return an OTel tracer for manual span instrumentation."""
    from opentelemetry import trace
    return trace.get_tracer(name)
