from __future__ import annotations

import os

import phoenix as px
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from src.services.config import get_monitoring_settings


class MonitoringClient:
    def init(
        self,
        project_name: str | None = None,
        port: int | None = None,
        *,
        instrument_langchain: bool | None = None,
        instrument_openai: bool | None = None,
    ) -> None:
        settings = get_monitoring_settings()
        selected_project_name = project_name or settings.project_name
        selected_port = settings.port if port is None else port
        use_langchain = (
            settings.instrument_langchain
            if instrument_langchain is None
            else instrument_langchain
        )
        use_openai = (
            settings.instrument_openai
            if instrument_openai is None
            else instrument_openai
        )

        phoenix_host = os.environ.get("PHOENIX_HOST", "")
        if phoenix_host:
            endpoint = f"http://{phoenix_host}:{selected_port}/v1/traces"
            print(f"Phoenix -> {endpoint}")
        else:
            os.environ.setdefault("PHOENIX_PORT", str(selected_port))
            px.launch_app()
            endpoint = f"http://localhost:{selected_port}/v1/traces"
            print(f"Phoenix UI -> http://localhost:{selected_port}")

        exporter = OTLPSpanExporter(endpoint=endpoint)
        tracer_provider = TracerProvider(
            resource=Resource.create({"service.name": selected_project_name})
        )
        tracer_provider.add_span_processor(BatchSpanProcessor(exporter))

        from opentelemetry import trace as otel_trace

        otel_trace.set_tracer_provider(tracer_provider)

        if use_langchain:
            LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
        if use_openai:
            OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

    def get_tracer(self, name: str | None = None):
        from opentelemetry import trace

        tracer_name = name or get_monitoring_settings().default_tracer_name
        return trace.get_tracer(tracer_name)
