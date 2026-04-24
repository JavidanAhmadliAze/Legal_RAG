from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

from fastapi import FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from src.agent.graph import build_graph
from src.api.routers import chat
from src.db.session import create_tables
from src.monitoring.tracing import init as init_tracing

# LangGraph's PostgresSaver uses psycopg3 — no +asyncpg driver prefix
_CHECKPOINT_DSN = os.getenv(
    "CHECKPOINT_DSN",
    "postgresql://legal:legal@localhost:5433/legal_rag",
)


async def _warmup_models() -> None:
    """Load embedding model and reranker into memory before the first request."""
    from src.index.embeddings import embed_query
    from src.rag.reranker import rerank
    vec = await asyncio.to_thread(embed_query, "warmup")
    await asyncio.to_thread(rerank, "warmup", [{"text": "warmup"}], top_k=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_tracing(project_name="legal-rag", port=6006)
    await create_tables()
    await _warmup_models()
    async with AsyncPostgresSaver.from_conn_string(_CHECKPOINT_DSN) as checkpointer:
        await checkpointer.setup()   # creates LangGraph checkpoint tables if absent
        app.state.graph = build_graph(checkpointer)
        yield


app = FastAPI(title="Legal RAG API", version="0.1.0", lifespan=lifespan)
app.include_router(chat.router)
