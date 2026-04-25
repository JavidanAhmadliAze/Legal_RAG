# Legal RAG — Polish Immigration Law Assistant

A retrieval-augmented generation (RAG) system for answering questions about Polish immigration and foreigners law. The assistant covers residence permits, temporary protection, border control, visas, repatriation, and related administrative procedures, drawing exclusively from currently valid (`obowiązujący`) acts published in the Dziennik Ustaw and Monitor Polski.

---

## Architecture

```
User query
    │
    ▼
Guardrail (word count + LLM topic filter)
    │
    ▼
Query parser  ──►  preserves Polish legal terms, maps English phrases
    │                to canonical Polish equivalents
    ▼
Translation   ──►  DeepSeek translates query to Polish keyword form
    │
    ▼
Retriever
  ├── BM25 (OpenSearch `polish_standard` analyser)
  ├── k-NN  (multilingual-e5-base embeddings, HNSW cosine)
  └── RRF merge
    │
    ▼
Cross-encoder rerank  (sdadas/polish-reranker-roberta-v3, Polish–Polish)
    │
    ▼
Relevance threshold filter  (score < 1.5 → OUT_OF_SCOPE)
    │
    ▼
DeepSeek LLM  ──►  answer with legal citations
    │
    ▼
FastAPI REST  (port 8000)  +  Gradio chat UI  (port 7860)
```

**Stores**

| Store | Purpose |
|---|---|
| OpenSearch 2.13 | BM25 full-text + k-NN vector search |
| PostgreSQL 16 | Conversation history + chunk metadata |
| Redis 7 | Retrieval result cache |

**Observability**: Arize Phoenix traces on port 6006.

---

## Data scope

- **Source**: [isap.sejm.gov.pl](https://isap.sejm.gov.pl) via the public ELI REST API
- **Journals**: Dziennik Ustaw (DU) and Monitor Polski (MP)
- **Years ingested**: 2024–2026
- **Status filter**: only `obowiązujący` (currently in force) acts are ingested
- **Domain filter**: immigration, foreigners law, residence permits, visas, border control, Ukrainian temporary protection, repatriation, citizenship

---

## Quick start (Docker)

### Prerequisites

- Docker & Docker Compose
- A [DeepSeek](https://platform.deepseek.com) API key

### 1. Environment

Create `.env` in the project root:

```env
DEEPSEEK_API_KEY=sk-...
```

### 2. Start all services

```bash
docker compose up -d
```

This starts the full stack: OpenSearch (`:9200`), PostgreSQL (`:5433`), Redis (`:6379`), Phoenix (`:6006`), the REST API (`:8000`), and the Gradio UI (`:7860`).

### 3. Ingest legal acts

```bash
docker compose exec api python scripts/ingest_and_index_foreign.py --years 2024 2025 2026 --journals DU MP
```

### 4. Open the chat UI

[http://localhost:7860](http://localhost:7860)

---

## Local development setup

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv)
- Docker & Docker Compose (for infrastructure)

### 1. Install dependencies

```bash
uv sync --extra index --extra ui --extra monitoring --extra agent
```

### 2. Start infrastructure only

```bash
docker compose up -d opensearch postgres redis phoenix
```

### 3. Start the API

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### 4. Start the Gradio UI

```bash
API_BASE=http://localhost:8000 python scripts/gradio_ui.py
```

---

## Ingestion scripts

| Script | Purpose |
|---|---|
| `scripts/ingest_and_index_foreign.py` | Main pipeline — fetch, parse, chunk, embed, index |
| `scripts/ingest_specific.py` | Ingest a single act by ELI id (e.g. `WDU20260000203`) |
| `scripts/chunk_docs.py` | Re-chunk already-downloaded documents |
| `scripts/index_chunks.py` | Re-index already-chunked documents into OpenSearch |
| `scripts/eval_retrieval.py` | Retrieval evaluation |

**Ingest options:**

| Flag | Default | Description |
|---|---|---|
| `--years` | 2022–2026 | Calendar years to fetch |
| `--journals` | `DU MP` | ELI journal codes |
| `--dry-run` | off | List matching acts without downloading |

---

## Project layout

```
src/
  agent/          # LangGraph agent graph and nodes
  api/            # FastAPI app, routers, schemas
  cache/          # Redis retrieval cache
  chunk/          # Page-based chunker (preserves article anchors)
  db/             # PostgreSQL models and session
  eval/           # Retrieval evaluation helpers
  fetch/          # HTTP client, robots.txt compliance, rate limiting
  index/          # OpenSearch client, embeddings, indexing
  io/             # Raw document storage and metadata
  monitoring/     # OpenTelemetry / Phoenix tracing
  parse/          # PDF → clean text
  rag/            # Chain, retriever, reranker, query parser, guardrail
  sources/        # ISAP ELI API client, keyword filters per domain

scripts/          # CLI entry points (see table above)

data/             # gitignored
  raw/            # Downloaded PDFs + metadata JSON
  parsed/         # Extracted plain text
  chunks/         # Per-document JSONL chunk files
```

---

## How retrieval works

1. **Guardrail** — rejects queries over 200 words or off-topic (non-Polish-law) using a fast LLM classifier.
2. **Subquestion splitting** — complex multi-part questions are broken into up to 3 focused retrieval units.
3. **Query parsing** — English legal phrases are mapped to canonical Polish equivalents (e.g. `"temporary protection"` → `ochrona czasowa`) so BM25 can match them exactly.
4. **Translation** — DeepSeek generates a short Polish keyword query (5–12 words) per unit; preserved terms are appended if the translator omits them.
5. **Hybrid retrieval** — BM25 and k-NN each return up to 5 candidates; Reciprocal Rank Fusion merges them.
6. **Cross-encoder reranking** — `sdadas/polish-reranker-roberta-v3` scores each `(Polish query, Polish chunk)` pair. Chunks scoring below **1.5** are filtered out; if nothing passes the threshold the system returns an out-of-scope message.
7. **Answer generation** — DeepSeek produces a cited answer in English, referencing specific articles and Dz.U./M.P. publication numbers.

---

## Guardrail behaviour

| Condition | Response |
|---|---|
| Query > 200 words | Rejected with word-count message |
| Off-topic (tax, labour, civil law, etc.) | "Outside my specialisation" |
| No relevant chunks found (score < 1.5) | Out-of-scope message |
| Relevant chunks found | Answer with legal citations |

---

## Tech stack

| Component | Library / Version |
|---|---|
| LLM | DeepSeek (`deepseek-chat`) via OpenAI-compatible API |
| Agent framework | LangGraph + LangChain |
| Embeddings | `intfloat/multilingual-e5-base` (768-dim) |
| Reranker | `sdadas/polish-reranker-roberta-v3` (Polish RoBERTa cross-encoder) |
| Vector store | OpenSearch 2.13 (Lucene HNSW) |
| REST API | FastAPI + Uvicorn |
| Chat UI | Gradio 4 |
| Tracing | Arize Phoenix + OpenTelemetry |
| PDF parsing | pdfplumber |
| HTTP | httpx |
| Checkpointing | LangGraph PostgreSQL checkpointer |
