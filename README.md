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
Cross-encoder rerank  (mmarco-mMiniLMv2-L12-H384-v1, Polish–Polish)
    │
    ▼
Relevance threshold filter  (score < 1.5 → OUT_OF_SCOPE)
    │
    ▼
DeepSeek LLM  ──►  answer with legal citations
    │
    ▼
Gradio chat UI  (port 7860)
```

**Stores**

| Store | Purpose |
|---|---|
| OpenSearch 2.13 | BM25 full-text + k-NN vector search |
| PostgreSQL 16 | Document and chunk metadata |
| Redis 7 | Retrieval result cache |

**Observability**: Phoenix / OpenTelemetry traces on port 6006.

---

## Data scope

- **Source**: [isap.sejm.gov.pl](https://isap.sejm.gov.pl) via the public ELI REST API
- **Journals**: Dziennik Ustaw (DU) and Monitor Polski (MP)
- **Years ingested**: 2024–2026
- **Status filter**: only `obowiązujący` (currently in force) acts are ingested; expired or superseded acts are rejected at ingest time
- **Domain filter**: immigration, foreigners law, residence permits, visas, border control, Ukrainian temporary protection, repatriation, citizenship

---

## Setup

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- A [DeepSeek](https://platform.deepseek.com) API key

### 1. Clone and install

```bash
git clone <repo>
cd Legal_RAG
python -m venv .venv && source .venv/bin/activate
pip install -e ".[index,ui,monitoring]"
```

### 2. Environment

Create `.env` in the project root:

```env
DEEPSEEK_API_KEY=sk-...
```

### 3. Start infrastructure

```bash
docker compose up -d
```

Starts OpenSearch (`:9200`), PostgreSQL (`:5433`), and Redis (`:6379`). All three report healthy before the ingestion pipeline can run.

### 4. Ingest legal acts

```bash
python scripts/ingest_and_index_foreign.py --years 2024 2025 2026 --journals DU MP
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--years` | 2022–2026 | Calendar years to fetch |
| `--journals` | `DU MP` | ELI journal codes |
| `--dry-run` | off | List matching acts without downloading |

The pipeline fetches PDF → parses text → chunks by page → embeds (multilingual-e5-base) → indexes into OpenSearch + PostgreSQL.

### 5. Start the chat UI

```bash
python scripts/chat_ui.py
```

Open [http://localhost:7860](http://localhost:7860).  
Phoenix traces: [http://localhost:6006](http://localhost:6006).

---

## Project layout

```
src/
  sources/        # ISAP ELI API client, keyword filters per domain
  fetch/          # HTTP client, robots.txt compliance, rate limiting
  parse/          # PDF → clean text
  chunk/          # Page-based chunker (preserves article anchors)
  index/          # OpenSearch client, PostgreSQL client, embeddings
  io/             # Raw document storage and metadata
  cache/          # Redis retrieval cache
  rag/            # Chain, retriever, reranker, query parser, guardrail
  monitoring/     # OpenTelemetry / Phoenix tracing

scripts/
  ingest_and_index_foreign.py   # Main ingestion pipeline
  chat_ui.py                    # Gradio chat interface
  ingest_specific.py            # One-off ingest for a single act
  evaluate.py                   # Retrieval evaluation

data/             # gitignored
  raw/            # Downloaded PDFs + metadata JSON
  parsed/         # Extracted plain text
  chunks/         # Per-document JSONL chunk files
```

---

## How retrieval works

Each user query goes through several stages before the LLM sees any context:

1. **Guardrail** — rejects queries over 120 words or off-topic (non-Polish-law) using a fast LLM classifier.
2. **Subquestion splitting** — complex multi-part questions are broken into up to 3 focused retrieval units.
3. **Query parsing** — English legal phrases are mapped to their canonical Polish equivalents (e.g. `"temporary protection"` → `ochrona czasowa`, `"permanent residence permit"` → `zezwolenie na pobyt stały`) so BM25 can match them exactly against Polish legal text.
4. **Translation** — DeepSeek generates a short Polish keyword query (5–12 words) for each unit; preserved terms are appended if the translator omits them.
5. **Hybrid retrieval** — BM25 and k-NN each return up to 20 candidates; Reciprocal Rank Fusion merges them.
6. **Cross-encoder reranking** — the multilingual cross-encoder scores each `(Polish query, Polish chunk)` pair. Chunks scoring below **1.5** are treated as irrelevant and the system returns an out-of-scope signal instead of a guess.
7. **Answer generation** — DeepSeek produces a cited answer in English, referencing specific articles and Dz.U. publication numbers.

---

## Guardrail behaviour

| Condition | Response |
|---|---|
| Query > 120 words | Rejected with word-count message |
| Off-topic (tax, labour, civil law, etc.) | "Outside my specialisation" |
| No relevant chunks found (score < 1.5) | Same out-of-scope message |
| Relevant chunks found | Answer with legal citations |

---

## Adding new acts manually

To ingest a single act by its ELI address (e.g. `WDU20260000203`):

```bash
python scripts/ingest_specific.py WDU20260000203
```

---

## Tech stack

| Component | Library |
|---|---|
| LLM | DeepSeek via OpenAI-compatible API (`langchain-openai`) |
| Embeddings | `intfloat/multilingual-e5-base` |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| Vector store | OpenSearch 2.13 (Lucene HNSW) |
| Chat UI | Gradio 4 |
| Tracing | Arize Phoenix + OpenTelemetry |
| PDF parsing | pdfplumber |
| HTTP | httpx |
