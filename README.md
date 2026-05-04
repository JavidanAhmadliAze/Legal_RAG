# Legal RAG — Polish Immigration Law Assistant

A retrieval-augmented generation system for answering questions about Polish
immigration and foreigners law. The assistant covers residence permits,
temporary protection, border control, visas, repatriation, citizenship, and
related administrative procedures, drawing exclusively from currently valid
(`obowiązujący`) acts published in the *Dziennik Ustaw* and *Monitor Polski*.

---

## Architecture

```
User query
    │
    ▼
Guardrail              (word-count + LLM topic filter)
    │
    ▼
Query parser           (preserves Polish legal terms;
    │                   maps English phrases to canonical Polish)
    ▼
Translation            (DeepSeek → focused Polish keyword query)
    │
    ▼
Retriever
  ├── BM25  (OpenSearch polish_standard analyser)
  ├── k-NN  (multilingual-e5-large embeddings, HNSW cosine)
  └── RRF merge
    │
    ▼
Cross-encoder rerank   (sdadas/polish-reranker-bge-v2, max 1024 tokens)
    │
    ▼
Relevance threshold    (sigmoid score < threshold → OUT_OF_SCOPE)
    │
    ▼
DeepSeek LLM           (cited answer in English with Dz.U./M.P. citations)
    │
    ▼
FastAPI REST  (:8000)  +  Gradio chat UI  (:7860)
```

### Stores

| Store          | Purpose                                       |
| -------------- | --------------------------------------------- |
| OpenSearch 2.x | BM25 full-text + k-NN vector search           |
| PostgreSQL 16  | Conversation history + chunk metadata         |
| Redis 7        | Retrieval result cache (`rag:retrieve:*`)     |

### Observability

Arize Phoenix on `http://localhost:6006` — every LangChain / LangGraph /
OpenAI call is instrumented via OpenTelemetry. Custom evaluation spans
(`eval.batch`, `eval.query`, `eval.retrieve`) emit per-query Precision,
Recall, Hit, MRR, MAP, NDCG, XDCG, and Fidelity.

---

## Data scope

- **Source:** [isap.sejm.gov.pl](https://isap.sejm.gov.pl) ELI REST API
- **Journals:** *Dziennik Ustaw* (DU) and *Monitor Polski* (MP)
- **Years ingested:** 2022 – 2026
- **Status filter:** only `obowiązujący` (in force) acts are kept
- **Domain filter:** immigration, foreigners law, residence permits, visas,
  border control, Ukrainian temporary protection, repatriation, citizenship

---

## Quick start (Docker)

### Prerequisites

- Docker & Docker Compose
- A [DeepSeek](https://platform.deepseek.com) API key

### 1. Environment

Create `.env` in the project root:

```env
DEEPSEEK_API_KEY=sk-...
# optional overrides
EMBEDDING_MODEL_NAME=intfloat/multilingual-e5-large
RERANKING_MODEL_NAME=sdadas/polish-reranker-bge-v2
RERANKING_MAX_LENGTH=1024
```

### 2. Start the full stack

```bash
docker compose up -d
```

Services started: OpenSearch (`:9200`), PostgreSQL (`:5433`), Redis (`:6379`),
Phoenix (`:6006`), the REST API (`:8000`), and the Gradio UI (`:7860`).

### 3. Ingest legal acts

Ingestion is orchestrated by Airflow at
[http://localhost:8080](http://localhost:8080) (admin / admin):

```
DAG: ingest_new_acts
```

The DAG fetches new acts from ELI, parses, chunks, embeds, and indexes them
into OpenSearch + PostgreSQL.

### 4. Open the chat UI

[http://localhost:7860](http://localhost:7860)

---

## Local development

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv)
- Docker & Docker Compose (for infrastructure)

### Install

```bash
uv sync --extra index --extra ui --extra monitoring --extra agent --extra dev
```

### Run infrastructure only

```bash
docker compose up -d opensearch postgres redis phoenix
```

### Run the API and UI from source

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
API_BASE=http://localhost:8000 python -m src.api.ui
```

### Tests

```bash
pytest                                         # unit + integration
pytest tests/unit -q                           # unit only
ruff check . && ruff format --check .          # lint + format
```

---

## Evaluation pipeline

The retrieval pipeline ships with a full metric suite and a synthetic
ground-truth generator.

| Command                                             | Purpose                                                                                              |
| --------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `python -m src.eval.cli.generate_synthetic_eval`    | Auto-generate questions from indexed chunks via the LLM; writes `data/eval/ground_truth.json`        |
| `python -m src.eval.cli.build_eval_dataset`         | Curated ground-truth builder with hard-coded `QUERIES` (graded relevance, build report)              |
| `python -m src.eval.cli.eval_retrieval`             | Batch retrieval eval against the ground-truth (Precision, Recall, Hit, MRR, MAP, NDCG, XDCG, Fidelity) |
| `python -m src.eval.cli.evaluate --query "..."`     | Single-query end-to-end evaluation                                                                    |
| `python -m src.eval.cli.run_full_eval`              | Full eval; writes `data/eval/results/run_<timestamp>.json` + `latest.json`                            |

**Ingest CLI flags** (see `python -m src.services.fetcher.cli --help`):

| Flag         | Default       | Description                          |
| ------------ | ------------- | ------------------------------------ |
| `--years`    | 2022 – 2026   | Calendar years to fetch              |
| `--journals` | `DU MP`       | ELI journal codes                    |
| `--dry-run`  | off           | List matching acts without download  |

### Headline metrics (66-query synthetic eval, k=15)

| Metric           | Score   |
| ---------------- | ------- |
| Hit@15           | 0.985   |
| MRR              | 0.831   |
| NDCG@15          | 0.869   |
| MAP@15           | 0.831   |
| Recall@15        | 0.985   |
| Holes (no hit)   | 1 / 66  |

Run config: `fetch_k=50`, `RERANKING_MAX_LENGTH=1024`, `max_per_doc=5`.

---

## Project layout

```
src/
  api/                    FastAPI app, routers, schemas, Gradio UI
  db/                     PostgreSQL models and session
  eval/                   Retrieval evaluation helpers
    cli/                    generate_synthetic_eval, build_eval_dataset,
                            eval_retrieval, evaluate, run_full_eval
  io/                     Raw document storage + metadata
  sources/                ISAP ELI API client, per-domain keyword filters
  services/
    agents/               LangGraph agent (graph, nodes, chain, retriever,
                          query_parser, guardrail, prompts, state)
    cache/                Redis retrieval cache
    chunking/             Structure-aware chunker + cli
    embedding/            multilingual-e5-large client
    fetcher/              HTTP client, robots.txt, rate limiting
    indexing/             OpenSearch + Postgres bulk indexing + cli
    llm/                  DeepSeek client + prompts
    monitoring/           OpenTelemetry / Phoenix tracing
    opensearch/           Client + index settings
    pdf_parser/           PDF → clean text
    reranking/            Polish cross-encoder reranker

airflow/dags/             Ingestion orchestration
tests/
  unit/                   Fast unit tests (pure logic, mocked I/O)
  integration/            PDF parsing + IO integration tests

data/                     gitignored
  raw/                    Downloaded PDFs + metadata JSON
  parsed/                 Extracted plain text
  chunks/                 Per-document JSONL chunk files
  eval/
    ground_truth.json     Active retrieval ground truth
    results/              Per-run metric snapshots
```

---

## How retrieval works

1. **Guardrail** — rejects queries over 200 words or off-topic (non-Polish-law)
   using a fast LLM classifier.
2. **Subquestion splitting** — complex multi-part questions are broken into
   up to 3 focused retrieval units.
3. **Query parsing** — English legal phrases are mapped to canonical Polish
   (e.g. *"temporary protection"* → `ochrona czasowa`) so BM25 can match
   them exactly. Slot extraction populates soft metadata filters
   (`law_domain`, `year`, `act_type`).
4. **Translation** — DeepSeek produces a focused Polish keyword query.
5. **Hybrid retrieval** — BM25 and k-NN each return `fetch_k` candidates;
   RRF merges them. Metadata filters are applied Python-side (soft for
   `law_domain`, hard for explicit `year` / `act_type`).
6. **Cross-encoder reranking** — `sdadas/polish-reranker-bge-v2` scores
   `(Polish query, Polish chunk)` pairs at up to 1024 tokens; per-document
   diversity is bounded by `max_per_doc`.
7. **Answer generation** — DeepSeek produces a cited English answer
   referencing specific articles and Dz.U./M.P. publication numbers.

---

## Tech stack

| Component        | Library / Version                                              |
| ---------------- | -------------------------------------------------------------- |
| LLM              | DeepSeek (`deepseek-chat`) via OpenAI-compatible API           |
| Agent framework  | LangGraph + LangChain                                          |
| Embeddings       | `intfloat/multilingual-e5-large` (1024-dim, normalized)        |
| Reranker         | `sdadas/polish-reranker-bge-v2` (1024-token max length)        |
| Vector store     | OpenSearch 2.x (Lucene HNSW, cosine)                           |
| REST API         | FastAPI + Uvicorn                                              |
| Chat UI          | Gradio 4                                                       |
| Tracing          | Arize Phoenix + OpenTelemetry (LangChain & OpenAI auto-instrumentation) |
| PDF parsing      | pdfplumber                                                     |
| HTTP             | httpx                                                          |
| Checkpointing    | LangGraph PostgreSQL checkpointer                              |
| Orchestration    | Airflow (Dockerised)                                           |

---

## Guardrail behaviour

| Condition                              | Response                          |
| -------------------------------------- | --------------------------------- |
| Query > 200 words                      | Rejected with word-count message  |
| Off-topic (tax, civil, criminal law…)  | "Outside my specialisation"       |
| No relevant chunks above threshold     | Out-of-scope message              |
| Relevant chunks found                  | Cited answer                      |

---

## Continuous integration

GitHub Actions runs lint + unit tests on every push and PR — see
[`.github/workflows/ci.yml`](.github/workflows/ci.yml).
