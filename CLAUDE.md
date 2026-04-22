# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

Ingestion pipeline for Polish-language documents from trusted Polish sources.
Current scope is intentionally narrow: **download documents, then chunk them.**
Nothing downstream (embeddings, vector DB, retrieval, UI) is in scope yet — do
not add it unless asked.

## Scope

In scope:
- Fetching documents from trusted Polish websites (see "Sources" below).
- Persisting the raw documents locally (original format + metadata).
- Splitting documents into chunks suitable for later embedding.

Out of scope (do not build yet):
- Embedding generation
- Vector store / retrieval
- RAG / chat / UI
- Auth, multi-user features
- Any non-Polish sources

## Sources

Only trusted, official Polish sources. Examples of the kind of domain that
qualifies:
- `isap.sejm.gov.pl` — legal acts
- `sejm.gov.pl` — parliament
- `gov.pl` and ministry subdomains
- `stat.gov.pl` — GUS statistics
- `nbp.pl` — central bank

Before adding a new source, confirm with the user that it counts as "trusted."
Respect `robots.txt` and reasonable rate limits. Save the source URL and fetch
timestamp as metadata for every document.

## Ingestion modes

Two supported modes — pick based on the task:

1. **One-off / manual** — a small script that takes a URL or list of URLs and
   downloads + chunks them. Good default for experimentation and single docs.
2. **Airflow DAG** — for scheduled or bulk ingestion. Use only when the user
   explicitly asks for orchestration, or when a batch clearly justifies it.

Keep the fetching, parsing, and chunking logic as plain Python functions so the
same code can be called from either a CLI script or an Airflow task. Do not
couple business logic to Airflow.

## Chunking

Chunking is a first-class step, not an afterthought. Keep it swappable — the
chunker should be a function/class that takes text (+ optional metadata) and
returns a list of chunks with metadata preserved.

Defaults unless the user says otherwise:
- Work on cleaned plain text (strip HTML/PDF artifacts first).
- Preserve document structure when available (headings, articles, paragraphs,
  for legal acts: `art.`, `§`, `ust.`, `pkt`).
- Keep Polish diacritics intact — verify encoding end-to-end (UTF-8).
- Each chunk carries: source URL, document id, position/index, and any
  structural anchor (e.g. article number).

Ask the user before picking chunk size / overlap — the right values depend on
what the chunks will be used for, which isn't decided yet.

## Language notes

- All document text is Polish. Tokenization, sentence splitting, and any
  future NLP must handle Polish correctly (diacritics, inflection).
- Prefer libraries with explicit Polish support (e.g. spaCy `pl_core_news_*`,
  stanza) over English-only defaults.

## Tech preferences

- Python 3.11+
- Keep dependencies minimal at this stage. Add a library only when it replaces
  meaningful custom code.
- For PDFs: prefer `pypdf` or `pdfplumber`; for HTML: `httpx` + `selectolax`
  or `beautifulsoup4`. Confirm before introducing heavier stacks.
- Formatting/linting: `ruff`. Type hints on public functions.

## Repo conventions

Suggested layout (create as needed, don't scaffold empty dirs):
```
src/
  sources/        # one module per source site
  fetch/          # HTTP, retries, robots.txt
  parse/          # HTML/PDF -> clean text
  chunk/          # chunking strategies
  io/             # local persistence + metadata
scripts/          # one-off CLI entry points
dags/             # Airflow DAGs (only if/when used)
data/
  raw/            # original downloads (gitignored)
  chunks/         # chunked output (gitignored)
tests/
```

## Working rules for Claude

- Stay inside current scope. If a change implies embeddings, retrieval, or UI,
  stop and ask first.
- Don't invent source URLs. If unsure whether a site is trusted or what
  endpoint to hit, ask.
- Don't pick chunking parameters unilaterally — ask or mark clearly as a
  placeholder.
- Keep fetch, parse, and chunk steps independently testable.
- Never commit downloaded documents or scraped data to git.