# arXiv RAG v2 -> v3 Progress (2026-03-13)

## Goal
Move the project from the v2 metadata/retrieval layout to the v3 layout:
- metadata: Local PostgreSQL
- vectors: Qdrant
- retrieval: Qdrant-first
- parsing/reprocessing: local scripts with explicit DB status transitions

## Completed Today

### 1. Repository / runtime baseline
- Created and switched to new GitHub repo: `arxiv-rag-v3-codex`
- Repointed `origin` to the new repo
- Brought up local `Qdrant`
- Added local PostgreSQL to Docker and connected the codebase to it
- Added local DB env defaults in `.env` / `.env.example`

### 2. Metadata backend migration
- Added local schema bootstrap: `scripts/init_local_db.sql`
- Added local PG client: `src/storage/postgres_client.py`
- Added DB factory: `src/storage/__init__.py::get_db_client()`
- Migrated `papers` metadata from Supabase into local PostgreSQL
- Current paper count in local PG: `2500`

### 3. Active codepath transition to `get_db_client()`
Active runtime / pipeline files were moved to the DB factory path so v3 runs on Local PG by default.

Representative files:
- `src/rag/api.py`
- `src/api/routes/chat.py`
- `src/api/routes/papers.py`
- `src/ui/streamlit_app.py`
- `scripts/01_collect.py`
- `scripts/02_download.py`
- `scripts/02_parse.py`
- `scripts/03_embed.py`
- `scripts/08_generate_synthetic_benchmark.py`
- `scripts/collect_extended.py`
- `scripts/compute_scores.py`
- `scripts/fetch_citations.py`
- `scripts/semantic_filter.py`
- `scripts/verify_v2_architecture.py`

Direct `get_supabase_client()` use now remains only in fallback / migration code:
- `src/storage/supabase_client.py`
- `src/storage/__init__.py` fallback branch
- `scripts/migrate_to_local_pg.py`

### 4. Legacy retrieval cleanup
- Removed active runtime dependency on legacy Supabase retrievers
- `src/rag/retriever.py` now mainly provides shared result models and legacy guardrails
- `scripts/06_evaluate.py` now validates v3 Qdrant modes only
- `scripts/tune_weights.py` now enforces Qdrant-only tuning
- `scripts/migrate_to_qdrant.py` converted into a deprecation stub for v3

### 5. Parser improvements (Phase 1)
Improved `src/parsing/latex_parser.py` with:
- macro extraction / substitution
- noisy LaTeX environment stripping
- paragraph-level LaTeX noise filtering
- safer archive detection (`tar`, `zip`, and early rejection of PDF masquerading as `.tar.gz`)

Single-paper validation was done successfully on:
- `2501.12948v2`

### 6. Reprocess pipeline (Phase 2)
Added:
- `scripts/11_reprocess_pipeline.py`

Capabilities:
- single-paper or batch reparse
- selection by DB status / file list / offset / limit
- parsed JSON backup
- parse-only or parse+embed flows
- DB status updates (`parsed`, `failed`, `embedded`)
- Qdrant/chunk reset before re-embedding

## Parsing status after today

### Full LaTeX reparse
Executed a parse-only reprocess across the LaTeX-available set.

Result:
- selected LaTeX-path papers: `2497`
- parse success: `2402`
- parse failure: `95`

### Failure analysis
The `95` parse failures are not LaTeX-parser failures.
They are all papers whose `latex_path` file is actually a `PDF` stored with a `.tar.gz` filename.

Separate existing failures:
- `3` papers have no `latex_path` at all

So the current DB state is:
- `total=2500`
- `parsed=2402`
- `failed=98`
- `embedded=0`
- `pending=0`

Interpretation:
- All papers with real LaTeX source were parsed successfully in this pass.
- Remaining failures are input-format issues, not LaTeX parser regressions.
- PDF fallback via Marker is intentionally not part of the current WSL execution path because GPU OOM already reproduces there.

## Notes on embeddings
- One earlier single-paper end-to-end test left `89` chunks in the metadata DB / Qdrant for `2501.12948v2`
- The bulk work today was parse-only, so full corpus embeddings have **not** been regenerated yet
- The next operational step is full embedding/reindexing on the parsed set

## Recommended next step
Run embedding for the parsed corpus with the v3 path:

```bash
set -a; source .env; set +a
.venv/bin/python scripts/03_embed.py --limit 2402
```

For a controlled batch rollout, use smaller batches first or drive it through:

```bash
set -a; source .env; set +a
.venv/bin/python scripts/11_reprocess_pipeline.py --limit 20 --device cpu
```

## Important artifacts
- `docs/v3_bootstrap_checklist.md`
- `docs/phase2_reprocess_execution_plan.md`
- `scripts/11_reprocess_pipeline.py`
- `src/parsing/latex_parser.py`
- `src/storage/postgres_client.py`


## TODO

### Next evaluation work
- [ ] Add `RAGAS`-based answer-quality evaluation for v3
- [ ] Build an evaluation set that measures whether the system retrieves the right evidence from the right paper, not just the right paper itself
- [ ] Generate v2 and v3 answers on the same question set and compare:
  - faithfulness
  - answer relevancy
  - context precision
  - context recall

### GraphRAG work
- [ ] Design and implement `GraphRAG` on top of the v3 Local PG + Qdrant architecture
- [ ] Add graph metadata flow for `citation_edges` and `entities`
- [ ] Decide whether graph-enriched retrieval requires payload-only refresh, reindexing, or full re-embedding

### Post-GraphRAG validation
- [ ] Create GraphRAG-focused benchmark queries
- [ ] Run comparative evaluation for:
  - current v3 baseline
  - v3 + GraphRAG
- [ ] Measure both retrieval metrics and answer-quality metrics after GraphRAG integration

### Recommended order
1. Add `RAGAS` evaluation and validate current v3 answer quality
2. Implement `GraphRAG`
3. Build GraphRAG-specific benchmark queries
4. Re-run evaluation on both retrieval quality and answer quality
