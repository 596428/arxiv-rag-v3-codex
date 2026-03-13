# arXiv RAG v3 - 파이프라인 재처리 및 고도화 계획

## Context

arXiv RAG v1의 검색 품질을 저하시키는 이슈들(`docs/code_issues.md` + `docs/improvement_plan.md` 1-2번)을 수정하고, 전체 파이프라인을 재실행한다. **v3로 버전업**하여 새 GitHub repo `arxiv-rag-v3-codex`로 이전하며, 로컬 PostgreSQL 전환과 GraphRAG 확장을 포함한다.

**현황**: 2,497 LaTeX 소스 로컬 보유 / DB: papers 2,500 + chunks 124,001 (Supabase 634MB)
**제외**: improvement_plan.md 3번(검색 로직), reranker, marker PDF 파싱, 수식 청크

### 버전 전환 전략
- **v2** (Supabase + GitHub Pages): 배포용 동결. Supabase DB 유지.
- **v3** (로컬 PG + Qdrant): 현재 디렉토리 그대로 사용. git remote만 `arxiv-rag-v3-codex`로 변경.

### code_issues.md 반영 사항
| # | 이슈 | 해결 Phase |
|---|------|-----------|
| 1 | `to_db_dict()` chunk_index/token_count top-level 누락 | Phase 2-1 |
| 2 | `get_chunks_by_paper()` 정렬이 metadata 의존 | Phase 0-4 (PG client에서 수정) |
| 3 | OpenAI 임베딩 차원 주석 1024→3072 불일치 | Phase 2-1 (주석 수정) |
| 4 | Gemini 모델명 설정/하드코딩 불일치 | Phase 0-1 (초기 정리) |

### 설계 결정 사항
| 항목 | 결정 | 근거 |
|------|------|------|
| PG 스키마 컬럼 | Supabase 동형 (벡터 컬럼만 제거) | `02_parse.py`가 `parse_status`, `latex_path` 등에 의존 |
| papers 마이그레이션 범위 | 전체 컬럼 (arxiv_id~updated_at) | 파이프라인 호환성 유지 |
| LaTeX 없는 3편 | `parse_status='failed'`로 갱신, Marker 사용하지 않음 | 제외 범위에 명시 (marker PDF 파싱 제외) |
| Legacy Supabase 벡터 검색 | `retriever.py` 전체 삭제 (6개 클래스 모두 Supabase RPC 의존) | 로컬 PG에 벡터 컬럼/RPC 없음, `qdrant_retriever.py`가 대체 |
| `api.py` legacy 분기 | `else` 분기(line 236-238) 제거, 기본값을 `adaptive`로 | 모든 검색은 Qdrant 경유 |
| 매크로 폴백 | 제거 (제목 추측 안 함) | 오탐 위험, `_resolve_inputs()`가 외부 파일 매크로 이미 커버 |
| figure/table 청크화 | v3에서는 현행 유지 (텍스트 청크만) | 향후 improvement_plan.md 3번에서 캡션 검색 추가 |
| `is_latex_noisy()` 호출 위치 | `_strip_noisy_environments()` 내부에서 단락 단위로 호출 | 파서 레이어에서 완결, chunker 수정 불필요 |
| citation_edges ID | 버전 없는 arxiv_id 사용 | `semantic_scholar.py`의 기존 로직과 일치, 그래프 분산 방지 |
| Qdrant payload 배치 업데이트 | 1,000 points/배치, 3회 재시도 (exponential backoff) | 120K points ÷ 1K = 120 배치, 타임아웃 방지 |
| 엔티티 정규화 | v3 초기: lowercase + strip만 | 매핑 테이블은 향후 확장 |
| LocalPG connection pool | `minconn=2, maxconn=10`, 연결 타임아웃 30초 | 스크립트(단일 프로세스)와 API(멀티 요청) 모두 커버 |
| 마이그레이션 인증/배치 | anon key 사용, 1,000행/배치, 배치 간 1초 대기 | Supabase 무료 tier rate limit 준수 |

---

## Phase 0: 인프라 세팅

### 0-1. 새 GitHub repo 생성 & 초기 정리
- GitHub에 `arxiv-rag-v3-codex` repo 생성 (gh CLI)
- `git remote set-url origin <new-repo-url>`
- `.env` → `.gitignore` 확인
- **code_issues #4 수정**: `src/api/routes/chat.py` line 149 `"gemini-2.0-flash"` 하드코딩 → `settings.gemini_model` 참조
- 초기 커밋 & push

### 0-2. docker-compose에 PostgreSQL 추가
- **파일**: `docker-compose.yml`
- `pgvector/pgvector:pg16` 이미지 추가
  - port: 5432, volume: `postgres_data`
  - 환경변수: `POSTGRES_DB=arxiv_rag`, `POSTGRES_USER`, `POSTGRES_PASSWORD`
  - healthcheck: `pg_isready`
- API 서비스에 `PG_*` 환경변수 전달 추가

### 0-3. DB 스키마 생성
- **신규**: `scripts/init_local_db.sql`
- **Supabase 스키마 동형** (`supabase/migrations/20260212000000_init_schema.sql` 기반):
  - papers: 전체 컬럼 유지 (arxiv_id, title, authors[], abstract, categories[], published_date, citation_count, download_count, pdf_path, latex_path, parse_status, parse_method, created_at, updated_at)
  - chunks: 벡터 컬럼 3개 제거 (embedding_dense, embedding_sparse, embedding_openai) — Qdrant가 담당. 나머지 유지 (chunk_id, paper_id, content, section_title, chunk_type, chunk_index, token_count, metadata)
  - equations/figures: 벡터 컬럼 제거, 나머지 유지
  - **추가**: `citation_edges(source_arxiv_id, target_arxiv_id, created_at)` — Phase 3용
  - **추가**: `entities(entity_id SERIAL, name, type, paper_id FK, chunk_id, metadata JSONB)` — Phase 3용
  - Supabase 전용 함수(match_chunks, match_equations, RLS) 제외
  - updated_at 트리거 유지

### 0-4. LocalPGClient 구현
- **신규**: `src/storage/postgres_client.py`
- `supabase_client.py`의 사용되는 메서드와 **동일 인터페이스** 구현:
  ```
  insert_paper, upsert_paper, get_paper, get_paper_count,
  get_papers_by_status, get_papers_for_parsing, get_papers_for_embedding,
  batch_insert_chunks_metadata, get_chunks_by_paper, get_chunks_by_ids,
  get_chunks_by_ids_ordered, delete_chunks_by_paper, get_papers_with_chunks
  ```
- `psycopg2` + connection pooling (`psycopg2.pool.SimpleConnectionPool(minconn=2, maxconn=10)`, 연결 타임아웃 30초)
- **code_issues #2 수정**: `get_chunks_by_paper()` → `ORDER BY chunk_index` (top-level 컬럼)
- **파일 수정**: `src/storage/__init__.py`
  - `get_db_client()` 팩토리 추가 (`DB_BACKEND` 환경변수: `local`|`supabase`, 기본값 `local`)
  - 기존 `get_supabase_client()` export 유지 (하위 호환)
- **의존성**: `requirements.txt`에 `psycopg2-binary>=2.9.0` 추가

### 0-5. Legacy Supabase 코드 정리 & get_db_client() 전환

#### A. `src/rag/retriever.py` — Legacy 벡터 검색 클래스 전체 삭제
- 삭제 대상 (모두 Supabase RPC `match_chunks_*` 의존, 로컬 PG에서 동작 불가):
  - `DenseRetriever` (line 49-138) — `match_chunks_dense` / `match_chunks` RPC
  - `SparseRetriever` (line 141-223) — `match_chunks_sparse` RPC
  - `HybridRetriever` (line 226-446) — Dense + Sparse 조합
  - `ColBERTRetriever` (line 448-534) — `match_chunks_colbert` RPC
  - `HybridFullRetriever` (line 537-732) — Dense + Sparse + ColBERT 조합
  - `OpenAIRetriever` (line 768+) — Supabase OpenAI 벡터 검색
  - 편의 함수 `hybrid_search`, `dense_search`, `sparse_search` (line 735-765)
- **유지**: `SearchResult`, `SearchResponse` dataclass (다른 모듈에서 import)
- 파일을 삭제하지 않고 데이터 클래스만 남김 (또는 `src/rag/models.py`로 이동)

#### B. `src/rag/api.py` — Legacy 검색 분기 제거
- line 236-238 `else` 분기 (legacy `HybridRetriever` 호출) 삭제
- `search_mode` 유효값을 `adaptive`, `qdrant_hybrid`, `dense`, `sparse`로 제한
- `get_retriever()` 함수 (line 48-53) 삭제, `_retriever` 전역변수 삭제
- `from .retriever import HybridRetriever` import 제거
- line 162, 306, 337, 383, 408, 431, 476의 `get_supabase_client()` → `get_db_client()`

#### C. LaTeX 없는 3편 처리
- `scripts/migrate_to_local_pg.py`에서 마이그레이션 후 LaTeX 미보유 3편의 `parse_status` → `'failed'` 갱신
- Marker PDF 파싱 시도하지 않음 (제외 범위)

#### D. 나머지 get_supabase_client() → get_db_client() 전환
- 대상 파일:
  ```
  # 파이프라인 스크립트 (9개)
  scripts/01_collect.py, 02_download.py, 02_parse.py, 03_embed.py,
  08_generate_synthetic_benchmark.py, collect_extended.py,
  compute_scores.py, fetch_citations.py, semantic_filter.py

  # API (2개)
  src/api/routes/chat.py (line 19), src/api/routes/papers.py

  # UI (1개)
  src/ui/streamlit_app.py

  # 유틸 (2개) — 참조만, 필요시 전환
  scripts/migrate_to_qdrant.py, scripts/verify_v2_architecture.py
  ```
- `qdrant_retriever.py`는 Supabase 무의존 → **전환 불필요**
- import 변경: `from ..storage.supabase_client import get_supabase_client` → `from ..storage import get_db_client`

### 0-6. papers 데이터 마이그레이션
- **신규**: `scripts/migrate_to_local_pg.py`
- Supabase REST API (anon key) 로 papers 전체 컬럼 2,500행 export → 로컬 PG import
- **배치**: 1,000행/배치, 배치 간 1초 대기 (Supabase 무료 tier rate limit 준수)
- 마이그레이션 컬럼: `arxiv_id, title, authors, abstract, categories, published_date, citation_count, download_count, pdf_path, latex_path, parse_status, parse_method, created_at, updated_at`
- 마이그레이션 후 LaTeX 미보유 3편 `parse_status` → `'failed'` 갱신
- chunks/equations/figures는 Phase 2에서 재생성하므로 마이그레이션 불필요

### 검증
```bash
docker-compose up -d postgres qdrant
python3 -c "from src.storage import get_db_client; c = get_db_client(); print(c.get_paper_count())"
# → 2500
```

---

## Phase 1: 파서 수정

### 1-1. LaTeX 매크로 치환 (CRITICAL)
- **파일**: `src/parsing/latex_parser.py`
- **적용 위치**: `_parse_tex_file()` 내부 (line 207-252)
  ```python
  # 현재 flow (line 218-221):
  content = self._resolve_inputs(content, tex_path.parent)
  body = self._extract_document_body(content)

  # 수정 flow:
  content = self._resolve_inputs(content, tex_path.parent)  # line 218
  macros = self._extract_macros(content)                     # ← 신규
  content = self._apply_macros(content, macros)              # ← 신규
  body = self._extract_document_body(content)                # line 221
  ```
- 신규 메서드 `_extract_macros(content) -> dict[str, tuple[str, int]]`:
  - 대상: `\newcommand`, `\renewcommand`, `\providecommand`, `\def`, `\DeclareMathOperator`
  - 반환: `{macro_name: (replacement_body, num_args)}`
  - 0-arg와 N-arg 매크로 구분
- 신규 메서드 `_apply_macros(content, macros) -> str`:
  - 0-arg: 단순 문자열 치환
  - N-arg: `\name{arg1}{arg2}` 패턴 매칭 → `#1`, `#2` 대입 (balanced brace 파싱 — `_extract_title()`의 기존 패턴 재사용)
  - 중첩 매크로: 최대 3회 반복 치환
- **폴백 없음**: 매크로 정의 미발견 시 치환하지 않음 (`_resolve_inputs()`가 외부 파일 매크로 이미 커버)

### 1-2. LaTeX 원시 코드 필터링 (MEDIUM)
- **파일**: `src/parsing/latex_parser.py`
- **적용 위치**: `_parse_tex_file()` 내부, body 추출 후 섹션 파싱 전 (line 221-237)
- figure/table **메타데이터 추출은 유지** — 필터링 전 body에서 수행
  ```python
  # 수정된 flow:
  body = self._extract_document_body(content)
  equations = self._extract_equations(body, arxiv_id)      # 필터링 전
  figures = self._extract_figures(body, arxiv_id, ...)     # 필터링 전
  tables = self._extract_tables(body, arxiv_id)            # 필터링 전
  body_clean = self._strip_noisy_environments(body)        # ← 신규
  sections = self._parse_sections(body_clean, arxiv_id)    # 필터링 후 텍스트만
  ```
- 신규 메서드 `_strip_noisy_environments(body) -> str`:
  - `re.DOTALL`로 `\\begin{ENV}...\\end{ENV}` 제거
  - 대상 환경:
    ```
    tikzpicture, pgfpicture, pspicture,
    algorithm, algorithmic, algorithmicx,
    lstlisting, minted, verbatim,
    figure, figure*, wrapfigure,
    table, table*
    ```
- 신규 함수 `is_latex_noisy(text, threshold=0.3) -> bool`:
  - backslash 명령어 비율 30% 초과 시 True → 해당 단락 제외
  - **호출 위치**: `_strip_noisy_environments()` 내부에서 환경 제거 후 남은 텍스트를 단락(`\n\n`) 단위로 분할하여 각 단락에 적용. 파서 레이어에서 완결되므로 chunker 수정 불필요

### 1-3. 저자 정보 수집 (LOW, 병렬 가능)
- **신규**: `scripts/10_fetch_authors.py`
- 기존 `src/collection/semantic_scholar.py`의 `SemanticScholarClient` 재사용 (batch API)
- papers 테이블 `authors` 컬럼 업데이트
- Phase 2와 병렬 실행 (API bound, GPU와 독립)

### 검증
```bash
# 매크로 치환
python3 scripts/02_parse.py --arxiv-ids 2501.12948v2 --latex-only
grep "DeepSeek-R1" data/parsed/2501.12948v2.json | head -3
# → 결과 있어야 함

# LaTeX 코드 유출
grep -rl "tikzpicture\|\\\\begin{tabular}" data/parsed/*.json
# → 0 결과
```

---

## Phase 2: 청킹 & 재임베딩

### 2-1. 데이터 모델 수정
- **파일**: `src/embedding/models.py`
- `Chunk.to_db_dict()` (line 59-77): top-level에 `chunk_index`, `token_count` 추가
  ```python
  def to_db_dict(self) -> dict:
      return {
          "chunk_id": self.chunk_id,
          "paper_id": self.paper_id,
          "content": self.content,
          "section_title": self.section_title,
          "chunk_type": self.chunk_type.value,
          "chunk_index": self.chunk_index,      # ← 추가 (code_issues #1)
          "token_count": self.token_count,       # ← 추가 (code_issues #1)
          "metadata": { ... },
      }
  ```
- **code_issues #3 수정**: line 178 주석 `"OpenAI vector (1024 dims, MRL)"` → `"OpenAI vector (3072 dims, text-embedding-3-large)"`

### 2-2. 헤더 편향 제거
- `ChunkingConfig.add_paper_context`는 이미 `default=False` (line 270)
- 재실행 시 `--add-paper-context` 미사용 → 코드 변경 불필요

### 2-3. 전체 재처리 실행
- **신규**: `scripts/11_reprocess_pipeline.py`
- 기존 `02_parse.py`, `03_embed.py`를 **import해서 래핑** (로직 중복 방지)
- 실행 순서:
  1. `data/parsed/` → `data/parsed_backup_YYYYMMDD/` 백업
  2. 2,497 LaTeX 아카이브 재파싱 (수정된 파서) — `02_parse.py` 로직 호출
  3. Qdrant 컬렉션 recreate (`ensure_collection(recreate=True)`)
  4. 로컬 PG chunks 테이블 truncate
  5. BGE-M3 임베딩 (dense + sparse, GPU) — `03_embed.py` 로직 호출, `--add-paper-context` 없이
  6. OpenAI 3-large 임베딩 (API)
  7. Qdrant upsert + 로컬 PG metadata 적재

### 예상 소요 시간
| 단계 | 소요 시간 |
|------|----------|
| 파싱 (2,497 archives) | ~30분 |
| BGE-M3 임베딩 (~120K chunks) | ~2-3시간 (GPU) |
| OpenAI 임베딩 (~120K chunks) | ~1-2시간 (API) |

### 검증
```sql
SELECT count(*) FROM chunks WHERE chunk_index IS NULL;  -- → 0
SELECT paper_id, count(*) FROM chunks GROUP BY paper_id ORDER BY count(*) DESC LIMIT 5;
```

---

## Phase 3: GraphRAG 확장

### 3-1. 인용 그래프 구축
- **파일 수정**: `scripts/fetch_citations.py` — 인용 *관계* 수집 추가 (현재는 count만)
- Semantic Scholar API: `paper/{id}/references` + `paper/{id}/citations`
- **ID 정규화**: 버전 없는 arxiv_id 사용 (예: `2501.12948`, not `2501.12948v2`) — `semantic_scholar.py`의 기존 로직과 일치
- **테이블**: `citation_edges` (Phase 0-3에서 생성)
- **신규**: `src/graph/citation_graph.py`
  - `CitationGraph` 클래스 (networkx DiGraph)
  - `get_related_papers(arxiv_id, hops=2)`, `get_pagerank_scores()`, `get_communities()` (Louvain)
  - `save(path)` / `load(path)` — JSON 직렬화
- API 예산: ~5,000 호출 (2,500 papers × references + citations), 무료 tier ~4시간
- **Phase 2와 병렬 실행 가능** (API bound vs GPU bound)
- **의존성**: `requirements.txt`에 `networkx>=3.0`, `python-louvain>=0.16` 추가

### 3-2. 엔티티 추출
- **신규**: `scripts/12_extract_entities.py`
- Gemini Flash로 청크에서 구조화 엔티티 추출
  - 타입: Model, Dataset, Method, Metric, Task
- **정규화**: v3 초기에는 `lowercase + strip`만 적용 (향후 매핑 테이블로 확장)
- **저장**: 로컬 PG `entities` 테이블 + Qdrant payload에 `entities` 필드 추가
- 비용: ~120K chunks × Gemini Flash ≈ $1-2
- **Phase 2 완료 후 실행** (새 청크 필요)

### 3-3. 그래프 메타데이터 Qdrant 반영
- **방식**: `scroll()` → `update_payload()` (기존 메서드, `qdrant_client.py` line 663)
  - 벡터 불변, 메타데이터만 업데이트
  - 전체 upsert 재실행 불필요
  - **배치 정책**: 1,000 points/배치 scroll, 배치당 `set_payload` 호출. 3회 재시도 (exponential backoff: 1s, 2s, 4s). 120K ÷ 1K = 120 배치, 예상 소요 ~5-10분
- 각 청크 Qdrant payload에 추가:
  - `paper_pagerank: float` — 인용 그래프 기반 영향력
  - `paper_community: int` — 논문 클러스터 ID
  - `entities: list[str]` — 추출된 엔티티명 (정규화 후)
  - `cited_by_count: int` — 피인용 수
- 향후 검색 로직 개선(improvement_plan.md 3번)에서 이 필드 활용

### 검증
```python
# 인용 그래프
graph = CitationGraph.load("data/citation_graph.json")
print(f"Nodes: {len(graph.nodes)}, Edges: {len(graph.edges)}")

# Qdrant payload 확인
from src.storage.qdrant_client import get_qdrant_client
q = get_qdrant_client()
points = q.scroll_chunks(limit=1)
print(points[0].payload.keys())  # paper_pagerank, entities 등 포함 확인
```

---

## 실행 순서 & 병렬화

```
Phase 0 (인프라 + repo 전환 + DB client 전환)
    │
    v
Phase 1.1-1.2 (파서 수정) ──────── Phase 1.3 (저자 수집, 병렬)
    │
    v
Phase 2 (재파싱 + 재임베딩) ─────── Phase 3.1 (인용 그래프, API 병렬)
    │
    v
Phase 3.2 (엔티티 추출)
    │
    v
Phase 3.3 (그래프 메타데이터 Qdrant 반영)
    │
    v
전체 검증
```

---

## 파일 목록

### 수정 파일
| 파일 | 변경 |
|------|------|
| `src/parsing/latex_parser.py` | 매크로 치환 (`_extract_macros`, `_apply_macros`), 환경 필터링 (`_strip_noisy_environments`, `is_latex_noisy`) |
| `src/embedding/models.py` | `to_db_dict()` chunk_index/token_count top-level, OpenAI 주석 3072 수정 |
| `docker-compose.yml` | PostgreSQL 서비스 추가 |
| `src/storage/__init__.py` | `get_db_client()` 팩토리 추가 |
| `src/rag/retriever.py` | Legacy 벡터 검색 클래스 6개 삭제, `SearchResult`/`SearchResponse` dataclass만 유지 |
| `src/rag/api.py` | legacy 검색 분기 제거, `get_supabase_client()` → `get_db_client()` |
| `scripts/fetch_citations.py` | 인용 관계 수집 확장 |
| `src/api/routes/chat.py` | line 149 Gemini 모델명 하드코딩 → `settings.gemini_model`, Supabase→DB client 전환 |
| `.env` | `DB_BACKEND=local`, PG 접속 정보 |
| `requirements.txt` | psycopg2-binary, networkx, python-louvain 추가 |
| 14개 스크립트/모듈 | `get_supabase_client()` → `get_db_client()` 전환 (0-5.D 참조) |

### 신규 파일
| 파일 | 용도 |
|------|------|
| `src/storage/postgres_client.py` | 로컬 PG 클라이언트 (SupabaseClient 동일 인터페이스) |
| `scripts/init_local_db.sql` | 로컬 PG 스키마 (Supabase 동형 + citation_edges/entities) |
| `scripts/migrate_to_local_pg.py` | Supabase → 로컬 PG papers 전체 컬럼 마이그레이션 |
| `scripts/10_fetch_authors.py` | Semantic Scholar 저자 수집 (`SemanticScholarClient` 재사용) |
| `scripts/11_reprocess_pipeline.py` | 재처리 오케스트레이션 (`02_parse` + `03_embed` import 래핑) |
| `src/graph/citation_graph.py` | 인용 그래프 (networkx DiGraph) |
| `scripts/12_extract_entities.py` | Gemini Flash 엔티티 추출 |
