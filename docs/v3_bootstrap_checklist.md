# V3 Bootstrap Checklist

`deep-orbiting-boot.md`를 기준으로 현재 코드베이스와 대조한 실행 체크리스트.

## Current State

- [x] GitHub repo `596428/arxiv-rag-v3-codex` 생성
- [x] `origin` remote를 `https://github.com/596428/arxiv-rag-v3-codex.git`로 변경
- [x] 현재 `master` 브랜치 첫 push 완료
- [x] Docker에서 `qdrant` 컨테이너 기동 및 `healthy` 확인
- [x] 로컬 `.env`에 `QDRANT_URL=http://localhost:6333` 추가

## Scope Checklist

### Phase 0-1. Repo / Config Cleanup

- [x] 새 repo 이름을 `arxiv-rag-v3-codex`로 확정
- [x] 새 repo 생성 및 remote 교체
- [x] `src/api/routes/chat.py`의 Gemini 하드코딩 제거
- [ ] `.env` 외 환경변수 문서와 실제 설정 차이 점검

### Phase 0-2. Docker Compose PostgreSQL

- [x] `docker-compose.yml`에 PostgreSQL 서비스 추가
- [x] `postgres_data` volume 추가
- [x] API 서비스에 `PG_*` 환경변수 전달
- [x] PostgreSQL healthcheck 추가

### Phase 0-3. Local PG Schema

- [x] `scripts/init_local_db.sql` 신규 생성
- [ ] `papers` 테이블을 Supabase 동형으로 정의
- [ ] `chunks` 테이블에서 벡터 컬럼 제거 버전 정의
- [ ] `equations`, `figures` 테이블 정의
- [ ] `citation_edges`, `entities` 테이블 추가
- [ ] `updated_at` 트리거 유지

### Phase 0-4. LocalPGClient

- [x] `src/storage/postgres_client.py` 신규 생성
- [ ] `SupabaseClient` 사용 메서드와 동일 인터페이스 제공
- [x] `get_chunks_by_paper()`를 `ORDER BY chunk_index`로 구현
- [x] connection pool(`minconn=2`, `maxconn=10`) 적용
- [x] `src/storage/__init__.py`에 `get_db_client()` 팩토리 추가
- [x] `requirements.txt` 또는 `pyproject.toml`에 `psycopg2-binary` 추가

### Phase 0-5. Legacy Supabase 정리

- [x] `src/rag/retriever.py`를 데이터 모델 중심으로 축소
- [x] `src/rag/api.py`의 legacy `HybridRetriever` 분기 제거
- [x] `search_mode` 허용값을 `adaptive`, `qdrant_hybrid`, `dense`, `sparse`로 제한
- [x] `get_supabase_client()` 호출부를 `get_db_client()`로 전환
- [ ] LaTeX 미보유 3편 후처리 정책을 마이그레이션 스크립트에 반영

### Phase 0-6. Papers Migration

- [x] `scripts/migrate_to_local_pg.py` 신규 생성
- [x] Supabase `papers` 컬럼 마이그레이션
- [x] 1,000행 배치 + 배치 간 1초 대기 적용
- [x] LaTeX 미보유 3편 `parse_status='failed'` 갱신

### Phase 1. Parser Fixes

- [ ] `src/parsing/latex_parser.py`에 매크로 추출/치환 추가
- [ ] noisy environment 제거 로직 추가
- [ ] `is_latex_noisy()` 단락 필터 추가
- [ ] figure/table 메타데이터 추출 순서 유지
- [ ] `scripts/10_fetch_authors.py` 신규 생성

### Phase 2. Reprocessing

- [ ] `src/embedding/models.py` 주석과 DB dict 구조 정리
- [ ] `scripts/11_reprocess_pipeline.py` 신규 생성
- [ ] 파싱 재실행 + Qdrant recreate + chunks truncate 자동화
- [ ] BGE-M3 / OpenAI 3-large 재임베딩 연결

## Codebase Comparison

### Already Present

- `src/embedding/models.py`의 `Chunk.to_db_dict()`는 이미 top-level `chunk_index`, `token_count`를 포함함
- `src/embedding/models.py`의 `openai_dimensions`는 이미 `3072`
- `src/rag/qdrant_retriever.py`가 Qdrant 기반 검색을 이미 담당
- `docker-compose.yml`에 `qdrant` 서비스는 이미 존재

### Remaining Gaps

- `scripts/10_fetch_authors.py` 없음
- `scripts/11_reprocess_pipeline.py` 없음
- `src/parsing/latex_parser.py`에는 `_extract_macros()`, `_apply_macros()`, `_strip_noisy_environments()`가 아직 없음
- `src/storage/supabase_client.py`의 `get_chunks_by_paper()`는 아직 `metadata->chunk_index` 정렬
- 일부 보조 스크립트는 여전히 legacy Supabase 경로 또는 평가용 legacy retriever import에 의존

## Phase 0 Immediate Execution Plan

1. `src/api/routes/chat.py` Gemini 하드코딩 제거
2. `psycopg2-binary` 의존성 추가
3. `src/storage/postgres_client.py` 초안 작성
4. `src/storage/__init__.py`에 `get_db_client()` 도입
5. `docker-compose.yml`에 PostgreSQL 서비스 추가
6. `scripts/init_local_db.sql` 생성
7. `src/rag/api.py`에서 legacy 분기 제거
8. `src/rag/retriever.py`를 데이터 모델 위주로 축소
9. `get_supabase_client()` 호출부를 우선순위 높은 파일부터 `get_db_client()`로 교체
10. `scripts/migrate_to_local_pg.py` 작성 및 papers 마이그레이션 검증
