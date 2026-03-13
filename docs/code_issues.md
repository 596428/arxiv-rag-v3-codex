# 기존 코드/주석 개선 필요사항

작성일: 2026-03-12

본 문서는 deep-orbiting-boot.md와의 불일치 여부와 무관하게, **현재 코드 또는 주석 자체에 문제가 있는 부분**만 정리한다.

## 1) chunk_index/token_count 저장 누락(실제 동작 버그)
- `Chunk.to_db_dict()`가 `chunk_index`, `token_count`를 top-level 컬럼에 저장하지 않고 `metadata`에만 저장한다.
- Supabase 스키마에는 `chunk_index`, `token_count` 컬럼이 존재하므로, 현재 저장 로직은 스키마와 불일치한다.
- 결과적으로 `chunks.chunk_index`는 NULL로 남으며, 쿼리 정렬/연속성 복원이 약해진다.

관련 파일:
- /home/ajh428/projects/arxiv-rag-v1/src/embedding/models.py
- /home/ajh428/projects/arxiv-rag-v1/supabase/migrations/20260212000000_init_schema.sql

개선 필요:
- `to_db_dict()`에 top-level `chunk_index`, `token_count` 포함.
- 기존 데이터는 재임베딩 또는 backfill로 보정.

---

## 2) chunk_index 정렬 경로가 메타데이터에만 의존
- `get_chunks_by_paper()`가 `metadata->chunk_index`로 정렬한다.
- top-level `chunk_index` 컬럼을 제대로 채울 경우, 정렬 기준을 컬럼으로 바꾸는 것이 일관적이다.

관련 파일:
- /home/ajh428/projects/arxiv-rag-v1/src/storage/supabase_client.py

개선 필요:
- 컬럼 채움 완료 후 정렬 기준을 `chunk_index`로 전환.

---

## 3) OpenAI 임베딩 차원 주석 불일치
- `embedding_openai` 주석은 1024 dims(MRL)로 표기돼 있으나, 실제 설정과 Qdrant 스키마는 3072 dims이다.
- 주석과 실제가 불일치하므로 유지보수 시 혼란을 유발한다.

관련 파일:
- /home/ajh428/projects/arxiv-rag-v1/src/embedding/models.py
- /home/ajh428/projects/arxiv-rag-v1/src/storage/qdrant_client.py
- /home/ajh428/projects/arxiv-rag-v1/src/embedding/openai_embedder.py

개선 필요:
- 주석을 실제 설정(3072)과 일치하도록 정리.

---

## 4) Gemini 모델명 설정과 실제 호출 불일치
- 설정 기본값은 `gemini-3-flash-preview`인데, API 라우트에서는 `gemini-2.0-flash`를 하드코딩한다.
- `gemini-3-flash-preview` 로 통일 필요

관련 파일:
- /home/ajh428/projects/arxiv-rag-v1/src/utils/config.py
- /home/ajh428/projects/arxiv-rag-v1/src/api/routes/chat.py

개선 필요:
- 모델명을 설정값으로 통일.


추가 질문 사항

  1. 문서에는 LocalPGClient와 get_db_client() 전환이 계획되어 있는데, 현재 코드는 전부 get_supabase_client() 직접 사용입니다. 계획상 전환 범
     위가 어디까지인지 확정이 필요합니다. deep-orbiting-boot.md storage/init.py
  2. 문서에 “docker-compose에 PostgreSQL 추가”가 있는데 현재 compose에는 Qdrant/API만 있습니다. v3에서 compose를 바꿀 예정인가요? deep-
     orbiting-boot.md docker-compose.yml
  3. 문서에 “LaTeX 매크로 치환”이 핵심인데 현재 파서는 매크로 치환 로직이 없습니다. 실제로 어느 레이어(파서/클리너/사전처리)에 넣을지 확정됐
     나요? deep-orbiting-boot.md latex_parser.py
  4. 문서에 “LaTeX 환경 필터링(tikz/algorithm/figure/table 등)”이 있는데, 현재 파서는 figure/table을 추출합니다. 필터링은 “텍스트 청크 전
     용”으로만 적용할 예정인가요? deep-orbiting-boot.md latex_parser.py
  5. 문서에 “로컬 PG 스키마”가 있는데, 현재 Supabase 스키마와 다릅니다(예: pdf_path, latex_path, parse_method 등). 로컬 스키마를 Supabase와
     동형으로 갈 건가요? deep-orbiting-boot.md 20260212000000_init_schema.sql
  6. 문서에는 scripts/11_reprocess_pipeline.py가 새로 생기는데, 현재는 02_parse.py/03_embed.py가 중심입니다. 기존 스크립트를 래핑할 건가요,
     완전 새로 구성할 건가요? deep-orbiting-boot.md 02_parse.py 03_embed.py
  7. 문서의 GraphRAG에서 networkx/Louvain을 사용한다 했지만 requirements에 없습니다. 의존성 추가 예정인가요? deep-orbiting-boot.md
     requirements.txt