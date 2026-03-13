# Phase 2 Reprocess Execution Plan

`Phase 2`의 목표는 수정된 LaTeX 파서로 전체 LaTeX 보유 논문을 다시 파싱하고, 청킹/임베딩/Qdrant 적재를 재실행해 메타데이터와 벡터를 재정렬하는 것이다.

## Current Script State

- 오케스트레이터: `scripts/11_reprocess_pipeline.py`
- 지원 기능:
  - 단건 실행: `--arxiv-id`
  - 배치 실행: `--limit`, `--offset`
  - ID 파일 기반 실행: `--paper-ids-file`
  - DB 선택 필터: `--status-filter`
  - parse only: `--skip-embed`
  - dry-run: `--dry-run`
  - Qdrant recreate: `--recreate-qdrant`
  - CPU/GPU 선택: `--device`
  - BGE batch size 제어: `--bge-batch-size`
- 상태 전이:
  - parse 성공 시 `parse_status='parsed'`, `parse_method='latex'`
  - parse 실패 시 `parse_status='failed'`
  - embed 완료 시 `03_embed.py` 경유로 `parse_status='embedded'`

## Recommended Batch Strategy

### 1. Preflight

실행 전 확인:

```bash
python3 -m py_compile src/parsing/latex_parser.py scripts/11_reprocess_pipeline.py scripts/03_embed.py
```

서비스 상태:

```bash
docker compose up -d postgres qdrant
```

로컬 PG 기본 확인:

```bash
.venv/bin/python - <<'PY'
from src.storage import get_db_client
c = get_db_client()
print(c.get_paper_count())
print(c.get_chunk_count())
PY
```

### 2. 단건 검증

대표 논문 1편으로 먼저 검증:

```bash
.venv/bin/python scripts/11_reprocess_pipeline.py \
  --arxiv-id 2501.12948v2 \
  --device cpu
```

확인 포인트:
- `data/parsed/<id>.json` 갱신
- `parse_method='latex'`
- `paper_chunk_count > 0`
- Qdrant에 해당 `paper_id` hit 존재

### 3. 샘플 배치 검증

GPU 사용 전 20편 정도로 샘플 실행:

```bash
.venv/bin/python scripts/11_reprocess_pipeline.py \
  --limit 20 \
  --device cpu \
  --recreate-qdrant
```

목표:
- 전체 흐름의 실패 유형 파악
- parse 실패 비율 확인
- chunk 수 분포 확인
- BGE CPU 처리 속도 대략 측정

### 4. 본 배치 실행

전체 2,497편은 한 번에 돌리지 말고 `citation_count DESC` 기준 분할 배치로 실행.

권장 배치 크기:
- GPU 사용 시: `100 ~ 150`
- CPU 사용 시: `20 ~ 30`

권장 순서:
1. 첫 배치만 `--recreate-qdrant`
2. 이후 배치는 `--offset`만 증가시키며 이어서 실행
3. OpenAI 3-large는 BGE 재처리 안정화 후 별도 배치로 추가

예시:

```bash
# Batch 1
.venv/bin/python scripts/11_reprocess_pipeline.py \
  --limit 100 \
  --offset 0 \
  --device cuda \
  --recreate-qdrant

# Batch 2
.venv/bin/python scripts/11_reprocess_pipeline.py \
  --limit 100 \
  --offset 100 \
  --device cuda

# Batch 3
.venv/bin/python scripts/11_reprocess_pipeline.py \
  --limit 100 \
  --offset 200 \
  --device cuda
```

### 5. OpenAI 3-large pass

BGE 재처리 완료 후 OpenAI를 붙일 경우:

```bash
.venv/bin/python scripts/11_reprocess_pipeline.py \
  --paper-ids-file batch_ids.txt \
  --device cuda \
  --with-openai
```

주의:
- OpenAI는 비용/시간이 커서 전체 BGE 완료 후 별도 pass 권장
- 장애 시 재시작 범위를 좁히기 위해 `paper_ids_file` 배치를 권장

## Operational Notes

- `data/parsed_backup_YYYYMMDD_HHMMSS/`가 자동 생성되므로, 배치 시작 전 디스크 여유 확인 필요
- Qdrant는 현재 `qdrant-client 1.17.0` vs server `1.12.1` 경고가 있으나 단건 검증은 통과함
- CPU 환경에서는 `torch.cuda.is_available()`가 false라 전체 실행은 매우 오래 걸릴 수 있음
- `03_embed.py`가 paper 단위로 `embedded` 상태를 갱신하므로, 배치 중단 시 일부 paper만 `embedded`인 상태가 남는 것은 정상

## Acceptance Checks Per Batch

배치 종료 후 확인:

```bash
.venv/bin/python - <<'PY'
from src.storage import get_db_client
c = get_db_client()
print('papers', c.get_paper_count())
print('chunks', c.get_chunk_count())
print(c.get_collection_stats())
PY
```

샘플 paper 확인:

```bash
.venv/bin/python - <<'PY'
from src.storage import get_db_client
c = get_db_client()
p = c.get_paper('2501.12948v2')
print(p['parse_status'], p.get('parse_method'))
print(len(c.get_chunks_by_paper('2501.12948v2')))
PY
```

Qdrant 샘플 확인:

```bash
.venv/bin/python - <<'PY'
from src.storage import get_qdrant_client
q = get_qdrant_client()
rows = q.search_dense([0.0]*1024, vector_name='dense_bge', top_k=1, paper_id_filter='2501.12948v2')
print(len(rows))
PY
```

## Recommended Execution Order From Here

1. 단건 검증 유지
2. 20편 CPU 샘플 배치
3. GPU 가능 여부 확인 후 100편 단위 본 배치
4. 전체 BGE 완료 후 OpenAI 3-large 별도 배치
5. 전체 완료 후 chunk count / status distribution / 검색 품질 smoke test
