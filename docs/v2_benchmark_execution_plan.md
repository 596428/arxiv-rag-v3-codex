# V2 Benchmark Plan: 검색 품질 평가

## 목표
v2 아키텍처(Qdrant + BGE-M3 + OpenAI 3-large)의 검색 품질을 벤치마크하고 v1 결과와 비교

## 현재 상태

### V1 벤치마크 결과 (1,822 쿼리)
| Mode | MRR | NDCG@10 | P@10 | Time(ms) |
|------|-----|---------|------|----------|
| sparse | 0.772 | 0.762 | 0.436 | 2,599 |
| openai (1024d) | 0.757 | 0.756 | 0.413 | 225 |
| hybrid | 0.614 | 0.681 | 0.370 | 3,578 |
| dense (BGE) | 0.433 | 0.434 | 0.140 | 440 |

### V2 현재 구성
- **Qdrant**: 105,876 chunks
- **Vectors**: dense_bge (1024d), dense_3large (3072d), sparse_bge
- **기존 쿼리셋**: `data/eval/synthetic_queries.json` (1,822 queries)

## 구현 계획

### Step 1: OpenAI 3-large Retriever 추가
**파일**: `src/rag/qdrant_retriever.py`

```python
class Qdrant3LargeRetriever:
    """Dense retrieval using OpenAI text-embedding-3-large (3072d)."""

    def search(self, query: str, top_k: int = 20) -> list[SearchResult]:
        # OpenAI embedder로 쿼리 임베딩
        # dense_3large 벡터로 검색
```

- `QdrantHybridRetriever`에 `search_3large_only()` 메서드 추가
- `search()` 메서드에 `vector_name` 파라미터 추가 (bge vs 3large 선택)

### Step 2: 평가 스크립트 업데이트
**파일**: `scripts/06_evaluate.py`

새로운 모드 추가:
- `qdrant_3large`: OpenAI 3-large dense only
- `qdrant_hybrid_3large`: 3-large + sparse RRF fusion

### Step 3: 벤치마크 실행
```bash
# V2 모드들 비교
python scripts/06_evaluate.py \
  --queries data/eval/synthetic_queries.json \
  --modes qdrant_dense qdrant_3large qdrant_sparse qdrant_hybrid qdrant_hybrid_3large \
  --output results/v2_benchmark.json
```

### Step 4: 쿼리셋 검증 (필요시)
기존 synthetic_queries.json의 `relevant_papers`가 v2 Qdrant에 존재하는지 확인:
```python
# 유효 쿼리만 필터링
valid_queries = [q for q in queries if q["relevant_papers"][0] in indexed_papers]
```

## 수정 대상 파일
1. `src/rag/qdrant_retriever.py` - 3-large retriever 추가
2. `scripts/06_evaluate.py` - 새 모드 추가
3. `results/v2_benchmark.json` - 결과 저장

## 예상 결과 비교

| Mode | 예상 MRR | 예상 NDCG@10 | 비고 |
|------|---------|-------------|------|
| qdrant_sparse | ~0.77 | ~0.76 | v1과 유사 |
| qdrant_3large | ~0.75-0.80 | ~0.75-0.80 | 3072d 고품질 |
| qdrant_dense (BGE) | ~0.43 | ~0.43 | v1과 유사 |
| qdrant_hybrid_3large | ~0.80+ | ~0.80+ | 최고 성능 예상 |

## 검증 방법
1. 벤치마크 실행 후 MRR, NDCG, Precision 지표 비교
2. v1 대비 성능 변화 확인
3. 3-large vs BGE-M3 dense 품질 비교
4. Hybrid fusion 효과 검증

## 사용자 결정
- **쿼리셋**: Gemini로 새로 생성 (v2 데이터셋 기준)
- **Reranker**: 포함 (BGE-reranker 테스트)

---

## 최종 구현 계획

### Phase 1: Gemini 쿼리 생성 (10-15분)
**스크립트**: `scripts/08_generate_synthetic_benchmark.py` (기존 재사용)

```bash
source .venv/bin/activate
python scripts/08_generate_synthetic_benchmark.py \
  --limit 2400 \
  --output data/eval/v2_synthetic_queries.json \
  --batch-size 10 \
  --delay 0.5
```

- v2 Qdrant에 임베딩된 2,400개 논문 대상
- 논문당 1-2개 쿼리 → 예상 3,000-4,000개 쿼리

### Phase 2: OpenAI 3-large Retriever 추가
**파일**: `src/rag/qdrant_retriever.py`

```python
class Qdrant3LargeRetriever:
    """Dense retrieval using OpenAI text-embedding-3-large (3072d)."""

    def __init__(self, client=None):
        self.client = client or get_qdrant_client()
        self._embedder = None  # OpenAIEmbedder

    def search(self, query: str, top_k: int = 20) -> list[SearchResult]:
        # OpenAI embedder로 쿼리 임베딩 (3072d)
        # Qdrant dense_3large 벡터로 검색
```

`QdrantHybridRetriever` 확장:
- `search_3large_only()` 메서드 추가
- `search_hybrid_3large()` 메서드 추가 (3large + sparse RRF)

### Phase 3: 평가 스크립트 업데이트
**파일**: `scripts/06_evaluate.py`

새로운 모드 추가:
| 모드 | 설명 |
|-----|------|
| `qdrant_3large` | OpenAI 3-large dense only |
| `qdrant_hybrid_3large` | 3-large + sparse RRF |
| `qdrant_3large+rerank` | 3-large + BGE reranker |
| `qdrant_hybrid_3large+rerank` | hybrid + reranker |

### Phase 4: 전체 벤치마크 실행
```bash
# 모든 모드 비교 (reranker 포함)
python scripts/06_evaluate.py \
  --queries data/eval/v2_synthetic_queries.json \
  --modes qdrant_dense qdrant_3large qdrant_sparse \
          qdrant_hybrid qdrant_hybrid_3large \
          qdrant_dense+rerank qdrant_3large+rerank \
          qdrant_hybrid+rerank qdrant_hybrid_3large+rerank \
  --top-k 10 \
  --output results/v2_benchmark_full.json
```

## 수정 대상 파일
1. `src/rag/qdrant_retriever.py` - 3-large retriever 추가
2. `scripts/06_evaluate.py` - 새 모드 및 reranker 통합
3. `data/eval/v2_synthetic_queries.json` - 새 쿼리셋
4. `results/v2_benchmark_full.json` - 결과 저장

## 예상 소요 시간
- Phase 1 (쿼리 생성): 10-15분
- Phase 2 (retriever 구현): 10분
- Phase 3 (evaluate 업데이트): 10분
- Phase 4 (벤치마크 실행): 30-60분 (reranker 포함)

## 검증 방법
1. 쿼리 생성 후 JSON 형식 검증
2. 각 retriever 개별 동작 테스트
3. 전체 벤치마크 실행 및 결과 비교
4. v1 대비 성능 변화 분석
