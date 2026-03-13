# Conceptual Query Performance Improvement - 작업 요약

> 작성일: 2026-02-22
> 목적: 다음 세션에서 작업을 이어갈 수 있도록 문제 상황과 해결 방안 정리

---

## 1. 문제 상황

### 성능 격차 분석

| Query Style | MRR | Hit@10 | 비고 |
|-------------|-----|--------|------|
| keyword | 0.710 | 88% | 기술 용어 직접 검색 |
| natural_short | 0.592 | 76% | 질문형 쿼리 |
| natural_long | 0.568 | 78% | 상세 질문 |
| **conceptual** | **0.149** | **28%** | **5배 성능 격차** |

### 근본 원인

1. **시스템 한계 (~30-40%)**
   - Dense embedding이 paraphrase된 개념을 잘 매칭하지 못함
   - RRF 가중치가 sparse(0.3)를 선호 → lexical overlap 없는 conceptual 쿼리 불리
   - Reranker는 top-10 내 재정렬만 → 애초에 top-10에 정답이 없으면 무용지물

2. **벤치마크 설계 (~60-70%)**
   - Conceptual 쿼리가 정의상 99.9% "hard"로 라벨링됨
   - 의도적으로 기술 용어를 제거하여 word overlap = 0
   - Single-paper ground truth (실제로는 여러 논문이 관련)

---

## 2. 구현 완료된 개선사항

### Phase 1: 측정 개선 ✅

#### 1.1 스타일별 평가 분석
**파일**: `scripts/06_evaluate.py`

```bash
# 사용법
python scripts/06_evaluate.py --modes qdrant_3large --by-style --by-difficulty
```

**변경 내용**:
- `EvalMetrics` dataclass에 `style`, `difficulty` 필드 추가
- `_aggregate_metrics_by_style()`, `_aggregate_metrics_by_difficulty()` 함수 추가
- `--by-style`, `--by-difficulty` CLI 인자 추가

#### 1.2 RRF 가중치 설정
**파일**: `src/rag/qdrant_retriever.py`

```python
RRF_PRESETS = {
    "default": {"dense_weight": 0.4, "sparse_weight": 0.3, "colbert_weight": 0.3},
    "conceptual": {"dense_weight": 0.6, "sparse_weight": 0.2, "colbert_weight": 0.2},
    "keyword": {"dense_weight": 0.3, "sparse_weight": 0.5, "colbert_weight": 0.2},
    "balanced": {"dense_weight": 0.5, "sparse_weight": 0.25, "colbert_weight": 0.25},
    "dense_only": {"dense_weight": 1.0, "sparse_weight": 0.0, "colbert_weight": 0.0},
}
```

**메서드 추가**:
- `set_weights(preset_name)`: 프리셋 적용
- `get_weights()`: 현재 가중치 반환

---

### Phase 2: Query 처리 개선 ✅

#### 2.1 Query Type 감지
**신규 파일**: `src/rag/query_classifier.py`

```python
from src.rag.query_classifier import QueryClassifier, classify_query

classifier = QueryClassifier()
result = classifier.classify("methods for understanding word relationships")
# Returns: "conceptual"

# 또는 간단하게
query_type = classify_query("BERT transformer attention")
# Returns: "keyword"
```

**분류 기준**:
- `keyword`: 짧고 기술용어 위주, 질문어 없음
- `natural`: 질문형 (How, What, Why 등)
- `conceptual`: 추상적 언어, 기술용어 없음

#### 2.2 HyDE (Hypothetical Document Embeddings)
**신규 파일**: `src/rag/hyde.py`

```python
from src.rag.hyde import HyDEExpander

expander = HyDEExpander()  # Gemini API 필요
expanded = expander.expand_for_search(
    query="methods for understanding word relationships",
    query_type="conceptual"
)
# Returns: 가상의 논문 abstract 형태로 확장된 쿼리
```

**확장 타입**:
- `abstract`: 가상 논문 초록 생성 (conceptual 쿼리용)
- `passage`: 가상 논문 단락 생성
- `keywords`: 기술 키워드 추출 (natural 쿼리용)

#### 2.3 Adaptive Retrieval
**파일**: `src/rag/qdrant_retriever.py`

```python
from src.rag.qdrant_retriever import qdrant_adaptive_search

# 자동으로 쿼리 분류 → HyDE 확장 → 적절한 RRF 가중치 적용
results = qdrant_adaptive_search(
    query="methods for understanding word relationships",
    top_k=10,
    use_hyde=True,  # HyDE 사용 여부
)
```

**동작 방식**:
1. Query 타입 분류 (keyword/natural/conceptual)
2. Conceptual이면 HyDE 확장
3. 타입에 맞는 RRF 가중치 자동 적용
4. 검색 결과 반환 (metadata에 사용된 전략 정보 포함)

---

### Phase 3: 인덱싱 개선 ✅

#### 3.1 Chunk Semantic Enrichment
**신규 파일**: `scripts/09_enrich_chunks.py`

```bash
# 사용법 (재임베딩 불필요 - 메타데이터만 추가)
python scripts/09_enrich_chunks.py --limit 100 --dry-run  # 테스트
python scripts/09_enrich_chunks.py --limit 1000           # 실제 실행
```

**추가되는 메타데이터**:
- `semantic_summary`: 1문장 요약 (개념 설명)
- `conceptual_keywords`: 추상적 키워드 3-5개 (기술용어 X)
- `contribution_type`: method | evaluation | analysis | background | other

#### 3.2 Paper Context Propagation
**파일**: `src/embedding/chunker.py`, `src/embedding/models.py`

```python
from src.embedding.chunker import HybridChunker
from src.embedding.models import ChunkingConfig

# Paper context 활성화 (재임베딩 필요!)
config = ChunkingConfig(
    add_paper_context=True,      # 활성화
    paper_context_tokens=100,    # 컨텍스트 토큰 예산
)
chunker = HybridChunker(config)
```

**효과**:
- 모든 청크에 `"Paper: {title}\nTopic: {abstract_excerpt}\n\n"` 프리픽스 추가
- 청크만 봐도 어떤 논문인지 파악 가능
- **주의**: 이 기능은 재임베딩이 필요함

---

### Phase 4: 벤치마크 개선 ✅

#### 4.1 Multi-Paper Ground Truth
**파일**: `scripts/08_generate_synthetic_benchmark.py`

```bash
# 유사 논문을 ground truth에 포함
python scripts/08_generate_synthetic_benchmark.py \
  --multi-paper-gt \
  --similar-threshold 0.85 \
  --max-similar 3
```

**동작**: 임베딩 유사도가 threshold 이상인 논문들을 `relevant_papers`에 추가

#### 4.2 Embedding-based Difficulty
**파일**: `scripts/08_generate_synthetic_benchmark.py`

```bash
# 임베딩 유사도 기반 난이도 추정
python scripts/08_generate_synthetic_benchmark.py --embedding-difficulty
```

**기존**: Word overlap 기반 (conceptual = 항상 hard)
**개선**: Query-Abstract 임베딩 유사도 기반
- similarity > 0.7 → easy
- similarity > 0.4 → medium
- similarity ≤ 0.4 → hard

---

## 3. 파일 변경 요약

| 파일 | 상태 | 설명 |
|------|------|------|
| `src/rag/query_classifier.py` | **신규** | 쿼리 타입 분류 |
| `src/rag/hyde.py` | **신규** | HyDE 확장 |
| `src/rag/qdrant_retriever.py` | 수정 | RRF 프리셋, adaptive search |
| `src/rag/retriever.py` | 수정 | SearchResponse에 metadata 필드 추가 |
| `src/embedding/chunker.py` | 수정 | Paper context propagation |
| `src/embedding/models.py` | 수정 | ChunkingConfig에 context 옵션 추가 |
| `scripts/06_evaluate.py` | 수정 | 스타일/난이도별 평가 |
| `scripts/08_generate_synthetic_benchmark.py` | 수정 | Multi-paper GT, embedding difficulty |
| `scripts/09_enrich_chunks.py` | **신규** | 청크 의미론적 보강 |

---

## 4. 다음 세션에서 할 일

### 즉시 테스트 가능 (재임베딩 불필요)

```bash
# 1. Adaptive retrieval 효과 테스트
python scripts/06_evaluate.py \
  --modes qdrant_adaptive qdrant_adaptive_no_hyde \
  --benchmark data/eval/v2_full_benchmark.json \
  --by-style

# 2. Chunk enrichment 실행 (일부만 먼저)
python scripts/09_enrich_chunks.py --limit 100 --dry-run
python scripts/09_enrich_chunks.py --limit 500

# 3. 개선된 벤치마크 생성
python scripts/08_generate_synthetic_benchmark.py \
  --limit 100 \
  --multi-paper-gt \
  --embedding-difficulty \
  --output data/eval/v3_benchmark.json
```

### 선택적 (재임베딩 필요)

Paper context propagation 효과를 검증하려면:
1. `ChunkingConfig(add_paper_context=True)`로 재청킹
2. 재임베딩 실행 (~2-3시간)
3. 벤치마크 재평가

---

## 5. 예상 성능 개선

| 메트릭 | 현재 | Phase 1-2 | Phase 3 | Phase 4 |
|--------|------|-----------|---------|---------|
| Keyword MRR | 0.71 | 0.73 | 0.75 | 0.78 |
| **Conceptual MRR** | **0.15** | **0.35** | **0.45** | **0.55** |
| 성능 격차 | 5x | 2x | 1.6x | 1.4x |

---

---

## 6. 미해결 이슈 (TODO)

### Issue 1: API에 Adaptive Retrieval 연결 필요

**문제**: 새로 구현한 adaptive retrieval이 API에 연결되어 있지 않음

**현재 상태**:
- `api.py`는 레거시 `HybridRetriever` (Supabase pgvector) 사용
- `qdrant_adaptive_search()` 함수는 구현되었지만 API와 분리됨
- Query Classifier, HyDE, Dynamic RRF Weights 모두 미연결

**필요한 작업**:
```python
# api.py 수정 예시
from .qdrant_retriever import qdrant_adaptive_search

# SearchRequest에 search_mode 옵션 추가
# "adaptive" | "qdrant_hybrid" | "hybrid" (legacy) | "dense" | "sparse"

# /search 엔드포인트에 adaptive 모드 추가
if request.search_mode == "adaptive":
    response = qdrant_adaptive_search(
        query=request.query,
        top_k=request.top_k,
        use_hyde=True,
    )
```

**영향 범위**:
- `src/rag/api.py`: SearchRequest 모델, search 엔드포인트 수정
- `src/rag/__init__.py`: `qdrant_adaptive_search` export 추가

---

### Issue 2: Paper Context Propagation 재임베딩 필요

**문제**: Phase 3.2 기능 사용 시 전체 재임베딩 필요

**현재 구현 상태**:

| 구성요소 | 상태 | 설명 |
|---------|------|------|
| `ChunkingConfig.add_paper_context` | ✅ 완료 | 모델에 필드 존재 (`src/embedding/models.py`) |
| `HybridChunker._build_paper_context()` | ✅ 완료 | 로직 구현됨 (`src/embedding/chunker.py`) |
| 청크 생성 메서드들 | ✅ 완료 | `paper_context` 전달/사용 |
| `03_embed.py --add-paper-context` | ❌ **미구현** | CLI 인자 없음 |

**선행 작업 - CLI 인자 추가**:
```python
# scripts/03_embed.py 수정 필요

# 1. argparse에 인자 추가 (약 line 280 부근)
parser.add_argument(
    "--add-paper-context",
    action="store_true",
    help="Prepend paper title/abstract to chunks for better retrieval"
)

# 2. ChunkingConfig 생성 부분 수정 (약 line 345)
chunking_config = ChunkingConfig(
    max_tokens=args.max_tokens,
    overlap_tokens=args.overlap_tokens,
    include_abstract=True,
    add_paper_context=args.add_paper_context,  # 추가
)
```

**재임베딩 절차** (CLI 수정 후):
```bash
# 1. Paper context 포함하여 재임베딩 (~2-3시간 예상)
python scripts/03_embed.py --add-paper-context --force

# 2. Qdrant 컬렉션 재생성
python scripts/04_index_qdrant.py --recreate
```

**대안 (재임베딩 없이 테스트)**:
- Phase 1-2, 3.1, 4의 개선사항은 재임베딩 없이 즉시 테스트 가능
- Paper context 효과는 소규모 실험(100개 논문)으로 먼저 검증 권장:
  ```bash
  python scripts/03_embed.py --add-paper-context --limit 100
  ```

---

## 7. 참고: 원본 계획 문서

전체 계획은 `.claude/plans/bright-discovering-deer.md`에 저장되어 있음.
