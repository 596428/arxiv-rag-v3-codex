# RAG 시스템 개선 계획

> 분석 일자: 2026-03-03
> 현황: 2,500편 논문 / 124,001 청크 / BGE-M3 + OpenAI text-embedding-3-large

---

## 1. Supabase 데이터 적재 개선

### 1-1. LaTeX 매크로 미치환 문제 (심각)

**현상**
```
원문 LaTeX: "We introduce \modelname{} and its precursor \modelname{}-Zero..."
파싱 후:   "We introduce  and its precursor -Zero..."
```

논문에서 커스텀 매크로(`\modelname`, `\ours`, `\method` 등)로 정의된 모델명·방법명이 파싱 시 치환되지 않고 삭제됨.
DeepSeek-R1 논문의 경우 abstract와 본문 대부분 청크에서 "DeepSeek-R1"이 공백으로 처리됨.

**영향**
- 키워드 검색(sparse) 시 모델명으로 매칭 불가
- 사용자가 "DeepSeek-R1"로 검색해도 관련 청크가 낮은 순위로 밀림

**개선 방향**
- LaTeX 파서에서 `\newcommand`, `\def` 정의를 먼저 파싱해 매크로 치환 테이블 구성
- 치환 후 본문 파싱 진행
- 매크로 정의를 못 찾은 경우 논문 제목에서 후보 추출하여 휴리스틱 치환

---

### 1-2. LaTeX 원시 코드 청크 유입 (중간)

**현상**
```
청크 내용: "\begintikzpicture [anchor=south west] (image) at (0,0)..."
```

figure/table 환경의 LaTeX 코드가 text 청크로 잘못 분류되어 저장됨.

**영향**
- 노이즈 토큰이 임베딩 벡터를 오염
- 검색 결과에 의미 없는 청크가 노출

**개선 방향**
- 파싱 단계에서 `tikzpicture`, `tabular`, `algorithm` 환경 명시적 필터링
- 텍스트 청크 내 LaTeX 명령어 비율(%) 임계값으로 품질 필터 적용

---

### 1-3. authors 필드 누락 (낮음)

**현상**
```sql
papers.authors = NULL  -- 전체 2,500편
```

**영향**
- "Hinton이 쓴 논문" 같은 저자 기반 쿼리 대응 불가
- 인용 표시 시 저자 정보 없음

**개선 방향**
- arXiv API 또는 Semantic Scholar API로 저자 정보 재수집 후 업데이트

---

## 2. Chunking 및 재임베딩이 필요한 이유

### 2-1. 반복 헤더가 임베딩을 편향시킴 (심각)

**현상**

`--add-paper-context` 플래그를 켜고 임베딩하여, 모든 청크(equation 제외)에 동일한 헤더가 prefix로 삽입됨.

```
[청크 구성 - 512 토큰 기준]
├── 헤더 (~100 토큰, 20%): "Paper: [제목]\nTopic: [abstract 앞부분]..."
└── 실제 본문 (~412 토큰, 80%): 섹션 내용
```

같은 논문의 57개 청크가 모두 동일한 100 토큰을 공유 → 임베딩 벡터가 abstract 방향으로 편향.

**영향**
- 같은 논문 내 섹션 간 변별력 감소 (Introduction과 Experiment 벡터가 비정상적으로 유사)
- 개념적(conceptual) 쿼리에서 특히 취약: 모든 청크가 비슷한 점수로 반환
- 재임베딩의 직접적 동기: 헤더를 제거하면 청크 고유 내용만으로 벡터 생성 가능

**개선 방향**
- `add_paper_context=False`로 재청킹 및 재임베딩
- 논문 컨텍스트는 임베딩 단계가 아닌 **LLM 프롬프트 빌드 단계**에서 추가
- 필요시 abstract 청크만 별도 색인하여 paper-level 검색에 활용

---

### 2-2. chunk_index 컬럼 미저장 (중간)

**현상**

`Chunk.to_db_dict()`가 `chunk_index`를 top-level 컬럼이 아닌 `metadata` JSONB 안에만 저장:

```python
def to_db_dict(self):
    return {
        "chunk_id": ...,
        "content": ...,
        # chunk_index 컬럼 직접 할당 없음 ← 버그
        "metadata": {
            "chunk_index": self.chunk_index,  # JSONB 안에만 존재
        }
    }
```

→ DB의 `chunk_index` 컬럼은 전체 124,001개 청크에서 NULL.

**영향**
- `ORDER BY chunk_index` 정렬 불가 → 같은 논문의 청크 순서 보장 없음
- 문맥 연속성이 필요한 경우(앞뒤 청크 조합) 순서 복원 어려움

**개선 방향**
- `to_db_dict()`에 `"chunk_index": self.chunk_index` 추가
- 재청킹 시 함께 반영

---

### 2-3. 수식 청크 미활성 (낮음)

**현상**
```
equation 타입 청크: 0개 (124,001개 중)
```

`include_equations=False`(기본값)로 실행하여 Gemini 기반 수식 자연어 변환 파이프라인이 동작하지 않음.

**영향**
- 수식 내용에 대한 검색 불가
- 수학/알고리즘 중심 논문의 핵심 내용 누락

**개선 방향**
- `include_equations=True`로 재임베딩 (Gemini API 비용 고려하여 선택적 적용)
- 단, 수식 설명 품질 검증 후 적용 권장

---

## 3. Retrieval 로직 개선

### 3-1. Chat 엔드포인트에 Reranker 없음 (심각)

**현상**

`/search` API는 BGE reranker를 적용하지만, 실제 챗봇이 사용하는 `/chat` 엔드포인트에는 reranker 단계가 없음.

**영향**
- 벤치마크 기준 reranker 적용 시 NDCG@10이 +2~5%p 향상
- 챗봇이 검색 API보다 낮은 품질의 결과를 사용 중

**개선 방향**
- `/chat` 엔드포인트에 reranker 옵션 추가 (기본값 `True`)
- RRF top-30 → reranker → top-5 파이프라인 적용

---

### 3-2. 논문당 청크 1개 강제 중복 제거 (심각)

**현상**

```python
# chat.py
if paper_id in seen_papers:
    continue  # 같은 논문의 모든 후속 청크 버림
```

top_k=5 기준, 최대 5개 논문 × 1청크만 LLM 컨텍스트에 진입.

**영향**
- 관련 내용이 해당 논문의 다른 섹션에 있으면 완전 누락
- 1개 논문에서 여러 섹션을 참조해야 하는 질문에 취약

**개선 방향**
- 논문당 최대 2~3청크 허용 (단, 섹션이 다른 경우에 한해)
- 또는 중복 제거 없이 top-k 청크를 그대로 사용하고 reranker로 품질 보장

---

### 3-3. BGE Hybrid에서 Sparse 기여 미미 (중간)

**현상**

기본 가중치: `dense=0.4, sparse=0.3`

```
qdrant_hybrid NDCG@10 = 0.644
qdrant_dense  NDCG@10 = 0.630  ← 차이 +0.014만
```

- colbert_weight=0.3이 설정되어 있으나 `_rrf_fusion()`이 dense+sparse만 처리 → 가중치 합 0.7로 계산
- 반복 헤더로 인한 sparse 신호 희석

**개선 방향**
- 재임베딩 후 (헤더 제거) sparse 기여도 재측정
- colbert_weight를 실제 fusion에 반영하거나 파라미터 제거
- keyword 쿼리에서 sparse 가중치를 높이는 adaptive 튜닝 강화

---

### 3-4. OpenAI 임베딩 시 Hybrid가 Dense-only로 동작 (중간)

**현상**

```python
# chat.py
if request.embedding_model == "openai":
    sparse_indices, sparse_values = None, None  # sparse 없음

if request.search_mode == "hybrid" and sparse_indices:
    # sparse_indices가 None이므로 이 분기 진입 안 함
    ...
else:
    raw_results = qdrant.search_dense(...)  # dense-only로 fallback
```

`embedding_model="openai"`(기본값)면 `search_mode="hybrid"` 설정과 무관하게 항상 dense-only.

**영향**
- Hybrid-3L (OpenAI dense + BGE sparse) 구현은 있으나 chat에서 미사용
- 성능상 차이는 미미(NDCG@10 +0.004)하나 의도와 동작 불일치

**개선 방향**
- OpenAI dense + BGE sparse hybrid를 chat 엔드포인트에서도 명시적으로 호출
- 또는 주석으로 의도적 설계임을 명시

---

## 우선순위 요약

| 우선순위 | 항목 | 예상 효과 |
|---------|------|----------|
| 🔴 높음 | LaTeX 매크로 미치환 수정 | 키워드 검색 정확도 직접 향상 |
| 🔴 높음 | 헤더 제거 후 재임베딩 | 섹션 변별력 회복, 개념 쿼리 개선 |
| 🔴 높음 | Chat에 Reranker 추가 | NDCG@10 +2~5%p |
| 🔴 높음 | 논문당 다중 청크 허용 | 복잡한 질문 커버리지 향상 |
| 🟡 중간 | chunk_index 컬럼 수정 | 청크 순서 보장 |
| 🟡 중간 | LaTeX 코드 필터링 | 노이즈 청크 제거 |
| 🟡 중간 | Sparse 가중치 튜닝 | BGE hybrid 효과 향상 |
| 🟢 낮음 | authors 필드 수집 | 저자 기반 검색 지원 |
| 🟢 낮음 | 수식 청크 활성화 | 수학적 내용 검색 가능 |
