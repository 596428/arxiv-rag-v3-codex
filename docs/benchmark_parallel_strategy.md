# 9개 모드 벤치마크 병렬 실행 전략

## 개요

9개 retrieval 모드를 GPU 제약(최대 VRAM 압력 2) 내에서 효율적으로 병렬 실행하는 전략입니다.

---

## 모드 분류 (VRAM Pressure 기준)

| 모드 | Embedding | Reranker | VRAM 압력 |
|------|-----------|----------|-----------|
| qdrant_dense | BGE-M3 | - | 1 (GPU) |
| qdrant_3large | OpenAI API | - | 0 (API) |
| qdrant_sparse | BGE sparse | - | 1 (GPU) |
| qdrant_hybrid | BGE-M3 | - | 1 (GPU) |
| qdrant_hybrid_3large | OpenAI + BGE sparse | - | 1 (sparse GPU) |
| qdrant_dense+rerank | BGE-M3 | BGE-reranker | 2 (순차 로드) |
| qdrant_3large+rerank | OpenAI API | BGE-reranker | 1 (reranker만) |
| qdrant_hybrid+rerank | BGE-M3 | BGE-reranker | 2 (순차 로드) |
| qdrant_hybrid_3large+rerank | OpenAI + BGE sparse | BGE-reranker | 2 (순차 로드) |

**참고**: VRAM 압력 2는 "물리 GPU 2개"가 아님. 1개 GPU에서 embedding → unload → reranker 순차 로드로 총 VRAM 사용량이 높음을 의미.

---

## VRAM 리소스 그룹

```
Group A: High VRAM (Embedding + Reranker 순차 로드)
- qdrant_dense+rerank       (VRAM: 2)
- qdrant_hybrid+rerank      (VRAM: 2)
- qdrant_hybrid_3large+rerank (VRAM: 2)

Group B: Medium VRAM (Embedding 또는 Reranker)
- qdrant_dense             (VRAM: 1)
- qdrant_sparse            (VRAM: 1)
- qdrant_hybrid            (VRAM: 1)
- qdrant_hybrid_3large     (VRAM: 1)
- qdrant_3large+rerank     (VRAM: 1)

Group C: No VRAM (API만 사용)
- qdrant_3large            (VRAM: 0)
```

---

## 병렬 실행 순서 (6 Phases)

### 제약조건
- 총 VRAM 압력 ≤ 2
- 최대 2개 프로세스 동시 실행

### Phase 구성

```
Phase 1: [qdrant_3large] + [qdrant_dense]
         API only (0)     GPU embedding (1)
         → 병렬 실행 (VRAM: 1)

Phase 2: [qdrant_sparse] + [qdrant_3large+rerank]
         GPU sparse (1)   API + GPU reranker (1)
         → 병렬 실행 (VRAM: 2)

Phase 3: [qdrant_hybrid] + [qdrant_hybrid_3large]
         GPU hybrid (1)   API + GPU sparse (1)
         → 병렬 실행 (VRAM: 2)

Phase 4: [qdrant_dense+rerank]
         → 단독 실행 (VRAM: 2)

Phase 5: [qdrant_hybrid+rerank]
         → 단독 실행 (VRAM: 2)

Phase 6: [qdrant_hybrid_3large+rerank]
         → 단독 실행 (VRAM: 2)
```

---

## 실행 다이어그램

```
Time →
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 1  │ qdrant_3large ────────│ qdrant_dense ──────────────│
         │ (API, VRAM: 0)        │ (VRAM: 1)                  │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 2  │ qdrant_sparse ────────│ qdrant_3large+rerank ──────│
         │ (VRAM: 1)             │ (VRAM: 1)                  │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 3  │ qdrant_hybrid ────────│ qdrant_hybrid_3large ──────│
         │ (VRAM: 1)             │ (VRAM: 1)                  │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 4  │ qdrant_dense+rerank ──────────────────────────────│
         │ (VRAM: 2 - embedding → unload → reranker)         │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 5  │ qdrant_hybrid+rerank ─────────────────────────────│
         │ (VRAM: 2 - embedding → unload → reranker)         │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 6  │ qdrant_hybrid_3large+rerank ──────────────────────│
         │ (VRAM: 2 - sparse → unload → reranker)            │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 구현: Subprocess 기반 병렬 실행

### 설계 원칙

- **asyncio 대신 subprocess 사용**: CUDA context 충돌 방지
- **프로세스 격리**: OOM 발생 시 해당 프로세스만 실패
- **자동 GPU 메모리 해제**: 프로세스 종료 시 자동 정리 (empty_cache 불필요)

### 구현 코드

```python
# scripts/run_benchmark_parallel.py
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait
from pathlib import Path

PHASES = [
    # Phase 1: VRAM 1
    [("qdrant_3large", 0), ("qdrant_dense", 1)],
    # Phase 2: VRAM 2
    [("qdrant_sparse", 1), ("qdrant_3large+rerank", 1)],
    # Phase 3: VRAM 2
    [("qdrant_hybrid", 1), ("qdrant_hybrid_3large", 1)],
    # Phase 4-6: VRAM 2 (단독)
    [("qdrant_dense+rerank", 2)],
    [("qdrant_hybrid+rerank", 2)],
    [("qdrant_hybrid_3large+rerank", 2)],
]

def run_mode_subprocess(mode: str, benchmark_file: str, output_dir: str) -> dict:
    """별도 프로세스에서 단일 모드 실행 (CUDA 격리)"""
    output_file = Path(output_dir) / f"{mode.replace('+', '_')}_results.json"

    start_time = time.time()
    result = subprocess.run(
        [
            sys.executable, "scripts/06_evaluate.py",
            "--modes", mode,
            "--benchmark", benchmark_file,
            "--output", str(output_file),
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,  # 프로젝트 루트
    )
    elapsed = time.time() - start_time

    return {
        "mode": mode,
        "returncode": result.returncode,
        "elapsed_seconds": elapsed,
        "stdout": result.stdout[-1000:] if result.stdout else "",  # 마지막 1000자
        "stderr": result.stderr[-500:] if result.stderr else "",   # 마지막 500자
        "output_file": str(output_file),
    }

def run_benchmark_parallel(benchmark_file: str, output_dir: str = "data/eval/results"):
    """6 Phase 병렬 벤치마크 실행"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_results = []
    total_start = time.time()

    for phase_idx, phase in enumerate(PHASES):
        phase_num = phase_idx + 1
        modes = [mode for mode, _ in phase]
        vram_total = sum(vram for _, vram in phase)

        print(f"\n{'='*60}")
        print(f"Phase {phase_num}/6: {modes}")
        print(f"VRAM 압력: {vram_total}")
        print(f"{'='*60}")

        phase_start = time.time()

        # 병렬 subprocess 실행
        with ProcessPoolExecutor(max_workers=len(modes)) as executor:
            futures = [
                executor.submit(run_mode_subprocess, mode, benchmark_file, output_dir)
                for mode in modes
            ]
            done, _ = wait(futures)

            for future in done:
                result = future.result()
                all_results.append(result)
                status = "✓" if result["returncode"] == 0 else "✗"
                print(f"  {status} {result['mode']}: {result['elapsed_seconds']:.1f}s")

        phase_elapsed = time.time() - phase_start
        print(f"Phase {phase_num} 완료: {phase_elapsed:.1f}s")

        # subprocess 종료 시 GPU 메모리 자동 해제됨

    total_elapsed = time.time() - total_start

    # 요약 출력
    print(f"\n{'='*60}")
    print(f"전체 벤치마크 완료: {total_elapsed:.1f}s")
    print(f"{'='*60}")

    success_count = sum(1 for r in all_results if r["returncode"] == 0)
    print(f"성공: {success_count}/9")

    failed = [r for r in all_results if r["returncode"] != 0]
    if failed:
        print(f"\n실패한 모드:")
        for r in failed:
            print(f"  - {r['mode']}: {r['stderr'][:200]}")

    return all_results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="9개 모드 병렬 벤치마크 실행")
    parser.add_argument("--benchmark", required=True, help="벤치마크 파일 경로")
    parser.add_argument("--output-dir", default="data/eval/results", help="결과 저장 디렉토리")

    args = parser.parse_args()
    run_benchmark_parallel(args.benchmark, args.output_dir)
```

---

## 검증 방법

### 1. 병렬 실행 검증
```bash
# 별도 터미널에서 GPU 모니터링
watch -n 1 nvidia-smi

# 병렬 벤치마크 실행
python scripts/run_benchmark_parallel.py --benchmark data/eval/test_v2_benchmark.json
```

### 2. 프로세스 격리 검증
```bash
# Phase 실행 중 프로세스 확인
ps aux | grep 06_evaluate
# 기대: Phase당 최대 2개 Python 프로세스
```

### 3. 개별 모드 테스트
```bash
# 단일 모드 테스트
python scripts/06_evaluate.py --modes qdrant_dense --benchmark data/eval/test_v2_benchmark.json
```

---

## 예상 소요 시간

| Phase | 모드 | 예상 시간 (50 쿼리 기준) |
|-------|------|-------------------------|
| 1 | qdrant_3large, qdrant_dense | ~3분 (API 병목) |
| 2 | qdrant_sparse, qdrant_3large+rerank | ~4분 |
| 3 | qdrant_hybrid, qdrant_hybrid_3large | ~4분 |
| 4 | qdrant_dense+rerank | ~5분 (reranker 포함) |
| 5 | qdrant_hybrid+rerank | ~5분 |
| 6 | qdrant_hybrid_3large+rerank | ~5분 |
| **총계** | | **~26분** (순차: ~45분) |

병렬화로 약 **40% 시간 절감** 예상.

---

## 주의사항

1. **GPU 메모리 모니터링**: Phase 2-6에서 VRAM 사용량 주시
2. **API Rate Limit**: qdrant_3large 모드는 OpenAI API 호출 - rate limit 주의
3. **결과 파일 병합**: 각 모드별 결과 파일을 최종 분석 시 병합 필요
