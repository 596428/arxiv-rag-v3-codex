#!/usr/bin/env python3
"""
arXiv RAG v1 - Parallel Benchmark Runner

Runs 9 benchmark modes in parallel with GPU VRAM constraints.
Uses subprocess isolation to prevent CUDA context conflicts and ensure
proper GPU memory cleanup between phases.

Phase Strategy (respecting VRAM ≤ 2 constraint):
- Phase 1: qdrant_3large (API) + qdrant_dense (GPU 1) = VRAM 1
- Phase 2: qdrant_sparse (GPU 1) + qdrant_3large+rerank (GPU 1) = VRAM 2
- Phase 3: qdrant_hybrid (GPU 1) + qdrant_hybrid_3large (API + sparse) = VRAM 2
- Phase 4: qdrant_dense+rerank (GPU 2 - embed→unload→rerank) = VRAM 2
- Phase 5: qdrant_hybrid+rerank (GPU 2) = VRAM 2
- Phase 6: qdrant_hybrid_3large+rerank (GPU 2) = VRAM 2

Usage:
    python scripts/run_benchmark_parallel.py --benchmark data/eval/test_v2_benchmark.json
    python scripts/run_benchmark_parallel.py --benchmark data/eval/test_v2_benchmark.json --output data/eval/results.json
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PhaseConfig:
    """Configuration for a benchmark phase."""
    phase_num: int
    modes: list[tuple[str, int]]  # (mode_name, vram_pressure)
    description: str


# Phase configuration based on VRAM constraints
PHASES: list[PhaseConfig] = [
    PhaseConfig(
        phase_num=1,
        modes=[("qdrant_3large", 0), ("qdrant_dense", 1)],
        description="API-only + Dense embedding (VRAM: 1)",
    ),
    PhaseConfig(
        phase_num=2,
        modes=[("qdrant_sparse", 1), ("qdrant_3large+rerank", 1)],
        description="Sparse + API+Reranker (VRAM: 2)",
    ),
    PhaseConfig(
        phase_num=3,
        modes=[("qdrant_hybrid", 1), ("qdrant_hybrid_3large", 1)],
        description="BGE Hybrid + OpenAI Hybrid (VRAM: 2)",
    ),
    PhaseConfig(
        phase_num=4,
        modes=[("qdrant_dense+rerank", 2)],
        description="Dense + Rerank sequential (VRAM: 2)",
    ),
    PhaseConfig(
        phase_num=5,
        modes=[("qdrant_hybrid+rerank", 2)],
        description="Hybrid + Rerank sequential (VRAM: 2)",
    ),
    PhaseConfig(
        phase_num=6,
        modes=[("qdrant_hybrid_3large+rerank", 2)],
        description="OpenAI Hybrid + Rerank sequential (VRAM: 2)",
    ),
    PhaseConfig(
        phase_num=7,
        modes=[("qdrant_sparse+rerank", 2)],
        description="Sparse + Rerank sequential (VRAM: 2)",
    ),
]


def run_mode_subprocess(
    mode: str,
    benchmark_file: str,
    output_dir: Path,
    top_k: int = 10,
    rerank_top_k: int = 10,
) -> dict:
    """
    Run a single mode evaluation in a subprocess.

    Subprocess isolation ensures:
    1. CUDA context is process-local (no conflicts)
    2. GPU memory is fully released when process exits
    3. OOM in one mode doesn't crash others

    Returns:
        Dict with mode name, success status, and results/error
    """
    output_file = output_dir / f"{mode.replace('+', '_plus_')}_results.json"

    cmd = [
        sys.executable,
        "scripts/06_evaluate.py",
        "--modes", mode,
        "--benchmark", benchmark_file,
        "--top-k", str(top_k),
        "--rerank-top-k", str(rerank_top_k),
        "--output", str(output_file),
    ]

    start_time = time.time()
    print(f"  [{mode}] Starting subprocess...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout per mode
            cwd=Path(__file__).parent.parent,  # Project root
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            # Load results from output file
            if output_file.exists():
                with open(output_file) as f:
                    results = json.load(f)
                print(f"  [{mode}] ✓ Completed in {elapsed:.1f}s")
                return {
                    "mode": mode,
                    "success": True,
                    "elapsed_seconds": elapsed,
                    "results": results.get(mode, {}),
                }
            else:
                print(f"  [{mode}] ⚠ Completed but no output file")
                return {
                    "mode": mode,
                    "success": True,
                    "elapsed_seconds": elapsed,
                    "results": {},
                    "warning": "No output file generated",
                }
        else:
            print(f"  [{mode}] ✗ Failed (exit code {result.returncode})")
            print(f"    stderr: {result.stderr[:500]}...")
            return {
                "mode": mode,
                "success": False,
                "elapsed_seconds": elapsed,
                "error": result.stderr[:1000],
                "returncode": result.returncode,
            }

    except subprocess.TimeoutExpired:
        print(f"  [{mode}] ✗ Timeout after 1 hour")
        return {
            "mode": mode,
            "success": False,
            "error": "Timeout after 3600 seconds",
        }
    except Exception as e:
        print(f"  [{mode}] ✗ Exception: {e}")
        return {
            "mode": mode,
            "success": False,
            "error": str(e),
        }


def run_phase(
    phase: PhaseConfig,
    benchmark_file: str,
    output_dir: Path,
    top_k: int = 10,
    rerank_top_k: int = 10,
) -> list[dict]:
    """
    Run all modes in a phase in parallel.

    Each mode runs in its own subprocess for CUDA isolation.
    """
    modes = [mode for mode, _ in phase.modes]

    print(f"\n{'='*60}")
    print(f"Phase {phase.phase_num}: {phase.description}")
    print(f"Modes: {', '.join(modes)}")
    print("="*60)

    phase_start = time.time()
    results = []

    # Run modes in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=len(modes)) as executor:
        futures = {
            executor.submit(
                run_mode_subprocess,
                mode,
                benchmark_file,
                output_dir,
                top_k,
                rerank_top_k,
            ): mode
            for mode in modes
        }

        for future in as_completed(futures):
            mode = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"  [{mode}] ✗ Future exception: {e}")
                results.append({
                    "mode": mode,
                    "success": False,
                    "error": str(e),
                })

    phase_elapsed = time.time() - phase_start
    successful = sum(1 for r in results if r.get("success"))
    print(f"\nPhase {phase.phase_num} completed: {successful}/{len(modes)} successful in {phase_elapsed:.1f}s")

    return results


def run_benchmark_parallel(
    benchmark_file: str,
    output_file: Optional[str] = None,
    top_k: int = 10,
    rerank_top_k: int = 10,
    phases: Optional[list[int]] = None,
) -> dict:
    """
    Run full parallel benchmark across all phases.

    Args:
        benchmark_file: Path to benchmark queries JSON
        output_file: Path to save combined results
        top_k: Number of results per query
        rerank_top_k: Number of results after reranking
        phases: Optional list of phase numbers to run (default: all)

    Returns:
        Combined results dict with all mode metrics
    """
    benchmark_path = Path(benchmark_file)
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_file}")

    # Create output directory for intermediate results
    output_dir = Path("data/eval/parallel_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter phases if specified
    phases_to_run = PHASES
    if phases:
        phases_to_run = [p for p in PHASES if p.phase_num in phases]

    print("=" * 60)
    print("PARALLEL BENCHMARK RUNNER")
    print("=" * 60)
    print(f"Benchmark file: {benchmark_file}")
    print(f"Output directory: {output_dir}")
    print(f"Top-K: {top_k}, Rerank Top-K: {rerank_top_k}")
    print(f"Phases to run: {[p.phase_num for p in phases_to_run]}")

    total_start = time.time()
    all_results = {}
    phase_summaries = []

    for phase in phases_to_run:
        phase_results = run_phase(
            phase,
            benchmark_file,
            output_dir,
            top_k,
            rerank_top_k,
        )

        # Collect results
        for result in phase_results:
            mode = result["mode"]
            if result.get("success") and result.get("results"):
                all_results[mode] = result["results"]

        phase_summaries.append({
            "phase": phase.phase_num,
            "description": phase.description,
            "modes": [r["mode"] for r in phase_results],
            "successful": [r["mode"] for r in phase_results if r.get("success")],
            "failed": [r["mode"] for r in phase_results if not r.get("success")],
        })

    total_elapsed = time.time() - total_start

    # Print final summary
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    print(f"Successful modes: {len(all_results)}/{sum(len(p.modes) for p in phases_to_run)}")

    # Print comparison table
    if all_results:
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(f"{'Mode':<30} {'MRR':>8} {'NDCG@10':>10} {'P@10':>8} {'Time(ms)':>10}")
        print("-" * 70)

        # Sort by MRR descending
        sorted_modes = sorted(
            all_results.items(),
            key=lambda x: x[1].get("avg_mrr", 0),
            reverse=True
        )

        for mode, metrics in sorted_modes:
            print(f"{mode:<30} "
                  f"{metrics.get('avg_mrr', 0):>8.3f} "
                  f"{metrics.get('avg_ndcg@10', 0):>10.3f} "
                  f"{metrics.get('avg_precision@10', 0):>8.3f} "
                  f"{metrics.get('avg_search_time_ms', 0):>10.0f}")

    # Save combined results
    combined_output = {
        "metadata": {
            "benchmark_file": str(benchmark_file),
            "total_time_seconds": total_elapsed,
            "phases_run": [p.phase_num for p in phases_to_run],
            "top_k": top_k,
            "rerank_top_k": rerank_top_k,
        },
        "phase_summaries": phase_summaries,
        "results": all_results,
    }

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(combined_output, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    # Also save to default location
    default_output = output_dir / "combined_results.json"
    with open(default_output, "w") as f:
        json.dump(combined_output, f, indent=2)
    print(f"Results also saved to: {default_output}")

    return combined_output


def main():
    parser = argparse.ArgumentParser(
        description="Run parallel benchmark evaluation with VRAM constraints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phase Strategy (VRAM ≤ 2 constraint):
  Phase 1: qdrant_3large + qdrant_dense (parallel)
  Phase 2: qdrant_sparse + qdrant_3large+rerank (parallel)
  Phase 3: qdrant_hybrid + qdrant_hybrid_3large (parallel)
  Phase 4: qdrant_dense+rerank (sequential)
  Phase 5: qdrant_hybrid+rerank (sequential)
  Phase 6: qdrant_hybrid_3large+rerank (sequential)

Examples:
  # Run all phases
  python scripts/run_benchmark_parallel.py --benchmark data/eval/test_v2_benchmark.json

  # Run specific phases only
  python scripts/run_benchmark_parallel.py --benchmark data/eval/test_v2_benchmark.json --phases 1 2 3

  # Custom output location
  python scripts/run_benchmark_parallel.py --benchmark data/eval/test_v2_benchmark.json --output results.json
        """,
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help="Path to benchmark queries JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for combined results",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to retrieve (default: 10)",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=10,
        help="Number of results after reranking (default: 10)",
    )
    parser.add_argument(
        "--phases",
        type=int,
        nargs="+",
        help="Specific phases to run (default: all phases 1-6)",
    )

    args = parser.parse_args()

    try:
        results = run_benchmark_parallel(
            benchmark_file=args.benchmark,
            output_file=args.output,
            top_k=args.top_k,
            rerank_top_k=args.rerank_top_k,
            phases=args.phases,
        )
        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
