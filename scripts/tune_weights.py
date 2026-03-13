#!/usr/bin/env python3
"""
arXiv RAG v3 - RRF Weight Tuning

Grid search for optimal RRF weights across dense, sparse, and ColBERT retrievers.

Usage:
    python scripts/tune_weights.py                    # Run with default eval queries
    python scripts/tune_weights.py --queries queries.json  # Custom queries
    python scripts/tune_weights.py                    # Use Qdrant-based tuning
"""

import argparse
import json
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging import get_logger

logger = get_logger("tune_weights")


@dataclass
class EvalQuery:
    """A single evaluation query with ground truth."""
    query: str
    relevant_papers: list[str]
    relevant_chunks: list[str] = None
    category: str = ""

    def __post_init__(self):
        if self.relevant_chunks is None:
            self.relevant_chunks = []


@dataclass
class TuningResult:
    """Result of a single weight configuration test."""
    dense_weight: float
    sparse_weight: float
    colbert_weight: float
    avg_mrr: float
    avg_ndcg_10: float
    avg_precision_5: float
    avg_latency_ms: float
    num_queries: int


def calculate_mrr(relevance: list[float]) -> float:
    """Calculate Mean Reciprocal Rank."""
    for i, rel in enumerate(relevance):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


def calculate_ndcg(relevance: list[float], k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain at k."""
    import math

    def dcg(scores):
        return sum(rel / math.log2(i + 2) for i, rel in enumerate(scores[:k]))

    actual_dcg = dcg(relevance)
    ideal_dcg = dcg(sorted(relevance, reverse=True))
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def evaluate_weight_config_legacy(
    queries: list[EvalQuery],
    dense_weight: float,
    sparse_weight: float,
    colbert_weight: float,
    top_k: int = 10,
) -> TuningResult:
    """Deprecated legacy entry point kept only for explicit failure."""
    raise RuntimeError("Legacy Supabase tuning is removed in v3. Use Qdrant tuning only.")


def evaluate_weight_config_qdrant(
    queries: list[EvalQuery],
    dense_weight: float,
    sparse_weight: float,
    colbert_weight: float,
    top_k: int = 10,
) -> TuningResult:
    """Evaluate a weight configuration using Qdrant retriever."""
    import time
    from src.rag.qdrant_retriever import QdrantHybridRetriever

    retriever = QdrantHybridRetriever(
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        colbert_weight=colbert_weight,
    )

    mrr_scores = []
    ndcg_scores = []
    precision_scores = []
    latencies = []

    for eq in queries:
        start = time.time()
        response = retriever.search(eq.query, top_k=top_k)
        latency = (time.time() - start) * 1000
        latencies.append(latency)

        # Calculate relevance
        relevance = []
        for result in response.results:
            is_relevant = (
                result.paper_id in eq.relevant_papers or
                result.chunk_id in eq.relevant_chunks
            )
            relevance.append(1.0 if is_relevant else 0.0)

        mrr_scores.append(calculate_mrr(relevance))
        ndcg_scores.append(calculate_ndcg(relevance, 10))
        precision_scores.append(sum(relevance[:5]) / 5 if len(relevance) >= 5 else sum(relevance) / max(len(relevance), 1))

    # Cleanup
    retriever.unload_models()

    return TuningResult(
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        colbert_weight=colbert_weight,
        avg_mrr=sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0,
        avg_ndcg_10=sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0,
        avg_precision_5=sum(precision_scores) / len(precision_scores) if precision_scores else 0,
        avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
        num_queries=len(queries),
    )


def get_default_eval_queries() -> list[EvalQuery]:
    """Return default evaluation queries."""
    return [
        EvalQuery(
            query="What is RLHF and how does reinforcement learning from human feedback improve language models?",
            relevant_papers=["2501.01031v3", "2501.01336v1"],
            category="alignment",
        ),
        EvalQuery(
            query="How does retrieval augmented generation (RAG) improve LLM responses?",
            relevant_papers=["2501.00879v3", "2501.01031v3"],
            category="rag",
        ),
        EvalQuery(
            query="What techniques are used for multi-tool reasoning in LLMs?",
            relevant_papers=["2501.01290v1", "2501.00830v2"],
            category="reasoning",
        ),
        EvalQuery(
            query="How do multimodal large language models process both text and images?",
            relevant_papers=["2501.00750v2", "2501.01645v3"],
            category="multimodal",
        ),
        EvalQuery(
            query="How are LLM agents used for autonomous problem solving?",
            relevant_papers=["2501.01205v1", "2501.00750v2"],
            category="agents",
        ),
        EvalQuery(
            query="What benchmarks are used to evaluate LLM capabilities?",
            relevant_papers=["2501.01243v3", "2501.01290v1"],
            category="evaluation",
        ),
        EvalQuery(
            query="How do attention mechanisms work in transformer models?",
            relevant_papers=["2501.00759v3", "2501.01073v2"],
            category="architecture",
        ),
        EvalQuery(
            query="What methods improve trustworthiness and robustness of RAG systems?",
            relevant_papers=["2501.00879v3", "2501.00888v1"],
            category="rag",
        ),
    ]


def run_grid_search(
    queries: list[EvalQuery],
    use_qdrant: bool = True,
    weight_step: float = 0.1,
    top_k: int = 10,
) -> list[TuningResult]:
    """
    Run grid search over weight combinations.

    Args:
        queries: Evaluation queries
        use_qdrant: Use Qdrant backend
        weight_step: Step size for weight grid
        top_k: Number of results to retrieve

    Returns:
        List of tuning results sorted by MRR
    """
    # Generate weight combinations that sum to approximately 1.0
    weights = [round(w, 1) for w in [i * weight_step for i in range(11)]]

    # Filter combinations where sum is close to 1.0
    combinations = []
    for d, s, c in itertools.product(weights, weights, weights):
        total = d + s + c
        if 0.9 <= total <= 1.1 and total > 0:  # Allow slight variance
            combinations.append((d / total, s / total, c / total))  # Normalize

    # Remove duplicates
    combinations = list(set(combinations))

    logger.info(f"Testing {len(combinations)} weight combinations...")

    if not use_qdrant:
        raise RuntimeError("Legacy weight tuning is removed in v3. Use Qdrant mode.")
    evaluate_fn = evaluate_weight_config_qdrant

    results = []
    for i, (d, s, c) in enumerate(combinations):
        logger.info(f"[{i+1}/{len(combinations)}] Testing dense={d:.1f}, sparse={s:.1f}, colbert={c:.1f}")

        try:
            result = evaluate_fn(queries, d, s, c, top_k=top_k)
            results.append(result)
            logger.info(f"  MRR={result.avg_mrr:.3f}, NDCG@10={result.avg_ndcg_10:.3f}, Latency={result.avg_latency_ms:.0f}ms")
        except Exception as e:
            logger.error(f"  Failed: {e}")

    # Sort by MRR descending
    results.sort(key=lambda r: r.avg_mrr, reverse=True)

    return results


def main():
    parser = argparse.ArgumentParser(description="RRF Weight Tuning")
    parser.add_argument(
        "--queries",
        type=str,
        help="Path to JSON file with evaluation queries",
    )
    parser.add_argument(
        "--use-qdrant",
        action="store_true",
        help="Use Qdrant-based tuning (required in v3)",
    )
    parser.add_argument(
        "--weight-step",
        type=float,
        default=0.1,
        help="Weight grid step size (default: 0.1)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to retrieve (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Show top N configurations (default: 10)",
    )

    args = parser.parse_args()

    # Load queries
    if args.queries:
        with open(args.queries) as f:
            query_data = json.load(f)
            queries = [EvalQuery(**q) for q in query_data]
    else:
        queries = get_default_eval_queries()

    print(f"Running grid search with {len(queries)} queries")
    if not args.use_qdrant:
        raise SystemExit('Supabase tuning is deprecated in v3. Re-run with --use-qdrant.')

    print('Backend: Qdrant')
    print(f"Weight step: {args.weight_step}")

    # Run grid search
    results = run_grid_search(
        queries,
        use_qdrant=args.use_qdrant,
        weight_step=args.weight_step,
        top_k=args.top_k,
    )

    # Print top configurations
    print("\n" + "=" * 80)
    print(f"TOP {args.top_n} WEIGHT CONFIGURATIONS")
    print("=" * 80)
    print(f"{'Rank':<5} {'Dense':>7} {'Sparse':>8} {'ColBERT':>9} {'MRR':>8} {'NDCG@10':>10} {'P@5':>8} {'Latency':>10}")
    print("-" * 80)

    for i, r in enumerate(results[:args.top_n]):
        print(f"{i+1:<5} {r.dense_weight:>7.2f} {r.sparse_weight:>8.2f} {r.colbert_weight:>9.2f} "
              f"{r.avg_mrr:>8.3f} {r.avg_ndcg_10:>10.3f} {r.avg_precision_5:>8.3f} {r.avg_latency_ms:>9.0f}ms")

    print("=" * 80)

    # Best configuration recommendation
    if results:
        best = results[0]
        print(f"\n🏆 RECOMMENDED CONFIGURATION:")
        print(f"   dense_weight  = {best.dense_weight:.2f}")
        print(f"   sparse_weight = {best.sparse_weight:.2f}")
        print(f"   colbert_weight = {best.colbert_weight:.2f}")
        print(f"\n   Expected MRR: {best.avg_mrr:.3f}")
        print(f"   Expected NDCG@10: {best.avg_ndcg_10:.3f}")
        print(f"   Expected Latency: {best.avg_latency_ms:.0f}ms")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "best_config": {
                "dense_weight": results[0].dense_weight if results else 0.4,
                "sparse_weight": results[0].sparse_weight if results else 0.3,
                "colbert_weight": results[0].colbert_weight if results else 0.3,
                "avg_mrr": results[0].avg_mrr if results else 0,
                "avg_ndcg_10": results[0].avg_ndcg_10 if results else 0,
            },
            "all_results": [
                {
                    "dense_weight": r.dense_weight,
                    "sparse_weight": r.sparse_weight,
                    "colbert_weight": r.colbert_weight,
                    "avg_mrr": r.avg_mrr,
                    "avg_ndcg_10": r.avg_ndcg_10,
                    "avg_precision_5": r.avg_precision_5,
                    "avg_latency_ms": r.avg_latency_ms,
                }
                for r in results
            ],
            "num_queries": len(queries),
            "backend": "qdrant",
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
