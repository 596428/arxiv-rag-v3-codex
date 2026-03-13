#!/usr/bin/env python3
"""
Test qdrant_3large with HyDE expansion.

Applies HyDE (Hypothetical Document Embeddings) to queries before
searching with OpenAI text-embedding-3-large.
"""

import json
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.qdrant_retriever import QdrantHybridRetriever


class OpenAIHyDEExpander:
    """HyDE expander using OpenAI GPT-4o (faster than Gemini)."""

    PROMPT = """You are a research paper abstract generator. Given a search query, generate a hypothetical abstract (2-3 sentences) for a paper that would perfectly answer the query.

The abstract should:
1. Use technical terminology appropriate for academic ML/AI papers
2. Mention specific methods, architectures, or techniques
3. Include concrete claims about results or contributions
4. Be written in academic style

Query: {query}

Hypothetical Abstract:"""

    def __init__(self, model: str = "gpt-4o-mini"):
        import openai
        self.client = openai.OpenAI()
        self.model = model
        self._cache = {}

    def expand(self, query: str) -> str:
        """Expand query to hypothetical abstract."""
        if query in self._cache:
            return self._cache[query]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": self.PROMPT.format(query=query)}],
            max_tokens=200,
            temperature=0.7,
        )
        expanded = response.choices[0].message.content.strip()
        self._cache[query] = expanded
        return expanded

    def expand_detailed(self, query: str):
        """Expand with timing info."""
        import time
        from dataclasses import dataclass

        @dataclass
        class Result:
            expanded_text: str
            success: bool
            latency_ms: float = 0.0

        start = time.time()
        try:
            expanded = self.expand(query)
            return Result(expanded_text=expanded, success=True, latency_ms=(time.time() - start) * 1000)
        except Exception as e:
            return Result(expanded_text=query, success=False, latency_ms=(time.time() - start) * 1000)


@dataclass
class EvalQuery:
    """A single evaluation query with ground truth."""
    query: str
    relevant_papers: list[str] = field(default_factory=list)
    style: str = ""
    difficulty: str = ""


def dedupe_by_paper(results) -> list:
    """Deduplicate results by paper_id, keeping highest scoring chunk per paper."""
    seen = set()
    deduped = []
    for r in results:
        if r.paper_id not in seen:
            seen.add(r.paper_id)
            deduped.append(r)
    return deduped


def calculate_mrr(results, relevant_papers: list[str]) -> float:
    """Calculate Mean Reciprocal Rank (deduplicated by paper)."""
    deduped = dedupe_by_paper(results)
    for i, result in enumerate(deduped, 1):
        if result.paper_id in relevant_papers:
            return 1.0 / i
    return 0.0


def calculate_ndcg(results, relevant_papers: list[str], k: int = 10) -> float:
    """Calculate NDCG@k (deduplicated by paper)."""
    import math

    deduped = dedupe_by_paper(results)[:k]

    dcg = 0.0
    for i, result in enumerate(deduped, 1):
        if result.paper_id in relevant_papers:
            dcg += 1.0 / math.log2(i + 1)

    # Ideal DCG: assume we have min(num_relevant, k) relevant docs at top positions
    num_relevant = len(relevant_papers)
    if num_relevant == 0:
        return 0.0

    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(num_relevant, k) + 1))

    return dcg / idcg if idcg > 0 else 0.0


def calculate_precision(results, relevant_papers: list[str], k: int) -> float:
    """Calculate Precision@k (deduplicated by paper)."""
    deduped = dedupe_by_paper(results)[:k]
    hits = sum(1 for r in deduped if r.paper_id in relevant_papers)
    return hits / k


def calculate_hit_rate(results, relevant_papers: list[str], k: int = 10) -> float:
    """Calculate Hit Rate@k (whether any relevant doc is in top-k)."""
    deduped = dedupe_by_paper(results)[:k]
    for result in deduped:
        if result.paper_id in relevant_papers:
            return 1.0
    return 0.0


def run_benchmark(
    benchmark_file: str,
    limit: int = None,
    use_hyde: bool = True,
    hyde_for_all: bool = False,  # Apply HyDE to all queries, not just conceptual
):
    """Run benchmark with HyDE."""

    # Load benchmark
    with open(benchmark_file) as f:
        data = json.load(f)

    # Handle both list and dict formats
    queries = data if isinstance(data, list) else data.get("queries", [])
    if limit:
        queries = queries[:limit]

    print(f"Loaded {len(queries)} queries from {benchmark_file}")
    print(f"HyDE enabled: {use_hyde}, HyDE for all: {hyde_for_all}")
    print("=" * 70)

    # Initialize
    retriever = QdrantHybridRetriever()
    hyde_expander = OpenAIHyDEExpander(model="gpt-4o-mini") if use_hyde else None

    # Metrics accumulators
    all_metrics = []
    by_style = {}
    by_difficulty = {}

    total_hyde_time = 0.0
    hyde_count = 0

    for i, q in enumerate(queries):
        eval_query = EvalQuery(
            query=q["query"],
            relevant_papers=q.get("relevant_papers", []),
            style=q.get("style", ""),
            difficulty=q.get("difficulty", ""),
        )

        # Determine if we should apply HyDE
        apply_hyde = False
        if use_hyde:
            if hyde_for_all:
                apply_hyde = True
            elif eval_query.style == "conceptual":
                apply_hyde = True

        # Apply HyDE expansion if needed
        search_query = eval_query.query
        hyde_latency = 0.0

        if apply_hyde and hyde_expander:
            hyde_start = time.time()
            result = hyde_expander.expand_detailed(eval_query.query)
            hyde_latency = (time.time() - hyde_start) * 1000

            if result.success:
                search_query = result.expanded_text
                total_hyde_time += hyde_latency
                hyde_count += 1

        # Search
        start = time.time()
        response = retriever.search_dense_3large(search_query, top_k=10)
        search_time = (time.time() - start) * 1000

        # Calculate metrics
        mrr = calculate_mrr(response.results, eval_query.relevant_papers)
        ndcg_10 = calculate_ndcg(response.results, eval_query.relevant_papers, k=10)
        p_5 = calculate_precision(response.results, eval_query.relevant_papers, k=5)
        hit_10 = calculate_hit_rate(response.results, eval_query.relevant_papers, k=10)

        metrics = {
            "mrr": mrr,
            "ndcg@10": ndcg_10,
            "p@5": p_5,
            "hit@10": hit_10,
            "time_ms": search_time + hyde_latency,
            "hyde_applied": apply_hyde and hyde_expander is not None,
            "style": eval_query.style,
            "difficulty": eval_query.difficulty,
        }
        all_metrics.append(metrics)

        # Accumulate by style
        style = eval_query.style or "unknown"
        if style not in by_style:
            by_style[style] = []
        by_style[style].append(metrics)

        # Accumulate by difficulty
        diff = eval_query.difficulty or "unknown"
        if diff not in by_difficulty:
            by_difficulty[diff] = []
        by_difficulty[diff].append(metrics)

        # Progress
        if (i + 1) % 100 == 0 or i == 0:
            print(f"[{i+1}/{len(queries)}] MRR: {mrr:.3f} | P@5: {p_5:.3f} | NDCG@10: {ndcg_10:.3f} | Time: {search_time + hyde_latency:.0f}ms | HyDE: {apply_hyde}")

    # Calculate averages
    n = len(all_metrics)
    avg_mrr = sum(m["mrr"] for m in all_metrics) / n
    avg_ndcg = sum(m["ndcg@10"] for m in all_metrics) / n
    avg_p5 = sum(m["p@5"] for m in all_metrics) / n
    avg_hit = sum(m["hit@10"] for m in all_metrics) / n
    avg_time = sum(m["time_ms"] for m in all_metrics) / n

    mode_name = "qdrant_3large+hyde" if use_hyde else "qdrant_3large"
    if hyde_for_all:
        mode_name += "_all"

    print("\n" + "=" * 70)
    print(f"{mode_name.upper()} Summary:")
    print(f"  Avg MRR:      {avg_mrr:.3f}")
    print(f"  Avg NDCG@10:  {avg_ndcg:.3f}")
    print(f"  Avg P@5:      {avg_p5:.3f}")
    print(f"  Avg Hit@10:   {avg_hit:.1%}")
    print(f"  Avg Time:     {avg_time:.0f}ms")
    if hyde_count > 0:
        print(f"  HyDE Applied: {hyde_count} queries (avg {total_hyde_time/hyde_count:.0f}ms)")

    # By style
    print(f"\n{mode_name.upper()} By Style:")
    print(f"  {'Style':<20} {'MRR':>8} {'NDCG@10':>10} {'Hit@10':>10} {'Count':>8}")
    print("  " + "-" * 60)
    for style in ["keyword", "natural_short", "natural_long", "conceptual"]:
        if style in by_style:
            metrics = by_style[style]
            s_mrr = sum(m["mrr"] for m in metrics) / len(metrics)
            s_ndcg = sum(m["ndcg@10"] for m in metrics) / len(metrics)
            s_hit = sum(m["hit@10"] for m in metrics) / len(metrics)
            print(f"  {style:<20} {s_mrr:>8.3f} {s_ndcg:>10.3f} {s_hit:>9.0%} {len(metrics):>8}")

    # By difficulty
    print(f"\n{mode_name.upper()} By Difficulty:")
    print(f"  {'Difficulty':<15} {'MRR':>8} {'Hit@10':>10} {'Count':>8}")
    print("  " + "-" * 45)
    for diff in ["easy", "medium", "hard"]:
        if diff in by_difficulty:
            metrics = by_difficulty[diff]
            d_mrr = sum(m["mrr"] for m in metrics) / len(metrics)
            d_hit = sum(m["hit@10"] for m in metrics) / len(metrics)
            print(f"  {diff:<15} {d_mrr:>8.3f} {d_hit:>9.0%} {len(metrics):>8}")

    # Save results
    results = {
        mode_name: {
            "avg_mrr": avg_mrr,
            "avg_ndcg@10": avg_ndcg,
            "avg_precision@5": avg_p5,
            "avg_hit@10": avg_hit,
            "avg_time_ms": avg_time,
            "num_queries": n,
            "hyde_applied_count": hyde_count,
            "by_style": {
                style: {
                    "avg_mrr": sum(m["mrr"] for m in metrics) / len(metrics),
                    "avg_ndcg@10": sum(m["ndcg@10"] for m in metrics) / len(metrics),
                    "hit_rate@10": sum(m["hit@10"] for m in metrics) / len(metrics),
                    "count": len(metrics),
                }
                for style, metrics in by_style.items()
            },
            "by_difficulty": {
                diff: {
                    "avg_mrr": sum(m["mrr"] for m in metrics) / len(metrics),
                    "hit_rate@10": sum(m["hit@10"] for m in metrics) / len(metrics),
                    "count": len(metrics),
                }
                for diff, metrics in by_difficulty.items()
            },
        }
    }

    output_file = Path("data/eval/parallel_results") / f"{mode_name}_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test qdrant_3large with HyDE")
    parser.add_argument("--benchmark", default="data/eval/v2_full_benchmark.json")
    parser.add_argument("--limit", type=int, help="Limit number of queries")
    parser.add_argument("--no-hyde", action="store_true", help="Disable HyDE")
    parser.add_argument("--hyde-all", action="store_true", help="Apply HyDE to all queries")

    args = parser.parse_args()

    run_benchmark(
        benchmark_file=args.benchmark,
        limit=args.limit,
        use_hyde=not args.no_hyde,
        hyde_for_all=args.hyde_all,
    )
