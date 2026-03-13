#!/usr/bin/env python3
"""
arXiv RAG v1 - Search Quality Evaluation Script (Optimized)

Evaluates search quality using:
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)
- Precision@K
- Recall@K

Optimizations:
- RetrieverPool: Type-based caching to avoid reloading models per query
- 2-Pass Execution: Non-rerank modes first (pool reuse), then rerank modes separately
- GPU Memory Management: Proper unload lifecycle for reranker compatibility

Usage:
    python scripts/06_evaluate.py                    # Run with default test queries
    python scripts/06_evaluate.py --queries queries.json  # Custom queries
    python scripts/06_evaluate.py --modes qdrant_hybrid qdrant_dense  # Compare modes
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.retriever import (
    HybridRetriever,
    HybridFullRetriever,
    OpenAIRetriever,
    ColBERTRetriever,
    SearchResponse,
)
from src.rag.qdrant_retriever import (
    QdrantHybridRetriever,
)


@dataclass
class EvalQuery:
    """A single evaluation query with ground truth."""
    query: str
    relevant_papers: list[str] = field(default_factory=list)  # arxiv_ids
    relevant_chunks: list[str] = field(default_factory=list)  # chunk_ids
    category: str = ""  # Query category (e.g., "methodology", "results")
    original_relevant_count: int = 0  # Original count before filtering
    # v2 benchmark fields
    style: str = ""  # Query style (keyword, natural_short, natural_long, conceptual)
    hard_negatives: list[str] = field(default_factory=list)  # Hard negative paper IDs
    difficulty: str = ""  # Difficulty level (easy, medium, hard)
    metadata: dict = field(default_factory=dict)  # Additional metadata


@dataclass
class EvalMetrics:
    """Evaluation metrics for a single query."""
    query: str
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    recall_at_10: float = 0.0
    search_time_ms: float = 0.0
    num_results: int = 0
    # v2 fields for style-based analysis
    style: str = ""
    difficulty: str = ""


class RetrieverPool:
    """
    Type-based retriever pool for efficient model reuse.

    Caches retrievers by their underlying type (not mode), allowing
    multiple modes that use the same retriever type to share the loaded model.

    Important: Call unload_all() only after all non-rerank queries are complete.
    Rerank modes should NOT use this pool (they need fresh instances for unload cycle).
    """

    def __init__(self):
        self._hybrid: Optional[HybridRetriever] = None
        self._hybrid_full: Optional[HybridFullRetriever] = None
        self._qdrant: Optional[QdrantHybridRetriever] = None
        self._openai: Optional[OpenAIRetriever] = None
        self._colbert: Optional[ColBERTRetriever] = None

    @property
    def hybrid(self) -> HybridRetriever:
        """Get or create HybridRetriever (Supabase-based)."""
        if self._hybrid is None:
            print("  [Pool] Loading HybridRetriever (BGE-M3)...")
            self._hybrid = HybridRetriever()
        return self._hybrid

    @property
    def hybrid_full(self) -> HybridFullRetriever:
        """Get or create HybridFullRetriever (dense + sparse + ColBERT)."""
        if self._hybrid_full is None:
            print("  [Pool] Loading HybridFullRetriever (BGE-M3 + ColBERT)...")
            self._hybrid_full = HybridFullRetriever()
        return self._hybrid_full

    @property
    def qdrant(self) -> QdrantHybridRetriever:
        """Get or create QdrantHybridRetriever."""
        if self._qdrant is None:
            print("  [Pool] Loading QdrantHybridRetriever (BGE-M3)...")
            self._qdrant = QdrantHybridRetriever()
        return self._qdrant

    @property
    def openai(self) -> OpenAIRetriever:
        """Get or create OpenAIRetriever (API-based, no GPU)."""
        if self._openai is None:
            print("  [Pool] Loading OpenAIRetriever (API)...")
            self._openai = OpenAIRetriever()
        return self._openai

    @property
    def colbert(self) -> ColBERTRetriever:
        """Get or create ColBERTRetriever."""
        if self._colbert is None:
            print("  [Pool] Loading ColBERTRetriever (BGE-M3)...")
            self._colbert = ColBERTRetriever()
        return self._colbert

    def unload_all(self) -> None:
        """
        Unload all cached retrievers to free GPU memory.

        Call this ONLY after all non-rerank queries are complete.
        """
        print("  [Pool] Unloading all retrievers...")

        if self._hybrid is not None:
            try:
                self._hybrid.unload_models()
            except Exception:
                pass
            self._hybrid = None

        if self._hybrid_full is not None:
            try:
                self._hybrid_full.unload_models()
            except Exception:
                pass
            self._hybrid_full = None

        if self._qdrant is not None:
            try:
                self._qdrant.unload_models()
            except Exception:
                pass
            self._qdrant = None

        if self._colbert is not None:
            try:
                self._colbert.unload_models()
            except Exception:
                pass
            self._colbert = None

        # OpenAI doesn't need GPU unload (API-based)
        self._openai = None

        # Clear CUDA cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def calculate_dcg(relevance_scores: list[float], k: int) -> float:
    """Calculate Discounted Cumulative Gain at k."""
    import math
    dcg = 0.0
    for i, rel in enumerate(relevance_scores[:k]):
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1)=0
    return dcg


def calculate_ndcg(relevance_scores: list[float], k: int) -> float:
    """Calculate Normalized DCG at k."""
    dcg = calculate_dcg(relevance_scores, k)
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = calculate_dcg(ideal_scores, k)
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_query_with_pool(
    eval_query: EvalQuery,
    pool: RetrieverPool,
    top_k: int = 10,
    mode: str = "hybrid",
) -> EvalMetrics:
    """
    Evaluate a single query using the retriever pool.

    This function is for NON-RERANK modes only.
    Rerank modes should use evaluate_query_with_rerank() instead.
    """
    start_time = time.time()

    # Get the appropriate retriever from pool based on mode
    if mode == "hybrid":
        response = pool.hybrid.search(eval_query.query, top_k=top_k)
    elif mode == "dense":
        response = pool.hybrid.search_dense_only(eval_query.query, top_k=top_k)
    elif mode == "sparse":
        response = pool.hybrid.search_sparse_only(eval_query.query, top_k=top_k)
    elif mode == "openai":
        response = pool.openai.search(eval_query.query, top_k=top_k)
    elif mode == "colbert":
        import time as time_mod
        search_start = time_mod.time()
        results = pool.colbert.search(eval_query.query, top_k=top_k)
        response = SearchResponse(
            query=eval_query.query,
            results=results,
            total_found=len(results),
            colbert_count=len(results),
            search_time_ms=(time_mod.time() - search_start) * 1000,
        )
    elif mode == "hybrid_full":
        response = pool.hybrid_full.search(eval_query.query, top_k=top_k)
    elif mode == "qdrant_hybrid":
        response = pool.qdrant.search(eval_query.query, top_k=top_k)
    elif mode == "qdrant_dense":
        response = pool.qdrant.search_dense_only(eval_query.query, top_k=top_k)
    elif mode == "qdrant_sparse":
        response = pool.qdrant.search_sparse_only(eval_query.query, top_k=top_k)
    elif mode == "qdrant_3large":
        # OpenAI text-embedding-3-large via Qdrant (dense_3large vector)
        response = pool.qdrant.search_dense_3large(eval_query.query, top_k=top_k)
    elif mode == "qdrant_hybrid_3large":
        # OpenAI dense (3large) + BGE sparse hybrid via Qdrant
        response = pool.qdrant.search_hybrid_3large(eval_query.query, top_k=top_k)
    elif mode == "qdrant_adaptive":
        # Adaptive search with query classification + HyDE for conceptual
        response = pool.qdrant.search_adaptive(eval_query.query, top_k=top_k, use_hyde=True)
    elif mode == "qdrant_adaptive_no_hyde":
        # Adaptive search without HyDE expansion
        response = pool.qdrant.search_adaptive(eval_query.query, top_k=top_k, use_hyde=False)
    else:
        raise ValueError(f"Unknown non-rerank mode: {mode}")

    search_time = (time.time() - start_time) * 1000

    return _calculate_metrics(eval_query, response, search_time)


def evaluate_query_with_rerank(
    eval_query: EvalQuery,
    top_k: int = 10,
    mode: str = "hybrid+rerank",
    rerank_top_k: int = 10,
    lightweight_reranker: bool = False,
    reranker=None,  # Pass shared reranker instance for efficiency
    retriever=None,  # Pass shared retriever instance for efficiency
) -> EvalMetrics:
    """
    Evaluate a single query with reranking.

    If retriever is provided, it will be reused (caller manages lifecycle).
    If not provided, a new retriever is created per call (slow).

    If reranker is provided, it will be reused (caller manages lifecycle).
    If not provided, a new reranker is created and unloaded per call.
    """
    from src.rag.reranker import BGEReranker, LightweightReranker

    start_time = time.time()
    base_mode = mode.replace("+rerank", "")
    owns_reranker = reranker is None  # Track if we created it
    owns_retriever = retriever is None  # Track if we created it

    # Use provided retriever or create new one
    if base_mode in ("hybrid", "dense", "sparse"):
        if retriever is None:
            retriever = HybridRetriever()
        if base_mode == "hybrid":
            response = retriever.search(eval_query.query, top_k=top_k)
        elif base_mode == "dense":
            response = retriever.search_dense_only(eval_query.query, top_k=top_k)
        else:  # sparse
            response = retriever.search_sparse_only(eval_query.query, top_k=top_k)
        # Only unload if we created it
        if owns_retriever:
            retriever.unload_models()

    elif base_mode == "hybrid_full":
        if retriever is None:
            retriever = HybridFullRetriever()
        response = retriever.search(eval_query.query, top_k=top_k)
        if owns_retriever:
            retriever.unload_models()

    elif base_mode == "colbert":
        if retriever is None:
            retriever = ColBERTRetriever()
        results = retriever.search(eval_query.query, top_k=top_k)
        response = SearchResponse(
            query=eval_query.query,
            results=results,
            total_found=len(results),
            colbert_count=len(results),
            search_time_ms=0,
        )
        if owns_retriever:
            retriever.unload_models()

    elif base_mode in ("qdrant_hybrid", "qdrant_dense", "qdrant_sparse"):
        if retriever is None:
            retriever = QdrantHybridRetriever()
        if base_mode == "qdrant_hybrid":
            response = retriever.search(eval_query.query, top_k=top_k)
        elif base_mode == "qdrant_dense":
            response = retriever.search_dense_only(eval_query.query, top_k=top_k)
        else:  # qdrant_sparse
            response = retriever.search_sparse_only(eval_query.query, top_k=top_k)
        # Only unload if we created it
        if owns_retriever:
            retriever.unload_models()

    elif base_mode == "qdrant_3large":
        # OpenAI dense_3large via Qdrant (API-based, no GPU unload needed)
        if retriever is None:
            retriever = QdrantHybridRetriever()
        response = retriever.search_dense_3large(eval_query.query, top_k=top_k)

    elif base_mode == "qdrant_hybrid_3large":
        # OpenAI dense + BGE sparse hybrid via Qdrant
        if retriever is None:
            retriever = QdrantHybridRetriever()
        response = retriever.search_hybrid_3large(eval_query.query, top_k=top_k)
        # Only unload if we created it
        if owns_retriever:
            retriever.unload_models()

    else:
        raise ValueError(f"Unknown rerank mode: {mode}")

    # Clear CUDA cache before loading reranker
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # Apply reranking
    if response.results:
        # Create reranker only if not provided
        if reranker is None:
            if lightweight_reranker:
                reranker = LightweightReranker()
            else:
                reranker = BGEReranker()

        response.results = reranker.rerank(
            eval_query.query,
            response.results,
            top_k=rerank_top_k
        )

        # Only unload if we created it (caller manages shared instance)
        if owns_reranker:
            reranker.unload()

    search_time = (time.time() - start_time) * 1000

    return _calculate_metrics(eval_query, response, search_time)


def _calculate_metrics(
    eval_query: EvalQuery,
    response: SearchResponse,
    search_time_ms: float,
) -> EvalMetrics:
    """Calculate evaluation metrics from search response."""
    # Calculate relevance scores (binary for now)
    relevance = []
    for result in response.results:
        is_relevant = (
            result.paper_id in eval_query.relevant_papers or
            result.chunk_id in eval_query.relevant_chunks
        )
        relevance.append(1.0 if is_relevant else 0.0)

    # MRR: reciprocal rank of first relevant result
    mrr = 0.0
    for i, rel in enumerate(relevance):
        if rel > 0:
            mrr = 1.0 / (i + 1)
            break

    # Precision@K
    precision_at_5 = sum(relevance[:5]) / 5 if len(relevance) >= 5 else sum(relevance) / max(len(relevance), 1)
    precision_at_10 = sum(relevance[:10]) / 10 if len(relevance) >= 10 else sum(relevance) / max(len(relevance), 1)

    # Recall@10
    total_relevant = len(eval_query.relevant_papers) + len(eval_query.relevant_chunks)
    recall_at_10 = sum(relevance[:10]) / total_relevant if total_relevant > 0 else 0.0

    # NDCG
    ndcg_at_5 = calculate_ndcg(relevance, 5)
    ndcg_at_10 = calculate_ndcg(relevance, 10)

    return EvalMetrics(
        query=eval_query.query,
        mrr=mrr,
        ndcg_at_5=ndcg_at_5,
        ndcg_at_10=ndcg_at_10,
        precision_at_5=precision_at_5,
        precision_at_10=precision_at_10,
        recall_at_10=recall_at_10,
        search_time_ms=search_time_ms,
        num_results=len(response.results),
        style=eval_query.style,
        difficulty=eval_query.difficulty,
    )


def get_default_eval_queries() -> list[EvalQuery]:
    """Return default evaluation queries based on actually embedded papers."""
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


def run_evaluation(
    queries: list[EvalQuery],
    modes: list[str] = ["hybrid", "dense", "sparse"],
    top_k: int = 10,
    use_reranker: bool = False,
    lightweight_reranker: bool = False,
    rerank_top_k: int = 5,
    by_style: bool = False,
    by_difficulty: bool = False,
) -> dict:
    """
    Run full evaluation across modes using optimized 2-pass execution.

    Pass 1: Non-rerank modes (retriever pool reuse)
    Pass 2: Rerank modes (fresh retriever + unload + reranker cycle)
    """
    # Separate modes into non-rerank and rerank
    non_rerank_modes = [m for m in modes if "+rerank" not in m]
    rerank_modes = [m for m in modes if "+rerank" in m]

    # Also add rerank variants if use_reranker flag is set
    if use_reranker:
        for mode in list(non_rerank_modes):
            rerank_mode = f"{mode}+rerank"
            if rerank_mode not in rerank_modes:
                rerank_modes.append(rerank_mode)

    results = {}

    # ===== PASS 1: Non-rerank modes with pool reuse =====
    if non_rerank_modes:
        print("\n" + "=" * 60)
        print("PASS 1: Non-rerank modes (pool reuse)")
        print("=" * 60)

        pool = RetrieverPool()

        for mode in non_rerank_modes:
            print(f"\n{'='*60}")
            print(f"Evaluating mode: {mode.upper()}")
            print("="*60)

            mode_metrics = []

            for i, eq in enumerate(queries):
                print(f"\n[{i+1}/{len(queries)}] {eq.query[:60]}...")

                try:
                    metrics = evaluate_query_with_pool(
                        eq, pool, top_k=top_k, mode=mode
                    )
                    mode_metrics.append(metrics)

                    print(f"  MRR: {metrics.mrr:.3f} | P@5: {metrics.precision_at_5:.3f} | "
                          f"NDCG@10: {metrics.ndcg_at_10:.3f} | Time: {metrics.search_time_ms:.0f}ms")

                except Exception as e:
                    print(f"  ERROR: {e}")

            # Aggregate metrics
            if mode_metrics:
                if by_style:
                    results[mode] = _aggregate_metrics_by_style(mode_metrics, mode)
                    if by_difficulty:
                        results[mode]["by_difficulty"] = _aggregate_metrics_by_difficulty(mode_metrics, mode)
                elif by_difficulty:
                    results[mode] = _aggregate_metrics(mode_metrics, mode)
                    results[mode]["by_difficulty"] = _aggregate_metrics_by_difficulty(mode_metrics, mode)
                else:
                    results[mode] = _aggregate_metrics(mode_metrics, mode)

        # Unload pool after all non-rerank modes complete
        pool.unload_all()

    # ===== PASS 2: Rerank modes (fresh retriever per query) =====
    if rerank_modes:
        from src.rag.reranker import BGEReranker, LightweightReranker

        print("\n" + "=" * 60)
        print("PASS 2: Rerank modes (shared retriever + shared reranker)")
        print("=" * 60)

        # Create shared reranker instance (load once, reuse for all queries)
        print("\n[Pool] Loading reranker model...")
        if lightweight_reranker:
            shared_reranker = LightweightReranker()
        else:
            shared_reranker = BGEReranker()
        # Trigger lazy load
        _ = shared_reranker.model

        for mode in rerank_modes:
            print(f"\n{'='*60}")
            print(f"Evaluating mode: {mode.upper()}")
            print("="*60)

            # Create shared retriever for this mode (load once, reuse for all queries)
            base_mode = mode.replace("+rerank", "")
            shared_retriever = None

            if base_mode in ("hybrid", "dense", "sparse"):
                print("  [Pool] Loading HybridRetriever (BGE-M3)...")
                shared_retriever = HybridRetriever()
            elif base_mode == "hybrid_full":
                print("  [Pool] Loading HybridFullRetriever (BGE-M3 + ColBERT)...")
                shared_retriever = HybridFullRetriever()
            elif base_mode == "colbert":
                print("  [Pool] Loading ColBERTRetriever...")
                shared_retriever = ColBERTRetriever()
            elif base_mode in ("qdrant_hybrid", "qdrant_dense", "qdrant_sparse",
                               "qdrant_3large", "qdrant_hybrid_3large"):
                print("  [Pool] Loading QdrantHybridRetriever (BGE-M3)...")
                shared_retriever = QdrantHybridRetriever()

            mode_metrics = []

            for i, eq in enumerate(queries):
                print(f"\n[{i+1}/{len(queries)}] {eq.query[:60]}...")

                try:
                    metrics = evaluate_query_with_rerank(
                        eq,
                        top_k=top_k,
                        mode=mode,
                        rerank_top_k=rerank_top_k,
                        lightweight_reranker=lightweight_reranker,
                        reranker=shared_reranker,  # Pass shared reranker
                        retriever=shared_retriever,  # Pass shared retriever
                    )
                    mode_metrics.append(metrics)

                    print(f"  MRR: {metrics.mrr:.3f} | P@5: {metrics.precision_at_5:.3f} | "
                          f"NDCG@10: {metrics.ndcg_at_10:.3f} | Time: {metrics.search_time_ms:.0f}ms")

                except Exception as e:
                    print(f"  ERROR: {e}")

            # Aggregate metrics
            if mode_metrics:
                if by_style:
                    results[mode] = _aggregate_metrics_by_style(mode_metrics, mode)
                    if by_difficulty:
                        results[mode]["by_difficulty"] = _aggregate_metrics_by_difficulty(mode_metrics, mode)
                elif by_difficulty:
                    results[mode] = _aggregate_metrics(mode_metrics, mode)
                    results[mode]["by_difficulty"] = _aggregate_metrics_by_difficulty(mode_metrics, mode)
                else:
                    results[mode] = _aggregate_metrics(mode_metrics, mode)

            # Unload shared retriever after this mode completes
            if shared_retriever is not None:
                print(f"  [Pool] Unloading retriever for {mode}...")
                try:
                    shared_retriever.unload_models()
                except Exception:
                    pass

        # Unload shared reranker after all rerank modes complete
        print("\n[Pool] Unloading reranker model...")
        shared_reranker.unload()

    return results


def _aggregate_metrics(mode_metrics: list[EvalMetrics], mode: str) -> dict:
    """Aggregate metrics for a mode and print summary."""
    avg_mrr = sum(m.mrr for m in mode_metrics) / len(mode_metrics)
    avg_ndcg_5 = sum(m.ndcg_at_5 for m in mode_metrics) / len(mode_metrics)
    avg_ndcg_10 = sum(m.ndcg_at_10 for m in mode_metrics) / len(mode_metrics)
    avg_p5 = sum(m.precision_at_5 for m in mode_metrics) / len(mode_metrics)
    avg_p10 = sum(m.precision_at_10 for m in mode_metrics) / len(mode_metrics)
    avg_time = sum(m.search_time_ms for m in mode_metrics) / len(mode_metrics)

    result = {
        "avg_mrr": avg_mrr,
        "avg_ndcg@5": avg_ndcg_5,
        "avg_ndcg@10": avg_ndcg_10,
        "avg_precision@5": avg_p5,
        "avg_precision@10": avg_p10,
        "avg_search_time_ms": avg_time,
        "num_queries": len(mode_metrics),
    }

    print(f"\n{mode.upper()} Summary:")
    print(f"  Avg MRR:      {avg_mrr:.3f}")
    print(f"  Avg NDCG@5:   {avg_ndcg_5:.3f}")
    print(f"  Avg NDCG@10:  {avg_ndcg_10:.3f}")
    print(f"  Avg P@5:      {avg_p5:.3f}")
    print(f"  Avg P@10:     {avg_p10:.3f}")
    print(f"  Avg Time:     {avg_time:.0f}ms")

    return result


def _aggregate_metrics_by_style(mode_metrics: list[EvalMetrics], mode: str) -> dict:
    """
    Aggregate metrics by query style for detailed analysis.

    Returns dict with overall metrics plus per-style breakdowns.
    """
    # Group metrics by style
    by_style: dict[str, list[EvalMetrics]] = {}
    for m in mode_metrics:
        style = m.style or "unknown"
        if style not in by_style:
            by_style[style] = []
        by_style[style].append(m)

    # Calculate overall metrics
    overall = _aggregate_metrics(mode_metrics, mode)

    # Calculate per-style metrics
    style_results = {}
    styles_order = ["keyword", "natural_short", "natural_long", "conceptual", "unknown"]

    print(f"\n{mode.upper()} By Style:")
    print(f"  {'Style':<16} {'MRR':>8} {'NDCG@10':>10} {'Hit@10':>8} {'Count':>8}")
    print(f"  {'-'*52}")

    for style in styles_order:
        if style not in by_style:
            continue

        style_metrics = by_style[style]
        n = len(style_metrics)

        avg_mrr = sum(m.mrr for m in style_metrics) / n
        avg_ndcg_10 = sum(m.ndcg_at_10 for m in style_metrics) / n
        hit_rate = sum(1 for m in style_metrics if m.mrr > 0) / n

        style_results[style] = {
            "avg_mrr": avg_mrr,
            "avg_ndcg@10": avg_ndcg_10,
            "hit_rate@10": hit_rate,
            "count": n,
        }

        print(f"  {style:<16} {avg_mrr:>8.3f} {avg_ndcg_10:>10.3f} {hit_rate:>7.0%} {n:>8}")

    overall["by_style"] = style_results
    return overall


def _aggregate_metrics_by_difficulty(mode_metrics: list[EvalMetrics], mode: str) -> dict:
    """
    Aggregate metrics by difficulty level.
    """
    # Group by difficulty
    by_difficulty: dict[str, list[EvalMetrics]] = {}
    for m in mode_metrics:
        diff = m.difficulty or "unknown"
        if diff not in by_difficulty:
            by_difficulty[diff] = []
        by_difficulty[diff].append(m)

    diff_results = {}
    diff_order = ["easy", "medium", "hard", "unknown"]

    print(f"\n{mode.upper()} By Difficulty:")
    print(f"  {'Difficulty':<12} {'MRR':>8} {'Hit@10':>8} {'Count':>8}")
    print(f"  {'-'*38}")

    for diff in diff_order:
        if diff not in by_difficulty:
            continue

        diff_metrics = by_difficulty[diff]
        n = len(diff_metrics)

        avg_mrr = sum(m.mrr for m in diff_metrics) / n
        hit_rate = sum(1 for m in diff_metrics if m.mrr > 0) / n

        diff_results[diff] = {
            "avg_mrr": avg_mrr,
            "hit_rate@10": hit_rate,
            "count": n,
        }

        print(f"  {diff:<12} {avg_mrr:>8.3f} {hit_rate:>7.0%} {n:>8}")

    return diff_results


def main():
    parser = argparse.ArgumentParser(description="Search Quality Evaluation (Optimized)")
    parser.add_argument(
        "--queries", "--benchmark",
        type=str,
        dest="queries",
        help="Path to JSON file with evaluation queries",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["hybrid", "dense", "sparse"],
        help="""Search modes to evaluate. Available modes:
            Supabase: hybrid, dense, sparse, openai, colbert, hybrid_full
            Qdrant: qdrant_hybrid, qdrant_dense, qdrant_sparse, qdrant_3large, qdrant_hybrid_3large
            Adaptive: qdrant_adaptive (with HyDE), qdrant_adaptive_no_hyde
            Add '+rerank' suffix for reranking (e.g., qdrant_hybrid+rerank)""",
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
        "--use-reranker",
        action="store_true",
        default=False,
        help="Enable reranking for all modes (adds +rerank variants)",
    )
    parser.add_argument(
        "--lightweight-reranker",
        action="store_true",
        default=False,
        help="Use lightweight reranker (bge-reranker-base) instead of full model",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=5,
        help="Number of results after reranking (default: 5)",
    )
    parser.add_argument(
        "--by-style",
        action="store_true",
        default=False,
        help="Show metrics breakdown by query style (keyword, natural_short, natural_long, conceptual)",
    )
    parser.add_argument(
        "--by-difficulty",
        action="store_true",
        default=False,
        help="Show metrics breakdown by difficulty level (easy, medium, hard)",
    )

    args = parser.parse_args()

    # Load queries
    if args.queries:
        with open(args.queries) as f:
            query_data = json.load(f)
            queries = [EvalQuery(**q) for q in query_data]
    else:
        queries = get_default_eval_queries()

    print(f"Running evaluation with {len(queries)} queries")
    print(f"Modes: {args.modes}")
    print(f"Top-K: {args.top_k}")
    print(f"Reranker: {'lightweight' if args.lightweight_reranker else 'full' if args.use_reranker else 'disabled (unless +rerank in mode)'}")

    # Run evaluation
    results = run_evaluation(
        queries,
        modes=args.modes,
        top_k=args.top_k,
        use_reranker=args.use_reranker,
        lightweight_reranker=args.lightweight_reranker,
        rerank_top_k=args.rerank_top_k,
        by_style=args.by_style,
        by_difficulty=args.by_difficulty,
    )

    # Print comparison
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Mode':<25} {'MRR':>8} {'NDCG@10':>10} {'P@10':>8} {'Time(ms)':>10}")
    print("-"*60)
    for mode, metrics in results.items():
        print(f"{mode:<25} {metrics['avg_mrr']:>8.3f} {metrics['avg_ndcg@10']:>10.3f} "
              f"{metrics['avg_precision@10']:>8.3f} {metrics['avg_search_time_ms']:>10.0f}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
