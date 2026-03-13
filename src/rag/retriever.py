"""
arXiv RAG v1 - Hybrid Retriever

RRF (Reciprocal Rank Fusion) hybrid search combining dense and sparse retrieval.
"""

from dataclasses import dataclass, field
from typing import Optional

import asyncio
import json

from ..embedding.bge_embedder import BGEEmbedder
from ..embedding.openai_embedder import OpenAIEmbedder
from ..embedding.models import ColBERTVector, EmbeddingConfig, SparseVector
from ..storage.supabase_client import get_supabase_client, SupabaseClient
from ..utils.logging import get_logger

logger = get_logger("retriever")


@dataclass
class SearchResult:
    """A single search result with metadata."""
    chunk_id: str
    paper_id: str
    content: str
    section_title: Optional[str] = None
    score: float = 0.0
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    colbert_score: Optional[float] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResponse:
    """Search response with results and metadata."""
    query: str
    results: list[SearchResult]
    total_found: int = 0
    dense_count: int = 0
    sparse_count: int = 0
    colbert_count: int = 0
    search_time_ms: float = 0.0
    metadata: dict = field(default_factory=dict)  # Additional search metadata


class DenseRetriever:
    """
    Dense vector retrieval using pgvector cosine similarity.

    Uses BGE-M3 1024-dim embeddings stored in Supabase.
    """

    def __init__(
        self,
        client: SupabaseClient = None,
        embedder: BGEEmbedder = None,
    ):
        self.client = client or get_supabase_client()
        self._embedder = embedder

    @property
    def embedder(self) -> BGEEmbedder:
        """Lazy load BGE embedder."""
        if self._embedder is None:
            self._embedder = BGEEmbedder(EmbeddingConfig(use_openai=False))
        return self._embedder

    def search(
        self,
        query: str,
        top_k: int = 20,
        query_embedding: list[float] = None,
        match_threshold: float = 0.3,
    ) -> list[SearchResult]:
        """
        Search using dense embeddings.

        Args:
            query: Search query text
            top_k: Number of results to return
            query_embedding: Pre-computed query embedding (optional)
            match_threshold: Minimum similarity threshold

        Returns:
            List of search results ordered by similarity
        """
        # Get query embedding
        if query_embedding is None:
            dense, _, _ = self.embedder.embed_single(query)
            query_embedding = dense

        try:
            # Try new function first, fall back to legacy
            try:
                result = self.client.client.rpc(
                    "match_chunks_dense",
                    {
                        "query_embedding": query_embedding,
                        "match_count": top_k,
                    }
                ).execute()
            except Exception:
                # Fall back to legacy match_chunks function
                result = self.client.client.rpc(
                    "match_chunks",
                    {
                        "query_embedding": query_embedding,
                        "match_threshold": match_threshold,
                        "match_count": top_k,
                    }
                ).execute()

            if not result.data:
                logger.warning("Dense search returned no results")
                return []

            results = []
            for row in result.data:
                sr = SearchResult(
                    chunk_id=row["chunk_id"],
                    paper_id=row["paper_id"],
                    content=row["content"],
                    section_title=row.get("section_title"),
                    score=float(row.get("similarity", 0)),
                    dense_score=float(row.get("similarity", 0)),
                    metadata=row.get("metadata", {}),
                )
                results.append(sr)

            logger.debug(f"Dense search found {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []


class SparseRetriever:
    """
    Sparse vector retrieval using BM25-style token matching.

    Uses BGE-M3 lexical weights stored as JSONB in Supabase.
    """

    def __init__(
        self,
        client: SupabaseClient = None,
        embedder: BGEEmbedder = None,
    ):
        self.client = client or get_supabase_client()
        self._embedder = embedder

    @property
    def embedder(self) -> BGEEmbedder:
        """Lazy load BGE embedder."""
        if self._embedder is None:
            self._embedder = BGEEmbedder(EmbeddingConfig(use_openai=False))
        return self._embedder

    def search(
        self,
        query: str,
        top_k: int = 20,
        query_sparse: SparseVector = None,
    ) -> list[SearchResult]:
        """
        Search using sparse embeddings.

        Args:
            query: Search query text
            top_k: Number of results to return
            query_sparse: Pre-computed sparse vector (optional)

        Returns:
            List of search results ordered by BM25-style score
        """
        # Get query sparse vector
        if query_sparse is None:
            _, sparse, _ = self.embedder.embed_single(query)
            query_sparse = sparse

        if query_sparse is None:
            logger.warning("Could not generate sparse query vector")
            return []

        try:
            # Use Supabase RPC for sparse matching
            result = self.client.client.rpc(
                "match_chunks_sparse",
                {
                    "query_indices": query_sparse.indices,
                    "query_values": query_sparse.values,
                    "match_count": top_k,
                }
            ).execute()

            if not result.data:
                logger.debug("Sparse search returned no results (function may not exist yet)")
                return []

            results = []
            for row in result.data:
                sr = SearchResult(
                    chunk_id=row["chunk_id"],
                    paper_id=row["paper_id"],
                    content=row["content"],
                    section_title=row.get("section_title"),
                    score=float(row.get("score", 0)),
                    sparse_score=float(row.get("score", 0)),
                    metadata=row.get("metadata", {}),
                )
                results.append(sr)

            logger.debug(f"Sparse search found {len(results)} results")
            return results

        except Exception as e:
            # Sparse function may not be deployed yet - graceful degradation
            logger.debug(f"Sparse search not available: {e}")
            return []


class HybridRetriever:
    """
    Hybrid retrieval using RRF (Reciprocal Rank Fusion).

    Combines dense (semantic) and sparse (lexical) search results
    using the RRF formula: score = sum(1 / (k + rank))

    Reference: Cormack et al., 2009 - k=60 is the standard default.
    """

    def __init__(
        self,
        client: SupabaseClient = None,
        embedder: BGEEmbedder = None,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ):
        """
        Initialize hybrid retriever.

        Args:
            client: Supabase client
            embedder: BGE embedder (shared between dense/sparse)
            rrf_k: RRF constant (default 60)
            dense_weight: Weight for dense results
            sparse_weight: Weight for sparse results
        """
        self.client = client or get_supabase_client()
        self._embedder = embedder
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        # Initialize sub-retrievers with shared embedder
        self.dense_retriever = DenseRetriever(self.client, self._embedder)
        self.sparse_retriever = SparseRetriever(self.client, self._embedder)

    @property
    def embedder(self) -> BGEEmbedder:
        """Lazy load BGE embedder."""
        if self._embedder is None:
            self._embedder = BGEEmbedder(EmbeddingConfig(use_openai=False))
            # Share with sub-retrievers
            self.dense_retriever._embedder = self._embedder
            self.sparse_retriever._embedder = self._embedder
        return self._embedder

    def search(
        self,
        query: str,
        top_k: int = 10,
        dense_top_k: int = 20,
        sparse_top_k: int = 20,
    ) -> SearchResponse:
        """
        Hybrid search combining dense and sparse retrieval.

        Args:
            query: Search query text
            top_k: Number of final results to return
            dense_top_k: Number of dense results to fetch
            sparse_top_k: Number of sparse results to fetch

        Returns:
            SearchResponse with fused results
        """
        import time
        start = time.time()

        # Get query embeddings once
        dense_vec, sparse_vec, _ = self.embedder.embed_single(query)

        # Run both searches
        dense_results = self.dense_retriever.search(
            query,
            top_k=dense_top_k,
            query_embedding=dense_vec
        )
        sparse_results = self.sparse_retriever.search(
            query,
            top_k=sparse_top_k,
            query_sparse=sparse_vec
        )

        # Apply RRF fusion
        fused_results = self._rrf_fusion(
            dense_results,
            sparse_results,
            top_k
        )

        elapsed_ms = (time.time() - start) * 1000

        return SearchResponse(
            query=query,
            results=fused_results,
            total_found=len(fused_results),
            dense_count=len(dense_results),
            sparse_count=len(sparse_results),
            search_time_ms=elapsed_ms,
        )

    def _rrf_fusion(
        self,
        dense_results: list[SearchResult],
        sparse_results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """
        Apply Reciprocal Rank Fusion to combine results.

        RRF Score = sum(weight / (k + rank))

        Args:
            dense_results: Results from dense search
            sparse_results: Results from sparse search
            top_k: Number of results to return

        Returns:
            Fused and re-ranked results
        """
        # Build score map: chunk_id -> (rrf_score, result)
        scores: dict[str, tuple[float, SearchResult]] = {}

        # Process dense results
        for rank, result in enumerate(dense_results):
            chunk_id = result.chunk_id
            rrf_score = self.dense_weight / (self.rrf_k + rank + 1)

            if chunk_id in scores:
                # Combine scores, keep result with more info
                existing_score, existing_result = scores[chunk_id]
                combined_score = existing_score + rrf_score
                # Update dense score
                existing_result.dense_score = result.dense_score
                scores[chunk_id] = (combined_score, existing_result)
            else:
                result.score = rrf_score
                scores[chunk_id] = (rrf_score, result)

        # Process sparse results
        for rank, result in enumerate(sparse_results):
            chunk_id = result.chunk_id
            rrf_score = self.sparse_weight / (self.rrf_k + rank + 1)

            if chunk_id in scores:
                # Combine scores
                existing_score, existing_result = scores[chunk_id]
                combined_score = existing_score + rrf_score
                # Update sparse score
                existing_result.sparse_score = result.sparse_score
                existing_result.score = combined_score
                scores[chunk_id] = (combined_score, existing_result)
            else:
                result.score = rrf_score
                scores[chunk_id] = (rrf_score, result)

        # Sort by combined RRF score
        sorted_results = sorted(
            scores.values(),
            key=lambda x: x[0],
            reverse=True
        )

        # Return top-k with updated scores
        final_results = []
        for score, result in sorted_results[:top_k]:
            result.score = score
            final_results.append(result)

        return final_results

    def search_dense_only(
        self,
        query: str,
        top_k: int = 10,
    ) -> SearchResponse:
        """Search using only dense embeddings."""
        import time
        start = time.time()

        results = self.dense_retriever.search(query, top_k=top_k)
        elapsed_ms = (time.time() - start) * 1000

        return SearchResponse(
            query=query,
            results=results,
            total_found=len(results),
            dense_count=len(results),
            sparse_count=0,
            search_time_ms=elapsed_ms,
        )

    def search_sparse_only(
        self,
        query: str,
        top_k: int = 10,
    ) -> SearchResponse:
        """Search using only sparse embeddings."""
        import time
        start = time.time()

        results = self.sparse_retriever.search(query, top_k=top_k)
        elapsed_ms = (time.time() - start) * 1000

        return SearchResponse(
            query=query,
            results=results,
            total_found=len(results),
            dense_count=0,
            sparse_count=len(results),
            search_time_ms=elapsed_ms,
        )

    def unload_models(self) -> None:
        """Unload embedder models to free GPU memory."""
        if self._embedder is not None:
            self._embedder.unload()
            self._embedder = None


class ColBERTRetriever:
    """
    ColBERT token-level MaxSim retrieval.

    Uses BGE-M3 ColBERT vectors for late-interaction scoring:
    Score = sum(max(cos_sim(q_token, d_tokens)) for q_token in query)
    """

    def __init__(
        self,
        client: SupabaseClient = None,
        embedder: BGEEmbedder = None,
    ):
        self.client = client or get_supabase_client()
        self._embedder = embedder

    @property
    def embedder(self) -> BGEEmbedder:
        """Lazy load BGE embedder."""
        if self._embedder is None:
            self._embedder = BGEEmbedder(EmbeddingConfig(use_openai=False))
        return self._embedder

    def search(
        self,
        query: str,
        top_k: int = 20,
        query_tokens: list[list[float]] = None,
    ) -> list[SearchResult]:
        """
        Search using ColBERT MaxSim scoring.

        Args:
            query: Search query text
            top_k: Number of results to return
            query_tokens: Pre-computed query token embeddings (optional)

        Returns:
            List of search results ordered by MaxSim score
        """
        # Get query ColBERT embeddings
        if query_tokens is None:
            query_tokens = self.embedder.encode_colbert(query)

        if not query_tokens:
            logger.warning("Could not generate ColBERT query tokens")
            return []

        try:
            # Call Supabase RPC for MaxSim scoring
            result = self.client.client.rpc(
                "match_chunks_colbert",
                {
                    "query_tokens": json.dumps(query_tokens),
                    "match_count": top_k,
                }
            ).execute()

            if not result.data:
                logger.debug("ColBERT search returned no results")
                return []

            results = []
            for row in result.data:
                sr = SearchResult(
                    chunk_id=row["chunk_id"],
                    paper_id=row["paper_id"],
                    content=row["content"],
                    section_title=row.get("section_title"),
                    score=float(row.get("similarity", 0)),
                    colbert_score=float(row.get("similarity", 0)),
                    metadata=row.get("metadata", {}),
                )
                results.append(sr)

            logger.debug(f"ColBERT search found {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"ColBERT search failed: {e}")
            return []

    def unload_models(self) -> None:
        """Unload embedder models to free GPU memory."""
        if self._embedder is not None:
            self._embedder.unload()
            self._embedder = None


class HybridFullRetriever:
    """
    Full hybrid retrieval using 3-way RRF fusion.

    Combines dense (semantic), sparse (lexical), and ColBERT (late-interaction)
    using Reciprocal Rank Fusion with configurable weights.
    """

    def __init__(
        self,
        client: SupabaseClient = None,
        embedder: BGEEmbedder = None,
        rrf_k: int = 60,
        dense_weight: float = 0.4,
        sparse_weight: float = 0.3,
        colbert_weight: float = 0.3,
    ):
        """
        Initialize full hybrid retriever.

        Args:
            client: Supabase client
            embedder: BGE embedder (shared across all retrievers)
            rrf_k: RRF constant (default 60)
            dense_weight: Weight for dense results
            sparse_weight: Weight for sparse results
            colbert_weight: Weight for ColBERT results
        """
        self.client = client or get_supabase_client()
        self._embedder = embedder
        self.rrf_k = rrf_k
        self.weights = {
            'dense': dense_weight,
            'sparse': sparse_weight,
            'colbert': colbert_weight,
        }

        # Initialize sub-retrievers with shared embedder
        self.dense_retriever = DenseRetriever(self.client, self._embedder)
        self.sparse_retriever = SparseRetriever(self.client, self._embedder)
        self.colbert_retriever = ColBERTRetriever(self.client, self._embedder)

    @property
    def embedder(self) -> BGEEmbedder:
        """Lazy load BGE embedder."""
        if self._embedder is None:
            self._embedder = BGEEmbedder(EmbeddingConfig(use_openai=False))
            # Share with sub-retrievers
            self.dense_retriever._embedder = self._embedder
            self.sparse_retriever._embedder = self._embedder
            self.colbert_retriever._embedder = self._embedder
        return self._embedder

    def search(
        self,
        query: str,
        top_k: int = 10,
        dense_top_k: int = 20,
        sparse_top_k: int = 20,
        colbert_top_k: int = 20,
    ) -> SearchResponse:
        """
        Full hybrid search combining dense, sparse, and ColBERT retrieval.

        Args:
            query: Search query text
            top_k: Number of final results to return
            dense_top_k: Number of dense results to fetch
            sparse_top_k: Number of sparse results to fetch
            colbert_top_k: Number of ColBERT results to fetch

        Returns:
            SearchResponse with fused results
        """
        import time
        start = time.time()

        # Get all query embeddings at once
        dense_vec, sparse_vec, colbert_cv = self.embedder.embed_single(
            query, return_colbert=True
        )
        colbert_tokens = colbert_cv.token_embeddings if colbert_cv else []

        # Run all three searches
        dense_results = self.dense_retriever.search(
            query, top_k=dense_top_k, query_embedding=dense_vec
        )
        sparse_results = self.sparse_retriever.search(
            query, top_k=sparse_top_k, query_sparse=sparse_vec
        )
        colbert_results = self.colbert_retriever.search(
            query, top_k=colbert_top_k, query_tokens=colbert_tokens
        )

        # Apply 3-way RRF fusion
        fused_results = self._rrf_fusion_3way(
            dense_results, sparse_results, colbert_results, top_k
        )

        elapsed_ms = (time.time() - start) * 1000

        return SearchResponse(
            query=query,
            results=fused_results,
            total_found=len(fused_results),
            dense_count=len(dense_results),
            sparse_count=len(sparse_results),
            colbert_count=len(colbert_results),
            search_time_ms=elapsed_ms,
        )

    def _rrf_fusion_3way(
        self,
        dense_results: list[SearchResult],
        sparse_results: list[SearchResult],
        colbert_results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """
        Apply 3-way Reciprocal Rank Fusion.

        RRF Score = sum(weight / (k + rank)) for each retriever

        Args:
            dense_results: Results from dense search
            sparse_results: Results from sparse search
            colbert_results: Results from ColBERT search
            top_k: Number of results to return

        Returns:
            Fused and re-ranked results
        """
        # Build score map: chunk_id -> (rrf_score, result)
        scores: dict[str, tuple[float, SearchResult]] = {}

        # Process dense results
        for rank, result in enumerate(dense_results):
            chunk_id = result.chunk_id
            rrf_score = self.weights['dense'] / (self.rrf_k + rank + 1)

            if chunk_id in scores:
                existing_score, existing_result = scores[chunk_id]
                existing_result.dense_score = result.dense_score
                scores[chunk_id] = (existing_score + rrf_score, existing_result)
            else:
                result.score = rrf_score
                scores[chunk_id] = (rrf_score, result)

        # Process sparse results
        for rank, result in enumerate(sparse_results):
            chunk_id = result.chunk_id
            rrf_score = self.weights['sparse'] / (self.rrf_k + rank + 1)

            if chunk_id in scores:
                existing_score, existing_result = scores[chunk_id]
                existing_result.sparse_score = result.sparse_score
                existing_result.score = existing_score + rrf_score
                scores[chunk_id] = (existing_score + rrf_score, existing_result)
            else:
                result.score = rrf_score
                scores[chunk_id] = (rrf_score, result)

        # Process ColBERT results
        for rank, result in enumerate(colbert_results):
            chunk_id = result.chunk_id
            rrf_score = self.weights['colbert'] / (self.rrf_k + rank + 1)

            if chunk_id in scores:
                existing_score, existing_result = scores[chunk_id]
                existing_result.colbert_score = result.colbert_score
                existing_result.score = existing_score + rrf_score
                scores[chunk_id] = (existing_score + rrf_score, existing_result)
            else:
                result.score = rrf_score
                scores[chunk_id] = (rrf_score, result)

        # Sort by combined RRF score
        sorted_results = sorted(
            scores.values(),
            key=lambda x: x[0],
            reverse=True
        )

        # Return top-k with updated scores
        final_results = []
        for score, result in sorted_results[:top_k]:
            result.score = score
            final_results.append(result)

        return final_results

    def unload_models(self) -> None:
        """Unload embedder models to free GPU memory."""
        if self._embedder is not None:
            self._embedder.unload()
            self._embedder = None


# Convenience functions
def hybrid_search(
    query: str,
    top_k: int = 10,
    rrf_k: int = 60,
) -> SearchResponse:
    """
    Perform hybrid search with default settings.

    Args:
        query: Search query
        top_k: Number of results
        rrf_k: RRF constant

    Returns:
        Search response with results
    """
    retriever = HybridRetriever(rrf_k=rrf_k)
    return retriever.search(query, top_k=top_k)


def dense_search(query: str, top_k: int = 10) -> SearchResponse:
    """Perform dense-only search."""
    retriever = HybridRetriever()
    return retriever.search_dense_only(query, top_k=top_k)


def sparse_search(query: str, top_k: int = 10) -> SearchResponse:
    """Perform sparse-only search."""
    retriever = HybridRetriever()
    return retriever.search_sparse_only(query, top_k=top_k)


class OpenAIRetriever:
    """
    OpenAI embedding-based retrieval for comparison with BGE-M3.

    Uses text-embedding-3-large with MRL reduction (1024 dims) stored in Supabase.
    """

    def __init__(
        self,
        client: SupabaseClient = None,
        embedder: OpenAIEmbedder = None,
    ):
        self.client = client or get_supabase_client()
        self._embedder = embedder

    @property
    def embedder(self) -> OpenAIEmbedder:
        """Lazy load OpenAI embedder."""
        if self._embedder is None:
            self._embedder = OpenAIEmbedder(EmbeddingConfig(use_openai=True))
        return self._embedder

    def search(
        self,
        query: str,
        top_k: int = 10,
        query_embedding: list[float] = None,
    ) -> SearchResponse:
        """
        Search using OpenAI embeddings.

        Args:
            query: Search query text
            top_k: Number of results to return
            query_embedding: Pre-computed query embedding (optional)

        Returns:
            SearchResponse with results
        """
        import time
        start = time.time()

        # Get query embedding
        if query_embedding is None:
            query_embedding = self.embedder.embed_single(query)

        try:
            result = self.client.client.rpc(
                "match_chunks_openai",
                {
                    "query_embedding": query_embedding,
                    "match_count": top_k,
                }
            ).execute()

            if not result.data:
                logger.warning("OpenAI search returned no results")
                return SearchResponse(
                    query=query,
                    results=[],
                    total_found=0,
                    search_time_ms=(time.time() - start) * 1000,
                )

            results = []
            for row in result.data:
                sr = SearchResult(
                    chunk_id=row["chunk_id"],
                    paper_id=row["paper_id"],
                    content=row["content"],
                    section_title=row.get("section_title"),
                    score=float(row.get("similarity", 0)),
                    dense_score=float(row.get("similarity", 0)),
                    metadata=row.get("metadata", {}),
                )
                results.append(sr)

            elapsed_ms = (time.time() - start) * 1000
            logger.debug(f"OpenAI search found {len(results)} results in {elapsed_ms:.0f}ms")

            return SearchResponse(
                query=query,
                results=results,
                total_found=len(results),
                dense_count=len(results),
                sparse_count=0,
                search_time_ms=elapsed_ms,
            )

        except Exception as e:
            logger.error(f"OpenAI search failed: {e}")
            return SearchResponse(
                query=query,
                results=[],
                total_found=0,
                search_time_ms=(time.time() - start) * 1000,
            )


def openai_search(query: str, top_k: int = 10) -> SearchResponse:
    """Perform OpenAI embedding search."""
    retriever = OpenAIRetriever()
    return retriever.search(query, top_k=top_k)


def colbert_search(query: str, top_k: int = 10) -> SearchResponse:
    """Perform ColBERT MaxSim search."""
    import time
    start = time.time()

    retriever = ColBERTRetriever()
    results = retriever.search(query, top_k=top_k)
    elapsed_ms = (time.time() - start) * 1000

    return SearchResponse(
        query=query,
        results=results,
        total_found=len(results),
        colbert_count=len(results),
        search_time_ms=elapsed_ms,
    )


def hybrid_full_search(
    query: str,
    top_k: int = 10,
    rrf_k: int = 60,
) -> SearchResponse:
    """
    Perform full hybrid search (dense + sparse + ColBERT).

    Args:
        query: Search query
        top_k: Number of results
        rrf_k: RRF constant

    Returns:
        Search response with 3-way fused results
    """
    retriever = HybridFullRetriever(rrf_k=rrf_k)
    return retriever.search(query, top_k=top_k)
