"""
arXiv RAG v1 - Qdrant Hybrid Retriever

RRF (Reciprocal Rank Fusion) hybrid search using Qdrant vector database.
Combines dense, sparse, and ColBERT retrieval with native Qdrant indexing.
"""

from dataclasses import dataclass, field
from typing import Optional
import time

from ..embedding.bge_embedder import BGEEmbedder
from ..embedding.models import EmbeddingConfig, SparseVector
from ..storage.qdrant_client import QdrantVectorClient, get_qdrant_client
from ..utils.logging import get_logger
from .retriever import SearchResult, SearchResponse

logger = get_logger("qdrant_retriever")


# =============================================================================
# RRF Weight Presets for Different Query Types
# =============================================================================

RRF_PRESETS = {
    # Default: balanced for general queries
    "default": {
        "dense_weight": 0.4,
        "sparse_weight": 0.3,
        "colbert_weight": 0.3,
    },
    # Dense-heavy: better for conceptual/paraphrased queries
    "conceptual": {
        "dense_weight": 0.6,
        "sparse_weight": 0.2,
        "colbert_weight": 0.2,
    },
    # Sparse-heavy: better for keyword/technical term queries
    "keyword": {
        "dense_weight": 0.3,
        "sparse_weight": 0.5,
        "colbert_weight": 0.2,
    },
    # Balanced: equal weighting
    "balanced": {
        "dense_weight": 0.5,
        "sparse_weight": 0.25,
        "colbert_weight": 0.25,
    },
    # Dense-only: pure semantic matching
    "dense_only": {
        "dense_weight": 1.0,
        "sparse_weight": 0.0,
        "colbert_weight": 0.0,
    },
}


class QdrantDenseRetriever:
    """
    Dense vector retrieval using Qdrant.

    Uses BGE-M3 1024-dim embeddings with native HNSW indexing.
    """

    def __init__(
        self,
        client: QdrantVectorClient = None,
        embedder: BGEEmbedder = None,
    ):
        self.client = client or get_qdrant_client()
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
    ) -> list[SearchResult]:
        """
        Search using dense embeddings.

        Args:
            query: Search query text
            top_k: Number of results to return
            query_embedding: Pre-computed query embedding (optional)

        Returns:
            List of search results ordered by similarity
        """
        # Get query embedding
        if query_embedding is None:
            dense, _, _ = self.embedder.embed_single(query)
            query_embedding = dense

        try:
            results = self.client.search_dense(
                query_vector=query_embedding,
                vector_name="dense_bge",
                top_k=top_k,
            )

            search_results = []
            for row in results:
                sr = SearchResult(
                    chunk_id=row["chunk_id"],
                    paper_id=row["paper_id"],
                    content=row["content"],
                    section_title=row.get("section_title"),
                    score=float(row.get("score", 0)),
                    dense_score=float(row.get("score", 0)),
                    metadata=row.get("metadata", {}),
                )
                search_results.append(sr)

            logger.debug(f"Qdrant dense search found {len(search_results)} results")
            return search_results

        except Exception as e:
            logger.error(f"Qdrant dense search failed: {e}")
            return []


class QdrantSparseRetriever:
    """
    Sparse vector retrieval using Qdrant native sparse indexing.

    Uses BGE-M3 lexical weights with inverted index for O(log n) lookup.
    """

    def __init__(
        self,
        client: QdrantVectorClient = None,
        embedder: BGEEmbedder = None,
    ):
        self.client = client or get_qdrant_client()
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
            results = self.client.search_sparse(
                query_indices=query_sparse.indices,
                query_values=query_sparse.values,
                top_k=top_k,
            )

            search_results = []
            for row in results:
                sr = SearchResult(
                    chunk_id=row["chunk_id"],
                    paper_id=row["paper_id"],
                    content=row["content"],
                    section_title=row.get("section_title"),
                    score=float(row.get("score", 0)),
                    sparse_score=float(row.get("score", 0)),
                    metadata=row.get("metadata", {}),
                )
                search_results.append(sr)

            logger.debug(f"Qdrant sparse search found {len(search_results)} results")
            return search_results

        except Exception as e:
            logger.error(f"Qdrant sparse search failed: {e}")
            return []


class QdrantHybridRetriever:
    """
    Hybrid retrieval using RRF (Reciprocal Rank Fusion) with Qdrant.

    Combines dense (semantic), sparse (lexical), and optionally ColBERT
    using the RRF formula: score = sum(weight / (k + rank))

    Expected performance improvements over Supabase:
    - Sparse search: 2,599ms → <500ms (5x faster)
    - Hybrid search: 3,578ms → <700ms (5x faster)
    """

    def __init__(
        self,
        client: QdrantVectorClient = None,
        embedder: BGEEmbedder = None,
        rrf_k: int = 60,
        dense_weight: float = 0.4,
        sparse_weight: float = 0.3,
        colbert_weight: float = 0.3,
    ):
        """
        Initialize Qdrant hybrid retriever.

        Args:
            client: Qdrant client
            embedder: BGE embedder (shared between dense/sparse)
            rrf_k: RRF constant (default 60)
            dense_weight: Weight for dense results
            sparse_weight: Weight for sparse results
            colbert_weight: Weight for ColBERT results
        """
        self.client = client or get_qdrant_client()
        self._embedder = embedder
        self.rrf_k = rrf_k
        self.weights = {
            'dense': dense_weight,
            'sparse': sparse_weight,
            'colbert': colbert_weight,
        }

        # Initialize sub-retrievers with shared embedder
        self.dense_retriever = QdrantDenseRetriever(self.client, self._embedder)
        self.sparse_retriever = QdrantSparseRetriever(self.client, self._embedder)

    @property
    def embedder(self) -> BGEEmbedder:
        """Lazy load BGE embedder."""
        if self._embedder is None:
            self._embedder = BGEEmbedder(EmbeddingConfig(use_openai=False))
            # Share with sub-retrievers
            self.dense_retriever._embedder = self._embedder
            self.sparse_retriever._embedder = self._embedder
        return self._embedder

    def set_weights(
        self,
        dense_weight: float = None,
        sparse_weight: float = None,
        colbert_weight: float = None,
        preset: str = None,
    ) -> None:
        """
        Set RRF weights for hybrid fusion.

        Args:
            dense_weight: Weight for dense results (0-1)
            sparse_weight: Weight for sparse results (0-1)
            colbert_weight: Weight for ColBERT results (0-1)
            preset: Use a predefined preset ("default", "conceptual", "keyword", "balanced", "dense_only")
        """
        if preset:
            if preset not in RRF_PRESETS:
                raise ValueError(f"Unknown preset: {preset}. Available: {list(RRF_PRESETS.keys())}")
            config = RRF_PRESETS[preset]
            self.weights = {
                'dense': config['dense_weight'],
                'sparse': config['sparse_weight'],
                'colbert': config['colbert_weight'],
            }
            logger.info(f"Set RRF weights to preset '{preset}': {self.weights}")
        else:
            if dense_weight is not None:
                self.weights['dense'] = dense_weight
            if sparse_weight is not None:
                self.weights['sparse'] = sparse_weight
            if colbert_weight is not None:
                self.weights['colbert'] = colbert_weight
            logger.info(f"Set RRF weights: {self.weights}")

    def get_weights(self) -> dict:
        """Get current RRF weights."""
        return self.weights.copy()

    def search(
        self,
        query: str,
        top_k: int = 10,
        dense_top_k: int = 50,
        sparse_top_k: int = 50,
        use_reranker: bool = False,
        rerank_top_k: int = 10,
    ) -> SearchResponse:
        """
        Hybrid search combining dense and sparse retrieval.

        Args:
            query: Search query text
            top_k: Number of final results to return
            dense_top_k: Number of dense results to fetch
            sparse_top_k: Number of sparse results to fetch
            use_reranker: Whether to apply BGE reranker
            rerank_top_k: Number of results after reranking

        Returns:
            SearchResponse with fused results
        """
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
            top_k if not use_reranker else max(top_k * 3, 30),
        )

        # Apply reranking if enabled
        if use_reranker and fused_results:
            from .reranker import BGEReranker

            reranker = BGEReranker()
            fused_results = reranker.rerank(query, fused_results, top_k=rerank_top_k)
            reranker.unload()

        elapsed_ms = (time.time() - start) * 1000

        return SearchResponse(
            query=query,
            results=fused_results[:top_k],
            total_found=len(fused_results),
            dense_count=len(dense_results),
            sparse_count=len(sparse_results),
            search_time_ms=elapsed_ms,
        )

    def search_with_qdrant_hybrid(
        self,
        query: str,
        top_k: int = 10,
    ) -> SearchResponse:
        """
        Use Qdrant's native hybrid search (if available).

        Falls back to manual RRF if not supported.
        """
        start = time.time()

        # Get query embeddings
        dense_vec, sparse_vec, _ = self.embedder.embed_single(query)

        if sparse_vec is None:
            # Fallback to dense-only
            results = self.dense_retriever.search(query, top_k=top_k, query_embedding=dense_vec)
            elapsed_ms = (time.time() - start) * 1000
            return SearchResponse(
                query=query,
                results=results,
                total_found=len(results),
                dense_count=len(results),
                sparse_count=0,
                search_time_ms=elapsed_ms,
            )

        try:
            # Use Qdrant native hybrid search
            results = self.client.search_hybrid(
                dense_vector=dense_vec,
                sparse_indices=sparse_vec.indices,
                sparse_values=sparse_vec.values,
                top_k=top_k,
                dense_weight=self.weights['dense'],
                sparse_weight=self.weights['sparse'],
            )

            search_results = []
            for row in results:
                sr = SearchResult(
                    chunk_id=row["chunk_id"],
                    paper_id=row["paper_id"],
                    content=row["content"],
                    section_title=row.get("section_title"),
                    score=float(row.get("score", 0)),
                    dense_score=row.get("dense_score"),
                    sparse_score=row.get("sparse_score"),
                    metadata=row.get("metadata", {}),
                )
                search_results.append(sr)

            elapsed_ms = (time.time() - start) * 1000

            return SearchResponse(
                query=query,
                results=search_results,
                total_found=len(search_results),
                dense_count=len(search_results),
                sparse_count=len(search_results),
                search_time_ms=elapsed_ms,
            )

        except Exception as e:
            logger.warning(f"Qdrant native hybrid failed, falling back to RRF: {e}")
            return self.search(query, top_k=top_k)

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

    def search_dense_3large(
        self,
        query: str,
        top_k: int = 10,
    ) -> SearchResponse:
        """
        Search using OpenAI text-embedding-3-large vectors.

        Uses the dense_3large vector field in Qdrant (3072 dims).
        Requires OpenAI API for query embedding.
        """
        from ..embedding.openai_embedder import OpenAIEmbedder
        from ..embedding.models import EmbeddingConfig

        start = time.time()

        # Get OpenAI query embedding
        openai_embedder = OpenAIEmbedder(EmbeddingConfig(use_openai=True))
        query_embedding = openai_embedder.embed_single(query)

        try:
            results = self.client.search_dense(
                query_vector=query_embedding,
                vector_name="dense_3large",
                top_k=top_k,
            )

            search_results = []
            for row in results:
                score = float(row.get("similarity") or row.get("score") or 0)
                sr = SearchResult(
                    chunk_id=row["chunk_id"],
                    paper_id=row["paper_id"],
                    content=row["content"],
                    section_title=row.get("section_title"),
                    score=score,
                    dense_score=score,
                    metadata=row.get("metadata", {}),
                )
                search_results.append(sr)

            elapsed_ms = (time.time() - start) * 1000
            logger.debug(f"Qdrant dense_3large search found {len(search_results)} results")

            return SearchResponse(
                query=query,
                results=search_results,
                total_found=len(search_results),
                dense_count=len(search_results),
                sparse_count=0,
                search_time_ms=elapsed_ms,
            )

        except Exception as e:
            logger.error(f"Qdrant dense_3large search failed: {e}")
            return SearchResponse(
                query=query,
                results=[],
                total_found=0,
                dense_count=0,
                sparse_count=0,
                search_time_ms=(time.time() - start) * 1000,
            )

    def search_hybrid_3large(
        self,
        query: str,
        top_k: int = 10,
        dense_top_k: int = 50,
        sparse_top_k: int = 50,
    ) -> SearchResponse:
        """
        Hybrid search combining OpenAI dense (3large) and BGE sparse.

        Uses RRF fusion of dense_3large and sparse_bge vectors.
        """
        from ..embedding.openai_embedder import OpenAIEmbedder
        from ..embedding.models import EmbeddingConfig

        start = time.time()

        # Get OpenAI query embedding for dense
        openai_embedder = OpenAIEmbedder(EmbeddingConfig(use_openai=True))
        dense_embedding = openai_embedder.embed_single(query)

        # Get BGE sparse embedding
        _, sparse_vec, _ = self.embedder.embed_single(query)

        # Search dense_3large
        try:
            dense_results_raw = self.client.search_dense(
                query_vector=dense_embedding,
                vector_name="dense_3large",
                top_k=dense_top_k,
            )
            dense_results = []
            for row in dense_results_raw:
                score = float(row.get("similarity") or row.get("score") or 0)
                sr = SearchResult(
                    chunk_id=row["chunk_id"],
                    paper_id=row["paper_id"],
                    content=row["content"],
                    section_title=row.get("section_title"),
                    score=score,
                    dense_score=score,
                    metadata=row.get("metadata", {}),
                )
                dense_results.append(sr)
        except Exception as e:
            logger.error(f"Dense 3large search failed: {e}")
            dense_results = []

        # Search sparse
        sparse_results = self.sparse_retriever.search(
            query, top_k=sparse_top_k, query_sparse=sparse_vec
        )

        # Apply RRF fusion
        fused_results = self._rrf_fusion(dense_results, sparse_results, top_k)

        elapsed_ms = (time.time() - start) * 1000

        return SearchResponse(
            query=query,
            results=fused_results,
            total_found=len(fused_results),
            dense_count=len(dense_results),
            sparse_count=len(sparse_results),
            search_time_ms=elapsed_ms,
        )

    def search_adaptive(
        self,
        query: str,
        top_k: int = 10,
        use_hyde: bool = True,
        use_reranker: bool = False,
        rerank_top_k: int = 10,
    ) -> SearchResponse:
        """
        Adaptive search that adjusts strategy based on query type.

        Strategy by query type:
        - keyword: Sparse-heavy weights (0.3 dense, 0.5 sparse)
        - natural: Default weights (0.4 dense, 0.3 sparse)
        - conceptual: Dense-heavy + HyDE expansion (0.6 dense, 0.2 sparse)

        Args:
            query: Search query text
            top_k: Number of final results
            use_hyde: Enable HyDE expansion for conceptual queries
            use_reranker: Apply BGE reranker to results
            rerank_top_k: Number of results after reranking

        Returns:
            SearchResponse with results and metadata
        """
        from .query_classifier import classify_query_detailed

        # Classify query
        classification = classify_query_detailed(query)
        query_type = classification.query_type
        recommended_preset = classification.recommended_preset

        logger.info(f"Adaptive search: query_type={query_type}, preset={recommended_preset}, "
                   f"confidence={classification.confidence:.2f}")

        # Store original weights to restore later
        original_weights = self.weights.copy()

        # Apply recommended preset
        self.set_weights(preset=recommended_preset)

        # Expand query with HyDE for conceptual queries
        search_query = query
        hyde_used = False

        if use_hyde and query_type == "conceptual":
            try:
                from .hyde import expand_query
                expanded = expand_query(query, query_type)
                if expanded != query:
                    search_query = expanded
                    hyde_used = True
                    logger.info(f"HyDE expanded: {query[:30]}... -> {expanded[:50]}...")
            except Exception as e:
                logger.warning(f"HyDE expansion failed, using original: {e}")

        # Adjust search parameters for conceptual queries
        if query_type == "conceptual":
            # Fetch more candidates for conceptual queries
            dense_top_k = 100
            sparse_top_k = 50
        elif query_type == "keyword":
            dense_top_k = 30
            sparse_top_k = 80
        else:
            dense_top_k = 50
            sparse_top_k = 50

        # Execute search
        response = self.search(
            search_query,
            top_k=top_k if not use_reranker else max(top_k * 3, 30),
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
            use_reranker=use_reranker,
            rerank_top_k=rerank_top_k,
        )

        # Add adaptive search metadata
        response.metadata = response.metadata or {}
        response.metadata.update({
            "query_type": query_type,
            "query_type_confidence": classification.confidence,
            "rrf_preset": recommended_preset,
            "hyde_used": hyde_used,
            "original_query": query if hyde_used else None,
        })

        # Restore original weights
        self.weights = original_weights

        return response

    def unload_models(self) -> None:
        """Unload embedder models to free GPU memory."""
        if self._embedder is not None:
            self._embedder.unload()
            self._embedder = None


# Convenience functions
def qdrant_hybrid_search(
    query: str,
    top_k: int = 10,
    rrf_k: int = 60,
    use_reranker: bool = False,
) -> SearchResponse:
    """
    Perform hybrid search with Qdrant.

    Args:
        query: Search query
        top_k: Number of results
        rrf_k: RRF constant
        use_reranker: Apply BGE reranker

    Returns:
        Search response with results
    """
    retriever = QdrantHybridRetriever(rrf_k=rrf_k)
    return retriever.search(query, top_k=top_k, use_reranker=use_reranker)


def qdrant_dense_search(query: str, top_k: int = 10) -> SearchResponse:
    """Perform dense-only search with Qdrant."""
    retriever = QdrantHybridRetriever()
    return retriever.search_dense_only(query, top_k=top_k)


def qdrant_sparse_search(query: str, top_k: int = 10) -> SearchResponse:
    """Perform sparse-only search with Qdrant."""
    retriever = QdrantHybridRetriever()
    return retriever.search_sparse_only(query, top_k=top_k)


def qdrant_adaptive_search(
    query: str,
    top_k: int = 10,
    use_hyde: bool = True,
    use_reranker: bool = False,
) -> SearchResponse:
    """
    Perform adaptive search with automatic strategy selection.

    Classifies the query and adjusts RRF weights and search parameters
    accordingly. Uses HyDE expansion for conceptual queries.

    Args:
        query: Search query
        top_k: Number of results
        use_hyde: Enable HyDE for conceptual queries
        use_reranker: Apply BGE reranker

    Returns:
        Search response with results and adaptive metadata
    """
    retriever = QdrantHybridRetriever()
    return retriever.search_adaptive(
        query,
        top_k=top_k,
        use_hyde=use_hyde,
        use_reranker=use_reranker,
    )
