"""
arXiv RAG v1 - RAG Module

Hybrid retrieval and search functionality.
- Supabase-based retrievers (legacy pgvector)
- Qdrant-based retrievers (optimized with native sparse indexing)
"""

from .retriever import (
    SearchResult,
    SearchResponse,
    DenseRetriever,
    SparseRetriever,
    HybridRetriever,
    HybridFullRetriever,
    ColBERTRetriever,
    OpenAIRetriever,
    hybrid_search,
    dense_search,
    sparse_search,
    colbert_search,
    openai_search,
    hybrid_full_search,
)

from .qdrant_retriever import (
    QdrantDenseRetriever,
    QdrantSparseRetriever,
    QdrantHybridRetriever,
    qdrant_hybrid_search,
    qdrant_dense_search,
    qdrant_sparse_search,
    qdrant_adaptive_search,
)

from .reranker import (
    BGEReranker,
    LightweightReranker,
    rerank_results,
)

from .api import app as api_app

__all__ = [
    # Supabase Retriever (legacy)
    "SearchResult",
    "SearchResponse",
    "DenseRetriever",
    "SparseRetriever",
    "HybridRetriever",
    "HybridFullRetriever",
    "ColBERTRetriever",
    "OpenAIRetriever",
    "hybrid_search",
    "dense_search",
    "sparse_search",
    "colbert_search",
    "openai_search",
    "hybrid_full_search",
    # Qdrant Retriever (optimized)
    "QdrantDenseRetriever",
    "QdrantSparseRetriever",
    "QdrantHybridRetriever",
    "qdrant_hybrid_search",
    "qdrant_dense_search",
    "qdrant_sparse_search",
    "qdrant_adaptive_search",
    # Reranker
    "BGEReranker",
    "LightweightReranker",
    "rerank_results",
    # API
    "api_app",
]
