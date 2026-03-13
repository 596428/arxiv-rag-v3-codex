"""Shared retrieval result models used by Qdrant retrievers and rerankers."""

from dataclasses import dataclass, field
from typing import Optional


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
    metadata: dict = field(default_factory=dict)


class _LegacyRetrieverRemoved:
    """Compatibility shim for removed Supabase RPC retrievers."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "Legacy Supabase retrievers were removed in v3. Use `src.rag.qdrant_retriever` instead."
        )


class DenseRetriever(_LegacyRetrieverRemoved):
    pass


class SparseRetriever(_LegacyRetrieverRemoved):
    pass


class HybridRetriever(_LegacyRetrieverRemoved):
    pass


class ColBERTRetriever(_LegacyRetrieverRemoved):
    pass


class HybridFullRetriever(_LegacyRetrieverRemoved):
    pass


class OpenAIRetriever(_LegacyRetrieverRemoved):
    pass


def hybrid_search(*args, **kwargs):
    raise RuntimeError("Legacy Supabase hybrid search was removed in v3.")


def dense_search(*args, **kwargs):
    raise RuntimeError("Legacy Supabase dense search was removed in v3.")


def sparse_search(*args, **kwargs):
    raise RuntimeError("Legacy Supabase sparse search was removed in v3.")


def colbert_search(*args, **kwargs):
    raise RuntimeError("Legacy Supabase ColBERT search was removed in v3.")


def openai_search(*args, **kwargs):
    raise RuntimeError("Legacy Supabase OpenAI search was removed in v3.")


def hybrid_full_search(*args, **kwargs):
    raise RuntimeError("Legacy Supabase full hybrid search was removed in v3.")
