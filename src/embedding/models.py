"""
arXiv RAG v1 - Embedding Data Models

Chunk and embedding data structures for RAG system.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ChunkType(str, Enum):
    """Type of content in a chunk."""
    TEXT = "text"
    ABSTRACT = "abstract"
    EQUATION = "equation"
    FIGURE = "figure"
    TABLE = "table"
    MIXED = "mixed"  # Contains multiple types


class Chunk(BaseModel):
    """
    A text chunk ready for embedding.

    Represents a unit of text extracted from a parsed document,
    with metadata for retrieval and context.
    """
    chunk_id: str = Field(..., description="Unique chunk ID (paper_id_chunk_N)")
    paper_id: str = Field(..., description="Source paper arXiv ID")
    content: str = Field(..., description="Chunk text content")

    # Source context
    section_id: Optional[str] = Field(None, description="Source section ID")
    section_title: Optional[str] = Field(None, description="Section title for context")
    paragraph_ids: list[str] = Field(default_factory=list, description="Source paragraph IDs")

    # Chunk metadata
    chunk_type: ChunkType = Field(default=ChunkType.TEXT, description="Content type")
    chunk_index: int = Field(..., description="Chunk order within paper")
    token_count: int = Field(default=0, description="Approximate token count")
    char_count: int = Field(default=0, description="Character count")

    # Overlap tracking
    has_overlap_before: bool = Field(default=False, description="Has overlap with previous chunk")
    has_overlap_after: bool = Field(default=False, description="Has overlap with next chunk")
    overlap_tokens: int = Field(default=0, description="Number of overlap tokens")

    # Additional metadata (flexible)
    metadata: dict = Field(default_factory=dict, description="Additional metadata")

    @property
    def word_count(self) -> int:
        """Approximate word count."""
        return len(self.content.split())

    def to_db_dict(self) -> dict:
        """Convert to database-ready dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "paper_id": self.paper_id,
            "content": self.content,
            "section_title": self.section_title,
            "chunk_type": self.chunk_type.value,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
            "metadata": {
                "section_id": self.section_id,
                "paragraph_ids": self.paragraph_ids,
                "chunk_index": self.chunk_index,
                "token_count": self.token_count,
                "char_count": self.char_count,
                "has_overlap_before": self.has_overlap_before,
                "has_overlap_after": self.has_overlap_after,
                **self.metadata,
            },
        }


class SparseVector(BaseModel):
    """
    Sparse vector representation for BM25-style retrieval.

    Stores token IDs and their weights, filtered to top-K for efficiency.
    """
    indices: list[int] = Field(..., description="Token IDs (vocabulary indices)")
    values: list[float] = Field(..., description="Token weights")

    @classmethod
    def from_dict(cls, token_weights: dict[int, float], top_k: int = 128) -> "SparseVector":
        """
        Create sparse vector from token weight dict, keeping top-K by weight.

        Args:
            token_weights: {token_id: weight} mapping
            top_k: Number of top weights to keep (default 128 for 60% storage reduction)
        """
        # Sort by weight descending and take top-K
        sorted_items = sorted(token_weights.items(), key=lambda x: x[1], reverse=True)[:top_k]

        if not sorted_items:
            return cls(indices=[], values=[])

        indices, values = zip(*sorted_items)
        return cls(indices=list(indices), values=list(values))

    def to_dict(self) -> dict[int, float]:
        """Convert to {token_id: weight} dict."""
        return dict(zip(self.indices, self.values))

    def to_jsonb(self) -> dict[str, float]:
        """Convert to JSONB-compatible format (string keys)."""
        return {str(idx): val for idx, val in zip(self.indices, self.values)}

    @classmethod
    def from_jsonb(cls, data: dict[str, float]) -> "SparseVector":
        """Load from JSONB format."""
        indices = [int(k) for k in data.keys()]
        values = list(data.values())
        return cls(indices=indices, values=values)

    @property
    def nnz(self) -> int:
        """Number of non-zero elements."""
        return len(self.indices)


class ColBERTVector(BaseModel):
    """
    Token-level embeddings for ColBERT MaxSim retrieval.

    ColBERT uses late interaction: query and document tokens are
    encoded separately, then MaxSim computes similarity per query token.

    Score = sum(max(cos_sim(q_token, d_tokens)) for q_token in query)
    """
    token_embeddings: List[List[float]] = Field(
        ..., description="Token-level embeddings [num_tokens, 1024]"
    )
    token_count: int = Field(..., description="Number of tokens")

    def to_jsonb(self) -> dict:
        """Convert to JSONB-compatible format for Supabase."""
        return {
            "token_embeddings": self.token_embeddings,
            "token_count": self.token_count,
        }

    @classmethod
    def from_jsonb(cls, data: dict) -> "ColBERTVector":
        """Load from JSONB format."""
        return cls(
            token_embeddings=data["token_embeddings"],
            token_count=data["token_count"],
        )

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension per token."""
        if self.token_embeddings and len(self.token_embeddings) > 0:
            return len(self.token_embeddings[0])
        return 0


class EmbeddedChunk(BaseModel):
    """
    A chunk with computed embeddings.

    Contains both dense (BGE-M3) and sparse vectors for hybrid retrieval.
    """
    chunk: Chunk = Field(..., description="Source chunk")

    # BGE-M3 embeddings
    embedding_dense: Optional[list[float]] = Field(None, description="Dense vector (1024 dims)")
    embedding_sparse: Optional[SparseVector] = Field(None, description="Sparse vector (top-128)")

    # OpenAI embedding (comparison)
    embedding_openai: Optional[list[float]] = Field(
        None,
        description="OpenAI vector (3072 dims, text-embedding-3-large)",
    )

    # ColBERT embedding (token-level for MaxSim)
    embedding_colbert: Optional[ColBERTVector] = Field(None, description="ColBERT token embeddings")

    # Embedding metadata
    model_bge: str = Field(default="BAAI/bge-m3", description="BGE model used")
    model_openai: Optional[str] = Field(None, description="OpenAI model used")
    embedded_at: datetime = Field(default_factory=datetime.now, description="Embedding timestamp")

    def to_db_dict(self) -> dict:
        """Convert to database-ready dictionary for Supabase (v1 with embeddings)."""
        base = self.chunk.to_db_dict()

        # Add embeddings
        base["embedding_dense"] = self.embedding_dense
        base["embedding_sparse"] = self.embedding_sparse.to_jsonb() if self.embedding_sparse else None
        base["embedding_openai"] = self.embedding_openai
        base["embedding_colbert"] = self.embedding_colbert.to_jsonb() if self.embedding_colbert else None

        return base

    def to_supabase_dict(self) -> dict:
        """Convert to Supabase metadata dict (v2 architecture - no vectors)."""
        return self.chunk.to_db_dict()

    def to_qdrant_dict(self) -> dict:
        """
        Convert to Qdrant-ready dictionary (v2 architecture).

        Returns dict with:
            - chunk_id, paper_id, content, section_title, metadata (payload)
            - dense_bge, dense_openai (dense vectors)
            - sparse_indices, sparse_values (sparse vector)
            - colbert_tokens (ColBERT multi-vector)
        """
        result = {
            "chunk_id": self.chunk.chunk_id,
            "paper_id": self.chunk.paper_id,
            "content": self.chunk.content,
            "section_title": self.chunk.section_title,
            "chunk_type": self.chunk.chunk_type.value if self.chunk.chunk_type else "text",
            "metadata": self.chunk.metadata,
        }

        # Add dense vectors
        if self.embedding_dense:
            result["dense_bge"] = self.embedding_dense
        if self.embedding_openai:
            result["dense_3large"] = self.embedding_openai

        # Add sparse vector
        if self.embedding_sparse:
            result["sparse_indices"] = self.embedding_sparse.indices
            result["sparse_values"] = self.embedding_sparse.values

        # Add ColBERT tokens
        if self.embedding_colbert:
            result["colbert_tokens"] = self.embedding_colbert.token_embeddings

        return result

    @property
    def has_bge_embeddings(self) -> bool:
        """Check if BGE embeddings are present."""
        return self.embedding_dense is not None

    @property
    def has_openai_embeddings(self) -> bool:
        """Check if OpenAI embeddings are present."""
        return self.embedding_openai is not None

    @property
    def has_colbert_embeddings(self) -> bool:
        """Check if ColBERT embeddings are present."""
        return self.embedding_colbert is not None


class ChunkingConfig(BaseModel):
    """Configuration for chunking strategy."""
    max_tokens: int = Field(default=512, description="Maximum tokens per chunk")
    overlap_tokens: int = Field(default=50, description="Overlap between chunks (~10%)")
    min_chunk_tokens: int = Field(default=100, description="Minimum chunk size")

    # Strategy options
    section_based: bool = Field(default=True, description="Use section boundaries")
    paragraph_based: bool = Field(default=True, description="Use paragraph boundaries")
    include_abstract: bool = Field(default=True, description="Create separate abstract chunk")
    include_equations: bool = Field(default=False, description="Create equation chunks (with descriptions)")
    include_tables: bool = Field(default=False, description="Create table chunks")

    # Paper context propagation (helps with conceptual query matching)
    add_paper_context: bool = Field(default=False, description="Prepend paper title/abstract to chunks")
    paper_context_tokens: int = Field(default=100, description="Max tokens for paper context prefix")

    # Token counting
    tokenizer_name: str = Field(default="cl100k_base", description="tiktoken tokenizer name")


class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation."""
    # BGE-M3 settings
    use_bge: bool = Field(default=True, description="Generate BGE-M3 embeddings")
    bge_model: str = Field(default="BAAI/bge-m3", description="BGE model name")
    bge_use_fp16: bool = Field(default=True, description="Use FP16 for BGE")
    bge_batch_size: int = Field(default=32, description="BGE batch size")
    sparse_top_k: int = Field(default=128, description="Top-K sparse tokens to keep")

    # OpenAI settings
    use_openai: bool = Field(default=False, description="Generate OpenAI embeddings")
    openai_model: str = Field(default="text-embedding-3-large", description="OpenAI model")
    openai_batch_size: int = Field(default=100, description="OpenAI batch size")
    openai_dimensions: int = Field(default=3072, description="OpenAI embedding dimensions (full 3-large)")

    # Processing
    device: str = Field(default="cuda", description="Device for local models")
    max_concurrent_openai: int = Field(default=5, description="Max concurrent OpenAI requests")


class ChunkingStats(BaseModel):
    """Statistics from chunking process."""
    total_papers: int = 0
    total_chunks: int = 0
    total_tokens: int = 0

    # By type
    abstract_chunks: int = 0
    text_chunks: int = 0
    equation_chunks: int = 0
    table_chunks: int = 0

    # Distribution
    avg_tokens_per_chunk: float = 0.0
    min_tokens: int = 0
    max_tokens: int = 0

    # Overlap
    chunks_with_overlap: int = 0

    def update_avg(self) -> None:
        """Update average calculations."""
        if self.total_chunks > 0:
            self.avg_tokens_per_chunk = self.total_tokens / self.total_chunks

    def summary(self) -> str:
        """Get human-readable summary."""
        self.update_avg()
        return (
            f"Chunking Stats:\n"
            f"  Papers: {self.total_papers}\n"
            f"  Chunks: {self.total_chunks}\n"
            f"  Tokens: {self.total_tokens:,}\n"
            f"  Avg tokens/chunk: {self.avg_tokens_per_chunk:.1f}\n"
            f"  Min/Max tokens: {self.min_tokens}/{self.max_tokens}\n"
            f"  By type:\n"
            f"    - Abstract: {self.abstract_chunks}\n"
            f"    - Text: {self.text_chunks}\n"
            f"    - Equation: {self.equation_chunks}\n"
            f"    - Table: {self.table_chunks}\n"
            f"  With overlap: {self.chunks_with_overlap}"
        )


class EmbeddingStats(BaseModel):
    """Statistics from embedding process."""
    total_chunks: int = 0
    bge_embedded: int = 0
    openai_embedded: int = 0

    bge_failed: int = 0
    openai_failed: int = 0

    total_bge_time: float = 0.0
    total_openai_time: float = 0.0

    @property
    def bge_chunks_per_second(self) -> float:
        """BGE embedding speed."""
        if self.total_bge_time > 0:
            return self.bge_embedded / self.total_bge_time
        return 0.0

    def summary(self) -> str:
        """Get human-readable summary."""
        return (
            f"Embedding Stats:\n"
            f"  Total chunks: {self.total_chunks}\n"
            f"  BGE-M3:\n"
            f"    - Embedded: {self.bge_embedded}\n"
            f"    - Failed: {self.bge_failed}\n"
            f"    - Time: {self.total_bge_time:.1f}s\n"
            f"    - Speed: {self.bge_chunks_per_second:.1f} chunks/s\n"
            f"  OpenAI:\n"
            f"    - Embedded: {self.openai_embedded}\n"
            f"    - Failed: {self.openai_failed}\n"
            f"    - Time: {self.total_openai_time:.1f}s"
        )
