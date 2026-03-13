"""
arXiv RAG v1 - FastAPI Search API

REST API endpoints for hybrid search and paper retrieval.
"""

from datetime import date
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..storage.supabase_client import get_supabase_client
from ..utils.logging import get_logger
from .retriever import HybridRetriever, SearchResponse, SearchResult
from .reranker import BGEReranker
from .qdrant_retriever import (
    QdrantHybridRetriever,
    qdrant_adaptive_search,
    qdrant_hybrid_search,
)

logger = get_logger("api")

# FastAPI app
app = FastAPI(
    title="arXiv RAG API",
    description="Hybrid search API for arXiv LLM papers",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (lazy loaded)
_retriever: Optional[HybridRetriever] = None
_qdrant_retriever: Optional[QdrantHybridRetriever] = None
_reranker: Optional[BGEReranker] = None


def get_retriever() -> HybridRetriever:
    """Get or create retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def get_qdrant_retriever() -> QdrantHybridRetriever:
    """Get or create Qdrant retriever instance."""
    global _qdrant_retriever
    if _qdrant_retriever is None:
        _qdrant_retriever = QdrantHybridRetriever()
    return _qdrant_retriever


def get_reranker() -> BGEReranker:
    """Get or create reranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = BGEReranker()
    return _reranker


# =============================================================================
# Request/Response Models
# =============================================================================


class SearchRequest(BaseModel):
    """Search request body."""
    query: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results")
    use_reranker: bool = Field(default=True, description="Apply reranking")
    rerank_top_k: int = Field(default=5, ge=1, le=20, description="Results after reranking")
    search_mode: str = Field(
        default="adaptive",
        description="Search mode: adaptive (auto strategy), qdrant_hybrid, hybrid (legacy), dense, sparse"
    )
    use_hyde: bool = Field(default=True, description="Enable HyDE expansion for conceptual queries (adaptive mode)")


class ChunkResponse(BaseModel):
    """Single chunk in search results."""
    chunk_id: str
    paper_id: str
    content: str
    section_title: Optional[str] = None
    score: float
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    reranker_score: Optional[float] = None


class SearchResponseModel(BaseModel):
    """Search response body."""
    query: str
    results: list[ChunkResponse]
    total_found: int
    search_time_ms: float
    reranked: bool = False
    # Adaptive search metadata
    search_mode: Optional[str] = None
    query_type: Optional[str] = None
    query_type_confidence: Optional[float] = None
    rrf_preset: Optional[str] = None
    hyde_used: Optional[bool] = None


class PaperResponse(BaseModel):
    """Paper details response."""
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: Optional[str] = None
    categories: list[str]
    published_date: Optional[date] = None
    citation_count: int = 0
    chunk_count: int = 0


class PaperWithChunksResponse(BaseModel):
    """Paper with its chunks."""
    paper: PaperResponse
    chunks: list[ChunkResponse]


class StatsResponse(BaseModel):
    """Database statistics."""
    total_papers: int
    embedded_papers: int
    total_chunks: int
    status: str = "healthy"


# =============================================================================
# API Endpoints
# =============================================================================


@app.get("/")
async def root():
    """API root - health check."""
    return {
        "status": "ok",
        "service": "arXiv RAG API",
        "version": "1.0.0",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        client = get_supabase_client()
        count = client.get_chunk_count()
        return {
            "status": "healthy",
            "chunks_indexed": count,
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
        }


@app.post("/search", response_model=SearchResponseModel)
async def search(request: SearchRequest):
    """
    Search for relevant paper chunks.

    Supports multiple search modes:
    - adaptive: Auto-selects strategy based on query type (recommended)
    - qdrant_hybrid: Qdrant RRF hybrid search
    - hybrid: Legacy Supabase hybrid search
    - dense: Dense-only semantic search
    - sparse: Sparse-only lexical search
    """
    import time
    start = time.time()

    try:
        response = None
        adaptive_metadata = {}

        # Perform search based on mode
        if request.search_mode == "adaptive":
            # Adaptive search with auto query classification and HyDE
            qdrant_retriever = get_qdrant_retriever()
            response = qdrant_retriever.search_adaptive(
                request.query,
                top_k=request.top_k,
                use_hyde=request.use_hyde,
                use_reranker=request.use_reranker,
                rerank_top_k=request.rerank_top_k,
            )
            # Extract adaptive metadata
            if response.metadata:
                adaptive_metadata = {
                    "query_type": response.metadata.get("query_type"),
                    "query_type_confidence": response.metadata.get("query_type_confidence"),
                    "rrf_preset": response.metadata.get("rrf_preset"),
                    "hyde_used": response.metadata.get("hyde_used"),
                }

        elif request.search_mode == "qdrant_hybrid":
            # Qdrant RRF hybrid search
            qdrant_retriever = get_qdrant_retriever()
            response = qdrant_retriever.search(
                request.query,
                top_k=request.top_k,
                use_reranker=request.use_reranker,
                rerank_top_k=request.rerank_top_k,
            )

        elif request.search_mode == "dense":
            # Dense-only search (Qdrant)
            qdrant_retriever = get_qdrant_retriever()
            response = qdrant_retriever.search_dense_only(request.query, top_k=request.top_k)

        elif request.search_mode == "sparse":
            # Sparse-only search (Qdrant)
            qdrant_retriever = get_qdrant_retriever()
            response = qdrant_retriever.search_sparse_only(request.query, top_k=request.top_k)

        else:
            # Legacy Supabase hybrid search
            retriever = get_retriever()
            response = retriever.search(request.query, top_k=request.top_k)

        results = response.results
        reranked = False

        # Apply reranking if not already done by adaptive/qdrant_hybrid
        if request.search_mode not in ("adaptive", "qdrant_hybrid"):
            if request.use_reranker and len(results) > request.rerank_top_k:
                reranker = get_reranker()
                results = reranker.rerank(request.query, results, top_k=request.rerank_top_k)
                reranked = True
        else:
            reranked = request.use_reranker

        # Convert to response model
        chunks = []
        for r in results:
            chunk = ChunkResponse(
                chunk_id=r.chunk_id,
                paper_id=r.paper_id,
                content=r.content,
                section_title=r.section_title,
                score=r.score,
                dense_score=r.dense_score,
                sparse_score=r.sparse_score,
                reranker_score=r.metadata.get("reranker_score") if r.metadata else None,
            )
            chunks.append(chunk)

        elapsed_ms = (time.time() - start) * 1000

        return SearchResponseModel(
            query=request.query,
            results=chunks,
            total_found=len(chunks),
            search_time_ms=elapsed_ms,
            reranked=reranked,
            search_mode=request.search_mode,
            query_type=adaptive_metadata.get("query_type"),
            query_type_confidence=adaptive_metadata.get("query_type_confidence"),
            rrf_preset=adaptive_metadata.get("rrf_preset"),
            hyde_used=adaptive_metadata.get("hyde_used"),
        )

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search", response_model=SearchResponseModel)
async def search_get(
    q: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(default=10, ge=1, le=50),
    rerank: bool = Query(default=True),
):
    """Search via GET request (convenience endpoint)."""
    request = SearchRequest(
        query=q,
        top_k=top_k,
        use_reranker=rerank,
    )
    return await search(request)


@app.get("/papers/{arxiv_id}", response_model=PaperResponse)
async def get_paper(arxiv_id: str):
    """Get paper details by arXiv ID."""
    try:
        client = get_supabase_client()
        paper = client.get_paper(arxiv_id)

        if not paper:
            raise HTTPException(status_code=404, detail=f"Paper not found: {arxiv_id}")

        # Get chunk count
        chunks = client.get_chunks_by_paper(arxiv_id)

        return PaperResponse(
            arxiv_id=paper["arxiv_id"],
            title=paper["title"],
            authors=paper.get("authors", []),
            abstract=paper.get("abstract"),
            categories=paper.get("categories", []),
            published_date=paper.get("published_date"),
            citation_count=paper.get("citation_count", 0),
            chunk_count=len(chunks),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get paper {arxiv_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/papers/{arxiv_id}/chunks", response_model=PaperWithChunksResponse)
async def get_paper_chunks(arxiv_id: str):
    """Get paper with all its chunks."""
    try:
        client = get_supabase_client()
        paper = client.get_paper(arxiv_id)

        if not paper:
            raise HTTPException(status_code=404, detail=f"Paper not found: {arxiv_id}")

        chunks_data = client.get_chunks_by_paper(arxiv_id)

        paper_response = PaperResponse(
            arxiv_id=paper["arxiv_id"],
            title=paper["title"],
            authors=paper.get("authors", []),
            abstract=paper.get("abstract"),
            categories=paper.get("categories", []),
            published_date=paper.get("published_date"),
            citation_count=paper.get("citation_count", 0),
            chunk_count=len(chunks_data),
        )

        chunks = []
        for c in chunks_data:
            chunk = ChunkResponse(
                chunk_id=c["chunk_id"],
                paper_id=c["paper_id"],
                content=c["content"],
                section_title=c.get("section_title"),
                score=0.0,
            )
            chunks.append(chunk)

        return PaperWithChunksResponse(
            paper=paper_response,
            chunks=chunks,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get paper chunks {arxiv_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chunks/{chunk_id}", response_model=ChunkResponse)
async def get_chunk(chunk_id: str):
    """Get a single chunk by ID."""
    try:
        client = get_supabase_client()
        chunk = client.get_chunk(chunk_id)

        if not chunk:
            raise HTTPException(status_code=404, detail=f"Chunk not found: {chunk_id}")

        return ChunkResponse(
            chunk_id=chunk["chunk_id"],
            paper_id=chunk["paper_id"],
            content=chunk["content"],
            section_title=chunk.get("section_title"),
            score=0.0,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get chunk {chunk_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get database statistics."""
    try:
        client = get_supabase_client()
        stats = client.get_collection_stats()

        return StatsResponse(
            total_papers=stats.get("total", 0),
            embedded_papers=stats.get("embedded", 0),
            total_chunks=client.get_chunk_count(),
            status="healthy",
        )

    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/papers", response_model=list[PaperResponse])
async def list_papers(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status: Optional[str] = Query(default=None, description="Filter by status"),
):
    """List papers with pagination."""
    try:
        client = get_supabase_client()

        if status:
            from ..collection.models import PaperStatus
            try:
                paper_status = PaperStatus(status)
                papers = client.get_papers_by_status(paper_status, limit=limit, offset=offset)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        else:
            papers = client.get_top_papers_by_citations(limit=limit)

        result = []
        for p in papers:
            result.append(PaperResponse(
                arxiv_id=p["arxiv_id"],
                title=p["title"],
                authors=p.get("authors", []),
                abstract=p.get("abstract"),
                categories=p.get("categories", []),
                published_date=p.get("published_date"),
                citation_count=p.get("citation_count", 0),
                chunk_count=0,
            ))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list papers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Startup/Shutdown Events
# =============================================================================


@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    logger.info("arXiv RAG API starting...")
    # Pre-warm Supabase connection
    try:
        client = get_supabase_client()
        count = client.get_chunk_count()
        logger.info(f"Connected to Supabase: {count} chunks indexed")
    except Exception as e:
        logger.error(f"Failed to connect to Supabase: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown."""
    global _retriever, _qdrant_retriever, _reranker

    logger.info("arXiv RAG API shutting down...")

    if _retriever:
        _retriever.embedder.unload()
        _retriever = None

    if _qdrant_retriever:
        _qdrant_retriever.unload_models()
        _qdrant_retriever = None

    if _reranker:
        _reranker.unload()
        _reranker = None
