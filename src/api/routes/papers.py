"""Paper metadata endpoints using the configured metadata DB."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ...storage import get_db_client
from ...utils.logging import get_logger

logger = get_logger("api.papers")

router = APIRouter()


class PaperSummary(BaseModel):
    """Paper summary for list view."""

    arxiv_id: str
    title: str
    authors: list[str]
    published_date: Optional[str] = None
    categories: list[str] = []
    citation_count: Optional[int] = None
    parse_status: Optional[str] = None


class PaperDetail(BaseModel):
    """Full paper details."""

    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published_date: Optional[str] = None
    updated_date: Optional[str] = None
    categories: list[str] = []
    citation_count: Optional[int] = None
    pdf_url: Optional[str] = None
    parse_status: Optional[str] = None
    parse_method: Optional[str] = None
    chunk_count: Optional[int] = None


class PapersListResponse(BaseModel):
    """Papers list response."""

    papers: list[PaperSummary]
    total: int
    page: int
    page_size: int


@router.get("/papers", response_model=PapersListResponse)
async def list_papers(
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Results per page"),
    status: Optional[str] = Query(default=None, description="Filter by parse status"),
    sort_by: str = Query(default="citation_count", description="Sort field"),
    order: str = Query(default="desc", description="Sort order (asc/desc)"),
):
    """List papers with pagination."""
    try:
        client = get_db_client()
        offset = (page - 1) * page_size
        rows, total = client.list_papers(
            page_size=page_size,
            offset=offset,
            status=status,
            sort_by=sort_by,
            desc=order.lower() == "desc",
            fields=[
                "arxiv_id",
                "title",
                "authors",
                "published_date",
                "categories",
                "citation_count",
                "parse_status",
            ],
        )

        papers = [
            PaperSummary(
                arxiv_id=row["arxiv_id"],
                title=row["title"],
                authors=row.get("authors", []),
                published_date=row.get("published_date"),
                categories=row.get("categories", []),
                citation_count=row.get("citation_count"),
                parse_status=row.get("parse_status"),
            )
            for row in rows
        ]

        return PapersListResponse(papers=papers, total=total, page=page, page_size=page_size)
    except Exception as e:
        logger.error("Failed to list papers: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/papers/{arxiv_id}", response_model=PaperDetail)
async def get_paper(arxiv_id: str):
    """Get paper details by arXiv ID."""
    try:
        client = get_db_client()
        paper = client.get_paper(arxiv_id)
        if not paper:
            raise HTTPException(status_code=404, detail=f"Paper not found: {arxiv_id}")

        chunk_count = len(client.get_chunks_by_paper(arxiv_id))
        return PaperDetail(
            arxiv_id=paper["arxiv_id"],
            title=paper["title"],
            authors=paper.get("authors", []),
            abstract=paper.get("abstract", ""),
            published_date=paper.get("published_date"),
            updated_date=paper.get("updated_date"),
            categories=paper.get("categories", []),
            citation_count=paper.get("citation_count"),
            pdf_url=paper.get("pdf_url"),
            parse_status=paper.get("parse_status"),
            parse_method=paper.get("parse_method"),
            chunk_count=chunk_count,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get paper %s: %s", arxiv_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/papers/{arxiv_id}/chunks")
async def get_paper_chunks(
    arxiv_id: str,
    include_embeddings: bool = Query(default=False, description="Include embedding vectors"),
):
    """Get all chunks for a paper."""
    try:
        client = get_db_client()
        paper = client.get_paper(arxiv_id)
        if not paper:
            raise HTTPException(status_code=404, detail=f"Paper not found: {arxiv_id}")

        chunks = client.get_chunks_by_paper(arxiv_id)
        response_chunks = []
        for chunk in chunks:
            chunk_data = {
                "chunk_id": chunk["chunk_id"],
                "content": chunk["content"],
                "section_title": chunk.get("section_title"),
                "metadata": chunk.get("metadata", {}),
            }
            if include_embeddings:
                chunk_data["has_dense_embedding"] = chunk.get("embedding_dense") is not None
                chunk_data["has_sparse_embedding"] = chunk.get("embedding_sparse") is not None
                chunk_data["has_colbert_embedding"] = chunk.get("embedding_colbert") is not None
            response_chunks.append(chunk_data)

        return {
            "arxiv_id": arxiv_id,
            "title": paper["title"],
            "chunks": response_chunks,
            "total": len(response_chunks),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get chunks for %s: %s", arxiv_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats():
    """Get collection statistics."""
    try:
        client = get_db_client()
        stats = client.get_collection_stats()
        return {"papers": stats, "chunks": {"total": client.get_chunk_count()}}
    except Exception as e:
        logger.error("Failed to get stats: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
