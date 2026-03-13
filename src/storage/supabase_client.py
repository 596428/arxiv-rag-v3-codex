"""
arXiv RAG v1 - Supabase Client

Unified database client for papers, chunks, equations, and figures.
"""

from datetime import datetime
from typing import Optional

from supabase import create_client, Client

from ..utils.config import settings
from ..utils.logging import get_logger
from ..collection.models import Paper, PaperStatus, ParseMethod

logger = get_logger("supabase")


class SupabaseError(Exception):
    """Supabase operation error."""
    pass


class SupabaseClient:
    """
    Supabase client for arXiv RAG database operations.

    Tables:
    - papers: Paper metadata
    - chunks: Text chunks with embeddings
    - equations: LaTeX equations with descriptions
    - figures: Extracted figures with captions
    """

    def __init__(self, url: str = None, key: str = None):
        self.url = url or settings.supabase_url
        self.key = key or settings.supabase_key

        if not self.url or not self.key:
            raise ValueError(
                "Supabase not configured. Set SUPABASE_URL and SUPABASE_KEY in .env"
            )

        self._client: Optional[Client] = None
        logger.info(f"SupabaseClient initialized: {self.url[:50]}...")

    @property
    def client(self) -> Client:
        """Lazy initialization of Supabase client."""
        if self._client is None:
            self._client = create_client(self.url, self.key)
        return self._client

    # =========================================================================
    # Papers Table Operations
    # =========================================================================

    def insert_paper(self, paper: Paper) -> dict:
        """
        Insert a new paper into the database.

        Args:
            paper: Paper object to insert

        Returns:
            Inserted record as dict

        Raises:
            SupabaseError: Insert failed
        """
        data = paper.to_db_dict()

        try:
            result = (
                self.client.table("papers")
                .insert(data)
                .execute()
            )

            if result.data:
                logger.debug(f"Inserted paper: {paper.arxiv_id}")
                return result.data[0]
            else:
                raise SupabaseError(f"Insert returned no data: {paper.arxiv_id}")

        except Exception as e:
            # Check for duplicate key error
            if "duplicate key" in str(e).lower():
                logger.debug(f"Paper already exists: {paper.arxiv_id}")
                return self.get_paper(paper.arxiv_id)
            raise SupabaseError(f"Failed to insert paper: {e}")

    def upsert_paper(self, paper: Paper) -> dict:
        """
        Insert or update a paper.

        Args:
            paper: Paper object

        Returns:
            Upserted record as dict
        """
        data = paper.to_db_dict()

        try:
            result = (
                self.client.table("papers")
                .upsert(data, on_conflict="arxiv_id")
                .execute()
            )

            if result.data:
                logger.debug(f"Upserted paper: {paper.arxiv_id}")
                return result.data[0]
            else:
                raise SupabaseError(f"Upsert returned no data: {paper.arxiv_id}")

        except Exception as e:
            raise SupabaseError(f"Failed to upsert paper: {e}")

    def get_paper(self, arxiv_id: str) -> Optional[dict]:
        """
        Get a paper by arXiv ID.

        Args:
            arxiv_id: arXiv paper ID

        Returns:
            Paper record as dict, or None if not found
        """
        try:
            result = (
                self.client.table("papers")
                .select("*")
                .eq("arxiv_id", arxiv_id)
                .execute()
            )

            if result.data:
                return result.data[0]
            return None

        except Exception as e:
            logger.error(f"Failed to get paper {arxiv_id}: {e}")
            return None

    def update_paper(self, arxiv_id: str, data: dict) -> bool:
        """Update arbitrary paper fields."""
        if not data:
            return True

        payload = dict(data)
        payload.setdefault("updated_at", datetime.utcnow().isoformat())

        try:
            result = (
                self.client.table("papers")
                .update(payload)
                .eq("arxiv_id", arxiv_id)
                .execute()
            )
            return bool(result.data)
        except Exception as e:
            logger.error(f"Failed to update paper {arxiv_id}: {e}")
            return False

    def paper_exists(self, arxiv_id: str) -> bool:
        """Check if a paper exists in the database."""
        return self.get_paper(arxiv_id) is not None

    def get_papers(
        self,
        fields: list[str] | None = None,
        limit: int | None = 100,
        offset: int = 0,
        status: str | None = None,
        order_by: str = "citation_count",
        desc: bool = True,
        require_abstract: bool = False,
    ) -> list[dict]:
        """Get papers with optional filtering, sorting, and pagination."""
        select_fields = ", ".join(fields) if fields else "*"
        rows: list[dict] = []
        batch_size = limit if limit is not None else 1000
        current_offset = offset

        while True:
            query = self.client.table("papers").select(select_fields)

            if status:
                query = query.eq("parse_status", status)
            if require_abstract:
                query = query.not_.is_("abstract", "null")

            query = query.order(order_by, desc=desc)
            query = query.range(current_offset, current_offset + batch_size - 1)
            result = query.execute()
            batch = result.data or []
            rows.extend(batch)

            if limit is not None:
                return rows[:limit]
            if len(batch) < batch_size:
                return rows
            current_offset += batch_size

    def list_papers(
        self,
        page_size: int,
        offset: int = 0,
        status: str | None = None,
        sort_by: str = "citation_count",
        desc: bool = True,
        fields: list[str] | None = None,
    ) -> tuple[list[dict], int]:
        """List papers with total count for pagination."""
        select_fields = ", ".join(fields) if fields else "*"
        query = self.client.table("papers").select(select_fields, count="exact")

        if status:
            query = query.eq("parse_status", status)

        result = (
            query.order(sort_by, desc=desc)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        return result.data or [], result.count or 0

    def get_papers_by_status(
        self,
        status: PaperStatus,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """
        Get papers by processing status.

        Args:
            status: Paper status to filter by
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of paper records
        """
        try:
            result = (
                self.client.table("papers")
                .select("*")
                .eq("parse_status", status.value)
                .order("citation_count", desc=True)
                .range(offset, offset + limit - 1)
                .execute()
            )

            return result.data or []

        except Exception as e:
            logger.error(f"Failed to get papers by status: {e}")
            return []

    def update_paper_status(
        self,
        arxiv_id: str,
        status: PaperStatus,
        parse_method: ParseMethod = None,
    ) -> bool:
        """
        Update paper processing status.

        Args:
            arxiv_id: arXiv paper ID
            status: New status
            parse_method: Parsing method used (optional)

        Returns:
            True if successful
        """
        data = {"parse_status": status.value}

        if parse_method:
            data["parse_method"] = parse_method.value

        return self.update_paper(arxiv_id, data)

    def update_paper_paths(
        self,
        arxiv_id: str,
        pdf_path: str = None,
        latex_path: str = None,
    ) -> bool:
        """
        Update paper file paths after download.

        Args:
            arxiv_id: arXiv paper ID
            pdf_path: Local PDF path
            latex_path: Local LaTeX path

        Returns:
            True if successful
        """
        data = {}

        if pdf_path:
            data["pdf_path"] = pdf_path
        if latex_path:
            data["latex_path"] = latex_path

        return self.update_paper(arxiv_id, data)

    def batch_insert_papers(self, papers: list[Paper]) -> int:
        """
        Insert multiple papers in a batch.

        Uses upsert to handle duplicates gracefully.

        Args:
            papers: List of Paper objects

        Returns:
            Number of papers inserted/updated
        """
        if not papers:
            return 0

        data = [p.to_db_dict() for p in papers]

        try:
            result = (
                self.client.table("papers")
                .upsert(data, on_conflict="arxiv_id")
                .execute()
            )

            count = len(result.data) if result.data else 0
            logger.info(f"Batch upserted {count} papers")
            return count

        except Exception as e:
            logger.error(f"Batch insert failed: {e}")
            # Fall back to individual inserts
            count = 0
            for paper in papers:
                try:
                    self.upsert_paper(paper)
                    count += 1
                except SupabaseError:
                    pass
            return count

    def get_paper_count(self) -> int:
        """Get total number of papers in database."""
        try:
            result = (
                self.client.table("papers")
                .select("arxiv_id", count="exact")
                .execute()
            )
            return result.count or 0

        except Exception as e:
            logger.error(f"Failed to get paper count: {e}")
            return 0

    def get_papers_for_parsing(self, limit: int = 100) -> list[dict]:
        """
        Get papers that are ready for parsing.

        Returns papers with status='pending' that have pdf_path or latex_path.

        Args:
            limit: Maximum number of results

        Returns:
            List of paper records
        """
        try:
            result = (
                self.client.table("papers")
                .select("*")
                .eq("parse_status", "pending")
                .or_("pdf_path.neq.null,latex_path.neq.null")
                .order("citation_count", desc=True)
                .limit(limit)
                .execute()
            )

            return result.data or []

        except Exception as e:
            logger.error(f"Failed to get papers for parsing: {e}")
            return []

    def get_top_papers_by_citations(self, limit: int = 1000) -> list[dict]:
        """
        Get top papers ordered by citation count.

        Args:
            limit: Maximum number of results

        Returns:
            List of paper records ordered by citations
        """
        try:
            result = (
                self.client.table("papers")
                .select("*")
                .order("citation_count", desc=True)
                .limit(limit)
                .execute()
            )

            return result.data or []

        except Exception as e:
            logger.error(f"Failed to get top papers: {e}")
            return []

    # =========================================================================
    # Chunks Table Operations
    # =========================================================================

    def insert_chunk(self, chunk_data: dict) -> dict:
        """
        Insert a single chunk with embeddings.

        Args:
            chunk_data: Chunk data dict from EmbeddedChunk.to_db_dict()

        Returns:
            Inserted record as dict
        """
        try:
            result = (
                self.client.table("chunks")
                .insert(chunk_data)
                .execute()
            )

            if result.data:
                logger.debug(f"Inserted chunk: {chunk_data.get('chunk_id')}")
                return result.data[0]
            else:
                raise SupabaseError(f"Insert returned no data: {chunk_data.get('chunk_id')}")

        except Exception as e:
            if "duplicate key" in str(e).lower():
                logger.debug(f"Chunk already exists: {chunk_data.get('chunk_id')}")
                return chunk_data
            raise SupabaseError(f"Failed to insert chunk: {e}")

    def upsert_chunk(self, chunk_data: dict) -> dict:
        """
        Insert or update a chunk.

        Args:
            chunk_data: Chunk data dict

        Returns:
            Upserted record as dict
        """
        try:
            result = (
                self.client.table("chunks")
                .upsert(chunk_data, on_conflict="chunk_id")
                .execute()
            )

            if result.data:
                return result.data[0]
            else:
                raise SupabaseError(f"Upsert returned no data")

        except Exception as e:
            raise SupabaseError(f"Failed to upsert chunk: {e}")

    def batch_insert_chunks(self, chunks_data: list[dict]) -> int:
        """
        Insert multiple chunks in a batch.

        Args:
            chunks_data: List of chunk data dicts

        Returns:
            Number of chunks inserted/updated
        """
        if not chunks_data:
            return 0

        try:
            result = (
                self.client.table("chunks")
                .upsert(chunks_data, on_conflict="chunk_id")
                .execute()
            )

            count = len(result.data) if result.data else 0
            logger.info(f"Batch upserted {count} chunks")
            return count

        except Exception as e:
            logger.error(f"Batch chunk insert failed: {e}")
            # Fall back to individual inserts
            count = 0
            for chunk in chunks_data:
                try:
                    self.upsert_chunk(chunk)
                    count += 1
                except SupabaseError:
                    pass
            return count

    def batch_insert_chunks_metadata(self, chunks_data: list[dict]) -> int:
        """
        Insert chunk metadata only (v2 architecture).

        v2 Architecture: Supabase stores metadata, Qdrant stores vectors.
        This method inserts only the metadata fields without any embedding columns.

        Args:
            chunks_data: List of chunk dicts with keys:
                - chunk_id, paper_id, content, section_title, chunk_type, metadata

        Returns:
            Number of chunks inserted/updated
        """
        if not chunks_data:
            return 0

        # Filter to only metadata fields (no embeddings)
        metadata_fields = ["chunk_id", "paper_id", "content", "section_title",
                          "chunk_type", "chunk_index", "token_count", "metadata"]

        clean_chunks = []
        for chunk in chunks_data:
            clean = {k: v for k, v in chunk.items() if k in metadata_fields}
            clean_chunks.append(clean)

        return self.batch_insert_chunks(clean_chunks)

    def get_chunks_by_paper(self, paper_id: str) -> list[dict]:
        """
        Get all chunks for a paper.

        Args:
            paper_id: arXiv paper ID

        Returns:
            List of chunk records
        """
        try:
            result = (
                self.client.table("chunks")
                .select("*")
                .eq("paper_id", paper_id)
                .order("metadata->chunk_index")
                .execute()
            )

            return result.data or []

        except Exception as e:
            logger.error(f"Failed to get chunks for paper {paper_id}: {e}")
            return []

    def get_chunk(self, chunk_id: str) -> dict | None:
        """
        Get a chunk by ID.

        Args:
            chunk_id: Chunk ID

        Returns:
            Chunk record or None
        """
        try:
            result = (
                self.client.table("chunks")
                .select("*")
                .eq("chunk_id", chunk_id)
                .execute()
            )

            if result.data:
                return result.data[0]
            return None

        except Exception as e:
            logger.error(f"Failed to get chunk {chunk_id}: {e}")
            return None

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict]:
        """
        Get multiple chunks by their IDs.

        v2 Architecture: Used for fetching content after Qdrant vector search.
        Qdrant returns chunk_ids + scores, then we fetch full content from Supabase.

        Args:
            chunk_ids: List of chunk IDs to fetch

        Returns:
            List of chunk records (order may not match input)
        """
        if not chunk_ids:
            return []

        try:
            result = (
                self.client.table("chunks")
                .select("chunk_id, paper_id, content, section_title, chunk_type, metadata")
                .in_("chunk_id", chunk_ids)
                .execute()
            )

            if result.data:
                logger.debug(f"Fetched {len(result.data)} chunks by IDs")
                return result.data
            return []

        except Exception as e:
            logger.error(f"Failed to get chunks by IDs: {e}")
            return []

    def get_chunks_by_ids_ordered(self, chunk_ids: list[str]) -> list[dict]:
        """
        Get multiple chunks by IDs, preserving the input order.

        Useful when chunk_ids come from ranked search results.

        Args:
            chunk_ids: List of chunk IDs in desired order

        Returns:
            List of chunk records in same order as input
        """
        if not chunk_ids:
            return []

        chunks = self.get_chunks_by_ids(chunk_ids)

        # Build lookup map
        chunk_map = {c["chunk_id"]: c for c in chunks}

        # Return in original order, skipping missing
        ordered = []
        for cid in chunk_ids:
            if cid in chunk_map:
                ordered.append(chunk_map[cid])

        return ordered

    def delete_chunks_by_paper(self, paper_id: str) -> int:
        """
        Delete all chunks for a paper.

        Args:
            paper_id: arXiv paper ID

        Returns:
            Number of chunks deleted
        """
        try:
            result = (
                self.client.table("chunks")
                .delete()
                .eq("paper_id", paper_id)
                .execute()
            )

            count = len(result.data) if result.data else 0
            logger.info(f"Deleted {count} chunks for paper {paper_id}")
            return count

        except Exception as e:
            logger.error(f"Failed to delete chunks for paper {paper_id}: {e}")
            return 0

    def get_chunk_count(self) -> int:
        """Get total number of chunks in database."""
        try:
            result = (
                self.client.table("chunks")
                .select("chunk_id", count="exact")
                .execute()
            )
            return result.count or 0

        except Exception as e:
            logger.error(f"Failed to get chunk count: {e}")
            return 0

    def get_papers_with_chunks(self) -> list[str]:
        """Get list of paper IDs that have chunks."""
        try:
            result = (
                self.client.table("chunks")
                .select("paper_id")
                .execute()
            )

            if result.data:
                return list(set(r["paper_id"] for r in result.data))
            return []

        except Exception as e:
            logger.error(f"Failed to get papers with chunks: {e}")
            return []

    def get_papers_for_embedding(self, limit: int = 100) -> list[dict]:
        """
        Get papers that are parsed but not yet embedded.

        Returns papers with status='parsed' that don't have chunks yet.

        Args:
            limit: Maximum number of results

        Returns:
            List of paper records
        """
        try:
            # Get parsed papers
            result = (
                self.client.table("papers")
                .select("*")
                .eq("parse_status", "parsed")
                .order("citation_count", desc=True)
                .limit(limit)
                .execute()
            )

            return result.data or []

        except Exception as e:
            logger.error(f"Failed to get papers for embedding: {e}")
            return []

    def update_chunk_colbert(self, chunk_id: str, colbert_data: dict) -> bool:
        """
        Update a chunk with ColBERT embedding.

        Args:
            chunk_id: Chunk ID
            colbert_data: ColBERT embedding data (token_embeddings, token_count)

        Returns:
            True if successful
        """
        try:
            result = (
                self.client.table("chunks")
                .update({"embedding_colbert": colbert_data})
                .eq("chunk_id", chunk_id)
                .execute()
            )

            return bool(result.data)

        except Exception as e:
            logger.error(f"Failed to update chunk ColBERT {chunk_id}: {e}")
            return False

    def batch_update_colbert(self, updates: list[dict]) -> int:
        """
        Batch update chunks with ColBERT embeddings.

        Args:
            updates: List of dicts with chunk_id and embedding_colbert

        Returns:
            Number of chunks updated
        """
        if not updates:
            return 0

        count = 0
        for update in updates:
            if self.update_chunk_colbert(
                update["chunk_id"],
                update["embedding_colbert"]
            ):
                count += 1

        logger.info(f"Updated {count}/{len(updates)} chunks with ColBERT embeddings")
        return count

    def get_chunks_without_colbert(self, limit: int = 1000) -> list[dict]:
        """
        Get chunks that don't have ColBERT embeddings.

        Args:
            limit: Maximum number of results

        Returns:
            List of chunk records
        """
        try:
            result = (
                self.client.table("chunks")
                .select("chunk_id, paper_id, content")
                .is_("embedding_colbert", "null")
                .limit(limit)
                .execute()
            )

            return result.data or []

        except Exception as e:
            logger.error(f"Failed to get chunks without ColBERT: {e}")
            return []

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_collection_stats(self) -> dict:
        """
        Get collection statistics.

        Returns:
            Dict with counts by status
        """
        try:
            total = self.get_paper_count()

            pending = len(self.get_papers_by_status(PaperStatus.PENDING, limit=10000))
            collected = len(self.get_papers_by_status(PaperStatus.COLLECTED, limit=10000))
            parsed = len(self.get_papers_by_status(PaperStatus.PARSED, limit=10000))
            embedded = len(self.get_papers_by_status(PaperStatus.EMBEDDED, limit=10000))
            failed = len(self.get_papers_by_status(PaperStatus.FAILED, limit=10000))

            return {
                "total": total,
                "pending": pending,
                "collected": collected,
                "parsed": parsed,
                "embedded": embedded,
                "failed": failed,
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}


# Singleton client
_client: Optional[SupabaseClient] = None


def get_supabase_client() -> SupabaseClient:
    """Get or create the Supabase client singleton."""
    global _client
    if _client is None:
        _client = SupabaseClient()
    return _client
