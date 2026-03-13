"""
arXiv RAG v3 - Local PostgreSQL Client

Local PostgreSQL metadata client with a Supabase-like interface.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Any, Optional, Sequence

import psycopg2
from psycopg2 import sql
from psycopg2.extras import Json, RealDictCursor, execute_values
from psycopg2.pool import SimpleConnectionPool

from ..collection.models import Paper, PaperStatus, ParseMethod
from ..utils.config import settings
from ..utils.logging import get_logger

logger = get_logger("postgres")


class PostgresError(Exception):
    """Local PostgreSQL operation error."""


ALLOWED_PAPER_SORT_FIELDS = {
    "arxiv_id",
    "title",
    "published_date",
    "citation_count",
    "parse_status",
    "updated_at",
    "created_at",
}


class LocalPGClient:
    """Local PostgreSQL client for paper/chunk metadata."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        database: str | None = None,
        user: str | None = None,
        password: str | None = None,
        minconn: int = 2,
        maxconn: int = 10,
        connect_timeout: int = 30,
    ):
        self.host = host or settings.pg_host
        self.port = port or settings.pg_port
        self.database = database or settings.pg_database
        self.user = user or settings.pg_user
        self.password = password or settings.pg_password
        self.connect_timeout = connect_timeout

        if not all([self.host, self.port, self.database, self.user, self.password]):
            raise ValueError(
                "Local PostgreSQL not configured. Set PG_HOST, PG_PORT, PG_DATABASE, PG_USER, and PG_PASSWORD in .env"
            )

        self._pool = SimpleConnectionPool(
            minconn=minconn,
            maxconn=maxconn,
            host=self.host,
            port=self.port,
            dbname=self.database,
            user=self.user,
            password=self.password,
            connect_timeout=self.connect_timeout,
            cursor_factory=RealDictCursor,
        )
        logger.info("LocalPGClient initialized: %s:%s/%s", self.host, self.port, self.database)

    @contextmanager
    def connection(self):
        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    def close(self) -> None:
        """Close all pooled connections."""
        self._pool.closeall()

    @staticmethod
    def _normalize_write_row(row: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, dict):
                normalized[key] = Json(value)
            else:
                normalized[key] = value
        return normalized

    def _execute(self, query: sql.SQL, params: Sequence[Any] | None = None) -> None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)

    def _fetchone(self, query: sql.SQL, params: Sequence[Any] | None = None) -> Optional[dict]:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                row = cur.fetchone()
                return dict(row) if row else None

    def _fetchall(self, query: sql.SQL, params: Sequence[Any] | None = None) -> list[dict]:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall() or []
                return [dict(row) for row in rows]

    def _fetchval(self, query: sql.SQL, params: Sequence[Any] | None = None, default: Any = None) -> Any:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                row = cur.fetchone()
                if not row:
                    return default
                return next(iter(row.values()))

    # ---------------------------------------------------------------------
    # Papers
    # ---------------------------------------------------------------------

    def insert_paper(self, paper: Paper) -> dict:
        data = paper.to_db_dict()
        columns = list(data.keys())
        values = [self._normalize_write_row(data)[c] for c in columns]
        query = sql.SQL("""
            INSERT INTO papers ({fields})
            VALUES ({placeholders})
            RETURNING *
        """).format(
            fields=sql.SQL(", ").join(map(sql.Identifier, columns)),
            placeholders=sql.SQL(", ").join(sql.Placeholder() * len(columns)),
        )
        try:
            row = self._fetchone(query, values)
            if row:
                return row
            raise PostgresError(f"Insert returned no data: {paper.arxiv_id}")
        except psycopg2.errors.UniqueViolation:
            return self.get_paper(paper.arxiv_id) or data
        except Exception as e:
            raise PostgresError(f"Failed to insert paper: {e}") from e

    def upsert_paper(self, paper: Paper) -> dict:
        data = paper.to_db_dict()
        return self.upsert_paper_dict(data)

    def upsert_paper_dict(self, data: dict[str, Any]) -> dict:
        columns = list(data.keys())
        normalized = self._normalize_write_row(data)
        query = sql.SQL("""
            INSERT INTO papers ({fields})
            VALUES ({placeholders})
            ON CONFLICT (arxiv_id) DO UPDATE SET
            {updates}
            RETURNING *
        """).format(
            fields=sql.SQL(", ").join(map(sql.Identifier, columns)),
            placeholders=sql.SQL(", ").join(sql.Placeholder() * len(columns)),
            updates=sql.SQL(", ").join(
                sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(col), sql.Identifier(col))
                for col in columns if col != "arxiv_id"
            ),
        )
        row = self._fetchone(query, [normalized[c] for c in columns])
        if row:
            return row
        raise PostgresError(f"Upsert returned no data: {data.get('arxiv_id')}")

    def batch_upsert_paper_dicts(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0

        columns = list(rows[0].keys())
        values = [tuple(self._normalize_write_row(row).get(col) for col in columns) for row in rows]
        query = sql.SQL("""
            INSERT INTO papers ({fields}) VALUES %s
            ON CONFLICT (arxiv_id) DO UPDATE SET
            {updates}
        """).format(
            fields=sql.SQL(", ").join(map(sql.Identifier, columns)),
            updates=sql.SQL(", ").join(
                sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(col), sql.Identifier(col))
                for col in columns if col != "arxiv_id"
            ),
        )
        with self.connection() as conn:
            with conn.cursor() as cur:
                execute_values(cur, query.as_string(conn), values)
        return len(rows)

    def get_paper(self, arxiv_id: str) -> Optional[dict]:
        return self._fetchone(
            sql.SQL("SELECT * FROM papers WHERE arxiv_id = %s"),
            (arxiv_id,),
        )

    def update_paper(self, arxiv_id: str, data: dict[str, Any]) -> bool:
        if not data:
            return True

        payload = dict(data)
        payload.setdefault("updated_at", datetime.utcnow())
        normalized = self._normalize_write_row(payload)
        columns = list(normalized.keys())
        query = sql.SQL("""
            UPDATE papers
            SET {assignments}
            WHERE arxiv_id = %s
        """).format(
            assignments=sql.SQL(", ").join(
                sql.SQL("{} = {}").format(sql.Identifier(col), sql.Placeholder())
                for col in columns
            )
        )
        params = [normalized[col] for col in columns] + [arxiv_id]
        try:
            self._execute(query, params)
            return True
        except Exception as e:
            logger.error("Failed to update paper %s: %s", arxiv_id, e)
            return False

    def paper_exists(self, arxiv_id: str) -> bool:
        return self.get_paper(arxiv_id) is not None

    def update_paper_status(
        self,
        arxiv_id: str,
        status: PaperStatus,
        parse_method: ParseMethod | None = None,
    ) -> bool:
        data: dict[str, Any] = {"parse_status": status.value}
        if parse_method:
            data["parse_method"] = parse_method.value
        return self.update_paper(arxiv_id, data)

    def update_paper_paths(
        self,
        arxiv_id: str,
        pdf_path: str | None = None,
        latex_path: str | None = None,
    ) -> bool:
        data: dict[str, Any] = {}
        if pdf_path:
            data["pdf_path"] = pdf_path
        if latex_path:
            data["latex_path"] = latex_path
        return self.update_paper(arxiv_id, data)

    def batch_insert_papers(self, papers: list[Paper]) -> int:
        return self.batch_upsert_paper_dicts([paper.to_db_dict() for paper in papers])

    def get_paper_count(self) -> int:
        return int(self._fetchval(sql.SQL("SELECT COUNT(*) AS count FROM papers"), default=0) or 0)

    def get_papers_by_status(self, status: PaperStatus, limit: int = 100, offset: int = 0) -> list[dict]:
        return self._fetchall(
            sql.SQL(
                "SELECT * FROM papers WHERE parse_status = %s ORDER BY citation_count DESC NULLS LAST OFFSET %s LIMIT %s"
            ),
            (status.value, offset, limit),
        )

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
        sort_field = order_by if order_by in ALLOWED_PAPER_SORT_FIELDS else "citation_count"
        order_dir = sql.SQL("DESC") if desc else sql.SQL("ASC")
        select_fields = sql.SQL(", ").join(map(sql.Identifier, fields)) if fields else sql.SQL("*")
        clauses = []
        params: list[Any] = []
        if status:
            clauses.append(sql.SQL("parse_status = %s"))
            params.append(status)
        if require_abstract:
            clauses.append(sql.SQL("abstract IS NOT NULL"))
        where_clause = sql.SQL(" WHERE ") + sql.SQL(" AND ").join(clauses) if clauses else sql.SQL("")
        limit_clause = sql.SQL(" LIMIT %s") if limit is not None else sql.SQL("")
        if limit is not None:
            params.append(limit)
        params.insert(len(params) - (1 if limit is not None else 0), offset)
        query = sql.SQL("SELECT {fields} FROM papers{where_clause} ORDER BY {sort_field} {order_dir} NULLS LAST OFFSET %s{limit_clause}").format(
            fields=select_fields,
            where_clause=where_clause,
            sort_field=sql.Identifier(sort_field),
            order_dir=order_dir,
            limit_clause=limit_clause,
        )
        return self._fetchall(query, tuple(params))

    def list_papers(
        self,
        page_size: int,
        offset: int = 0,
        status: str | None = None,
        sort_by: str = "citation_count",
        desc: bool = True,
        fields: list[str] | None = None,
    ) -> tuple[list[dict], int]:
        sort_field = sort_by if sort_by in ALLOWED_PAPER_SORT_FIELDS else "citation_count"
        select_fields = sql.SQL(", ").join(map(sql.Identifier, fields)) if fields else sql.SQL("*")
        clauses = []
        params: list[Any] = []
        if status:
            clauses.append(sql.SQL("parse_status = %s"))
            params.append(status)
        where_clause = sql.SQL(" WHERE ") + sql.SQL(" AND ").join(clauses) if clauses else sql.SQL("")
        count_query = sql.SQL("SELECT COUNT(*) AS count FROM papers{where_clause}").format(where_clause=where_clause)
        total = int(self._fetchval(count_query, tuple(params), default=0) or 0)
        params_with_paging = params + [offset, page_size]
        order_dir = sql.SQL("DESC") if desc else sql.SQL("ASC")
        data_query = sql.SQL("""
            SELECT {fields}
            FROM papers{where_clause}
            ORDER BY {sort_field} {order_dir} NULLS LAST
            OFFSET %s LIMIT %s
        """).format(
            fields=select_fields,
            where_clause=where_clause,
            sort_field=sql.Identifier(sort_field),
            order_dir=order_dir,
        )
        return self._fetchall(data_query, tuple(params_with_paging)), total

    def get_papers_for_parsing(self, limit: int = 100) -> list[dict]:
        return self._fetchall(
            sql.SQL(
                """
                SELECT *
                FROM papers
                WHERE parse_status = 'pending'
                  AND (pdf_path IS NOT NULL OR latex_path IS NOT NULL)
                ORDER BY citation_count DESC NULLS LAST
                LIMIT %s
                """
            ),
            (limit,),
        )

    def get_top_papers_by_citations(self, limit: int = 1000) -> list[dict]:
        return self._fetchall(
            sql.SQL("SELECT * FROM papers ORDER BY citation_count DESC NULLS LAST LIMIT %s"),
            (limit,),
        )

    # ---------------------------------------------------------------------
    # Chunks
    # ---------------------------------------------------------------------

    def insert_chunk(self, chunk_data: dict) -> dict:
        return self.upsert_chunk(chunk_data)

    def upsert_chunk(self, chunk_data: dict) -> dict:
        columns = list(chunk_data.keys())
        normalized = self._normalize_write_row(chunk_data)
        query = sql.SQL("""
            INSERT INTO chunks ({fields})
            VALUES ({placeholders})
            ON CONFLICT (chunk_id) DO UPDATE SET
            {updates}
            RETURNING *
        """).format(
            fields=sql.SQL(", ").join(map(sql.Identifier, columns)),
            placeholders=sql.SQL(", ").join(sql.Placeholder() * len(columns)),
            updates=sql.SQL(", ").join(
                sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(col), sql.Identifier(col))
                for col in columns if col != "chunk_id"
            ),
        )
        row = self._fetchone(query, [normalized[c] for c in columns])
        if row:
            return row
        raise PostgresError(f"Upsert returned no data: {chunk_data.get('chunk_id')}")

    def batch_insert_chunks(self, chunks_data: list[dict]) -> int:
        if not chunks_data:
            return 0
        columns = list(chunks_data[0].keys())
        values = [tuple(self._normalize_write_row(row).get(col) for col in columns) for row in chunks_data]
        query = sql.SQL("""
            INSERT INTO chunks ({fields}) VALUES %s
            ON CONFLICT (chunk_id) DO UPDATE SET
            {updates}
        """).format(
            fields=sql.SQL(", ").join(map(sql.Identifier, columns)),
            updates=sql.SQL(", ").join(
                sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(col), sql.Identifier(col))
                for col in columns if col != "chunk_id"
            ),
        )
        with self.connection() as conn:
            with conn.cursor() as cur:
                execute_values(cur, query.as_string(conn), values)
        return len(chunks_data)

    def batch_insert_chunks_metadata(self, chunks_data: list[dict]) -> int:
        if not chunks_data:
            return 0
        metadata_fields = [
            "chunk_id",
            "paper_id",
            "content",
            "section_title",
            "chunk_type",
            "chunk_index",
            "token_count",
            "metadata",
        ]
        clean_chunks = [{k: v for k, v in row.items() if k in metadata_fields} for row in chunks_data]
        return self.batch_insert_chunks(clean_chunks)

    def get_chunks_by_paper(self, paper_id: str) -> list[dict]:
        return self._fetchall(
            sql.SQL("SELECT * FROM chunks WHERE paper_id = %s ORDER BY chunk_index ASC NULLS LAST, chunk_id ASC"),
            (paper_id,),
        )

    def get_chunk(self, chunk_id: str) -> dict | None:
        return self._fetchone(sql.SQL("SELECT * FROM chunks WHERE chunk_id = %s"), (chunk_id,))

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict]:
        if not chunk_ids:
            return []
        return self._fetchall(
            sql.SQL(
                "SELECT chunk_id, paper_id, content, section_title, chunk_type, chunk_index, token_count, metadata FROM chunks WHERE chunk_id = ANY(%s)"
            ),
            (chunk_ids,),
        )

    def get_chunks_by_ids_ordered(self, chunk_ids: list[str]) -> list[dict]:
        if not chunk_ids:
            return []
        chunks = self.get_chunks_by_ids(chunk_ids)
        chunk_map = {chunk["chunk_id"]: chunk for chunk in chunks}
        return [chunk_map[cid] for cid in chunk_ids if cid in chunk_map]

    def delete_chunks_by_paper(self, paper_id: str) -> int:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chunks WHERE paper_id = %s", (paper_id,))
                return cur.rowcount

    def get_chunk_count(self) -> int:
        return int(self._fetchval(sql.SQL("SELECT COUNT(*) AS count FROM chunks"), default=0) or 0)

    def get_papers_with_chunks(self) -> list[str]:
        rows = self._fetchall(sql.SQL("SELECT DISTINCT paper_id FROM chunks ORDER BY paper_id"))
        return [row["paper_id"] for row in rows]

    def get_papers_for_embedding(self, limit: int = 100) -> list[dict]:
        return self._fetchall(
            sql.SQL(
                """
                SELECT p.*
                FROM papers p
                LEFT JOIN chunks c ON c.paper_id = p.arxiv_id
                WHERE p.parse_status = 'parsed'
                GROUP BY p.arxiv_id, p.id
                HAVING COUNT(c.chunk_id) = 0
                ORDER BY p.citation_count DESC NULLS LAST
                LIMIT %s
                """
            ),
            (limit,),
        )

    # ---------------------------------------------------------------------
    # Stats
    # ---------------------------------------------------------------------

    def get_collection_stats(self) -> dict:
        rows = self._fetchall(
            sql.SQL(
                "SELECT parse_status, COUNT(*) AS count FROM papers GROUP BY parse_status"
            )
        )
        stats = {"total": self.get_paper_count(), "pending": 0, "collected": 0, "parsed": 0, "embedded": 0, "failed": 0}
        for row in rows:
            status = row["parse_status"]
            if status in stats:
                stats[status] = int(row["count"])
        return stats


_client: Optional[LocalPGClient] = None


def get_local_pg_client() -> LocalPGClient:
    """Get or create the LocalPG client singleton."""
    global _client
    if _client is None:
        _client = LocalPGClient()
    return _client
