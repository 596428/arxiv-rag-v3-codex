"""Storage factories for metadata and vector backends."""

from __future__ import annotations

from ..utils.config import settings
from .postgres_client import LocalPGClient, PostgresError, get_local_pg_client
from .qdrant_client import COLLECTION_NAME, QdrantConfig, QdrantVectorClient, get_qdrant_client
from .supabase_client import SupabaseClient, SupabaseError, get_supabase_client


def get_db_client() -> LocalPGClient | SupabaseClient:
    """Return the configured metadata database client."""
    backend = (settings.db_backend or "local").lower()
    if backend == "supabase":
        return get_supabase_client()
    if backend == "local":
        return get_local_pg_client()
    raise ValueError(f"Unsupported DB_BACKEND: {settings.db_backend}")


__all__ = [
    "SupabaseClient",
    "SupabaseError",
    "get_supabase_client",
    "LocalPGClient",
    "PostgresError",
    "get_local_pg_client",
    "get_db_client",
    "QdrantVectorClient",
    "QdrantConfig",
    "get_qdrant_client",
    "COLLECTION_NAME",
]
