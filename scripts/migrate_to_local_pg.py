#!/usr/bin/env python3
"""Migrate papers metadata from Supabase REST to local PostgreSQL."""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage import get_local_pg_client
from src.utils.config import settings
from src.utils.logging import get_logger, setup_logging

logger = get_logger("migrate_local_pg")

BATCH_SIZE = 1000
MIGRATION_COLUMNS = [
    "arxiv_id",
    "title",
    "authors",
    "abstract",
    "categories",
    "published_date",
    "citation_count",
    "download_count",
    "pdf_path",
    "latex_path",
    "parse_status",
    "parse_method",
    "created_at",
    "updated_at",
]


def fetch_supabase_batch(offset: int, limit: int) -> list[dict]:
    """Fetch a batch of papers from Supabase REST."""
    if not settings.supabase_url or not settings.supabase_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY are required for migration")

    select_clause = ",".join(MIGRATION_COLUMNS)
    query = urllib.parse.urlencode({"select": select_clause, "order": "arxiv_id.asc"})
    url = f"{settings.supabase_url.rstrip('/')}/rest/v1/papers?{query}"
    req = urllib.request.Request(
        url,
        headers={
            "apikey": settings.supabase_key,
            "Authorization": f"Bearer {settings.supabase_key}",
            "Range-Unit": "items",
            "Range": f"{offset}-{offset + limit - 1}",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def coerce_row(row: dict) -> dict:
    """Normalize a Supabase row before writing to local PG."""
    normalized = {col: row.get(col) for col in MIGRATION_COLUMNS}
    normalized.setdefault("authors", [])
    normalized.setdefault("categories", [])
    normalized.setdefault("citation_count", 0)
    normalized.setdefault("download_count", 0)
    return normalized


def migrate_papers() -> int:
    """Migrate papers from Supabase into local PostgreSQL."""
    pg = get_local_pg_client()
    total = 0
    offset = 0

    while True:
        batch = fetch_supabase_batch(offset, BATCH_SIZE)
        if not batch:
            break

        rows = [coerce_row(row) for row in batch]
        pg.batch_upsert_paper_dicts(rows)
        total += len(rows)
        logger.info("Migrated %s papers so far", total)

        if len(batch) < BATCH_SIZE:
            break

        offset += BATCH_SIZE
        time.sleep(1)

    return total


def mark_missing_latex_failed() -> int:
    """Mark papers without LaTeX archives as failed."""
    pg = get_local_pg_client()
    papers = pg.get_papers(fields=["arxiv_id", "latex_path"], limit=None, order_by="arxiv_id", desc=False)
    updated = 0
    for paper in papers:
        if not paper.get("latex_path"):
            if pg.update_paper(paper["arxiv_id"], {"parse_status": "failed"}):
                updated += 1
    return updated


def main() -> int:
    load_dotenv()
    setup_logging()

    try:
        migrated = migrate_papers()
        failed_updates = mark_missing_latex_failed()
        logger.info("Paper migration complete: %s rows upserted", migrated)
        logger.info("Marked %s papers without LaTeX as failed", failed_updates)
        return 0
    except urllib.error.HTTPError as e:
        logger.error("Supabase request failed: %s", e.read().decode("utf-8", errors="ignore"))
        return 1
    except Exception as e:
        logger.error("Migration failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
