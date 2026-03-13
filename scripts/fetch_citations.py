#!/usr/bin/env python3
"""
arXiv RAG v1 - Fetch Citation Counts

Fetches citation counts from Semantic Scholar for all papers in the database.
Uses batch API (500 papers per request) for efficiency.

Usage:
    python scripts/fetch_citations.py
    python scripts/fetch_citations.py --limit 1000  # Test with subset
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.collection.semantic_scholar import SemanticScholarClient, close_client
from src.storage import get_db_client
from src.utils.logging import get_logger, setup_logging

logger = get_logger("fetch_citations")


async def fetch_all_papers(client, limit: int = None) -> list[dict]:
    """Fetch all papers from database."""
    rows = client.get_papers(fields=["arxiv_id", "citation_count"], limit=limit, order_by="citation_count")
    logger.info(f"Fetched {len(rows)} papers from DB...")
    return rows


async def update_citations_batch(
    db_client,
    updates: list[tuple[str, int]],
    batch_size: int = 100,
) -> int:
    """Update citation counts in database."""
    updated = 0

    for i in range(0, len(updates), batch_size):
        batch = updates[i:i + batch_size]

        for arxiv_id, citation_count in batch:
            try:
                if db_client.update_paper(arxiv_id, {"citation_count": citation_count}):
                    updated += 1
            except Exception as e:
                logger.warning(f"Failed to update {arxiv_id}: {e}")

        if (i + batch_size) % 1000 == 0:
            logger.info(f"Updated {i + batch_size}/{len(updates)} papers in DB")

    return updated


async def main():
    parser = argparse.ArgumentParser(description="Fetch citation counts from Semantic Scholar")
    parser.add_argument("--limit", type=int, help="Limit number of papers (for testing)")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size for S2 API (max 500)")
    parser.add_argument("--dry-run", action="store_true", help="Don't update database")

    args = parser.parse_args()

    setup_logging()

    db_client = get_db_client()
    s2_client = SemanticScholarClient(batch_size=min(args.batch_size, 500))

    try:
        # Fetch papers from database
        logger.info("Fetching papers from database...")
        papers = await fetch_all_papers(db_client, limit=args.limit)
        logger.info(f"Total papers: {len(papers)}")

        # Get arxiv IDs
        arxiv_ids = [p["arxiv_id"] for p in papers]

        # Fetch citations from Semantic Scholar
        logger.info(f"Fetching citations from Semantic Scholar (batch_size={s2_client.batch_size})...")
        citations = await s2_client.batch_get_citations(arxiv_ids)

        # Statistics
        non_zero = sum(1 for c in citations.values() if c > 0)
        total_citations = sum(citations.values())
        max_citations = max(citations.values()) if citations else 0

        logger.info(f"Citation stats: {non_zero}/{len(citations)} papers have citations")
        logger.info(f"Total citations: {total_citations}, Max: {max_citations}")

        # Top cited papers
        top_cited = sorted(citations.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\n" + "=" * 50)
        print("TOP 10 MOST CITED PAPERS")
        print("=" * 50)
        for arxiv_id, count in top_cited:
            print(f"  [{count:5d}] {arxiv_id}")
        print("=" * 50)

        if args.dry_run:
            logger.info("Dry run - not updating database")
            return

        # Update database
        logger.info("Updating database with citation counts...")
        updates = [(arxiv_id, count) for arxiv_id, count in citations.items()]
        updated = await update_citations_batch(db_client, updates)

        logger.info(f"Updated {updated} papers in database")

        print("\n" + "=" * 50)
        print("CITATION FETCH COMPLETE")
        print("=" * 50)
        print(f"Papers processed: {len(papers)}")
        print(f"Papers with citations: {non_zero}")
        print(f"Total citations: {total_citations}")
        print(f"Database updated: {updated}")
        print("=" * 50)

    finally:
        await close_client()


if __name__ == "__main__":
    asyncio.run(main())
