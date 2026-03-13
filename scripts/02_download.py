#!/usr/bin/env python3
"""
arXiv RAG v1 - Download Script

Downloads PDF and LaTeX files for papers already saved in the database.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.collection import Paper, close_downloader, get_downloader
from src.storage import get_db_client
from src.utils.logging import get_logger, setup_logging

logger = get_logger("download")


async def load_papers_from_db(limit: int = None) -> list[Paper]:
    """Load papers from database."""
    client = get_db_client()
    rows = client.get_papers(limit=limit, order_by="citation_count", fields=[
        "arxiv_id", "title", "abstract", "authors", "categories", "published_date", "pdf_url", "citation_count"
    ])
    return [
        Paper(
            arxiv_id=row["arxiv_id"],
            title=row["title"],
            abstract=row.get("abstract", ""),
            authors=row.get("authors", []),
            categories=row.get("categories", []),
            published_date=row.get("published_date"),
            pdf_url=row.get("pdf_url"),
            citation_count=row.get("citation_count", 0),
        )
        for row in rows
    ]


async def download_papers(
    papers: list[Paper],
    download_pdf: bool = True,
    download_latex: bool = True,
) -> list[Paper]:
    """Download PDF and LaTeX files for papers."""
    logger.info("=" * 60)
    logger.info("Downloading Papers")
    logger.info("=" * 60)
    logger.info(f"Total papers: {len(papers)}")
    logger.info(f"Download PDF: {download_pdf}")
    logger.info(f"Download LaTeX: {download_latex}")

    downloader = get_downloader()

    try:
        downloaded = await downloader.download_batch(
            papers,
            download_pdf=download_pdf,
            download_latex=download_latex,
            skip_existing=True,
        )
    finally:
        await close_downloader()

    pdf_count = sum(1 for p in downloaded if p.pdf_path)
    latex_count = sum(1 for p in downloaded if p.latex_path)

    logger.info("=" * 60)
    logger.info("Download Complete")
    logger.info("=" * 60)
    logger.info(f"PDF downloaded: {pdf_count}/{len(papers)}")
    logger.info(f"LaTeX downloaded: {latex_count}/{len(papers)}")

    return downloaded


async def update_db_paths(papers: list[Paper]) -> int:
    """Update database with download paths."""
    client = get_db_client()
    updated = 0
    for paper in papers:
        if paper.pdf_path or paper.latex_path:
            client.update_paper_paths(
                paper.arxiv_id,
                pdf_path=str(paper.pdf_path) if paper.pdf_path else None,
                latex_path=str(paper.latex_path) if paper.latex_path else None,
            )
            updated += 1

    logger.info(f"Updated {updated} paper paths in database")
    return updated


async def load_papers_from_json(json_path: str) -> list[Paper]:
    """Load papers from JSON file (final_papers.json format)."""
    import json

    with open(json_path, "r") as f:
        data = json.load(f)

    papers_data = data.get("papers", data) if isinstance(data, dict) else data
    return [
        Paper(
            arxiv_id=row["arxiv_id"],
            title=row["title"],
            abstract=row.get("abstract", ""),
            authors=row.get("authors", []),
            categories=row.get("categories", []),
            published_date=row.get("published_date"),
            pdf_url=row.get("pdf_url"),
            citation_count=row.get("citation_count", 0),
        )
        for row in papers_data
    ]


async def main():
    parser = argparse.ArgumentParser(description="Download papers from database")
    parser.add_argument("--limit", type=int, help="Limit number of papers to download")
    parser.add_argument("--input", "-i", type=str, help="Input JSON file with papers (e.g., final_papers.json)")
    parser.add_argument("--pdf-only", action="store_true", help="Download PDF only")
    parser.add_argument("--latex-only", action="store_true", help="Download LaTeX only")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    import logging

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    if args.input:
        logger.info(f"Loading papers from {args.input}...")
        papers = await load_papers_from_json(args.input)
    else:
        logger.info("Loading papers from database...")
        papers = await load_papers_from_db(limit=args.limit)

    if args.limit and len(papers) > args.limit:
        papers = papers[:args.limit]

    logger.info(f"Loaded {len(papers)} papers")

    if args.dry_run:
        logger.info("Dry run - no files will be downloaded")
        for i, paper in enumerate(papers[:10], start=1):
            print(f"  {i}. {paper.arxiv_id}: {paper.title[:50]}...")
        if len(papers) > 10:
            print(f"  ... and {len(papers) - 10} more")
        return

    download_pdf = not args.latex_only
    download_latex = not args.pdf_only

    downloaded = await download_papers(papers, download_pdf, download_latex)
    await update_db_paths(downloaded)

    print("\nDownload complete!")


if __name__ == "__main__":
    asyncio.run(main())
