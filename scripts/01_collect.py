#!/usr/bin/env python3
"""
arXiv RAG v1 - Data Collection Script

Collects LLM papers from arXiv with 2-stage filtering:
1. Stage 1: Broad Recall (arXiv API + keyword filtering)
2. Stage 2a: Rule-based filtering (strong LLM indicators)
3. Stage 2b: LLM verification (edge cases via Gemini)
4. Stage 3: Ranking by citations and selection of top N

Usage:
    python scripts/01_collect.py --max-results 5000 --target 1000
    python scripts/01_collect.py --dry-run  # Preview without downloading
    python scripts/01_collect.py --resume   # Resume from existing DB state
"""

import argparse
import asyncio
import json
import sys
from datetime import date, datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.collection import (
    ArxivClient,
    Paper,
    PaperStatus,
    CollectionStats,
    CollectionState,
    generate_date_windows,
    get_semantic_scholar_client,
    close_semantic_scholar,
    get_downloader,
    close_downloader,
)
from src.storage import get_db_client
from src.utils.config import settings
from src.utils.logging import get_logger, setup_logging
from src.utils.gemini import get_gemini_client

# Checkpoint file location
CHECKPOINT_DIR = Path(__file__).parent.parent / "data" / "cache"
CHECKPOINT_FILE = CHECKPOINT_DIR / "collection_state.json"

# Stage 2b analysis output
ANALYSIS_DIR = Path(__file__).parent.parent / "data" / "analysis"
STAGE2B_VERIFIED_FILE = ANALYSIS_DIR / "stage2b_verified.json"
STAGE2B_REJECTED_FILE = ANALYSIS_DIR / "stage2b_rejected.json"
STAGE2A_EDGE_CASES_FILE = ANALYSIS_DIR / "stage2a_edge_cases.json"

logger = get_logger("collect")


def load_checkpoint() -> CollectionState | None:
    """Load checkpoint state from file."""
    if not CHECKPOINT_FILE.exists():
        return None
    try:
        with open(CHECKPOINT_FILE, "r") as f:
            data = json.load(f)
        state = CollectionState(**data)
        logger.info(f"Loaded checkpoint: {len(state.windows_completed)} windows completed, "
                    f"{state.papers_collected} papers collected")
        return state
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}")
        return None


def save_checkpoint(state: CollectionState) -> None:
    """Save checkpoint state to file."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(state.model_dump(mode="json"), f, indent=2, default=str)
        logger.debug(f"Checkpoint saved: {len(state.windows_completed)} windows completed")
    except Exception as e:
        logger.warning(f"Failed to save checkpoint: {e}")


def clear_checkpoint() -> None:
    """Remove checkpoint file."""
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logger.info("Checkpoint cleared")


def save_stage2b_results(
    verified: list[Paper],
    rejected: list[Paper],
    edge_cases: list[Paper] = None,
) -> None:
    """Save Stage 2b results to JSON files for analysis."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    def paper_to_dict(paper: Paper) -> dict:
        return {
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "abstract": paper.abstract[:500] + "..." if len(paper.abstract) > 500 else paper.abstract,
            "categories": paper.categories,
            "published_date": str(paper.published_date) if paper.published_date else None,
            "is_llm_relevant": paper.is_llm_relevant,
            "relevance_reason": paper.relevance_reason,
        }

    # Save verified papers
    with open(STAGE2B_VERIFIED_FILE, "w", encoding="utf-8") as f:
        json.dump([paper_to_dict(p) for p in verified], f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(verified)} verified papers to {STAGE2B_VERIFIED_FILE}")

    # Save rejected papers
    with open(STAGE2B_REJECTED_FILE, "w", encoding="utf-8") as f:
        json.dump([paper_to_dict(p) for p in rejected], f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(rejected)} rejected papers to {STAGE2B_REJECTED_FILE}")

    # Save edge cases (before verification)
    if edge_cases:
        with open(STAGE2A_EDGE_CASES_FILE, "w", encoding="utf-8") as f:
            json.dump([paper_to_dict(p) for p in edge_cases], f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(edge_cases)} edge cases to {STAGE2A_EDGE_CASES_FILE}")


async def stage1_broad_recall_windowed(
    arxiv_client: ArxivClient,
    start_date: date,
    end_date: date,
    window_days: int = 14,
    max_per_window: int = 10000,
    resume_state: CollectionState = None,
) -> tuple[list[Paper], CollectionState]:
    """
    Stage 1: Broad Recall using date windowing.

    Fetches papers using date windows to overcome arXiv API limitations.
    """
    logger.info("=" * 60)
    logger.info("Stage 1: Broad Recall (with Date Windowing)")
    logger.info("=" * 60)

    # Initialize or resume state
    state = resume_state or CollectionState()
    skip_windows = state.windows_completed if resume_state else []

    # Create checkpoint callback
    def on_window_complete(window_key: str, paper_count: int):
        state.mark_window_complete(window_key, paper_count)
        save_checkpoint(state)

    papers = await arxiv_client.search_with_windowing(
        start_date=start_date,
        end_date=end_date,
        window_days=window_days,
        max_per_window=max_per_window,
        on_window_complete=on_window_complete,
        skip_windows=skip_windows,
    )

    logger.info(f"Stage 1 complete: {len(papers)} papers fetched")
    return papers, state


async def stage1_broad_recall(
    arxiv_client: ArxivClient,
    start_date: date,
    end_date: date,
    max_results: int,
) -> list[Paper]:
    """
    Stage 1: Broad Recall from arXiv.

    Fetches papers matching LLM keywords from specified categories.
    """
    logger.info("=" * 60)
    logger.info("Stage 1: Broad Recall")
    logger.info("=" * 60)

    papers = await arxiv_client.search(
        start_date=start_date,
        end_date=end_date,
        max_results=max_results,
    )

    logger.info(f"Stage 1 complete: {len(papers)} papers fetched")
    return papers


async def stage2a_rule_based_filter(
    arxiv_client: ArxivClient,
    papers: list[Paper],
) -> tuple[list[Paper], list[Paper]]:
    """
    Stage 2a: Rule-based filtering.

    Separates clearly LLM papers from edge cases.
    """
    logger.info("=" * 60)
    logger.info("Stage 2a: Rule-based Filtering")
    logger.info("=" * 60)

    clearly_llm, edge_cases = arxiv_client.filter_stage2a(papers)

    logger.info(f"Stage 2a complete:")
    logger.info(f"  - Clearly LLM: {len(clearly_llm)}")
    logger.info(f"  - Edge cases: {len(edge_cases)}")

    return clearly_llm, edge_cases


async def stage2b_llm_verification(
    edge_cases: list[Paper],
    max_verify: int = 500,
) -> tuple[list[Paper], list[Paper]]:
    """
    Stage 2b: LLM verification for edge cases.

    Uses Gemini to verify if edge case papers are about LLM.
    """
    logger.info("=" * 60)
    logger.info("Stage 2b: LLM Verification")
    logger.info("=" * 60)

    if not edge_cases:
        return [], []

    # Limit verification to save API costs
    to_verify = edge_cases[:max_verify]
    logger.info(f"Verifying {len(to_verify)} edge cases (limit: {max_verify})")

    try:
        gemini = get_gemini_client()
    except ValueError as e:
        logger.warning(f"Gemini not available, skipping Stage 2b: {e}")
        return [], edge_cases

    verified = []
    rejected = []

    for i, paper in enumerate(to_verify):
        if (i + 1) % 50 == 0:
            logger.info(f"Progress: {i + 1}/{len(to_verify)}")

        try:
            is_llm, reason = await gemini.verify_llm_relevance(paper.abstract)
            paper.is_llm_relevant = is_llm
            paper.relevance_reason = reason

            if is_llm:
                verified.append(paper)
            else:
                rejected.append(paper)

        except Exception as e:
            logger.warning(f"Verification failed for {paper.arxiv_id}: {e}")
            # On error, include the paper (conservative approach)
            verified.append(paper)

    logger.info(f"Stage 2b complete:")
    logger.info(f"  - Verified LLM: {len(verified)}")
    logger.info(f"  - Rejected: {len(rejected)}")

    return verified, rejected


async def stage3_enrich_and_rank(
    papers: list[Paper],
    target_count: int,
) -> list[Paper]:
    """
    Stage 3: Enrich with citations and rank.

    Gets citation counts from Semantic Scholar and selects top papers.
    """
    logger.info("=" * 60)
    logger.info("Stage 3: Citation Enrichment & Ranking")
    logger.info("=" * 60)

    s2_client = get_semantic_scholar_client()

    try:
        enriched = await s2_client.enrich_papers_with_citations(papers)
    finally:
        await close_semantic_scholar()

    # Sort by citation count (descending), then by date (newest first)
    sorted_papers = sorted(
        enriched,
        key=lambda p: (p.citation_count, p.published_date or date.min),
        reverse=True,
    )

    # Select top N
    selected = sorted_papers[:target_count]

    logger.info(f"Stage 3 complete:")
    logger.info(f"  - Total papers: {len(enriched)}")
    logger.info(f"  - Selected top: {len(selected)}")

    if selected:
        top_citations = selected[0].citation_count
        min_citations = selected[-1].citation_count if len(selected) > 1 else 0
        logger.info(f"  - Citation range: {min_citations} - {top_citations}")

    return selected


async def download_papers(
    papers: list[Paper],
    download_pdf: bool = True,
    download_latex: bool = True,
) -> list[Paper]:
    """
    Download PDF and LaTeX files for papers.
    """
    logger.info("=" * 60)
    logger.info("Downloading Papers")
    logger.info("=" * 60)

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

    logger.info(f"Download complete:")
    logger.info(f"  - PDFs: {pdf_count}")
    logger.info(f"  - LaTeX: {latex_count}")

    return downloaded


async def save_to_database(papers: list[Paper]) -> int:
    """
    Save papers to Supabase database.
    """
    logger.info("=" * 60)
    logger.info("Saving to Database")
    logger.info("=" * 60)

    db = get_db_client()
    count = db.batch_insert_papers(papers)

    logger.info(f"Saved {count} papers to database")
    return count


async def run_collection_pipeline(
    max_results: int = 5000,
    target_count: int = 1000,
    start_date: date = None,
    end_date: date = None,
    dry_run: bool = False,
    skip_download: bool = False,
    max_verify: int = 500,
    use_windowing: bool = False,
    window_days: int = 14,
    resume: bool = False,
) -> CollectionStats:
    """
    Run the full collection pipeline.

    Args:
        max_results: Maximum papers to fetch from arXiv (per window if windowing enabled)
        target_count: Target number of papers to collect
        start_date: Start date filter (default: 2025-01-01)
        end_date: End date filter (default: today)
        dry_run: If True, don't download or save
        skip_download: If True, skip downloading files
        max_verify: Maximum papers to verify via LLM
        use_windowing: If True, use date windowing for large collections
        window_days: Size of date windows in days (default: 14)
        resume: If True, resume from last checkpoint

    Returns:
        Collection statistics
    """
    stats = CollectionStats()

    # Default date range
    if start_date is None:
        start_date = date(2025, 1, 1)
    if end_date is None:
        end_date = date.today()

    # Calculate window count for stats
    if use_windowing:
        windows = generate_date_windows(start_date, end_date, window_days)
        stats.windows_total = len(windows)

    logger.info("=" * 60)
    logger.info("arXiv RAG v1 - Collection Pipeline")
    logger.info("=" * 60)
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Max results: {max_results}")
    logger.info(f"Target count: {target_count}")
    logger.info(f"Dry run: {dry_run}")
    if use_windowing:
        logger.info(f"Windowing: {window_days} day windows ({stats.windows_total} total)")
        logger.info(f"Resume: {resume}")
    logger.info("")

    arxiv_client = ArxivClient()

    # Stage 1: Broad Recall
    if use_windowing:
        # Load checkpoint if resuming
        resume_state = load_checkpoint() if resume else None
        if resume_state:
            stats.windows_processed = len(resume_state.windows_completed)

        all_papers, collection_state = await stage1_broad_recall_windowed(
            arxiv_client,
            start_date=start_date,
            end_date=end_date,
            window_days=window_days,
            max_per_window=max_results,
            resume_state=resume_state,
        )
        stats.windows_processed = len(collection_state.windows_completed)
    else:
        all_papers = await stage1_broad_recall(
            arxiv_client,
            start_date=start_date,
            end_date=end_date,
            max_results=max_results,
        )

    stats.stage1_count = len(all_papers)
    stats.total_fetched = len(all_papers)

    if not all_papers:
        logger.warning("No papers found. Exiting.")
        return stats

    # Stage 2a: Rule-based filtering
    clearly_llm, edge_cases = await stage2a_rule_based_filter(
        arxiv_client,
        all_papers,
    )
    stats.stage2a_passed = len(clearly_llm)

    # Stage 2b: LLM verification (if needed and not dry run)
    verified_from_edges = []
    rejected = []
    if edge_cases and not dry_run:
        verified_from_edges, rejected = await stage2b_llm_verification(
            edge_cases,
            max_verify=max_verify,
        )
        stats.stage2b_verified = len(verified_from_edges)
        stats.stage2b_rejected = len(rejected)

        # Save Stage 2b results for analysis
        save_stage2b_results(verified_from_edges, rejected, edge_cases)

    # Combine verified papers
    all_verified = clearly_llm + verified_from_edges
    stats.total_filtered = len(all_verified)

    logger.info(f"Total verified papers: {len(all_verified)}")

    # Stage 3: Enrich with citations and rank
    selected = await stage3_enrich_and_rank(all_verified, target_count)

    if dry_run:
        logger.info("Dry run complete. No files downloaded or saved.")
        print("\n" + stats.summary())
        return stats

    # Download papers
    if not skip_download:
        selected = await download_papers(selected)
        stats.total_downloaded = sum(1 for p in selected if p.pdf_path)

    # Save to database
    await save_to_database(selected)

    print("\n" + stats.summary())
    return stats


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect LLM papers from arXiv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--max-results",
        type=int,
        default=5000,
        help="Maximum papers to fetch from arXiv (default: 5000)",
    )

    parser.add_argument(
        "--target",
        type=int,
        default=1000,
        help="Target number of papers to collect (default: 1000)",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default="2025-01-01",
        help="Start date (YYYY-MM-DD, default: 2025-01-01)",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD, default: today)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without downloading or saving",
    )

    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading PDF/LaTeX files",
    )

    parser.add_argument(
        "--max-verify",
        type=int,
        default=500,
        help="Maximum papers to verify via LLM (default: 500)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    # Windowing options
    parser.add_argument(
        "--window-days",
        type=int,
        default=14,
        help="Date window size in days (default: 14, enables windowing mode)",
    )

    parser.add_argument(
        "--use-windowing",
        action="store_true",
        help="Enable date windowing for large collections",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint (requires --use-windowing)",
    )

    parser.add_argument(
        "--clear-checkpoint",
        action="store_true",
        help="Clear existing checkpoint before starting",
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    import logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)

    # Parse dates
    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date) if args.end_date else date.today()

    # Handle checkpoint clearing
    if args.clear_checkpoint:
        clear_checkpoint()

    # Validate resume option
    if args.resume and not args.use_windowing:
        logger.warning("--resume requires --use-windowing, enabling windowing mode")
        args.use_windowing = True

    try:
        stats = await run_collection_pipeline(
            max_results=args.max_results,
            target_count=args.target,
            start_date=start_date,
            end_date=end_date,
            dry_run=args.dry_run,
            skip_download=args.skip_download,
            max_verify=args.max_verify,
            use_windowing=args.use_windowing,
            window_days=args.window_days,
            resume=args.resume,
        )

        # Clear checkpoint on successful completion
        if args.use_windowing and not args.dry_run:
            clear_checkpoint()

        # Exit with error if no papers collected
        if stats.total_fetched == 0:
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Collection interrupted by user")
        logger.info("Use --resume to continue from last checkpoint")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
