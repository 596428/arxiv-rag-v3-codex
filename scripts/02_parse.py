#!/usr/bin/env python3
"""
arXiv RAG v1 - Document Parsing Pipeline

Parse downloaded papers using LaTeX (priority) or Marker (fallback).

Usage:
    python scripts/02_parse.py                    # Parse all pending papers
    python scripts/02_parse.py --limit 100        # Parse first 100 papers
    python scripts/02_parse.py --arxiv-id 2501.12345  # Parse specific paper
    python scripts/02_parse.py --latex-only       # Only use LaTeX parser
    python scripts/02_parse.py --marker-only      # Only use Marker parser
    python scripts/02_parse.py --with-equations   # Generate equation descriptions
    python scripts/02_parse.py --dry-run          # Preview without saving
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import get_settings
from src.utils.logging import setup_logging, ProgressLogger
from src.storage import get_db_client
from src.collection.models import PaperStatus, ParseMethod
from src.parsing import (
    LatexParser,
    LatexParseError,
    MarkerParser,
    MarkerParseError,
    ParsedDocument,
    ParsingStats,
    filter_document,
    check_document_quality,
    get_equation_processor,
    get_figure_processor,
)

logger = logging.getLogger(__name__)


class ParsingPipeline:
    """
    Document parsing pipeline.

    Strategy:
    1. Try LaTeX parsing first (higher quality)
    2. Fall back to Marker PDF parsing if LaTeX fails
    3. Apply quality checks
    4. Optionally generate equation descriptions
    """

    def __init__(
        self,
        settings=None,
        latex_only: bool = False,
        marker_only: bool = False,
        with_equations: bool = False,
        max_equations_per_paper: int = 20,
    ):
        self.settings = settings or get_settings()
        self.settings.ensure_directories()

        self.latex_only = latex_only
        self.marker_only = marker_only
        self.with_equations = with_equations
        self.max_equations_per_paper = max_equations_per_paper

        # Initialize parsers
        self.latex_parser = LatexParser(figures_dir=self.settings.figures_dir)
        self.marker_parser = None  # Lazy load (heavy GPU models)

        # Initialize processors
        self.equation_processor = None
        if with_equations:
            self.equation_processor = get_equation_processor()

        self.figure_processor = get_figure_processor(self.settings.figures_dir)

        # Statistics
        self.stats = ParsingStats()

    def _get_marker_parser(self) -> MarkerParser:
        """Lazy load Marker parser."""
        if self.marker_parser is None:
            logger.info("Loading Marker models (this may take a moment)...")
            self.marker_parser = MarkerParser(
                figures_dir=self.settings.figures_dir,
                device="cuda",
            )
        return self.marker_parser

    def parse_paper(
        self,
        arxiv_id: str,
        latex_path: Path = None,
        pdf_path: Path = None,
    ) -> tuple[ParsedDocument | None, str | None]:
        """
        Parse a single paper.

        Args:
            arxiv_id: Paper ID
            latex_path: Path to LaTeX archive
            pdf_path: Path to PDF file

        Returns:
            (ParsedDocument, None) on success
            (None, error_message) on failure
        """
        doc = None
        error = None
        method_used = None

        # Strategy 1: Try LaTeX first (unless marker_only)
        if latex_path and latex_path.exists() and not self.marker_only:
            try:
                logger.debug(f"Trying LaTeX parser for {arxiv_id}")
                doc = self.latex_parser.parse_archive(latex_path, arxiv_id)
                method_used = ParseMethod.LATEX
                self.stats.latex_success += 1
                logger.info(f"LaTeX parsing successful: {arxiv_id}")
            except LatexParseError as e:
                logger.debug(f"LaTeX parsing failed for {arxiv_id}: {e}")
                self.stats.latex_failed += 1
                error = str(e)

        # Strategy 2: Fall back to Marker (unless latex_only)
        if doc is None and pdf_path and pdf_path.exists() and not self.latex_only:
            try:
                logger.debug(f"Trying Marker parser for {arxiv_id}")
                parser = self._get_marker_parser()
                doc = parser.parse_pdf(pdf_path, arxiv_id)
                method_used = ParseMethod.MARKER
                self.stats.marker_success += 1
                logger.info(f"Marker parsing successful: {arxiv_id}")
                error = None  # Clear previous error
            except MarkerParseError as e:
                logger.warning(f"Marker parsing failed for {arxiv_id}: {e}")
                self.stats.marker_failed += 1
                error = str(e)

        if doc is None:
            return None, error or "No parser available"

        # Post-processing
        doc = self._post_process(doc)

        return doc, None

    def _post_process(self, doc: ParsedDocument) -> ParsedDocument:
        """Apply post-processing to parsed document."""

        # 1. Filter excluded sections (references, acknowledgments, etc.)
        doc = filter_document(doc)

        # 2. Quality check
        report = check_document_quality(doc)
        if not report.passed:
            doc.has_quality_issues = True
            doc.quality_issues = [
                f"{i.severity}: {i.issue_type} - {i.description}"
                for i in report.issues
            ]
            self.stats.quality_issues_count += 1

        # 3. Update counts
        doc.update_counts()
        self.stats.total_sections += doc.total_sections
        self.stats.total_equations += doc.total_equations
        self.stats.total_figures += doc.total_figures
        self.stats.total_tables += doc.total_tables

        # 4. Generate equation descriptions (if enabled)
        if self.equation_processor and doc.equations:
            logger.debug(f"Generating descriptions for {len(doc.equations)} equations")
            doc = self.equation_processor.process_document(
                doc, max_equations=self.max_equations_per_paper
            )

        # 5. Process figures
        doc = self.figure_processor.process_document_figures(doc)

        return doc

    def save_parsed_document(self, doc: ParsedDocument) -> Path:
        """Save parsed document to JSON file."""
        output_path = self.settings.parsed_dir / f"{doc.arxiv_id.replace('/', '_')}.json"
        doc.to_json_file(str(output_path))
        return output_path

    def run(
        self,
        papers: list[dict],
        dry_run: bool = False,
        progress_callback=None,
    ) -> ParsingStats:
        """
        Run parsing pipeline on a list of papers.

        Args:
            papers: List of paper dicts with arxiv_id, pdf_path, latex_path
            dry_run: If True, don't save results
            progress_callback: Optional callback(current, total)

        Returns:
            ParsingStats
        """
        self.stats = ParsingStats()
        self.stats.total_papers = len(papers)

        logger.info(f"Starting parsing pipeline for {len(papers)} papers")
        start_time = time.time()

        for i, paper in enumerate(papers):
            arxiv_id = paper.get("arxiv_id")
            pdf_path = Path(paper.get("pdf_path", "")) if paper.get("pdf_path") else None
            latex_path = Path(paper.get("latex_path", "")) if paper.get("latex_path") else None

            # Parse
            doc, error = self.parse_paper(arxiv_id, latex_path, pdf_path)

            if doc:
                if not dry_run:
                    # Save to file
                    output_path = self.save_parsed_document(doc)
                    logger.debug(f"Saved: {output_path}")
            else:
                logger.warning(f"Failed to parse {arxiv_id}: {error}")

            if progress_callback:
                progress_callback(i + 1, len(papers))

        elapsed = time.time() - start_time
        logger.info(f"Parsing complete in {elapsed:.1f}s")
        logger.info(self.stats.summary())

        return self.stats


def get_papers_to_parse(
    supabase_client,
    limit: int = None,
    arxiv_id: str = None,
) -> list[dict]:
    """Get papers that need parsing from database."""

    if arxiv_id:
        # Get specific paper
        paper = supabase_client.get_paper(arxiv_id)
        return [paper] if paper else []

    return supabase_client.get_papers(
        fields=["arxiv_id", "pdf_path", "latex_path", "citation_count"],
        limit=limit,
        status="pending",
        order_by="citation_count",
    )


def update_paper_status(
    supabase_client,
    arxiv_id: str,
    status: str,
    parse_method: str = None,
):
    """Update paper status in database."""
    update_data = {"parse_status": status}
    if parse_method:
        update_data["parse_method"] = parse_method

    supabase_client.update_paper(arxiv_id, update_data)


def main():
    parser = argparse.ArgumentParser(description="Parse downloaded papers")
    parser.add_argument("--limit", type=int, help="Maximum papers to parse")
    parser.add_argument("--arxiv-id", type=str, help="Parse specific paper")
    parser.add_argument("--latex-only", action="store_true", help="Only use LaTeX parser")
    parser.add_argument("--marker-only", action="store_true", help="Only use Marker parser")
    parser.add_argument("--with-equations", action="store_true", help="Generate equation descriptions")
    parser.add_argument("--max-equations", type=int, default=20, help="Max equations per paper")
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving")
    parser.add_argument("--skip-db-update", action="store_true", help="Don't update database status")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)

    settings = get_settings()
    settings.ensure_directories()

    # Get papers to parse
    if args.arxiv_id:
        # Parse specific paper - look for files locally
        papers = [{
            "arxiv_id": args.arxiv_id,
            "pdf_path": str(settings.pdf_dir / f"{args.arxiv_id.replace('/', '_')}.pdf"),
            "latex_path": str(settings.latex_dir / f"{args.arxiv_id.replace('/', '_')}.tar.gz"),
        }]
    else:
        # Get from database
        supabase = get_db_client()
        papers = get_papers_to_parse(supabase, limit=args.limit, arxiv_id=args.arxiv_id)

        if not papers:
            logger.info("No papers to parse")
            return

    logger.info(f"Found {len(papers)} papers to parse")

    # Create pipeline
    pipeline = ParsingPipeline(
        settings=settings,
        latex_only=args.latex_only,
        marker_only=args.marker_only,
        with_equations=args.with_equations,
        max_equations_per_paper=args.max_equations,
    )

    # Progress tracking
    progress = ProgressLogger(name="Parsing", total=len(papers), log_every=10)

    def on_progress(current, total):
        progress.update(current)

    # Run pipeline
    stats = pipeline.run(papers, dry_run=args.dry_run, progress_callback=on_progress)

    # Update database status
    if not args.dry_run and not args.skip_db_update and not args.arxiv_id:
        supabase = get_db_client()
        success_count = stats.latex_success + stats.marker_success
        logger.info(f"Updating database status for {success_count} papers")

        # Note: In production, batch this update
        for paper in papers[:success_count]:
            arxiv_id = paper["arxiv_id"]
            # Check which method was used (simplified)
            parsed_file = settings.parsed_dir / f"{arxiv_id.replace('/', '_')}.json"
            if parsed_file.exists():
                update_paper_status(supabase, arxiv_id, "parsed")

    # Print summary
    print("\n" + "=" * 50)
    print(stats.summary())
    print("=" * 50)


if __name__ == "__main__":
    main()
