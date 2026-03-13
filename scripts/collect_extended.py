#!/usr/bin/env python3
"""
arXiv RAG v1 - Extended Data Collection (14 Months)

Collects papers over a 14-month date range with iterative NG keyword learning.

Pipeline:
1. Category + Positive Keyword filtering (arXiv API)
2. Rule-based filtering (Stage 2a)
3. NG Keyword auto-filtering (Stage 2b)
4. Gemini classification + NG keyword extraction (Stage 2c)
5. Auto-update NG keywords
6. Repeat for each month

Usage:
    python scripts/collect_extended.py                           # Start from beginning
    python scripts/collect_extended.py --resume                  # Resume from last checkpoint
    python scripts/collect_extended.py --month 2025-01           # Specific month
    python scripts/collect_extended.py --dry-run                 # Preview only
    python scripts/collect_extended.py --skip-gemini             # Skip Gemini classification
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.collection.arxiv_client import ArxivClient
from src.collection.ng_keywords import get_ng_keywords_manager, filter_by_ng_keywords
from src.collection.models import Paper
from src.storage import get_db_client
from src.utils.logging import get_logger

logger = get_logger("collect_extended")

# Date range for collection (14 months: Jan 2025 ~ Feb 2026)
START_DATE = date(2025, 1, 1)
END_DATE = date(2026, 2, 28)

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "collection"

# Gemini classification prompt
CLASSIFICATION_PROMPT = """
You are an expert in LLM/NLP research. Classify whether this paper is suitable for an LLM research dataset.

Paper Title: {title}
Abstract: {abstract}

Classification criteria:
SUITABLE papers focus on:
- Large Language Models (LLM), foundation models, pretraining
- Model architecture, transformer improvements
- Alignment, RLHF, safety, reasoning
- Multimodal models with language focus
- RAG, retrieval-augmented generation
- Prompt engineering, in-context learning
- Model compression, quantization, efficiency for LLMs

NOT SUITABLE papers focus on:
- Domain-specific applications (medical, biology, chemistry, robotics, climate, finance)
- Pure computer vision without language (image classification, object detection)
- Traditional ML without LLM connection
- Hardware, systems, networking
- Signal processing, time series without NLP
- Other non-LLM deep learning

Output format (JSON only, no markdown):
{{
  "suitable": true/false,
  "confidence": "high"/"medium"/"low",
  "reason": "Brief explanation",
  "ng_keywords": ["keyword1", "keyword2"] // Only if NOT suitable - extract 2-3 domain keywords
}}

Respond with JSON only, no other text.
"""


class CollectionState:
    """Tracks collection progress across months."""

    def __init__(self, filepath: Path = None):
        self.filepath = filepath or (OUTPUT_DIR / "collection_state.json")
        self._state: Optional[dict] = None

    @property
    def state(self) -> dict:
        if self._state is None:
            self._state = self._load()
        return self._state

    def _load(self) -> dict:
        if self.filepath.exists():
            with open(self.filepath, 'r') as f:
                return json.load(f)
        return {
            "completed_months": [],
            "current_month": None,
            "total_collected": 0,
            "total_filtered": 0,
            "total_ng_keywords_added": 0,
            "papers_by_month": {},
        }

    def save(self):
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self.filepath, 'w') as f:
            json.dump(self._state, f, indent=2)

    def mark_month_complete(self, month: str, stats: dict):
        self.state["completed_months"].append(month)
        self.state["papers_by_month"][month] = stats
        self.state["total_collected"] += stats.get("collected", 0)
        self.state["total_filtered"] += stats.get("filtered", 0)
        self.state["total_ng_keywords_added"] += stats.get("ng_keywords_added", 0)
        self.state["current_month"] = None
        self.save()

    def is_month_complete(self, month: str) -> bool:
        return month in self.state.get("completed_months", [])


def generate_months(start: date, end: date) -> list[tuple[str, date, date]]:
    """Generate list of (month_key, start_date, end_date) tuples."""
    months = []
    current = start.replace(day=1)

    while current <= end:
        # Get last day of month
        if current.month == 12:
            next_month = current.replace(year=current.year + 1, month=1)
        else:
            next_month = current.replace(month=current.month + 1)
        last_day = next_month - timedelta(days=1)

        month_key = current.strftime("%Y-%m")
        months.append((month_key, current, min(last_day, end)))

        current = next_month

    return months


async def classify_paper_gemini(
    title: str,
    abstract: str,
    api_key: str,
) -> Optional[dict]:
    """
    Classify a single paper using Gemini.

    Returns:
        Classification result dict or None if failed
    """
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")

        prompt = CLASSIFICATION_PROMPT.format(
            title=title,
            abstract=abstract[:2000],  # Truncate long abstracts
        )

        response = model.generate_content(prompt)
        text = response.text.strip()

        # Parse JSON response
        # Handle potential markdown code blocks
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        result = json.loads(text)
        return result

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse Gemini response: {e}")
        return None
    except Exception as e:
        logger.warning(f"Gemini classification failed: {e}")
        return None


async def classify_edge_cases(
    papers: list,
    api_key: str,
    batch_size: int = 10,
) -> tuple[list, list, set]:
    """
    Classify edge case papers using Gemini.

    Args:
        papers: List of Paper objects
        api_key: Gemini API key
        batch_size: Concurrent requests

    Returns:
        Tuple of (suitable_papers, unsuitable_papers, new_ng_keywords)
    """
    suitable = []
    unsuitable = []
    new_ng_keywords = set()

    total = len(papers)

    # Process in batches with rate limiting
    for i in range(0, total, batch_size):
        batch = papers[i:i + batch_size]

        tasks = [
            classify_paper_gemini(
                p.title if hasattr(p, 'title') else p.get('title', ''),
                p.abstract if hasattr(p, 'abstract') else p.get('abstract', ''),
                api_key,
            )
            for p in batch
        ]

        results = await asyncio.gather(*tasks)

        for paper, result in zip(batch, results):
            if result is None:
                # Classification failed - be conservative, add to suitable
                suitable.append(paper)
                continue

            if result.get("suitable", False):
                suitable.append(paper)
            else:
                unsuitable.append(paper)

                # Extract NG keywords
                ng_kws = result.get("ng_keywords", [])
                for kw in ng_kws:
                    if kw and len(kw) > 2:
                        new_ng_keywords.add(kw.lower().strip())

        # Progress
        processed = min(i + batch_size, total)
        if processed % 20 == 0 or processed == total:
            logger.info(f"Gemini classification: {processed}/{total}")

        # Rate limiting
        await asyncio.sleep(1)

    return suitable, unsuitable, new_ng_keywords


async def collect_month(
    month_key: str,
    start_date: date,
    end_date: date,
    dry_run: bool = False,
    skip_gemini: bool = False,
) -> dict:
    """
    Collect papers for a single month.

    Returns:
        Stats dict with collected, filtered, edge_cases counts
    """
    stats = {
        "month": month_key,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "stage1_count": 0,  # From arXiv API
        "stage2a_clearly_llm": 0,
        "stage2a_filtered": 0,
        "stage2a_edge_cases": 0,
        "stage2b_ng_filtered": 0,
        "stage2b_edge_cases": 0,
        "stage2c_gemini_suitable": 0,
        "stage2c_gemini_filtered": 0,
        "ng_keywords_added": 0,
        "collected": 0,
        "filtered": 0,
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"Collecting: {month_key} ({start_date} to {end_date})")
    logger.info(f"{'='*60}")

    # Stage 1: arXiv API search
    client = ArxivClient()
    all_papers = await client.search_paginated(
        start_date=start_date,
        end_date=end_date,
        max_results=10000,  # arXiv limit per window
    )

    stats["stage1_count"] = len(all_papers)
    logger.info(f"Stage 1 (arXiv API): {len(all_papers)} papers")

    if dry_run:
        logger.info("DRY RUN - stopping here")
        return stats

    # Stage 2a: Rule-based filtering
    clearly_llm, edge_cases = client.filter_stage2a(all_papers)
    stats["stage2a_clearly_llm"] = len(clearly_llm)
    stats["stage2a_edge_cases"] = len(edge_cases)
    stats["stage2a_filtered"] = len(all_papers) - len(clearly_llm) - len(edge_cases)

    logger.info(f"Stage 2a (Rule-based): {len(clearly_llm)} clearly LLM, {len(edge_cases)} edge cases")

    # Stage 2b: NG keyword filtering (on edge cases)
    ng_filtered, remaining_edge_cases = filter_by_ng_keywords(edge_cases)
    stats["stage2b_ng_filtered"] = len(ng_filtered)
    stats["stage2b_edge_cases"] = len(remaining_edge_cases)

    logger.info(f"Stage 2b (NG keywords): {len(ng_filtered)} filtered, {len(remaining_edge_cases)} edge cases remain")

    # Stage 2c: Gemini classification (on remaining edge cases)
    gemini_suitable = []
    gemini_filtered = []
    new_ng_keywords = set()

    if not skip_gemini and remaining_edge_cases:
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            logger.info(f"Stage 2c (Gemini): Classifying {len(remaining_edge_cases)} edge cases...")

            gemini_suitable, gemini_filtered, new_ng_keywords = await classify_edge_cases(
                remaining_edge_cases,
                api_key,
                batch_size=10,
            )

            stats["stage2c_gemini_suitable"] = len(gemini_suitable)
            stats["stage2c_gemini_filtered"] = len(gemini_filtered)

            logger.info(f"Stage 2c (Gemini): {len(gemini_suitable)} suitable, {len(gemini_filtered)} filtered")

            # Auto-update NG keywords
            if new_ng_keywords:
                ng_manager = get_ng_keywords_manager()
                added = ng_manager.add_keywords(
                    list(new_ng_keywords),
                    category="gemini_extracted",
                    reason=f"Auto-extracted from {month_key} edge cases"
                )
                ng_manager.save()
                stats["ng_keywords_added"] = added
                logger.info(f"Added {added} new NG keywords: {list(new_ng_keywords)[:5]}...")
        else:
            logger.warning("GEMINI_API_KEY not set - skipping Gemini classification")
            gemini_suitable = remaining_edge_cases  # Keep all if can't classify
    elif skip_gemini:
        logger.info("Stage 2c (Gemini): Skipped by user request")
        gemini_suitable = remaining_edge_cases  # Keep all edge cases
    else:
        logger.info("Stage 2c (Gemini): No edge cases to classify")

    # Combine all suitable papers
    final_papers = clearly_llm + gemini_suitable
    stats["collected"] = len(final_papers)
    stats["filtered"] = len(all_papers) - len(final_papers)

    logger.info(f"Final: {len(final_papers)} papers collected, {stats['filtered']} filtered")

    # Save to database
    if final_papers:
        db_client = get_db_client()
        inserted = db_client.batch_insert_papers(final_papers)
        logger.info(f"Inserted {inserted} papers to database")

    # Save edge cases for human review (only Gemini-filtered ones)
    if gemini_filtered:
        edge_cases_file = OUTPUT_DIR / f"gemini_filtered_{month_key}.json"
        edge_cases_file.parent.mkdir(parents=True, exist_ok=True)

        edge_cases_data = [
            {
                "arxiv_id": p.arxiv_id if hasattr(p, 'arxiv_id') else p.get('arxiv_id'),
                "title": p.title if hasattr(p, 'title') else p.get('title'),
                "abstract": (p.abstract if hasattr(p, 'abstract') else p.get('abstract', ''))[:500],
                "categories": p.categories if hasattr(p, 'categories') else p.get('categories'),
            }
            for p in gemini_filtered[:100]  # Limit for review
        ]

        with open(edge_cases_file, 'w') as f:
            json.dump(edge_cases_data, f, indent=2)

        logger.info(f"Saved {len(edge_cases_data)} Gemini-filtered papers for review: {edge_cases_file}")

    # Save new NG keywords for review
    if new_ng_keywords:
        ng_file = OUTPUT_DIR / f"new_ng_keywords_{month_key}.json"
        with open(ng_file, 'w') as f:
            json.dump({
                "month": month_key,
                "keywords": sorted(new_ng_keywords),
                "count": len(new_ng_keywords),
                "status": "auto_added",
            }, f, indent=2)
        logger.info(f"Saved {len(new_ng_keywords)} new NG keywords to {ng_file}")

    return stats


async def collect_all(
    resume: bool = False,
    specific_month: str = None,
    dry_run: bool = False,
    skip_gemini: bool = False,
):
    """
    Collect papers for all months.

    Args:
        resume: Resume from last checkpoint
        specific_month: Collect only this month (format: YYYY-MM)
        dry_run: Preview without writing
        skip_gemini: Skip Gemini classification stage
    """
    state = CollectionState()
    months = generate_months(START_DATE, END_DATE)

    logger.info(f"Collection range: {START_DATE} to {END_DATE}")
    logger.info(f"Total months: {len(months)}")

    if specific_month:
        months = [(m, s, e) for m, s, e in months if m == specific_month]
        if not months:
            logger.error(f"Month not found in range: {specific_month}")
            return

    for month_key, start_date, end_date in months:
        # Skip completed months if resuming
        if resume and state.is_month_complete(month_key):
            logger.info(f"Skipping completed month: {month_key}")
            continue

        stats = await collect_month(
            month_key, start_date, end_date,
            dry_run=dry_run,
            skip_gemini=skip_gemini
        )

        if not dry_run:
            state.mark_month_complete(month_key, stats)

        # Progress report
        if not dry_run:
            logger.info(f"\nProgress: {len(state.state['completed_months'])}/{len(months)} months")
            logger.info(f"Total collected: {state.state['total_collected']}")
            logger.info(f"Total filtered: {state.state['total_filtered']}")
            logger.info(f"Total NG keywords added: {state.state.get('total_ng_keywords_added', 0)}")

    # Final summary
    print("\n" + "=" * 60)
    print("COLLECTION SUMMARY")
    print("=" * 60)
    print(f"Months processed: {len(state.state.get('completed_months', []))}")
    print(f"Total papers collected: {state.state.get('total_collected', 0)}")
    print(f"Total papers filtered: {state.state.get('total_filtered', 0)}")
    print(f"Total NG keywords added: {state.state.get('total_ng_keywords_added', 0)}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Extended data collection over 14 months with NG keyword learning"
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--month", "-m",
        type=str,
        help="Collect specific month only (format: YYYY-MM)"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Preview without writing to database"
    )
    parser.add_argument(
        "--skip-gemini",
        action="store_true",
        help="Skip Gemini classification (use only rule-based + NG keywords)"
    )
    parser.add_argument(
        "--show-state",
        action="store_true",
        help="Show current collection state and exit"
    )

    args = parser.parse_args()

    if args.show_state:
        state = CollectionState()
        print(json.dumps(state.state, indent=2))
        return 0

    # Run collection
    asyncio.run(collect_all(
        resume=args.resume,
        specific_month=args.month,
        dry_run=args.dry_run,
        skip_gemini=args.skip_gemini,
    ))

    return 0


if __name__ == "__main__":
    exit(main())
