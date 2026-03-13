#!/usr/bin/env python3
"""
arXiv RAG v1 - Chunk Semantic Enrichment Script

Adds LLM-generated semantic metadata to chunks for improved conceptual query retrieval.

Enrichment includes:
- semantic_summary: 1-sentence summary of what the chunk explains
- conceptual_keywords: Abstract keywords (not technical terms) for conceptual matching
- contribution_type: method | evaluation | analysis | background | other

Usage:
    python scripts/09_enrich_chunks.py --limit 100 --batch-size 10
    python scripts/09_enrich_chunks.py --paper-id 2501.12345v1
    python scripts/09_enrich_chunks.py --dry-run --limit 5
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

from src.storage.qdrant_client import get_qdrant_client
from src.utils.config import settings
from src.utils.logging import get_logger

logger = get_logger("enrich_chunks")


# =============================================================================
# Enrichment Prompt
# =============================================================================

ENRICHMENT_PROMPT = """Analyze this chunk from a machine learning research paper and provide semantic metadata.

Chunk content:
---
{chunk_content}
---

Output JSON with these fields:
1. semantic_summary: A single sentence describing what concept or idea this chunk explains (avoid technical jargon)
2. conceptual_keywords: 3-5 abstract/conceptual keywords that describe the chunk WITHOUT using specific model names, acronyms, or technical terms. Focus on what the concept DOES, not what it's called.
3. contribution_type: One of [method, evaluation, analysis, background, implementation, other]

Examples of good conceptual_keywords:
- Instead of "BERT attention": use "text understanding", "word relationships"
- Instead of "RLHF": use "learning from feedback", "human preference alignment"
- Instead of "transformer architecture": use "sequence processing", "parallel computation"

Output only valid JSON (no markdown):
{{"semantic_summary": "...", "conceptual_keywords": ["...", "...", "..."], "contribution_type": "..."}}"""


VALID_CONTRIBUTION_TYPES = {"method", "evaluation", "analysis", "background", "implementation", "other"}


# =============================================================================
# Enrichment Result
# =============================================================================

class EnrichmentResult:
    """Result of enriching a single chunk."""

    def __init__(
        self,
        chunk_id: str,
        success: bool,
        semantic_summary: str = None,
        conceptual_keywords: list[str] = None,
        contribution_type: str = None,
        error: str = None,
        latency_ms: float = 0.0,
    ):
        self.chunk_id = chunk_id
        self.success = success
        self.semantic_summary = semantic_summary
        self.conceptual_keywords = conceptual_keywords or []
        self.contribution_type = contribution_type
        self.error = error
        self.latency_ms = latency_ms

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "semantic_summary": self.semantic_summary,
            "conceptual_keywords": self.conceptual_keywords,
            "contribution_type": self.contribution_type,
        }


# =============================================================================
# Chunk Enricher
# =============================================================================

class ChunkEnricher:
    """
    Enriches chunks with semantic metadata using LLM.

    Adds conceptual descriptions and keywords to help with
    non-technical/paraphrased query matching.
    """

    def __init__(self, model_name: str = None, dry_run: bool = False):
        """
        Initialize enricher.

        Args:
            model_name: Gemini model name (default from config)
            dry_run: If True, don't update database
        """
        self.model_name = model_name or settings.gemini_model
        self.dry_run = dry_run
        self._model = None
        self._qdrant = None

        # Stats
        self.stats = {
            "processed": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "total_latency_ms": 0.0,
        }

        logger.info(f"Initialized ChunkEnricher: model={self.model_name}, dry_run={dry_run}")

    @property
    def model(self):
        """Lazy load Gemini model."""
        if self._model is None:
            api_key = settings.gemini_api_key
            if not api_key:
                raise ValueError("GEMINI_API_KEY not set")

            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(self.model_name)
            logger.info(f"Loaded Gemini model: {self.model_name}")

        return self._model

    @property
    def qdrant(self):
        """Lazy load Qdrant client."""
        if self._qdrant is None:
            self._qdrant = get_qdrant_client()
            if not self._qdrant.health_check():
                raise ConnectionError("Could not connect to Qdrant")
            logger.info("Connected to Qdrant")

        return self._qdrant

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def enrich_chunk(self, chunk_id: str, content: str) -> EnrichmentResult:
        """
        Enrich a single chunk with semantic metadata.

        Args:
            chunk_id: Chunk identifier
            content: Chunk text content

        Returns:
            EnrichmentResult with metadata or error
        """
        start = time.time()

        # Skip very short chunks
        if len(content.strip()) < 50:
            return EnrichmentResult(
                chunk_id=chunk_id,
                success=False,
                error="Content too short",
            )

        # Truncate very long content
        content_truncated = content[:3000] if len(content) > 3000 else content

        try:
            prompt = ENRICHMENT_PROMPT.format(chunk_content=content_truncated)
            response = self.model.generate_content(prompt)
            text = response.text.strip()

            # Clean up response
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()

            # Parse JSON
            data = json.loads(text)

            # Validate and extract fields
            semantic_summary = data.get("semantic_summary", "").strip()
            conceptual_keywords = data.get("conceptual_keywords", [])
            contribution_type = data.get("contribution_type", "other").lower()

            # Validate contribution type
            if contribution_type not in VALID_CONTRIBUTION_TYPES:
                contribution_type = "other"

            # Validate keywords
            if isinstance(conceptual_keywords, str):
                conceptual_keywords = [k.strip() for k in conceptual_keywords.split(",")]
            conceptual_keywords = [k.strip() for k in conceptual_keywords if k.strip()][:5]

            latency_ms = (time.time() - start) * 1000

            return EnrichmentResult(
                chunk_id=chunk_id,
                success=True,
                semantic_summary=semantic_summary,
                conceptual_keywords=conceptual_keywords,
                contribution_type=contribution_type,
                latency_ms=latency_ms,
            )

        except json.JSONDecodeError as e:
            latency_ms = (time.time() - start) * 1000
            logger.warning(f"JSON parse error for {chunk_id}: {e}")
            return EnrichmentResult(
                chunk_id=chunk_id,
                success=False,
                error=f"JSON parse error: {e}",
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            logger.error(f"Enrichment failed for {chunk_id}: {e}")
            return EnrichmentResult(
                chunk_id=chunk_id,
                success=False,
                error=str(e),
                latency_ms=latency_ms,
            )

    def update_chunk_metadata(self, chunk_id: str, enrichment: EnrichmentResult) -> bool:
        """
        Update chunk metadata in Qdrant.

        Args:
            chunk_id: Chunk identifier
            enrichment: Enrichment result with metadata

        Returns:
            True if update succeeded
        """
        if self.dry_run:
            logger.debug(f"[DRY RUN] Would update {chunk_id}: {enrichment.to_dict()}")
            return True

        try:
            # Update payload in Qdrant
            self.qdrant.update_payload(
                chunk_id=chunk_id,
                payload={
                    "semantic_summary": enrichment.semantic_summary,
                    "conceptual_keywords": enrichment.conceptual_keywords,
                    "contribution_type": enrichment.contribution_type,
                    "enriched": True,
                },
            )
            return True

        except Exception as e:
            logger.error(f"Failed to update {chunk_id}: {e}")
            return False

    def get_unenriched_chunks(
        self,
        limit: int = 100,
        paper_id: str = None,
    ) -> list[dict]:
        """
        Get chunks that haven't been enriched yet.

        Args:
            limit: Maximum chunks to return
            paper_id: Filter by specific paper

        Returns:
            List of chunk dicts with chunk_id and content
        """
        try:
            # Query Qdrant for chunks without enrichment
            filter_conditions = [
                {"key": "enriched", "match": {"value": False}},
            ]

            if paper_id:
                filter_conditions.append({
                    "key": "paper_id",
                    "match": {"value": paper_id},
                })

            # Use scroll to get chunks
            chunks = self.qdrant.scroll_chunks(
                limit=limit,
                filter_conditions=filter_conditions,
                with_payload=True,
            )

            return chunks

        except Exception as e:
            logger.error(f"Failed to get unenriched chunks: {e}")

            # Fallback: get all chunks and filter
            logger.info("Falling back to unfiltered chunk retrieval...")
            chunks = self.qdrant.scroll_chunks(
                limit=limit,
                with_payload=True,
            )

            # Filter out already enriched
            unenriched = [
                c for c in chunks
                if not c.get("payload", {}).get("enriched", False)
            ]

            return unenriched[:limit]

    def process_chunks(
        self,
        chunks: list[dict],
        batch_size: int = 10,
        delay: float = 0.5,
    ) -> list[EnrichmentResult]:
        """
        Process multiple chunks with rate limiting.

        Args:
            chunks: List of chunk dicts
            batch_size: Chunks per batch for progress logging
            delay: Delay between batches in seconds

        Returns:
            List of enrichment results
        """
        results = []
        total = len(chunks)

        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            content = chunk.get("content") or chunk.get("payload", {}).get("content", "")

            if not chunk_id or not content:
                self.stats["skipped"] += 1
                continue

            # Check if already enriched
            if chunk.get("payload", {}).get("enriched"):
                self.stats["skipped"] += 1
                continue

            # Enrich
            result = self.enrich_chunk(chunk_id, content)
            self.stats["processed"] += 1
            self.stats["total_latency_ms"] += result.latency_ms

            if result.success:
                # Update database
                if self.update_chunk_metadata(chunk_id, result):
                    self.stats["success"] += 1
                    logger.info(f"[{i+1}/{total}] {chunk_id}: {result.semantic_summary[:50]}...")
                else:
                    self.stats["failed"] += 1
            else:
                self.stats["failed"] += 1
                logger.warning(f"[{i+1}/{total}] {chunk_id}: FAILED - {result.error}")

            results.append(result)

            # Rate limiting
            if (i + 1) % batch_size == 0:
                avg_latency = self.stats["total_latency_ms"] / max(self.stats["processed"], 1)
                logger.info(f"Progress: {i+1}/{total}, success={self.stats['success']}, "
                           f"avg_latency={avg_latency:.0f}ms")
                time.sleep(delay)

        return results

    def get_stats(self) -> dict:
        """Get processing statistics."""
        return {
            **self.stats,
            "avg_latency_ms": (
                self.stats["total_latency_ms"] / max(self.stats["processed"], 1)
            ),
        }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Enrich chunks with semantic metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum chunks to process (default: 100)",
    )
    parser.add_argument(
        "--paper-id",
        type=str,
        help="Process only chunks from specific paper",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for progress logging (default: 10)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between batches in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Gemini model name (default: from config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't update database, just show what would be done",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save enrichment results to JSON file",
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("Chunk Semantic Enrichment")
    print(f"{'='*60}")
    print(f"Limit: {args.limit}")
    print(f"Paper ID: {args.paper_id or 'all'}")
    print(f"Dry run: {args.dry_run}")
    print()

    # Initialize enricher
    enricher = ChunkEnricher(
        model_name=args.model,
        dry_run=args.dry_run,
    )

    # Get chunks to process
    logger.info("Fetching unenriched chunks...")
    chunks = enricher.get_unenriched_chunks(
        limit=args.limit,
        paper_id=args.paper_id,
    )

    print(f"Found {len(chunks)} unenriched chunks")

    if not chunks:
        print("No chunks to process")
        return

    # Process chunks
    start_time = time.time()
    results = enricher.process_chunks(
        chunks,
        batch_size=args.batch_size,
        delay=args.delay,
    )
    elapsed = time.time() - start_time

    # Print stats
    stats = enricher.get_stats()

    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    print(f"Processed: {stats['processed']}")
    print(f"Success: {stats['success']}")
    print(f"Failed: {stats['failed']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Avg latency: {stats['avg_latency_ms']:.0f}ms")
    print(f"Total time: {elapsed:.1f}s")

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "stats": stats,
            "results": [
                {
                    "chunk_id": r.chunk_id,
                    "success": r.success,
                    **r.to_dict(),
                    "error": r.error,
                }
                for r in results
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
