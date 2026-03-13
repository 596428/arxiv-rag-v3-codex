#!/usr/bin/env python3
"""
arXiv RAG v2 - Embedding Pipeline

Chunks parsed documents and generates embeddings using BGE-M3.
Optionally adds OpenAI embeddings for comparison.

v2 Architecture:
- Supabase: metadata only (no vectors)
- Qdrant: vector storage (dense, sparse, colbert)

Usage:
    python scripts/03_embed.py                     # Process all parsed papers
    python scripts/03_embed.py --arxiv-id 2501.12948v2  # Single paper
    python scripts/03_embed.py --with-openai       # Include OpenAI embeddings
    python scripts/03_embed.py --dry-run           # Preview only
    python scripts/03_embed.py --limit 10          # Process first N papers
    python scripts/03_embed.py --supabase-only     # Skip Qdrant (metadata only)
"""

import argparse
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedding import (
    HybridChunker,
    BGEEmbedder,
    OpenAIEmbedder,
    ChunkingConfig,
    EmbeddingConfig,
    ChunkingStats,
    EmbeddingStats,
)
from src.parsing.models import ParsedDocument
from src.storage import get_supabase_client, get_qdrant_client
from src.utils.config import settings
from src.utils.logging import setup_logging, get_logger

logger = get_logger("embed_pipeline")


def load_parsed_documents(
    parsed_dir: Path,
    arxiv_id: str = None,
    limit: int = None,
) -> list[ParsedDocument]:
    """
    Load parsed documents from JSON files.

    Args:
        parsed_dir: Directory containing parsed JSON files
        arxiv_id: Optional specific paper to load
        limit: Maximum number of papers to load

    Returns:
        List of ParsedDocument objects
    """
    documents = []

    if arxiv_id:
        # Load specific paper
        json_path = parsed_dir / f"{arxiv_id}.json"
        if json_path.exists():
            try:
                doc = ParsedDocument.from_json_file(str(json_path))
                documents.append(doc)
                logger.info(f"Loaded: {arxiv_id}")
            except Exception as e:
                logger.error(f"Failed to load {arxiv_id}: {e}")
        else:
            logger.error(f"Parsed file not found: {json_path}")
    else:
        # Load all parsed files
        json_files = sorted(parsed_dir.glob("*.json"))

        if limit:
            json_files = json_files[:limit]

        for json_path in json_files:
            if json_path.name == "CLAUDE.md":
                continue

            try:
                doc = ParsedDocument.from_json_file(str(json_path))
                documents.append(doc)
                logger.debug(f"Loaded: {doc.arxiv_id}")
            except Exception as e:
                logger.error(f"Failed to load {json_path.name}: {e}")

    logger.info(f"Loaded {len(documents)} parsed documents")
    return documents


def run_embedding_pipeline(
    documents: list[ParsedDocument],
    chunking_config: ChunkingConfig,
    embedding_config: EmbeddingConfig,
    with_openai: bool = False,
    dry_run: bool = False,
    save_to_db: bool = True,
    supabase_only: bool = False,
) -> tuple[ChunkingStats, EmbeddingStats]:
    """
    Run the full embedding pipeline (v2 architecture).

    v2 Storage Strategy:
    - Supabase: metadata only (chunk_id, paper_id, content, section_title, etc.)
    - Qdrant: vectors only (dense_bge, sparse, dense_openai)

    Args:
        documents: List of parsed documents
        chunking_config: Chunking configuration
        embedding_config: Embedding configuration
        with_openai: Whether to add OpenAI embeddings
        dry_run: Preview only, don't save
        save_to_db: Whether to save to databases
        supabase_only: Skip Qdrant, save metadata to Supabase only

    Returns:
        Tuple of (chunking stats, embedding stats)
    """
    # Initialize components
    chunker = HybridChunker(chunking_config)
    bge_embedder = BGEEmbedder(embedding_config)
    openai_embedder = OpenAIEmbedder(embedding_config) if with_openai else None

    # v2: Initialize both storage clients
    supabase_client = get_supabase_client() if save_to_db and not dry_run else None
    qdrant_client = get_qdrant_client() if save_to_db and not dry_run and not supabase_only else None

    if qdrant_client:
        # Ensure collection exists
        try:
            qdrant_client.ensure_collection()
            logger.info("Qdrant collection ready")
        except Exception as e:
            logger.warning(f"Qdrant collection setup failed: {e}")
            qdrant_client = None

    total_chunks_supabase = 0
    total_chunks_qdrant = 0
    embedding_stats = EmbeddingStats()

    for doc in documents:
        logger.info(f"Processing: {doc.arxiv_id} - {doc.title[:50]}...")

        # Step 1: Chunk the document
        try:
            chunks = chunker.chunk_document(doc)
            logger.info(f"  Created {len(chunks)} chunks")

            if not chunks:
                logger.warning(f"  No chunks created for {doc.arxiv_id}")
                continue

        except Exception as e:
            logger.error(f"  Chunking failed: {e}")
            continue

        if dry_run:
            # Preview mode - show chunk info
            for i, chunk in enumerate(chunks[:3]):
                logger.info(f"  Chunk {i}: {chunk.token_count} tokens, {chunk.section_title}")
            if len(chunks) > 3:
                logger.info(f"  ... and {len(chunks) - 3} more chunks")
            continue

        # Step 2: Generate BGE-M3 embeddings
        try:
            start_time = time.time()
            embedded_chunks = bge_embedder.embed_chunks(chunks)
            bge_time = time.time() - start_time

            embedding_stats.bge_embedded += len(embedded_chunks)
            embedding_stats.total_bge_time += bge_time
            logger.info(f"  BGE-M3: {len(embedded_chunks)} chunks in {bge_time:.1f}s")

        except Exception as e:
            logger.error(f"  BGE embedding failed: {e}")
            embedding_stats.bge_failed += len(chunks)
            continue

        # Step 3: Add OpenAI embeddings (optional)
        if with_openai and openai_embedder:
            try:
                start_time = time.time()
                embedded_chunks = openai_embedder.embed_chunks(embedded_chunks)
                openai_time = time.time() - start_time

                embedding_stats.openai_embedded += len(embedded_chunks)
                embedding_stats.total_openai_time += openai_time
                logger.info(f"  OpenAI: {len(embedded_chunks)} chunks in {openai_time:.1f}s")

            except Exception as e:
                logger.error(f"  OpenAI embedding failed: {e}")
                embedding_stats.openai_failed += len(embedded_chunks)

        # Step 4a: Save metadata to Supabase (v2 - no vectors)
        if supabase_client:
            try:
                # v2: Convert to metadata-only format
                metadata_chunks = [ec.to_supabase_dict() for ec in embedded_chunks]

                # Batch insert metadata only
                inserted = supabase_client.batch_insert_chunks_metadata(metadata_chunks)
                total_chunks_supabase += inserted
                logger.info(f"  Supabase: {inserted} chunks (metadata)")

                # Update paper status
                from src.collection.models import PaperStatus
                supabase_client.update_paper_status(doc.arxiv_id, PaperStatus.EMBEDDED)

            except Exception as e:
                logger.error(f"  Supabase save failed: {e}")

        # Step 4b: Save vectors to Qdrant (v2)
        if qdrant_client:
            try:
                # v2: Convert to Qdrant format with vectors
                qdrant_points = [ec.to_qdrant_dict() for ec in embedded_chunks]

                # Batch upsert to Qdrant
                upserted = qdrant_client.upsert_chunks(qdrant_points)
                total_chunks_qdrant += upserted
                logger.info(f"  Qdrant: {upserted} chunks (vectors)")

            except Exception as e:
                logger.error(f"  Qdrant save failed: {e}")

    # Cleanup
    bge_embedder.unload()

    embedding_stats.total_chunks = chunker.stats.total_chunks

    logger.info(f"\nPipeline complete.")
    logger.info(f"  Supabase: {total_chunks_supabase} chunks (metadata)")
    logger.info(f"  Qdrant: {total_chunks_qdrant} chunks (vectors)")

    return chunker.stats, embedding_stats


def main():
    parser = argparse.ArgumentParser(
        description="Embed parsed papers using BGE-M3 and optionally OpenAI"
    )

    # Input options
    parser.add_argument(
        "--arxiv-id",
        type=str,
        help="Process specific paper by arXiv ID",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of papers to process",
    )
    parser.add_argument(
        "--parsed-dir",
        type=str,
        default="data/parsed",
        help="Directory containing parsed JSON files",
    )

    # Embedding options
    parser.add_argument(
        "--with-openai",
        action="store_true",
        help="Also generate OpenAI embeddings for comparison",
    )
    parser.add_argument(
        "--bge-batch-size",
        type=int,
        default=32,
        help="Batch size for BGE-M3 (default: 32)",
    )
    parser.add_argument(
        "--sparse-top-k",
        type=int,
        default=128,
        help="Top-K sparse tokens to keep (default: 128)",
    )

    # Chunking options
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens per chunk (default: 512)",
    )
    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=50,
        help="Overlap tokens between chunks (default: 50)",
    )
    parser.add_argument(
        "--add-paper-context",
        action="store_true",
        help="Prepend paper title/abstract to chunks for better conceptual query retrieval",
    )
    parser.add_argument(
        "--paper-context-tokens",
        type=int,
        default=100,
        help="Max tokens for paper context prefix (default: 100)",
    )

    # Processing options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview only, don't generate embeddings or save",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Don't save to database",
    )
    parser.add_argument(
        "--supabase-only",
        action="store_true",
        help="Save metadata to Supabase only, skip Qdrant (v2)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device for BGE-M3 (default: cuda)",
    )

    # Logging
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)

    logger.info("=" * 60)
    logger.info("arXiv RAG v1 - Embedding Pipeline")
    logger.info("=" * 60)

    # Configuration
    chunking_config = ChunkingConfig(
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
        include_abstract=True,
        add_paper_context=args.add_paper_context,
        paper_context_tokens=args.paper_context_tokens,
    )

    embedding_config = EmbeddingConfig(
        use_bge=True,
        bge_batch_size=args.bge_batch_size,
        sparse_top_k=args.sparse_top_k,
        device=args.device,
        use_openai=args.with_openai,
    )

    logger.info(f"Chunking: max_tokens={args.max_tokens}, overlap={args.overlap_tokens}")
    if args.add_paper_context:
        logger.info(f"Paper context: enabled ({args.paper_context_tokens} tokens)")
    logger.info(f"Embedding: device={args.device}, sparse_top_k={args.sparse_top_k}")

    if args.dry_run:
        logger.info("DRY RUN MODE - no embeddings or saves")

    # Load documents
    parsed_dir = Path(args.parsed_dir)
    if not parsed_dir.exists():
        logger.error(f"Parsed directory not found: {parsed_dir}")
        sys.exit(1)

    documents = load_parsed_documents(
        parsed_dir,
        arxiv_id=args.arxiv_id,
        limit=args.limit,
    )

    if not documents:
        logger.warning("No documents to process")
        sys.exit(0)

    # Run pipeline
    start_time = time.time()

    chunking_stats, embedding_stats = run_embedding_pipeline(
        documents=documents,
        chunking_config=chunking_config,
        embedding_config=embedding_config,
        with_openai=args.with_openai,
        dry_run=args.dry_run,
        save_to_db=not args.no_db,
        supabase_only=args.supabase_only,
    )

    elapsed = time.time() - start_time

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EMBEDDING PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info(chunking_stats.summary())
    logger.info("")
    logger.info(embedding_stats.summary())
    logger.info(f"\nTotal time: {elapsed:.1f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
