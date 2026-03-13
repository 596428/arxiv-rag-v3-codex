#!/usr/bin/env python3
"""
arXiv RAG v1 - Semantic Filtering

Filters papers by semantic similarity to an anchor query representing
ideal LLM research topics.

Usage:
    python scripts/semantic_filter.py --threshold 0.55
    python scripts/semantic_filter.py --input data/collection/all_papers.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedding.bge_embedder import BGEEmbedder
from src.embedding.models import EmbeddingConfig
from src.storage.supabase_client import get_supabase_client
from src.utils.logging import get_logger

logger = get_logger("semantic_filter")

# Anchor query representing ideal LLM research
ANCHOR_QUERY = """
State of the art research in large language models, deep learning architectures,
transformer optimization, training efficiency, model scaling, multimodal models,
vision-language models, alignment techniques, reinforcement learning from human feedback,
reasoning capabilities, chain-of-thought prompting, in-context learning,
retrieval-augmented generation, knowledge distillation, model compression,
quantization, foundation models, instruction tuning, fine-tuning methods.
"""

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "collection"


def compute_similarities(
    papers: list[dict],
    embedder: BGEEmbedder,
    anchor_embedding: list[float],
    batch_size: int = 32,
) -> list[tuple[dict, float]]:
    """
    Compute cosine similarity between papers and anchor.

    Args:
        papers: List of paper dicts with 'abstract' key
        embedder: BGE embedder instance
        anchor_embedding: Pre-computed anchor embedding
        batch_size: Batch size for embedding

    Returns:
        List of (paper, similarity) tuples
    """
    results = []
    anchor_np = np.array(anchor_embedding)
    anchor_norm = np.linalg.norm(anchor_np)

    for i in range(0, len(papers), batch_size):
        batch = papers[i:i + batch_size]
        texts = [
            f"{p.get('title', '')} {p.get('abstract', '')}"
            for p in batch
        ]

        # Get embeddings
        dense_vecs, _, _ = embedder.embed_texts(texts, return_sparse=False)

        for paper, dense_vec in zip(batch, dense_vecs):
            # Cosine similarity
            vec_np = np.array(dense_vec)
            similarity = np.dot(anchor_np, vec_np) / (anchor_norm * np.linalg.norm(vec_np))
            results.append((paper, float(similarity)))

        # Progress
        logger.info(f"Processed {min(i + batch_size, len(papers))}/{len(papers)} papers")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Filter papers by semantic similarity to anchor query"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.55,
        help="Minimum similarity threshold (default: 0.55)"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input JSON file with papers (default: fetch from DB)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output JSON file (default: semantic_filtered.json)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Batch size for embedding (default: 32)"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5000,
        help="Maximum papers to keep (default: 5000)"
    )

    args = parser.parse_args()

    # Load papers
    if args.input:
        with open(args.input, 'r') as f:
            papers = json.load(f)
        logger.info(f"Loaded {len(papers)} papers from {args.input}")
    else:
        # Fetch all papers from database
        supabase = get_supabase_client()
        # Use pagination to fetch all papers (Supabase has 1000 row limit per request)
        all_papers = []
        offset = 0
        batch_size = 1000
        while True:
            result = (
                supabase.client.table("papers")
                .select("arxiv_id, title, abstract, categories, citation_count, published_date")
                .range(offset, offset + batch_size - 1)
                .execute()
            )
            batch = result.data or []
            if not batch:
                break
            all_papers.extend(batch)
            logger.info(f"Fetched {len(all_papers)} papers...")
            offset += batch_size
        papers = all_papers
        logger.info(f"Loaded {len(papers)} papers from database")

    if not papers:
        logger.error("No papers to process")
        return 1

    # Initialize embedder
    logger.info("Loading BGE-M3 embedder...")
    embedder = BGEEmbedder(EmbeddingConfig(use_openai=False))

    # Compute anchor embedding
    logger.info("Computing anchor embedding...")
    anchor_embedding, _, _ = embedder.embed_single(ANCHOR_QUERY)

    # Compute similarities
    logger.info(f"Computing similarities for {len(papers)} papers...")
    results = compute_similarities(papers, embedder, anchor_embedding, args.batch_size)

    # Filter by threshold
    above_threshold = [(p, s) for p, s in results if s >= args.threshold]
    below_threshold = [(p, s) for p, s in results if s < args.threshold]

    logger.info(f"Above threshold ({args.threshold}): {len(above_threshold)}")
    logger.info(f"Below threshold: {len(below_threshold)}")

    # Sort by similarity and take top-k
    above_threshold.sort(key=lambda x: x[1], reverse=True)
    final_papers = above_threshold[:args.top_k]

    # Add similarity scores to papers
    for paper, similarity in final_papers:
        paper["semantic_similarity"] = round(similarity, 4)

    # Unload model
    embedder.unload()

    # Save results
    output_file = args.output or (OUTPUT_DIR / "semantic_filtered.json")
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump([p for p, _ in final_papers], f, indent=2)

    logger.info(f"Saved {len(final_papers)} papers to {output_file}")

    # Summary statistics
    similarities = [s for _, s in final_papers]
    print("\n" + "=" * 50)
    print("SEMANTIC FILTERING SUMMARY")
    print("=" * 50)
    print(f"Input papers:        {len(papers)}")
    print(f"Above threshold:     {len(above_threshold)}")
    print(f"Final (top-{args.top_k}):     {len(final_papers)}")
    print(f"Similarity range:    {min(similarities):.3f} - {max(similarities):.3f}")
    print(f"Mean similarity:     {np.mean(similarities):.3f}")
    print(f"Output file:         {output_file}")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    exit(main())
