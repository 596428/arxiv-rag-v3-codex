#!/usr/bin/env python3
"""
Test search functionality with new embeddings.

Tests:
1. Dense BGE-M3 (1024d) search
2. Dense OpenAI 3-large (3072d) search
3. Sparse BGE search
4. Hybrid (dense + sparse) search
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.qdrant_client import get_qdrant_client
from src.embedding import BGEEmbedder, OpenAIEmbedder, EmbeddingConfig
from src.utils.logging import setup_logging, get_logger

setup_logging(level="INFO")
logger = get_logger("test_search")


def test_dense_bge_search(query: str, top_k: int = 5):
    """Test dense BGE-M3 search."""
    logger.info(f"\n[1] Dense BGE-M3 Search (1024d)")
    logger.info(f"Query: {query}")

    # Embed query
    config = EmbeddingConfig(use_bge=True, device="cuda")
    embedder = BGEEmbedder(config)
    dense_vec, sparse_vec, _ = embedder.embed_single(query)
    embedder.unload()

    # Search
    client = get_qdrant_client()
    results = client.search_dense(dense_vec, vector_name="dense_bge", top_k=top_k)

    logger.info(f"Results ({len(results)} hits):")
    for i, r in enumerate(results):
        logger.info(f"  {i+1}. [{r['paper_id']}] {r['content'][:80]}... (score: {r['similarity']:.4f})")

    return results


def test_dense_3large_search(query: str, top_k: int = 5):
    """Test dense OpenAI 3-large search."""
    logger.info(f"\n[2] Dense OpenAI 3-large Search (3072d)")
    logger.info(f"Query: {query}")

    # Embed query
    config = EmbeddingConfig(
        use_openai=True,
        openai_model="text-embedding-3-large",
        openai_dimensions=3072,
    )
    embedder = OpenAIEmbedder(config)
    query_vec = embedder.embed_single(query)

    # Search
    client = get_qdrant_client()
    results = client.search_dense(query_vec, vector_name="dense_3large", top_k=top_k)

    logger.info(f"Results ({len(results)} hits):")
    for i, r in enumerate(results):
        logger.info(f"  {i+1}. [{r['paper_id']}] {r['content'][:80]}... (score: {r['similarity']:.4f})")

    return results


def test_sparse_search(query: str, top_k: int = 5):
    """Test sparse BGE search."""
    logger.info(f"\n[3] Sparse BGE Search")
    logger.info(f"Query: {query}")

    # Embed query
    config = EmbeddingConfig(use_bge=True, device="cuda")
    embedder = BGEEmbedder(config)
    _, sparse_vec, _ = embedder.embed_single(query)
    embedder.unload()

    # Search
    client = get_qdrant_client()
    results = client.search_sparse(sparse_vec.indices, sparse_vec.values, top_k=top_k)

    logger.info(f"Results ({len(results)} hits):")
    for i, r in enumerate(results):
        logger.info(f"  {i+1}. [{r['paper_id']}] {r['content'][:80]}... (score: {r['score']:.4f})")

    return results


def test_hybrid_search(query: str, top_k: int = 5):
    """Test hybrid (dense + sparse) search with RRF fusion."""
    logger.info(f"\n[4] Hybrid Search (BGE dense + sparse, RRF fusion)")
    logger.info(f"Query: {query}")

    # Embed query
    config = EmbeddingConfig(use_bge=True, device="cuda")
    embedder = BGEEmbedder(config)
    dense_vec, sparse_vec, _ = embedder.embed_single(query)
    embedder.unload()

    # Search
    client = get_qdrant_client()
    results = client.search_hybrid(
        dense_vector=dense_vec,
        sparse_indices=sparse_vec.indices,
        sparse_values=sparse_vec.values,
        vector_name="dense_bge",
        top_k=top_k,
        dense_weight=0.5,
        sparse_weight=0.5,
    )

    logger.info(f"Results ({len(results)} hits):")
    for i, r in enumerate(results):
        dense_score = r.get('dense_score', 0)
        sparse_score = r.get('sparse_score', 0)
        logger.info(f"  {i+1}. [{r['paper_id']}] {r['content'][:80]}...")
        logger.info(f"      RRF: {r['score']:.4f} (dense: {dense_score:.4f}, sparse: {sparse_score:.4f})")

    return results


def main():
    logger.info("=" * 70)
    logger.info("Search Test - BGE-M3 (dense, sparse) + OpenAI 3-large (3072d)")
    logger.info("=" * 70)

    # Test queries
    queries = [
        "How does chain-of-thought prompting improve reasoning in language models?",
        "What are the best techniques for retrieval-augmented generation?",
        "transformer architecture attention mechanism optimization",
    ]

    for query in queries[:1]:  # Test with first query
        test_dense_bge_search(query)
        test_dense_3large_search(query)
        test_sparse_search(query)
        test_hybrid_search(query)

    logger.info("\n" + "=" * 70)
    logger.info("✓ All search tests completed")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
